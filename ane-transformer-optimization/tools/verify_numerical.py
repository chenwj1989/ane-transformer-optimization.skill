#!/usr/bin/env python3
"""
Validate the ANE-refactored PyTorch model against the original PyTorch model.

This is **Step 2** of the production workflow (after refactoring, before CoreML
export): confirm that the refactored model is numerically equivalent to the
original on the same inputs, in fp32.

Supports:
  - Single-tensor or multi-tensor (list / tuple / dict) inputs (passed as *args).
  - Single or multiple outputs (Tensor / list / tuple / dict).
  - JIT-saved (.pt / .ts) models, or eager `module_path:ClassName` specs with a
    custom builder callable.
  - Tolerance gating via PSNR floor and/or absolute-error ceiling.
  - JSON report for CI integration.

Usage:
    # Simple case — single input, single output, two JIT models
    python tools/verify_numerical.py \\
        --original baseline.pt --ane ane.pt --input sample.pt

    # Multi-input model (e.g. encoder-decoder): pass a .pt file holding a
    # list/tuple/dict of tensors. They are unpacked as positional args.
    python tools/verify_numerical.py \\
        --original orig_decoder.pt --ane ane_decoder.pt \\
        --input decoder_inputs.pt \\
        --report report.json --psnr-min 40 --max-err-max 1e-3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────

def compute_metrics(ref: torch.Tensor, cand: torch.Tensor) -> Dict[str, Any]:
    """Per-output metrics: shape, max/mean abs err, MSE, PSNR (dB), SNR (dB), cosine.

    PSNR is computed against the reference peak (signal range); SNR uses
    reference signal power. Both reported in dB. Cosine similarity is computed
    over flattened tensors.
    """
    shape_ok = tuple(ref.shape) == tuple(cand.shape)
    ref_f = ref.detach().float()
    cand_f = cand.detach().float()

    if not shape_ok:
        return {
            "shape_ref": list(ref.shape),
            "shape_cand": list(cand.shape),
            "shape_ok": False,
        }

    diff = (ref_f - cand_f).abs()
    max_err = float(diff.max().item())
    mean_err = float(diff.mean().item())
    mse = float((ref_f - cand_f).pow(2).mean().item())
    peak = float(ref_f.abs().max().item()) ** 2
    sig_pwr = float(ref_f.pow(2).mean().item())

    psnr_db = float(10 * np.log10(peak / mse)) if mse > 0 and peak > 0 else float("inf")
    snr_db = float(10 * np.log10(sig_pwr / mse)) if mse > 0 and sig_pwr > 0 else float("inf")
    cos = float(
        F.cosine_similarity(ref_f.flatten().unsqueeze(0),
                            cand_f.flatten().unsqueeze(0)).item()
    )
    return {
        "shape": list(ref.shape),
        "shape_ok": True,
        "max_err": max_err,
        "mean_err": mean_err,
        "mse": mse,
        "psnr_db": psnr_db,
        "snr_db": snr_db,
        "cos_sim": cos,
    }


# ─────────────────────────────────────────────────────────────────────
# Output flattening (handles Tensor / tuple / list / dict / nested)
# ─────────────────────────────────────────────────────────────────────

def _flatten_outputs(out: Any, prefix: str = "out") -> List[Tuple[str, torch.Tensor]]:
    """Walk an output structure, returning [(label, tensor), ...] pairs."""
    flat: List[Tuple[str, torch.Tensor]] = []
    if isinstance(out, torch.Tensor):
        flat.append((prefix, out))
    elif isinstance(out, dict):
        for k, v in out.items():
            flat.extend(_flatten_outputs(v, f"{prefix}.{k}"))
    elif isinstance(out, (list, tuple)):
        for i, v in enumerate(out):
            flat.extend(_flatten_outputs(v, f"{prefix}[{i}]"))
    else:
        # Non-tensor outputs are skipped silently (e.g. caches we don't compare).
        pass
    return flat


# ─────────────────────────────────────────────────────────────────────
# Input loading
# ─────────────────────────────────────────────────────────────────────

def _load_inputs(path: str) -> List[torch.Tensor]:
    """Load model inputs from a .pt file. Accepts Tensor / list / tuple / dict.

    Dict values are passed positionally in insertion order (keys are kept only
    for documentation purposes).
    """
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, torch.Tensor):
        return [obj]
    if isinstance(obj, (list, tuple)):
        return [t for t in obj]
    if isinstance(obj, dict):
        return [v for v in obj.values()]
    raise TypeError(f"--input {path}: unsupported type {type(obj).__name__} "
                    "(expected Tensor / list / tuple / dict)")


# ─────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────

def _fmt(v: Any, w: int = 11, prec: int = 6) -> str:
    if isinstance(v, float):
        if v == float("inf"):
            return f"{'inf':>{w}s}"
        return f"{v:{w}.{prec}e}" if abs(v) < 1e-2 or abs(v) >= 1e4 else f"{v:{w}.{prec}f}"
    return f"{str(v):>{w}s}"


def print_report(per_output: List[Dict[str, Any]], thresholds: Dict[str, Any],
                 passed: bool) -> None:
    print("\n" + "=" * 88)
    print("  PyTorch Parity Report  (original vs ANE-refactored, fp32)")
    print("=" * 88)
    print(f"  {'Output':<24s} {'Shape':<18s} {'MaxErr':>11s} {'MeanErr':>11s} "
          f"{'PSNR(dB)':>9s} {'SNR(dB)':>9s} {'Cos':>7s}")
    print("  " + "-" * 86)
    for entry in per_output:
        label = entry["label"]
        m = entry["metrics"]
        if not m.get("shape_ok", False):
            print(f"  {label:<24s} SHAPE MISMATCH  ref={m.get('shape_ref')} "
                  f"cand={m.get('shape_cand')}")
            continue
        print(f"  {label:<24s} {str(m['shape']):<18s} "
              f"{_fmt(m['max_err'])} {_fmt(m['mean_err'])} "
              f"{m['psnr_db']:9.2f} {m['snr_db']:9.2f} {m['cos_sim']:7.4f}")
    print("  " + "-" * 86)

    print(f"\n  Thresholds: psnr_min={thresholds.get('psnr_min')}  "
          f"max_err_max={thresholds.get('max_err_max')}")
    tag = "PASS" if passed else "FAIL"
    print(f"  Result:     {tag}")
    print("=" * 88)


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate ANE-refactored PyTorch model vs original PyTorch model.")
    parser.add_argument("--original", required=True,
                        help="Original PyTorch model — JIT-saved .pt / .ts file.")
    parser.add_argument("--ane", required=True,
                        help="ANE-refactored PyTorch model — JIT-saved .pt / .ts file.")
    parser.add_argument("--input", required=True,
                        help="Sample input(s) as .pt: Tensor, or list/tuple/dict of tensors. "
                             "Dict / list values are unpacked positionally.")
    parser.add_argument("--report", default=None,
                        help="Write JSON report to this path.")
    parser.add_argument("--atol", type=float, default=None,
                        help="(legacy) Absolute tolerance for backward compat — if set, also "
                             "enforces torch.allclose(atol=...) on the first output.")
    parser.add_argument("--psnr-min", type=float, default=None,
                        help="Fail if any output PSNR (dB) falls below this.")
    parser.add_argument("--max-err-max", type=float, default=None,
                        help="Fail if any output max absolute error exceeds this.")
    args = parser.parse_args()

    orig = torch.jit.load(args.original).eval()
    ane = torch.jit.load(args.ane).eval()
    inputs = _load_inputs(args.input)

    with torch.no_grad():
        out_o = orig(*inputs)
        out_a = ane(*inputs)

    pairs_o = _flatten_outputs(out_o, prefix="out")
    pairs_a = _flatten_outputs(out_a, prefix="out")

    if len(pairs_o) != len(pairs_a):
        print(f"ERROR: output structure differs — original has {len(pairs_o)} tensor "
              f"output(s), ANE has {len(pairs_a)}.", file=sys.stderr)
        return 2

    per_output: List[Dict[str, Any]] = []
    for (lo, to), (la, ta) in zip(pairs_o, pairs_a):
        if lo != la:
            label = f"{lo} / {la}"
        else:
            label = lo
        per_output.append({"label": label, "metrics": compute_metrics(to, ta)})

    failures: List[str] = []
    for entry in per_output:
        m = entry["metrics"]
        if not m.get("shape_ok", False):
            failures.append(f"{entry['label']}: shape mismatch")
            continue
        if args.psnr_min is not None and m["psnr_db"] < args.psnr_min:
            failures.append(f"{entry['label']}: PSNR {m['psnr_db']:.2f} dB "
                            f"< {args.psnr_min:.2f} dB")
        if args.max_err_max is not None and m["max_err"] > args.max_err_max:
            failures.append(f"{entry['label']}: max_err {m['max_err']:.3e} "
                            f"> {args.max_err_max:.3e}")

    if args.atol is not None and pairs_o:
        if not torch.allclose(pairs_o[0][1].float(), pairs_a[0][1].float(),
                              atol=args.atol):
            failures.append(f"{pairs_o[0][0]}: torch.allclose(atol={args.atol}) failed")

    passed = len(failures) == 0
    thresholds = {"psnr_min": args.psnr_min, "max_err_max": args.max_err_max,
                  "atol": args.atol}

    print_report(per_output, thresholds, passed)
    if failures:
        print("\nFailures:")
        for f in failures:
            print(f"  - {f}")

    if args.report:
        report = {
            "original": str(Path(args.original).resolve()),
            "ane": str(Path(args.ane).resolve()),
            "input": str(Path(args.input).resolve()),
            "thresholds": thresholds,
            "outputs": per_output,
            "passed": passed,
            "failures": failures,
        }
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(report, indent=2))
        print(f"\nWrote JSON report: {args.report}")

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
