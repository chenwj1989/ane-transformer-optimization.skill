#!/usr/bin/env python3
"""
End-to-end CoreML report: accuracy and latency for baseline vs ANE CoreML
packages, optionally measured against a PyTorch reference and on real inputs.

This is **Step 4** of the production workflow (Steps 1–3 produce: a lint pass,
a refactored PyTorch model with PyTorch parity verified, and two `.mlpackage`
files — baseline and ANE). This tool consumes those artifacts and produces a
single report covering all four quadrants the user cares about:

  Accuracy:
    - baseline CoreML  vs  PyTorch reference   (sanity of the conversion)
    - ANE CoreML       vs  PyTorch reference   (sanity of the ANE refactor)
  Latency:
    - baseline CoreML  on CPU_AND_NE
    - ANE CoreML       on CPU_AND_NE
    - speedup (baseline / ANE)
  Real-input run (optional):
    - PyTorch / baseline / ANE outputs on the same real input, with metrics
      between each CoreML output and the PyTorch reference.

Modes:
  ml — compare two already-exported .mlpackage files (most production case).
  py — build + trace + export from PyTorch class specs, then compare. Only
       suitable for models whose constructor matches `ModelCls(**dims)`. For
       constructor patterns that take a single config object (e.g. Whisper's
       `ModelDimensions`) write a thin wrapper module and use `ml` mode.

Usage:
    # Common case — compare two .mlpackage files against a PyTorch reference
    python tools/verify_coreml.py ml \\
        --original-ml baseline.mlpackage \\
        --ane-ml      ane.mlpackage \\
        --inputs-pt   inputs.pt \\
        --ref-pt      ref_outputs.pt \\
        --report      report.json \\
        --md          report.md \\
        --warmup 5 --iters 50

    # Random inputs, baseline-as-reference (legacy behaviour)
    python tools/verify_coreml.py ml \\
        --original-ml baseline.mlpackage --ane-ml ane.mlpackage \\
        --input '{"mel":[1,80,3000]}'

`--inputs-pt` / `--ref-pt` formats: a single `.pt` file containing a dict
mapping name → tensor (CoreML input name → array, output name → tensor).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

try:
    import coremltools as ct
    HAS_COREMLTOOLS = True
except ImportError:
    HAS_COREMLTOOLS = False


# ═════════════════════════════════════════════════════════════════════
# Metrics
# ═════════════════════════════════════════════════════════════════════

def compute_metrics(ref: torch.Tensor, cand: torch.Tensor) -> Dict[str, Any]:
    """Per-output: shape, max/mean err, MSE, PSNR (dB), SNR (dB), cosine sim."""
    if tuple(ref.shape) != tuple(cand.shape):
        return {"shape_ref": list(ref.shape), "shape_cand": list(cand.shape),
                "shape_ok": False}
    r = ref.detach().float()
    c = cand.detach().float()
    diff = (r - c).abs()
    mse = float((r - c).pow(2).mean().item())
    peak = float(r.abs().max().item()) ** 2
    sig_pwr = float(r.pow(2).mean().item())
    psnr = float(10 * np.log10(peak / mse)) if mse > 0 and peak > 0 else float("inf")
    snr = float(10 * np.log10(sig_pwr / mse)) if mse > 0 and sig_pwr > 0 else float("inf")
    cos = float(F.cosine_similarity(r.flatten().unsqueeze(0),
                                    c.flatten().unsqueeze(0)).item())
    return {
        "shape": list(ref.shape),
        "shape_ok": True,
        "max_err": float(diff.max().item()),
        "mean_err": float(diff.mean().item()),
        "mse": mse,
        "psnr_db": psnr,
        "snr_db": snr,
        "cos_sim": cos,
    }


# ═════════════════════════════════════════════════════════════════════
# Latency benchmark (CoreML CPU_AND_NE)
# ═════════════════════════════════════════════════════════════════════

def benchmark(mlmodel, inputs: Dict[str, np.ndarray],
              warmup: int = 5, iters: int = 50) -> Dict[str, float]:
    for _ in range(warmup):
        mlmodel.predict(inputs)
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        mlmodel.predict(inputs)
        times.append((time.perf_counter() - t0) * 1000)
    arr = np.array(times)
    return {
        "avg_ms": float(arr.mean()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "min_ms": float(arr.min()),
        "iters": int(iters),
        "warmup": int(warmup),
    }


# ═════════════════════════════════════════════════════════════════════
# Input / reference loading
# ═════════════════════════════════════════════════════════════════════

def _to_numpy(t: Any) -> np.ndarray:
    if isinstance(t, np.ndarray):
        return t
    if isinstance(t, torch.Tensor):
        x = t.detach().cpu().numpy()
        # Token-like inputs may be int64; caller can override via the .pt file.
        return x
    raise TypeError(f"cannot convert {type(t).__name__} to numpy")


def _cast_int_to_int32(d: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """coremltools rejects int64 — cast token / index inputs to int32."""
    out = {}
    for k, v in d.items():
        if isinstance(v, np.ndarray) and v.dtype == np.int64:
            out[k] = v.astype(np.int32)
        else:
            out[k] = v
    return out


def _load_inputs_pt(path: str) -> Dict[str, np.ndarray]:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(obj, dict):
        raise TypeError(f"--inputs-pt {path}: expected dict[name → tensor], "
                        f"got {type(obj).__name__}")
    return _cast_int_to_int32({k: _to_numpy(v) for k, v in obj.items()})


def _load_ref_pt(path: str) -> Dict[str, torch.Tensor]:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict):
        return {k: (v if isinstance(v, torch.Tensor) else torch.as_tensor(v))
                for k, v in obj.items()}
    if isinstance(obj, torch.Tensor):
        # Single-output reference: name it "output_0"
        return {"output_0": obj}
    raise TypeError(f"--ref-pt {path}: expected Tensor or dict, "
                    f"got {type(obj).__name__}")


def _ml_output_dict(raw: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    return {k: torch.as_tensor(v) for k, v in raw.items()}


def _align_output_names(ml_out: Dict[str, torch.Tensor],
                        ref: Dict[str, torch.Tensor]) -> List[Tuple[str, torch.Tensor, torch.Tensor]]:
    """Pair ML outputs with reference outputs. Match by name first, fall back
    to positional pairing when the spec output names don't match (common when
    converters rename outputs to `var_*`)."""
    pairs: List[Tuple[str, torch.Tensor, torch.Tensor]] = []
    used = set()
    for k, ref_t in ref.items():
        if k in ml_out:
            pairs.append((k, ref_t, ml_out[k]))
            used.add(k)
    if pairs:
        return pairs
    # Positional fallback — same length expected.
    ml_keys = list(ml_out.keys())
    ref_keys = list(ref.keys())
    if len(ml_keys) != len(ref_keys):
        raise ValueError(f"Reference has {len(ref_keys)} output(s) but CoreML "
                         f"produced {len(ml_keys)} (names {ml_keys}). "
                         "Rename ref keys to match CoreML output names.")
    for rk, mk in zip(ref_keys, ml_keys):
        pairs.append((f"{rk}↔{mk}", ref[rk], ml_out[mk]))
    return pairs


# ═════════════════════════════════════════════════════════════════════
# Core: compare two CoreML models against a (PyTorch) reference
# ═════════════════════════════════════════════════════════════════════

def compare_coreml_models(orig_ml_path: str,
                          ane_ml_path: str,
                          inputs: Dict[str, np.ndarray],
                          ref_outputs: Optional[Dict[str, torch.Tensor]] = None,
                          warmup: int = 5,
                          iters: int = 50,
                          real_inputs: Optional[Dict[str, np.ndarray]] = None,
                          real_ref: Optional[Dict[str, torch.Tensor]] = None,
                          ) -> Dict[str, Any]:
    """Returns:
        {
          "accuracy_vs_ref": { "<output>": { "baseline": {...}, "ane": {...} } },
          "latency":         { "baseline": {...}, "ane": {...}, "speedup": float },
          "real_input":      { ... }  # optional
        }
    """
    if not HAS_COREMLTOOLS:
        print("ERROR: coremltools not installed. `pip install coremltools`.",
              file=sys.stderr)
        sys.exit(2)

    cu = ct.ComputeUnit.CPU_AND_NE
    ml_orig = ct.models.MLModel(orig_ml_path, compute_units=cu)
    ml_ane = ct.models.MLModel(ane_ml_path, compute_units=cu)

    out_o = _ml_output_dict(ml_orig.predict(inputs))
    out_a = _ml_output_dict(ml_ane.predict(inputs))

    # ── Accuracy ──
    if ref_outputs is None:
        ref_outputs = out_o  # legacy: use baseline CoreML as reference

    pairs_b = _align_output_names(out_o, ref_outputs)
    pairs_a = _align_output_names(out_a, ref_outputs)

    accuracy: Dict[str, Dict[str, Any]] = {}
    for (kb, rb, cb), (ka, ra, ca) in zip(pairs_b, pairs_a):
        label = kb if kb == ka else f"{kb} / {ka}"
        accuracy[label] = {
            "baseline_vs_ref": compute_metrics(rb, cb),
            "ane_vs_ref":      compute_metrics(ra, ca),
        }

    # ── Latency ──
    lat_o = benchmark(ml_orig, inputs, warmup, iters)
    lat_a = benchmark(ml_ane, inputs, warmup, iters)
    speedup = lat_o["avg_ms"] / lat_a["avg_ms"] if lat_a["avg_ms"] > 0 else 0.0

    result: Dict[str, Any] = {
        "accuracy_vs_ref": accuracy,
        "latency": {
            "baseline": lat_o,
            "ane": lat_a,
            "speedup_baseline_over_ane": speedup,
            "compute_unit": "CPU_AND_NE",
        },
        "inputs": {k: list(v.shape) for k, v in inputs.items()},
        "ref_source": "PyTorch" if ref_outputs is not None and id(ref_outputs) != id(out_o)
                      else "baseline CoreML (no --ref-pt)",
    }

    # ── Real-input run (optional) ──
    if real_inputs is not None:
        ro = _ml_output_dict(ml_orig.predict(real_inputs))
        ra = _ml_output_dict(ml_ane.predict(real_inputs))
        real_block: Dict[str, Any] = {
            "inputs": {k: list(v.shape) for k, v in real_inputs.items()},
            "outputs": {
                "baseline_coreml": {k: list(v.shape) for k, v in ro.items()},
                "ane_coreml":      {k: list(v.shape) for k, v in ra.items()},
            },
        }
        if real_ref is not None:
            pairs_rb = _align_output_names(ro, real_ref)
            pairs_ra = _align_output_names(ra, real_ref)
            real_block["accuracy_vs_pytorch"] = {}
            for (kb, rb, cb), (ka, ra_t, ca) in zip(pairs_rb, pairs_ra):
                label = kb if kb == ka else f"{kb} / {ka}"
                real_block["accuracy_vs_pytorch"][label] = {
                    "baseline_vs_ref": compute_metrics(rb, cb),
                    "ane_vs_ref":      compute_metrics(ra_t, ca),
                }
        result["real_input"] = real_block

    return result


# ═════════════════════════════════════════════════════════════════════
# Reporting (table + markdown)
# ═════════════════════════════════════════════════════════════════════

def _fmt_metrics(m: Dict[str, Any]) -> str:
    if not m.get("shape_ok", False):
        return (f"SHAPE MISMATCH ref={m.get('shape_ref')} "
                f"cand={m.get('shape_cand')}")
    return (f"max={m['max_err']:.3e}  mean={m['mean_err']:.3e}  "
            f"PSNR={m['psnr_db']:6.2f}dB  SNR={m['snr_db']:6.2f}dB  "
            f"cos={m['cos_sim']:.4f}")


def print_report(res: Dict[str, Any]) -> None:
    print("\n" + "=" * 88)
    print("  CoreML Production Report")
    print("=" * 88)
    print(f"  Reference: {res.get('ref_source', '?')}")
    print(f"  Inputs:    {res['inputs']}")

    print("\n── Accuracy (CoreML vs reference) ──")
    for out_name, m in res["accuracy_vs_ref"].items():
        print(f"  Output: {out_name}")
        print(f"    baseline : {_fmt_metrics(m['baseline_vs_ref'])}")
        print(f"    ANE      : {_fmt_metrics(m['ane_vs_ref'])}")

    print("\n── Latency on CPU_AND_NE ──")
    lat = res["latency"]
    print(f"  {'Model':<10s} {'Avg (ms)':>10s} {'P50':>10s} {'P95':>10s} {'Min':>10s}")
    for k, label in [("baseline", "Baseline"), ("ane", "ANE")]:
        L = lat[k]
        print(f"  {label:<10s} {L['avg_ms']:10.2f} {L['p50_ms']:10.2f} "
              f"{L['p95_ms']:10.2f} {L['min_ms']:10.2f}")
    s = lat["speedup_baseline_over_ane"]
    tag = "ANE faster" if s > 1.05 else ("Tie" if s > 0.95 else "Baseline faster")
    print(f"  Speedup (baseline / ANE): {s:.2f}x  [{tag}]")

    if "real_input" in res:
        ri = res["real_input"]
        print("\n── Real-input run ──")
        print(f"  Inputs:  {ri['inputs']}")
        print(f"  Baseline outputs: {ri['outputs']['baseline_coreml']}")
        print(f"  ANE      outputs: {ri['outputs']['ane_coreml']}")
        if "accuracy_vs_pytorch" in ri:
            for out_name, m in ri["accuracy_vs_pytorch"].items():
                print(f"  Output: {out_name}")
                print(f"    baseline vs PyTorch : {_fmt_metrics(m['baseline_vs_ref'])}")
                print(f"    ANE      vs PyTorch : {_fmt_metrics(m['ane_vs_ref'])}")
    print("=" * 88)


def write_markdown(res: Dict[str, Any], path: str,
                   orig_ml: str, ane_ml: str) -> None:
    lines: List[str] = []
    lines.append("# CoreML Production Report\n")
    lines.append(f"- **Baseline `.mlpackage`:** `{orig_ml}`")
    lines.append(f"- **ANE `.mlpackage`:** `{ane_ml}`")
    lines.append(f"- **Reference:** {res.get('ref_source', '?')}")
    lines.append(f"- **Inputs:** `{res['inputs']}`\n")

    lines.append("## Accuracy (CoreML vs reference)\n")
    lines.append("| Output | Variant | Shape | MaxErr | MeanErr | PSNR (dB) | SNR (dB) | Cos |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|")
    for out_name, m in res["accuracy_vs_ref"].items():
        for variant, key in [("Baseline", "baseline_vs_ref"), ("ANE", "ane_vs_ref")]:
            x = m[key]
            if not x.get("shape_ok", False):
                lines.append(f"| `{out_name}` | {variant} | shape mismatch "
                             f"ref={x.get('shape_ref')} vs cand={x.get('shape_cand')} "
                             "| - | - | - | - | - |")
                continue
            lines.append(f"| `{out_name}` | {variant} | `{x['shape']}` | "
                         f"{x['max_err']:.3e} | {x['mean_err']:.3e} | "
                         f"{x['psnr_db']:.2f} | {x['snr_db']:.2f} | {x['cos_sim']:.4f} |")

    lines.append("\n## Latency (CPU_AND_NE)\n")
    lines.append("| Variant | Avg (ms) | P50 | P95 | Min | Iters |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for k, label in [("baseline", "Baseline"), ("ane", "ANE")]:
        L = res["latency"][k]
        lines.append(f"| {label} | {L['avg_ms']:.2f} | {L['p50_ms']:.2f} | "
                     f"{L['p95_ms']:.2f} | {L['min_ms']:.2f} | {L['iters']} |")
    s = res["latency"]["speedup_baseline_over_ane"]
    lines.append(f"\n**Speedup (baseline / ANE): {s:.2f}x**\n")

    if "real_input" in res:
        ri = res["real_input"]
        lines.append("## Real-input Run\n")
        lines.append(f"- Inputs: `{ri['inputs']}`")
        lines.append(f"- Baseline output shapes: `{ri['outputs']['baseline_coreml']}`")
        lines.append(f"- ANE output shapes: `{ri['outputs']['ane_coreml']}`\n")
        if "accuracy_vs_pytorch" in ri:
            lines.append("### Accuracy on real input (vs PyTorch)\n")
            lines.append("| Output | Variant | MaxErr | PSNR (dB) | SNR (dB) | Cos |")
            lines.append("|---|---|---:|---:|---:|---:|")
            for out_name, m in ri["accuracy_vs_pytorch"].items():
                for variant, key in [("Baseline", "baseline_vs_ref"), ("ANE", "ane_vs_ref")]:
                    x = m[key]
                    if not x.get("shape_ok", False):
                        continue
                    lines.append(f"| `{out_name}` | {variant} | "
                                 f"{x['max_err']:.3e} | {x['psnr_db']:.2f} | "
                                 f"{x['snr_db']:.2f} | {x['cos_sim']:.4f} |")

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n")


# ═════════════════════════════════════════════════════════════════════
# `py` mode — kept for back-compat, suitable for `Model(**dims)` ctors
# ═════════════════════════════════════════════════════════════════════

def compare_pytorch_models(original_module_spec: str,
                           ane_module_spec: str,
                           dims: dict,
                           mel_shape: list,
                           token_shape: Optional[list],
                           warmup: int = 5, iters: int = 50,
                           out_dir: str = "/tmp/verify_coreml") -> Dict[str, Any]:
    if not HAS_COREMLTOOLS:
        print("ERROR: coremltools not installed.", file=sys.stderr)
        sys.exit(2)

    def load_class(filepath, classname, module_name):
        spec = importlib.util.spec_from_file_location(module_name, filepath)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = mod
        spec.loader.exec_module(mod)
        return getattr(mod, classname)

    orig_path, orig_cls = original_module_spec.split(":")
    ane_path, ane_cls = ane_module_spec.split(":")
    OrigCls = load_class(orig_path, orig_cls, "_orig_model")
    AneCls = load_class(ane_path, ane_cls, "_ane_model")

    torch.manual_seed(42)
    orig_model = OrigCls(**dims).eval()
    ane_model = AneCls(**dims).eval()

    mel = torch.randn(mel_shape)
    tokens = torch.randint(0, 100, token_shape) if token_shape else None
    inputs = (mel,) if tokens is None else (mel, tokens)

    with torch.no_grad():
        ref_out = orig_model(*inputs)

    with torch.no_grad():
        traced_orig = torch.jit.trace(orig_model, inputs)
        traced_ane = torch.jit.trace(ane_model, inputs)

    os.makedirs(out_dir, exist_ok=True)
    input_types = [ct.TensorType(name="mel", shape=mel_shape)]
    if token_shape:
        input_types.append(ct.TensorType(name="tokens", shape=token_shape,
                                         dtype=np.int32))

    ml_o = ct.convert(traced_orig, convert_to="mlprogram", inputs=input_types,
                      minimum_deployment_target=ct.target.macOS14,
                      compute_precision=ct.precision.FLOAT16,
                      compute_units=ct.ComputeUnit.ALL)
    orig_path_ml = f"{out_dir}/original.mlpackage"
    ml_o.save(orig_path_ml)

    ml_a = ct.convert(traced_ane, convert_to="mlprogram", inputs=input_types,
                      minimum_deployment_target=ct.target.macOS14,
                      compute_precision=ct.precision.FLOAT16,
                      compute_units=ct.ComputeUnit.ALL)
    ane_path_ml = f"{out_dir}/ane.mlpackage"
    ml_a.save(ane_path_ml)

    ml_inputs: Dict[str, np.ndarray] = {"mel": mel.numpy()}
    if token_shape:
        ml_inputs["tokens"] = tokens.numpy().astype(np.int32)

    ref_dict = {"output_0": ref_out if isinstance(ref_out, torch.Tensor) else ref_out[0]}
    return compare_coreml_models(orig_path_ml, ane_path_ml, ml_inputs,
                                 ref_outputs=ref_dict,
                                 warmup=warmup, iters=iters)


# ═════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════

def main() -> int:
    parser = argparse.ArgumentParser(
        description="CoreML production report: baseline vs ANE, with optional "
                    "PyTorch reference and real-input run.")
    sub = parser.add_subparsers(dest="mode", required=True)

    p_ml = sub.add_parser("ml", help="Compare two .mlpackage files.")
    p_ml.add_argument("--original-ml", required=True)
    p_ml.add_argument("--ane-ml", required=True)
    p_ml.add_argument("--input", default=None,
                      help='JSON: {"name": shape_list}. Random fp32 inputs (or '
                           'int32 if name contains "tok"/"id"/"int"/"pos"). '
                           'Mutually exclusive with --inputs-pt.')
    p_ml.add_argument("--inputs-pt", default=None,
                      help="Path to .pt file containing dict[name → tensor]. "
                           "int64 entries are auto-cast to int32 for CoreML.")
    p_ml.add_argument("--ref-pt", default=None,
                      help="Path to .pt file containing PyTorch reference "
                           "outputs (dict[name → Tensor] or single Tensor).")
    p_ml.add_argument("--real-inputs-pt", default=None,
                      help="Optional real input .pt for a representative sample.")
    p_ml.add_argument("--real-ref-pt", default=None,
                      help="Optional PyTorch reference outputs for the real input.")
    p_ml.add_argument("--report", default=None, help="Write JSON report.")
    p_ml.add_argument("--md", default=None, help="Write Markdown summary.")
    p_ml.add_argument("--warmup", type=int, default=5)
    p_ml.add_argument("--iters", type=int, default=50)
    p_ml.add_argument("--psnr-min", type=float, default=None,
                      help="Fail if any ANE output PSNR (dB) falls below this.")
    p_ml.add_argument("--max-err-max", type=float, default=None,
                      help="Fail if any ANE output max absolute error exceeds this.")
    p_ml.add_argument("--min-speedup", type=float, default=None,
                      help="Fail if baseline/ANE latency speedup < this.")

    p_py = sub.add_parser("py", help="Build+trace+export+compare from PyTorch.")
    p_py.add_argument("--original-model", required=True)
    p_py.add_argument("--ane-model", required=True)
    p_py.add_argument("--dims", required=True, help='JSON model dimensions (passed as **dims).')
    p_py.add_argument("--mel-shape", required=True)
    p_py.add_argument("--token-shape", default=None)
    p_py.add_argument("--warmup", type=int, default=5)
    p_py.add_argument("--iters", type=int, default=50)
    p_py.add_argument("-o", "--out-dir", default="/tmp/verify_coreml")
    p_py.add_argument("--report", default=None)
    p_py.add_argument("--md", default=None)

    args = parser.parse_args()

    if args.mode == "ml":
        if args.input and args.inputs_pt:
            print("ERROR: pass only one of --input / --inputs-pt", file=sys.stderr)
            return 2
        if args.inputs_pt:
            inputs = _load_inputs_pt(args.inputs_pt)
        elif args.input:
            inputs = {}
            for name, shape in json.loads(args.input).items():
                lower = name.lower()
                if any(s in lower for s in ("id", "tok", "int", "pos")):
                    inputs[name] = np.random.randint(0, 100, size=shape).astype(np.int32)
                else:
                    inputs[name] = np.random.randn(*shape).astype(np.float32)
        else:
            print("ERROR: provide either --input or --inputs-pt", file=sys.stderr)
            return 2

        ref_outputs = _load_ref_pt(args.ref_pt) if args.ref_pt else None
        real_inputs = _load_inputs_pt(args.real_inputs_pt) if args.real_inputs_pt else None
        real_ref = _load_ref_pt(args.real_ref_pt) if args.real_ref_pt else None

        res = compare_coreml_models(args.original_ml, args.ane_ml, inputs,
                                    ref_outputs=ref_outputs,
                                    warmup=args.warmup, iters=args.iters,
                                    real_inputs=real_inputs,
                                    real_ref=real_ref)
        print_report(res)

        if args.md:
            write_markdown(res, args.md, args.original_ml, args.ane_ml)
            print(f"\nWrote Markdown: {args.md}")
        if args.report:
            Path(args.report).parent.mkdir(parents=True, exist_ok=True)
            Path(args.report).write_text(json.dumps(res, indent=2))
            print(f"Wrote JSON: {args.report}")

        # Gating
        failures: List[str] = []
        for out_name, m in res["accuracy_vs_ref"].items():
            ane = m["ane_vs_ref"]
            if not ane.get("shape_ok", False):
                failures.append(f"{out_name}: ANE shape mismatch")
                continue
            if args.psnr_min is not None and ane["psnr_db"] < args.psnr_min:
                failures.append(f"{out_name}: ANE PSNR {ane['psnr_db']:.2f}dB "
                                f"< {args.psnr_min:.2f}dB")
            if args.max_err_max is not None and ane["max_err"] > args.max_err_max:
                failures.append(f"{out_name}: ANE max_err {ane['max_err']:.3e} "
                                f"> {args.max_err_max:.3e}")
        if args.min_speedup is not None:
            s = res["latency"]["speedup_baseline_over_ane"]
            if s < args.min_speedup:
                failures.append(f"speedup {s:.2f}x < {args.min_speedup:.2f}x")
        if failures:
            print("\nFailures:")
            for f in failures:
                print(f"  - {f}")
            return 1
        return 0

    # py mode
    dims = json.loads(args.dims)
    mel_shape = json.loads(args.mel_shape)
    token_shape = json.loads(args.token_shape) if args.token_shape else None
    res = compare_pytorch_models(args.original_model, args.ane_model, dims,
                                 mel_shape, token_shape,
                                 warmup=args.warmup, iters=args.iters,
                                 out_dir=args.out_dir)
    print_report(res)
    if args.md:
        write_markdown(res, args.md, f"{args.out_dir}/original.mlpackage",
                       f"{args.out_dir}/ane.mlpackage")
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
