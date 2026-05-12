#!/usr/bin/env python3
"""
Export baseline (original OpenAI Whisper) and ANE encoder/decoder CoreML packages,
measure accuracy vs original PyTorch Whisper outputs, benchmark latency, and (when
`whisper/tests/jfk.flac` exists) run `transcribe` on that real clip for original PyTorch,
baseline CoreML, and ANE CoreML — results go under `real_audio_transcription` in
`benchmark_results.json`.

Accuracy: each CoreML output is compared to the reference from the original
`whisper.model` PyTorch model (encoder / decoder logits), fp32 tensors.

Latency: CoreML predict() with models loaded using compute_units=CPU_AND_NE.

Run:
  cd examples/whisper-ane && .venv/bin/python benchmark_coreml.py -o .output/coreml_benchmark
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "whisper"))

import whisper  # noqa: E402
from whisper.model import disable_sdpa  # noqa: E402

from coreml_whisper_adapter import CoreMLWhisper, dec_io_triplet, enc_io_pair  # noqa: E402
from model_ane import (  # noqa: E402
    WhisperANE,
    WhisperDecoderCoreMLWrapper,
    WhisperEncoderCoreMLWrapper,
    trace_with_roundtrip,
)


class OriginalWhisperEncoderWrapper(nn.Module):
    """TorchScript/CoreML export: mel -> encoder output."""

    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        return self.encoder(mel)


class OriginalWhisperDecoderWrapper(nn.Module):
    """TorchScript/CoreML export: int32 tokens + encoder_out -> logits."""

    def __init__(self, decoder: nn.Module):
        super().__init__()
        self.decoder = decoder

    def forward(self, tokens: torch.Tensor, encoder_out: torch.Tensor) -> torch.Tensor:
        return self.decoder(tokens.to(torch.long), encoder_out)


def _metrics(ref: torch.Tensor, cand: torch.Tensor):
    diff = (ref.float() - cand.float()).abs()
    max_err = float(diff.max().item())
    mean_err = float(diff.mean().item())
    mse = float((ref.float() - cand.float()).pow(2).mean().item())
    peak = float(ref.abs().max().item()) ** 2
    psnr_db = float(10 * np.log10(peak / mse)) if mse > 0 else float("inf")
    return {"max_err": max_err, "mean_err": mean_err, "psnr_db": psnr_db}


def _benchmark(ml_model, inputs: dict, warmup: int, iters: int):
    for _ in range(warmup):
        ml_model.predict(inputs)
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        ml_model.predict(inputs)
        times.append((time.perf_counter() - t0) * 1000.0)
    arr = np.array(times)
    return {
        "avg_ms": float(arr.mean()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "min_ms": float(arr.min()),
    }


def _convert(ts_module, inputs_spec, prec, out_path: str):
    import coremltools as ct

    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    ml = ct.convert(
        ts_module,
        convert_to="mlprogram",
        inputs=inputs_spec,
        minimum_deployment_target=ct.target.macOS14,
        compute_precision=prec,
        compute_units=ct.ComputeUnit.ALL,
    )
    ml.save(out_path)


def _export_flex_decoders(
    ct,
    prec,
    out_subdir: str,
    w,
    ane,
    base_dec_w: nn.Module,
    ane_dec_w: nn.Module,
    n_text_ctx: int,
    n_audio_ctx: int,
    n_audio_state: int,
):
    """Export baseline + ANE decoders with RangeDim(1, n_text_ctx) for full `transcribe`."""
    from coremltools.converters.mil.input_types import RangeDim

    os.makedirs(out_subdir, exist_ok=True)
    mel_trace = torch.randn(1, w.dims.n_mels, 3000)
    tok_full = torch.randint(0, min(5000, w.dims.n_vocab), (1, n_text_ctx), dtype=torch.int32)
    with torch.no_grad(), disable_sdpa():
        xa_b = w.encoder(mel_trace.to(w.device)).cpu().float()
    with torch.no_grad():
        xa_a = ane.encoder(mel_trace.to(next(ane.parameters()).device)).cpu().float()

    with torch.no_grad(), disable_sdpa():
        ts_bd_f = trace_with_roundtrip(base_dec_w, (tok_full, xa_b))
    with torch.no_grad():
        ts_ad_f = trace_with_roundtrip(ane_dec_w, (tok_full, xa_a))

    inputs_flex = [
        ct.TensorType(name="tokens", shape=(1, RangeDim(1, n_text_ctx)), dtype=np.int32),
        ct.TensorType(name="encoder_out", shape=(1, n_audio_ctx, n_audio_state)),
    ]
    p_base = os.path.join(out_subdir, "BaselineWhisperDecoder_flex.mlpackage")
    p_ane = os.path.join(out_subdir, "ANEWhisperDecoder_flex.mlpackage")
    _convert(ts_bd_f, inputs_flex, prec, p_base)
    _convert(ts_ad_f, inputs_flex, prec, p_ane)
    return {"baseline_decoder_flex": p_base, "ane_decoder_flex": p_ane}


def _norm_text(s: str) -> str:
    return " ".join(s.lower().split())


def run_real_audio_transcription(
    ct,
    prec,
    cu,
    w,
    ane,
    paths: dict,
    out_dir: str,
    audio_path: Path,
    model_name: str,
) -> dict:
    """Run `transcribe` on real audio: PyTorch original, CoreML baseline, CoreML ANE."""
    w_cpu = w.cpu().eval()
    ane_cpu = ane.cpu().eval()
    sub = os.path.join(out_dir, "transcribe_ml")
    flex_paths = _export_flex_decoders(
        ct,
        prec,
        sub,
        w_cpu,
        ane_cpu,
        OriginalWhisperDecoderWrapper(w_cpu.decoder).cpu().eval(),
        WhisperDecoderCoreMLWrapper(ane_cpu.decoder).cpu().eval(),
        w_cpu.dims.n_text_ctx,
        w_cpu.dims.n_audio_ctx,
        w_cpu.dims.n_audio_state,
    )

    ml_be = ct.models.MLModel(paths["baseline_encoder"], compute_units=cu)
    ml_ae = ct.models.MLModel(paths["ane_encoder"], compute_units=cu)
    ml_bdf = ct.models.MLModel(flex_paths["baseline_decoder_flex"], compute_units=cu)
    ml_adf = ct.models.MLModel(flex_paths["ane_decoder_flex"], compute_units=cu)

    wb = CoreMLWhisper(
        w_cpu, ml_be, ml_bdf, enc_io_pair(ml_be), dec_io_triplet(ml_bdf)
    ).cpu().eval()
    wa = CoreMLWhisper(
        w_cpu, ml_ae, ml_adf, enc_io_pair(ml_ae), dec_io_triplet(ml_adf)
    ).cpu().eval()

    language = "en" if model_name.endswith(".en") else None
    kw = dict(language=language, temperature=0.0, word_timestamps=False, verbose=False)

    with torch.no_grad(), disable_sdpa():
        r_torch = w_cpu.transcribe(str(audio_path), **kw)
    r_base = wb.transcribe(str(audio_path), **kw)
    r_ane = wa.transcribe(str(audio_path), **kw)

    t0, t1, t2 = r_torch["text"], r_base["text"], r_ane["text"]
    n0, n1, n2 = _norm_text(t0), _norm_text(t1), _norm_text(t2)
    phrases = ["my fellow americans", "your country", "do for you"]
    subchk = {p: (p in n0 and p in n1 and p in n2) for p in phrases}

    return {
        "audio_path": str(audio_path.resolve()),
        "transcribe_options": {
            "language": language,
            "temperature": 0.0,
            "word_timestamps": False,
            "note": "word_timestamps off: CoreML path has no PyTorch decoder blocks for alignment",
        },
        "language_detected": {
            "original_torch": r_torch.get("language"),
            "baseline_coreml": r_base.get("language"),
            "ane_coreml": r_ane.get("language"),
        },
        "text": {
            "original_torch": t0,
            "baseline_coreml": t1,
            "ane_coreml": t2,
        },
        "normalized_text_equal": {
            "torch_vs_baseline_coreml": n0 == n1,
            "torch_vs_ane_coreml": n0 == n2,
            "baseline_coreml_vs_ane_coreml": n1 == n2,
        },
        "test_audio_phrases_all_three": subchk,
        "flex_decoder_mlpackages": flex_paths,
    }


def main():
    ap = argparse.ArgumentParser(description="Baseline vs ANE CoreML accuracy & latency")
    ap.add_argument("--model", default="tiny", help="Whisper checkpoint name")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("-o", "--out-dir", default=str(_HERE / ".output/coreml_benchmark"))
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=80)
    ap.add_argument(
        "--fp32-coreml",
        action="store_true",
        help="FLOAT32 CoreML (tighter vs torch reference; slower)",
    )
    ap.add_argument("--mel-frames", type=int, default=3000)
    ap.add_argument("--token-len", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--no-real-audio",
        action="store_true",
        help="Skip jfk.flac-style transcribe validation (CoreML flex decoders).",
    )
    ap.add_argument(
        "--audio",
        default=str(_HERE / "whisper" / "tests" / "jfk.flac"),
        help="Audio for transcribe comparison (same fixture as whisper/tests/test_transcribe.py).",
    )
    args = ap.parse_args()

    try:
        import coremltools as ct
    except ImportError:
        print("Install coremltools: pip install coremltools", file=sys.stderr)
        sys.exit(1)

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    prec = ct.precision.FLOAT32 if args.fp32_coreml else ct.precision.FLOAT16
    prec_name = "fp32" if args.fp32_coreml else "fp16"

    print(f"Loading `{args.model}` …")
    w = whisper.load_model(args.model, device=device).eval()
    ane = WhisperANE.from_whisper(w).to(device).eval()

    mel = torch.randn(1, w.dims.n_mels, args.mel_frames, device=device)
    toks = torch.randint(
        0, min(5000, w.dims.n_vocab), (1, args.token_len), device=device, dtype=torch.long
    )

    with torch.no_grad(), disable_sdpa():
        ref_enc = w.encoder(mel)
        ref_logits = w.decoder(toks, ref_enc)

    mel_cpu = mel.cpu()
    toks_i32 = toks.cpu().to(torch.int32)
    ref_enc_cpu = ref_enc.cpu().float()

    # Trace args: decoder always uses original torch encoder output (apples-to-apples vs ref_logits)
    dec_trace_args = (toks_i32, ref_enc_cpu)

    base_enc_w = OriginalWhisperEncoderWrapper(w.encoder).cpu().eval()
    base_dec_w = OriginalWhisperDecoderWrapper(w.decoder).cpu().eval()
    ane_enc_w = WhisperEncoderCoreMLWrapper(ane.encoder).cpu().eval()
    ane_dec_w = WhisperDecoderCoreMLWrapper(ane.decoder).cpu().eval()

    print("TorchScript trace + round-trip (baseline + ANE) …")
    with torch.no_grad(), disable_sdpa():
        ts_base_enc = trace_with_roundtrip(base_enc_w, (mel_cpu,))
        ts_base_dec = trace_with_roundtrip(base_dec_w, dec_trace_args)
    with torch.no_grad():
        ts_ane_enc = trace_with_roundtrip(ane_enc_w, (mel_cpu,))
        ts_ane_dec = trace_with_roundtrip(ane_dec_w, dec_trace_args)

    mel_shape = list(mel_cpu.shape)
    tok_shape = list(toks_i32.shape)
    enc_shape = list(ref_enc_cpu.shape)

    mel_np = mel_cpu.numpy().astype(np.float32)
    dec_inputs_np = {
        "tokens": toks_i32.numpy().astype(np.int32),
        "encoder_out": ref_enc_cpu.numpy().astype(np.float32),
    }

    paths = {
        "baseline_encoder": os.path.join(args.out_dir, "BaselineWhisperEncoder.mlpackage"),
        "baseline_decoder": os.path.join(args.out_dir, "BaselineWhisperDecoder.mlpackage"),
        "ane_encoder": os.path.join(args.out_dir, "ANEWhisperEncoder.mlpackage"),
        "ane_decoder": os.path.join(args.out_dir, "ANEWhisperDecoder.mlpackage"),
    }

    print(f"CoreML convert ({prec_name}, compute_units=ALL) …")
    _convert(
        ts_base_enc,
        [ct.TensorType(name="mel", shape=mel_shape)],
        prec,
        paths["baseline_encoder"],
    )
    _convert(
        ts_base_dec,
        [
            ct.TensorType(name="tokens", shape=tok_shape, dtype=np.int32),
            ct.TensorType(name="encoder_out", shape=enc_shape),
        ],
        prec,
        paths["baseline_decoder"],
    )
    _convert(
        ts_ane_enc,
        [ct.TensorType(name="mel", shape=mel_shape)],
        prec,
        paths["ane_encoder"],
    )
    _convert(
        ts_ane_dec,
        [
            ct.TensorType(name="tokens", shape=tok_shape, dtype=np.int32),
            ct.TensorType(name="encoder_out", shape=enc_shape),
        ],
        prec,
        paths["ane_decoder"],
    )
    for k, p in paths.items():
        print(f"  {k}: {p}")

    cu = ct.ComputeUnit.CPU_AND_NE
    ml_be = ct.models.MLModel(paths["baseline_encoder"], compute_units=cu)
    ml_bd = ct.models.MLModel(paths["baseline_decoder"], compute_units=cu)
    ml_ae = ct.models.MLModel(paths["ane_encoder"], compute_units=cu)
    ml_ad = ct.models.MLModel(paths["ane_decoder"], compute_units=cu)

    ref_enc_t = ref_enc.cpu().float()
    ref_logits_t = ref_logits.cpu().float()

    def _first_out(d):
        k = list(d.keys())[0]
        return torch.from_numpy(np.array(d[k])).float()

    out_be = _first_out(ml_be.predict({"mel": mel_np}))
    out_ae = _first_out(ml_ae.predict({"mel": mel_np}))
    out_bd = _first_out(ml_bd.predict(dec_inputs_np))
    out_ad = _first_out(ml_ad.predict(dec_inputs_np))

    report = {
        "whisper_checkpoint": args.model,
        "coreml_precision": prec_name,
        "reference": "original_openai_whisper_pytorch_fp32 (disable_sdpa)",
        "mel_shape": mel_shape,
        "decoder_inputs": {"tokens": tok_shape, "encoder_out": enc_shape},
        "accuracy_vs_torch": {
            "encoder": {
                "baseline_coreml": _metrics(ref_enc_t, out_be),
                "ane_coreml": _metrics(ref_enc_t, out_ae),
            },
            "decoder_logits": {
                "baseline_coreml": _metrics(ref_logits_t, out_bd),
                "ane_coreml": _metrics(ref_logits_t, out_ad),
            },
        },
        "latency_ms": {
            "encoder": {
                "baseline": _benchmark(ml_be, {"mel": mel_np}, args.warmup, args.iters),
                "ane": _benchmark(ml_ae, {"mel": mel_np}, args.warmup, args.iters),
            },
            "decoder": {
                "baseline": _benchmark(ml_bd, dec_inputs_np, args.warmup, args.iters),
                "ane": _benchmark(ml_ad, dec_inputs_np, args.warmup, args.iters),
            },
        },
        "mlpackage_paths": paths,
    }

    eb = report["latency_ms"]["encoder"]["baseline"]["avg_ms"]
    ea = report["latency_ms"]["encoder"]["ane"]["avg_ms"]
    db = report["latency_ms"]["decoder"]["baseline"]["avg_ms"]
    da = report["latency_ms"]["decoder"]["ane"]["avg_ms"]
    report["latency_ms"]["encoder"]["speedup_baseline_over_ane"] = eb / ea if ea > 0 else None
    report["latency_ms"]["decoder"]["speedup_baseline_over_ane"] = db / da if da > 0 else None

    def _print_block(title, acc_b, acc_a, lat_b, lat_a, sp):
        print(f"\n── {title} ──")
        print(
            f"  Accuracy vs torch: baseline max_err={acc_b['max_err']:.6e} PSNR={acc_b['psnr_db']:.2f} dB"
        )
        print(
            f"                       ANE      max_err={acc_a['max_err']:.6e} PSNR={acc_a['psnr_db']:.2f} dB"
        )
        print(
            f"  Latency (avg / p50 / p95 ms): baseline {lat_b['avg_ms']:.3f} / {lat_b['p50_ms']:.3f} / {lat_b['p95_ms']:.3f}"
        )
        print(
            f"                                ANE      {lat_a['avg_ms']:.3f} / {lat_a['p50_ms']:.3f} / {lat_a['p95_ms']:.3f}"
        )
        if sp is not None:
            tag = "baseline slower" if sp > 1 else ("ANE slower" if sp < 1 else "tie")
            print(f"  Avg latency ratio (baseline / ANE) = {sp:.3f}x  ({tag})")

    print("\n" + "=" * 72)
    print("  CoreML benchmark: baseline vs ANE (reference = original PyTorch Whisper)")
    print("=" * 72)
    acc = report["accuracy_vs_torch"]
    lat = report["latency_ms"]
    _print_block(
        "Encoder",
        acc["encoder"]["baseline_coreml"],
        acc["encoder"]["ane_coreml"],
        lat["encoder"]["baseline"],
        lat["encoder"]["ane"],
        lat["encoder"]["speedup_baseline_over_ane"],
    )
    _print_block(
        "Decoder (encoder_out = torch reference)",
        acc["decoder_logits"]["baseline_coreml"],
        acc["decoder_logits"]["ane_coreml"],
        lat["decoder"]["baseline"],
        lat["decoder"]["ane"],
        lat["decoder"]["speedup_baseline_over_ane"],
    )

    audio_path = Path(args.audio)
    if not args.no_real_audio and audio_path.is_file():
        try:
            print("\n── Real audio transcription (test_transcribe.py audio) ──")
            report["real_audio_transcription"] = run_real_audio_transcription(
                ct,
                prec,
                cu,
                w,
                ane,
                paths,
                args.out_dir,
                audio_path,
                args.model,
            )
            ra = report["real_audio_transcription"]
            if "error" not in ra:
                print(f"  Audio: {ra['audio_path']}")
                for k in ("original_torch", "baseline_coreml", "ane_coreml"):
                    snippet = ra["text"][k][:120].replace("\n", " ")
                    print(f"  {k}: {snippet!r}{'...' if len(ra['text'][k]) > 120 else ''}")
                ne = ra["normalized_text_equal"]
                print(
                    f"  Normalized text match: torch==baseline {ne['torch_vs_baseline_coreml']}, "
                    f"torch==ANE {ne['torch_vs_ane_coreml']}, baseline==ANE {ne['baseline_coreml_vs_ane_coreml']}"
                )
                print(f"  Phrase checks (all three): {ra['test_audio_phrases_all_three']}")
            else:
                print(f"  ERROR: {ra.get('type')}: {ra.get('error')}")
        except Exception as e:
            report["real_audio_transcription"] = {"error": str(e), "type": type(e).__name__}
            print(f"\n  Real-audio validation failed: {type(e).__name__}: {e}")
    elif not args.no_real_audio:
        report["real_audio_transcription"] = {
            "skipped": True,
            "reason": "audio file not found",
            "path": str(audio_path),
        }
        print(f"\n── Real audio: skipped (not found: {audio_path}) ──")

    out_json = os.path.join(args.out_dir, "benchmark_results.json")
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {out_json}")
    print("=" * 72)


if __name__ == "__main__":
    main()
