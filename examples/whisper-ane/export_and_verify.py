#!/usr/bin/env python3
"""
Load OpenAI Whisper, build WhisperANE from weights, verify PyTorch parity,
export encoder and decoder to separate CoreML packages, and smoke-test predict().

Run from repo root or this directory:
  cd examples/whisper-ane && .venv/bin/python export_and_verify.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

# whisper package path
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "whisper"))

import whisper  # noqa: E402
from whisper.model import disable_sdpa  # noqa: E402

from model_ane import (  # noqa: E402
    WhisperANE,
    WhisperDecoderCoreMLWrapper,
    WhisperEncoderCoreMLWrapper,
    trace_with_roundtrip,
)


def _psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = (a - b).pow(2).mean().item()
    if mse == 0:
        return float("inf")
    peak = a.abs().max().item() ** 2
    return 10 * np.log10(peak / mse)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="tiny", help="Whisper checkpoint name (tiny, base, ...)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out-dir", default=str(_HERE / "coreml_out"))
    ap.add_argument("--skip-coreml", action="store_true")
    ap.add_argument(
        "--fp32-coreml",
        action="store_true",
        help="Use FLOAT32 CoreML precision (tighter vs PyTorch; slower / larger).",
    )
    ap.add_argument("--mel-frames", type=int, default=3000, help="Mel time dimension (3000 = 30s)")
    ap.add_argument("--token-len", type=int, default=32)
    args = ap.parse_args()

    device = torch.device(args.device)
    print(f"Loading OpenAI Whisper `{args.model}` …")
    w = whisper.load_model(args.model, device=device)
    w.eval()

    print("Building WhisperANE …")
    ane = WhisperANE.from_whisper(w).to(device).eval()

    n_ctx = w.dims.n_audio_ctx
    mel = torch.randn(1, w.dims.n_mels, args.mel_frames, device=device)
    toks = torch.randint(0, min(1000, w.dims.n_vocab), (1, args.token_len), device=device, dtype=torch.long)

    with torch.no_grad(), disable_sdpa():
        ref_enc = w.encoder(mel)
        ref_logits = w.decoder(toks, ref_enc)

    with torch.no_grad():
        ane_enc = ane.encoder(mel)
        ane_logits = ane.decoder(toks, ane_enc)

    enc_diff = (ref_enc - ane_enc).abs().max().item()
    log_diff = (ref_logits - ane_logits).abs().max().item()
    print(f"Encoder max abs diff vs OpenAI (SDPA off): {enc_diff:.6e}")
    print(f"Decoder logits max abs diff:              {log_diff:.6e}")
    print(f"Decoder logits PSNR (dB):                 {_psnr(ref_logits, ane_logits):.2f}")

    if enc_diff > 1e-3 or log_diff > 1e-2:
        print("WARNING: numerical gap larger than typical fp32 copy — inspect masks / layout.")

    # structure check (skill tool)
    skill_root = _HERE.parent.parent / "ane-transformer-optimization"
    check_script = skill_root / "tools" / "check_structure.py"
    if check_script.is_file():
        print(f"\nRunning {check_script} …")
        os.system(f"{sys.executable} {check_script} --model {_HERE / 'model_ane.py'}")

    if args.skip_coreml:
        print("--skip-coreml: done.")
        return

    try:
        import coremltools as ct
    except ImportError:
        print("coremltools not installed; skip CoreML export. pip install coremltools")
        return

    os.makedirs(args.out_dir, exist_ok=True)
    mel_cpu = mel.cpu()
    toks_cpu = toks.cpu().to(torch.int32)
    ane_enc_cpu = ane_enc.cpu()

    enc_wrap = WhisperEncoderCoreMLWrapper(ane.encoder).cpu().eval()
    dec_wrap = WhisperDecoderCoreMLWrapper(ane.decoder).cpu().eval()

    print("\nTorchScript trace + round-trip …")
    ts_enc = trace_with_roundtrip(enc_wrap, (mel_cpu,))
    ts_dec = trace_with_roundtrip(dec_wrap, (toks_cpu, ane_enc_cpu))

    prec = ct.precision.FLOAT32 if args.fp32_coreml else ct.precision.FLOAT16
    prec_name = "fp32" if args.fp32_coreml else "fp16"
    print(f"Converting to CoreML (mlprogram, {prec_name}, compute_units=ALL) …")
    enc_ml_path = os.path.join(args.out_dir, "WhisperEncoderANE.mlpackage")
    dec_ml_path = os.path.join(args.out_dir, "WhisperDecoderANE.mlpackage")

    ml_enc = ct.convert(
        ts_enc,
        convert_to="mlprogram",
        inputs=[ct.TensorType(name="mel", shape=list(mel_cpu.shape))],
        minimum_deployment_target=ct.target.macOS14,
        compute_precision=prec,
        compute_units=ct.ComputeUnit.ALL,
    )
    ml_enc.save(enc_ml_path)
    print(f"  Saved {enc_ml_path}")

    dec_shape_tokens = list(toks_cpu.shape)
    dec_shape_xa = list(ane_enc_cpu.shape)
    ml_dec = ct.convert(
        ts_dec,
        convert_to="mlprogram",
        inputs=[
            ct.TensorType(name="tokens", shape=dec_shape_tokens, dtype=np.int32),
            ct.TensorType(name="encoder_out", shape=dec_shape_xa),
        ],
        minimum_deployment_target=ct.target.macOS14,
        compute_precision=prec,
        compute_units=ct.ComputeUnit.ALL,
    )
    ml_dec.save(dec_ml_path)
    print(f"  Saved {dec_ml_path}")

    print("\nCoreML predict() smoke test (CPU_AND_NE at load time per skill) …")
    ml_e = ct.models.MLModel(enc_ml_path, compute_units=ct.ComputeUnit.CPU_AND_NE)
    ml_d = ct.models.MLModel(dec_ml_path, compute_units=ct.ComputeUnit.CPU_AND_NE)

    out_e = ml_e.predict({"mel": mel_cpu.numpy().astype(np.float32)})
    enc_key = list(out_e.keys())[0]
    enc_np = out_e[enc_key]
    enc_ml_t = torch.from_numpy(np.array(enc_np)).float()
    print(f"  Encoder CoreML output shape: {tuple(enc_ml_t.shape)}")
    print(
        f"  Encoder vs PyTorch(ANE) max abs: {(enc_ml_t - ane_enc.cpu().float()).abs().max().item():.4f}"
    )

    out_d = ml_d.predict(
        {
            "tokens": toks_cpu.numpy().astype(np.int32),
            "encoder_out": ane_enc_cpu.numpy().astype(np.float32),
        }
    )
    dk = list(out_d.keys())[0]
    log_np = out_d[dk]
    log_ml_t = torch.from_numpy(np.array(log_np)).float()
    print(f"  Decoder CoreML logits shape: {tuple(log_ml_t.shape)}")
    print(
        f"  Decoder vs PyTorch(ANE) logits max abs: {(log_ml_t - ane_logits.cpu().float()).abs().max().item():.4f}"
    )

    meta = {
        "whisper_checkpoint": args.model,
        "mel_shape": list(mel_cpu.shape),
        "token_shape": dec_shape_tokens,
        "encoder_out_shape": dec_shape_xa,
        "coreml_precision": prec_name,
        "decoder_mlpackage": dec_ml_path,
        "pytorch_encoder_max_abs_diff": enc_diff,
        "pytorch_logits_max_abs_diff": log_diff,
    }
    meta_path = os.path.join(args.out_dir, "export_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nWrote {meta_path}")


if __name__ == "__main__":
    main()
