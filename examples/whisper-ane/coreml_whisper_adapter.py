"""
Whisper-compatible wrapper that runs audio encoder + text decoder via CoreML,
while reusing the stock `transcribe` / `decode` loop with a no-KV-cache decoder
inference path (full-sequence CoreML decoder each step).

Decoder CoreML must be exported with a flexible token sequence dimension
(RangeDim(1, n_text_ctx)); see benchmark_coreml._export_decoder_flex.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Tuple

import numpy as np
import torch
import torch.nn as nn

from whisper.decoding import BeamSearchDecoder, DecodingOptions, DecodingTask, Inference
from whisper.decoding import detect_language as detect_language_function
from whisper.transcribe import transcribe as transcribe_function


class _CoreMLEncoder(nn.Module):
    def __init__(self, parent: "CoreMLWhisper"):
        super().__init__()
        # Parent is an nn.Module; assigning self._p = parent would register a
        # submodule cycle (CoreMLWhisper -> encoder -> _p -> CoreMLWhisper ...).
        object.__setattr__(self, "_p", parent)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        ml = self._p.ml_encoder
        in_name = self._p._enc_in
        out_name = self._p._enc_out
        mel_np = mel.detach().float().cpu().numpy().astype(np.float32)
        out = ml.predict({in_name: mel_np})
        x = torch.from_numpy(np.array(out[out_name])).float()
        return x.to(mel.device)


class CoreMLNoCacheInference(Inference):
    """Decoder via CoreML; full `tokens` each call (no KV cache)."""

    def __init__(
        self,
        ml_decoder: Any,
        initial_token_length: int,
        token_in: str,
        xa_in: str,
        out_name: str,
    ):
        super().__init__()
        self.ml_decoder = ml_decoder
        self.initial_token_length = initial_token_length
        self.token_in = token_in
        self.xa_in = xa_in
        self.out_name = out_name

    def logits(self, tokens: torch.Tensor, audio_features: torch.Tensor) -> torch.Tensor:
        to = tokens.detach().cpu().numpy().astype(np.int32)
        xf = audio_features.detach().float().cpu().numpy().astype(np.float32)
        out = self.ml_decoder.predict({self.token_in: to, self.xa_in: xf})
        logits = torch.from_numpy(np.array(out[self.out_name])).float()
        return logits.to(tokens.device)

    def rearrange_kv_cache(self, source_indices) -> None:
        return

    def cleanup_caching(self) -> None:
        return


def attach_coreml_inference(task: Any, ml_decoder: Any, dec_io: Tuple[str, str, str]) -> None:
    """Swap DecodingTask inference for CoreML; fix BeamSearchDecoder reference."""
    inf = CoreMLNoCacheInference(
        ml_decoder,
        len(task.initial_tokens),
        dec_io[0],
        dec_io[1],
        dec_io[2],
    )
    task.inference = inf
    if isinstance(task.decoder, BeamSearchDecoder):
        task.decoder.inference = inf


def decode_with_coreml_inference(model: "CoreMLWhisper", mel: torch.Tensor, options=None, **kwargs):
    """Same contract as `whisper.decoding.decode`, but swaps in CoreML decoder inference."""
    if options is None:
        options = DecodingOptions()
    if kwargs:
        options = replace(options, **kwargs)
    single = mel.ndim == 2
    if single:
        mel = mel.unsqueeze(0)
    ref = model._pytorch_whisper_for_decoding_task
    task = DecodingTask(ref, options)
    attach_coreml_inference(
        task, model.ml_decoder, (model._dec_tok, model._dec_xa, model._dec_out)
    )
    task.model = model
    result = task.run(mel)
    return result[0] if single else result


class CoreMLWhisper(nn.Module):
    """Minimal surface compatible with `transcribe` / `decode` / `detect_language`."""

    transcribe = transcribe_function
    detect_language = detect_language_function

    def __init__(
        self,
        reference: nn.Module,
        ml_encoder: Any,
        ml_decoder: Any,
        enc_io: Tuple[str, str],
        dec_io: Tuple[str, str, str],
    ):
        super().__init__()
        self.dims = reference.dims
        self.is_multilingual = reference.is_multilingual
        self.num_languages = reference.num_languages
        self.ml_encoder = ml_encoder
        self.ml_decoder = ml_decoder
        self._enc_in, self._enc_out = enc_io
        self._dec_tok, self._dec_xa, self._dec_out = dec_io
        dev = next(reference.parameters()).device
        self.register_buffer("_dev", torch.zeros((), dtype=torch.float32, device=dev))
        self.encoder = _CoreMLEncoder(self)
        # DecodingTask.__init__ introspects reference.decoder.blocks; we swap task.model after init.
        self._pytorch_whisper_for_decoding_task = reference

    @property
    def device(self) -> torch.device:
        return self._dev.device

    def decode(self, mel: torch.Tensor, options=None, **kwargs):
        return decode_with_coreml_inference(self, mel, options, **kwargs)

    def logits(self, tokens: torch.Tensor, audio_features: torch.Tensor) -> torch.Tensor:
        to = tokens.detach().cpu().numpy().astype(np.int32)
        xf = audio_features.detach().float().cpu().numpy().astype(np.float32)
        out = self.ml_decoder.predict({self._dec_tok: to, self._dec_xa: xf})
        logits = torch.from_numpy(np.array(out[self._dec_out])).float()
        return logits.to(tokens.device)


def dec_io_triplet(ml_decoder: Any) -> Tuple[str, str, str]:
    """Return (tokens_input_name, encoder_out_input_name, logits_output_name)."""
    ins = [n for n in ml_decoder.input_description]
    outs = [n for n in ml_decoder.output_description]
    if len(ins) < 2:
        raise ValueError("decoder model must have tokens + encoder_out inputs")
    tok = "tokens" if "tokens" in ins else ins[0]
    xa = "encoder_out" if "encoder_out" in ins else [n for n in ins if n != tok][0]
    return (tok, xa, outs[0])


def enc_io_pair(ml_encoder: Any) -> Tuple[str, str]:
    ins = [n for n in ml_encoder.input_description]
    outs = [n for n in ml_encoder.output_description]
    return (ins[0], outs[0])
