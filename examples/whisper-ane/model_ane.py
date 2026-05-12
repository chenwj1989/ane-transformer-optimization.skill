"""
Whisper refactored for Apple ANE / CoreML-friendly inference.

Follows ane-transformer-optimization/SKILL.md:
  BC1S layout in attention + Conv2d(1×1) projections, Conv2d stem from Conv1d,
  per-head attention (einsum), additive float masks, F.layer_norm (eps≥1e-7),
  no SDPA, GELU preserved for weight compatibility.

Encoder and decoder are separate nn.Modules for independent CoreML export.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# OpenAI whisper repo layout: examples/whisper-ane/whisper/whisper/
_WHISPER_SRC = Path(__file__).resolve().parent / "whisper"
if str(_WHISPER_SRC) not in sys.path:
    sys.path.insert(0, str(_WHISPER_SRC))

from whisper.model import ModelDimensions  # noqa: E402


def sinusoids(length: int, channels: int, max_timescale: float = 10000.0) -> Tensor:
    assert channels % 2 == 0
    log_timescale_increment = math.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, None] * inv_timescales[None, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)


class LayerNormANE(nn.Module):
    """Strategy A: F.layer_norm — CoreML-fused, loads Whisper LayerNorm weights."""

    def __init__(self, num_channels: int, eps: float = 1e-5):
        super().__init__()
        self.num_channels = num_channels
        self.eps = max(eps, 1e-7)
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x: Tensor) -> Tensor:
        return F.layer_norm(x, (self.num_channels,), self.weight, self.bias, self.eps)


def _bsc_to_bc1s(x: Tensor) -> Tensor:
    return x.transpose(1, 2).unsqueeze(2)


def _bc1s_to_bsc(x: Tensor) -> Tensor:
    return x.squeeze(2).transpose(1, 2)


class MultiHeadAttentionANEWhisper(nn.Module):
    """Whisper-style MHA with Conv2d projections and per-head ANE attention."""

    def __init__(self, n_state: int, n_head: int, key_has_bias: bool = False):
        super().__init__()
        self.n_head = n_head
        self.n_state = n_state
        self.d_head = n_state // n_head
        self.scale = float(self.d_head) ** -0.5

        self.q_proj = nn.Conv2d(n_state, n_state, kernel_size=1)
        self.k_proj = nn.Conv2d(n_state, n_state, kernel_size=1, bias=key_has_bias)
        self.v_proj = nn.Conv2d(n_state, n_state, kernel_size=1)
        self.out_proj = nn.Conv2d(n_state, n_state, kernel_size=1)

    def forward(
        self,
        x_bsc: Tensor,
        xa_bsc: Optional[Tensor],
        qk_mask_bc1s: Optional[Tensor],
        k_mask_bc1s: Optional[Tensor],
    ) -> Tensor:
        """
        x_bsc: (B, S_q, C) query side (decoder hidden or encoder hidden).
        xa_bsc: (B, S_k, C) cross key/value side; if None, use x_bsc for self-attn.
        qk_mask_bc1s: optional (S_k, 1, S_q) or (1, S_k, 1, S_q) additive float mask.
        k_mask_bc1s: optional (B, S_k, 1, 1) additive key padding mask.
        """
        kv_src = x_bsc if xa_bsc is None else xa_bsc
        q = self.q_proj(_bsc_to_bc1s(x_bsc))
        k = self.k_proj(_bsc_to_bc1s(kv_src))
        v = self.v_proj(_bsc_to_bc1s(kv_src))

        mh_q = q.split(self.d_head, dim=1)
        mh_k = k.transpose(1, 3).split(self.d_head, dim=3)
        mh_v = v.split(self.d_head, dim=1)

        attn_w = [
            torch.einsum("bchq,bkhc->bkhq", qi, ki) * self.scale
            for qi, ki in zip(mh_q, mh_k)
        ]
        if qk_mask_bc1s is not None:
            m = qk_mask_bc1s
            if m.dim() == 3:
                m = m.unsqueeze(0)
            for i in range(self.n_head):
                attn_w[i] = attn_w[i] + m
        if k_mask_bc1s is not None:
            for i in range(self.n_head):
                attn_w[i] = attn_w[i] + k_mask_bc1s

        attn_w = [w.softmax(dim=1) for w in attn_w]
        out = torch.cat(
            [
                torch.einsum("bkhq,bchk->bchq", wi, vi)
                for wi, vi in zip(attn_w, mh_v)
            ],
            dim=1,
        )
        out = self.out_proj(out)
        return _bc1s_to_bsc(out)


class ResidualAttentionBlockANE(nn.Module):
    def __init__(self, n_state: int, n_head: int, cross_attention: bool = False):
        super().__init__()
        self.attn = MultiHeadAttentionANEWhisper(n_state, n_head, key_has_bias=False)
        self.attn_ln = LayerNormANE(n_state)

        self.cross_attn = (
            MultiHeadAttentionANEWhisper(n_state, n_head, key_has_bias=False)
            if cross_attention
            else None
        )
        self.cross_attn_ln = LayerNormANE(n_state) if cross_attention else None

        n_mlp = n_state * 4
        self.mlp_fc1 = nn.Conv2d(n_state, n_mlp, kernel_size=1)
        self.mlp_fc2 = nn.Conv2d(n_mlp, n_state, kernel_size=1)
        self.mlp_ln = LayerNormANE(n_state)

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor],
        qk_mask_self: Optional[Tensor],
        k_mask_self: Optional[Tensor],
    ) -> Tensor:
        x = x + self.attn(
            self.attn_ln(x), None, qk_mask_self, k_mask_self
        )
        if self.cross_attn is not None:
            x = x + self.cross_attn(self.cross_attn_ln(x), xa, None, None)
        nx = self.mlp_ln(x)
        h = _bsc_to_bc1s(nx)
        h = self.mlp_fc2(F.gelu(self.mlp_fc1(h)))
        x = x + _bc1s_to_bsc(h)
        return x


class AudioEncoderANE(nn.Module):
    def __init__(
        self, n_mels: int, n_ctx: int, n_state: int, n_head: int, n_layer: int
    ):
        super().__init__()
        self.n_ctx = n_ctx
        self.conv1 = nn.Conv2d(
            n_mels, n_state, kernel_size=(1, 3), padding=(0, 1), bias=True
        )
        self.conv2 = nn.Conv2d(
            n_state,
            n_state,
            kernel_size=(1, 3),
            stride=(1, 2),
            padding=(0, 1),
            bias=True,
        )
        self.register_buffer("positional_embedding", sinusoids(n_ctx, n_state))
        self.blocks = nn.ModuleList(
            [ResidualAttentionBlockANE(n_state, n_head) for _ in range(n_layer)]
        )
        self.ln_post = LayerNormANE(n_state)

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, n_mels, T_mel) — T_mel must yield exactly n_ctx frames after CNN."""
        x = x.unsqueeze(2)
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.squeeze(2).transpose(1, 2)
        if not torch.jit.is_tracing():
            assert (
                x.shape[1] == self.n_ctx
                and x.shape[2] == self.positional_embedding.shape[1]
            ), "mel time dimension must match Whisper's expected n_audio_ctx after CNN"
        x = x + self.positional_embedding.to(x.dtype)
        for block in self.blocks:
            x = block(x, xa=None, qk_mask_self=None, k_mask_self=None)
        return self.ln_post(x)


class TextDecoderANE(nn.Module):
    def __init__(
        self, n_vocab: int, n_ctx: int, n_state: int, n_head: int, n_layer: int
    ):
        super().__init__()
        self.n_ctx = n_ctx
        self.n_state = n_state
        self.token_embedding = nn.Embedding(n_vocab, n_state)
        self.positional_embedding = nn.Parameter(torch.empty(n_ctx, n_state))
        self.blocks = nn.ModuleList(
            [
                ResidualAttentionBlockANE(n_state, n_head, cross_attention=True)
                for _ in range(n_layer)
            ]
        )
        self.ln = LayerNormANE(n_state)
        # Causal additive mask in (S_k, 1, S_q): -1e4 where key index > query index
        upper = torch.triu(torch.ones(n_ctx, n_ctx), diagonal=1)
        qk_sq_sk = upper * (-1e4)
        self.register_buffer("qk_mask_bc1s", qk_sq_sk.transpose(0, 1).unsqueeze(1))

    def forward(self, tokens: Tensor, xa: Tensor) -> Tensor:
        """
        tokens: (B, S) int64 — use int32 at CoreML boundary.
        xa: (B, n_audio_ctx, n_state) encoder output.
        """
        offset = 0
        x = (
            self.token_embedding(tokens)
            + self.positional_embedding[offset : offset + tokens.shape[1]]
        )
        x = x.to(xa.dtype)
        s = tokens.shape[1]
        qk = self.qk_mask_bc1s[:s, :, :s]
        for block in self.blocks:
            x = block(x, xa, qk_mask_self=qk, k_mask_self=None)
        x = self.ln(x)
        logits = F.linear(
            x, self.token_embedding.weight.to(x.dtype), bias=None
        ).float()
        return logits


class WhisperANE(nn.Module):
    """ANE-shaped Whisper; load weights via WhisperANE.from_whisper(whisper)."""

    def __init__(self, dims: ModelDimensions):
        super().__init__()
        self.dims = dims
        self.encoder = AudioEncoderANE(
            dims.n_mels,
            dims.n_audio_ctx,
            dims.n_audio_state,
            dims.n_audio_head,
            dims.n_audio_layer,
        )
        self.decoder = TextDecoderANE(
            dims.n_vocab,
            dims.n_text_ctx,
            dims.n_text_state,
            dims.n_text_head,
            dims.n_text_layer,
        )

    def embed_audio(self, mel: Tensor) -> Tensor:
        return self.encoder(mel)

    def logits(self, tokens: Tensor, audio_features: Tensor) -> Tensor:
        return self.decoder(tokens, audio_features)

    def forward(self, mel: Tensor, tokens: Tensor) -> Tensor:
        return self.decoder(tokens, self.encoder(mel))

    @classmethod
    def from_whisper(cls, w) -> "WhisperANE":
        """Copy weights from an OpenAI `whisper.model.Whisper` instance (Pattern B)."""
        from whisper.model import Whisper as OpenAIWhisper

        if not isinstance(w, OpenAIWhisper):
            raise TypeError("expected whisper.model.Whisper")
        ane = cls(w.dims)

        enc_o, enc_n = w.encoder, ane.encoder
        with torch.no_grad():
            enc_n.conv1.weight.copy_(enc_o.conv1.weight.unsqueeze(2))
            enc_n.conv1.bias.copy_(enc_o.conv1.bias)
            enc_n.conv2.weight.copy_(enc_o.conv2.weight.unsqueeze(2))
            enc_n.conv2.bias.copy_(enc_o.conv2.bias)
            enc_n.positional_embedding.copy_(enc_o.positional_embedding)
            enc_n.ln_post.weight.copy_(enc_o.ln_post.weight)
            enc_n.ln_post.bias.copy_(enc_o.ln_post.bias)

        for bo, bn in zip(w.encoder.blocks, ane.encoder.blocks):
            cls._copy_residual_block(bo, bn, cross=False)

        dec_o, dec_n = w.decoder, ane.decoder
        with torch.no_grad():
            dec_n.token_embedding.weight.copy_(dec_o.token_embedding.weight)
            dec_n.positional_embedding.copy_(dec_o.positional_embedding)
            dec_n.ln.weight.copy_(dec_o.ln.weight)
            dec_n.ln.bias.copy_(dec_o.ln.bias)

        for bo, bn in zip(w.decoder.blocks, ane.decoder.blocks):
            cls._copy_residual_block(bo, bn, cross=True)

        cls._sync_causal_mask_from_openai(ane, w)
        return ane

    @staticmethod
    def _sync_causal_mask_from_openai(ane: "WhisperANE", w) -> None:
        """Keep additive mask numerically aligned with OpenAI -inf mask (for softmax)."""
        with torch.no_grad():
            m = w.decoder.mask
            tri = torch.isinf(m) & (m < 0)
            qk = torch.zeros_like(m, dtype=torch.float32)
            qk = qk + tri.to(qk.dtype) * (-1e4)
            ane.decoder.qk_mask_bc1s.copy_(qk.transpose(0, 1).unsqueeze(1))

    @staticmethod
    def _copy_mha(openai_mha, ane_mha: MultiHeadAttentionANEWhisper) -> None:
        with torch.no_grad():
            ane_mha.q_proj.weight.copy_(openai_mha.query.weight[:, :, None, None])
            ane_mha.q_proj.bias.copy_(openai_mha.query.bias)
            ane_mha.k_proj.weight.copy_(openai_mha.key.weight[:, :, None, None])
            if openai_mha.key.bias is not None:
                ane_mha.k_proj.bias.copy_(openai_mha.key.bias)
            ane_mha.v_proj.weight.copy_(openai_mha.value.weight[:, :, None, None])
            ane_mha.v_proj.bias.copy_(openai_mha.value.bias)
            ane_mha.out_proj.weight.copy_(openai_mha.out.weight[:, :, None, None])
            ane_mha.out_proj.bias.copy_(openai_mha.out.bias)

    @staticmethod
    def _copy_residual_block(openai_block, ane_block: ResidualAttentionBlockANE, cross: bool) -> None:
        WhisperANE._copy_mha(openai_block.attn, ane_block.attn)
        with torch.no_grad():
            ane_block.attn_ln.weight.copy_(openai_block.attn_ln.weight)
            ane_block.attn_ln.bias.copy_(openai_block.attn_ln.bias)
        if cross and openai_block.cross_attn is not None:
            WhisperANE._copy_mha(openai_block.cross_attn, ane_block.cross_attn)
            with torch.no_grad():
                ane_block.cross_attn_ln.weight.copy_(openai_block.cross_attn_ln.weight)
                ane_block.cross_attn_ln.bias.copy_(openai_block.cross_attn_ln.bias)
        with torch.no_grad():
            ane_block.mlp_fc1.weight.copy_(openai_block.mlp[0].weight[:, :, None, None])
            ane_block.mlp_fc1.bias.copy_(openai_block.mlp[0].bias)
            ane_block.mlp_fc2.weight.copy_(openai_block.mlp[2].weight[:, :, None, None])
            ane_block.mlp_fc2.bias.copy_(openai_block.mlp[2].bias)
            ane_block.mlp_ln.weight.copy_(openai_block.mlp_ln.weight)
            ane_block.mlp_ln.bias.copy_(openai_block.mlp_ln.bias)


# ─── Thin wrappers for separate CoreML export (single forward each) ─────────


class WhisperEncoderCoreMLWrapper(nn.Module):
    """Trace/export: mel -> encoder_features."""

    def __init__(self, encoder: AudioEncoderANE):
        super().__init__()
        self.encoder = encoder

    def forward(self, mel: Tensor) -> Tensor:
        return self.encoder(mel)


class WhisperDecoderCoreMLWrapper(nn.Module):
    """Trace/export: tokens + encoder_out -> logits."""

    def __init__(self, decoder: TextDecoderANE):
        super().__init__()
        self.decoder = decoder

    def forward(self, tokens: Tensor, encoder_out: Tensor) -> Tensor:
        return self.decoder(tokens, encoder_out)


def trace_with_roundtrip(module: nn.Module, args: Tuple[Tensor, ...]) -> torch.jit.ScriptModule:
    """TorchScript save/load round-trip before CoreML (SKILL pitfall #3)."""
    import tempfile

    with torch.no_grad():
        traced = torch.jit.trace(module, args, strict=False)
    fd, path = tempfile.mkstemp(suffix=".pt")
    import os

    os.close(fd)
    try:
        torch.jit.save(traced, path)
        return torch.jit.load(path)
    finally:
        os.unlink(path)
