"""
Reference implementations of ANE-friendly transformer building blocks.

This file is **read on demand** by the ANE Transformer Optimization skill — it is
not auto-loaded into agent context. Read it when you need to:
  - Copy-paste a full reference module into a project
  - Verify exact tensor shape annotations and parameter conventions
  - Look up the bias/scale order or load_state_dict hook for LayerNormANE B

Each section maps 1:1 to a rule or optimization in ../SKILL.md. Modules are
runnable PyTorch and importable as a library:
    from references.ane_modules import MultiHeadAttentionANE, LayerNormANE_BSC

Tensor layout convention used throughout:
  BSC = (batch, sequence, channels)   — what PyTorch / HF normally produces
  BC1S = (batch, channels, 1, seq)    — what ANE prefers (Apple Principle 1)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# LayerNorm — Strategy A (default) — see SKILL.md Rule 1.4
# ─────────────────────────────────────────────────────────────────────────────

class LayerNormANE_BSC(nn.Module):
    """Default LayerNorm: stays in BSC, lets CoreML auto-fuse F.layer_norm.

    Loads any pretrained nn.LayerNorm state_dict directly. fp32-stable internal
    accumulator inside the fused CoreML op even under compute_precision=FLOAT16.
    """

    def __init__(self, num_channels: int, eps: float = 1e-5):
        super().__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, S, C)
        return F.layer_norm(x, (self.num_channels,), self.weight, self.bias, self.eps)


# ─────────────────────────────────────────────────────────────────────────────
# LayerNorm — Strategy B (only when justified) — see SKILL.md Rule 1.4
# ─────────────────────────────────────────────────────────────────────────────

def correct_bias_scale_order(state_dict, prefix, *_):
    """state_dict pre-hook: convert nn.LayerNorm bias to Apple's (out+bias)*weight order."""
    bk, wk = prefix + "bias", prefix + "weight"
    if bk in state_dict and wk in state_dict:
        state_dict[bk] = state_dict[bk] / state_dict[wk]


class LayerNormANE_BC1S(nn.Module):
    """Apple-reference style LayerNorm in BC1S layout with optional fp16 clamp.

    PITFALL: applies (out + bias) * weight, NOT out * weight + bias. Loading a
    trained nn.LayerNorm state_dict requires the bias-scale inversion hook
    (registered automatically below). Use only when one of these applies:
      1. You see fp16 LayerNorm overflow and need clip_mag.
      2. You measured a profiler win on your specific model (typically d_model >= 4096).
      3. Porting Apple's reference code 1:1.
    Otherwise prefer LayerNormANE_BSC.
    """

    def __init__(self, num_channels: int, eps: float = 1e-5, clip_mag: float | None = None):
        super().__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.clip_mag = clip_mag
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self._register_load_state_dict_pre_hook(correct_bias_scale_order)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, C, 1, S)
        if self.clip_mag is not None:
            x = x.clamp(-self.clip_mag, self.clip_mag)
        mu = x.mean(dim=1, keepdims=True)
        zm = x - mu
        var = (zm * zm).mean(dim=1, keepdims=True)
        out = zm * (var + self.eps).rsqrt()
        return (out + self.bias.view(1, -1, 1, 1)) * self.weight.view(1, -1, 1, 1)


# Default alias — most projects should use this name
LayerNormANE = LayerNormANE_BSC


# ─────────────────────────────────────────────────────────────────────────────
# Multi-head attention — see SKILL.md Rules 1.1, 1.2, 1.5 + Optimization 2.2
# ─────────────────────────────────────────────────────────────────────────────

class MultiHeadAttentionANE(nn.Module):
    """ANE-friendly multi-head attention. Inputs/outputs in BC1S layout.

    Mask shapes (both float, additive, -1e4 = mask out / 0 = keep):
      qk_mask: (B, S_k, 1, S_q)   — causal / attention mask
      k_mask:  (B, S_k, 1, 1)     — key padding mask
    """

    def __init__(self, embed_dim: int, n_head: int = 8,
                 d_qk: int | None = None, d_v: int | None = None, d_out: int | None = None):
        super().__init__()
        self.d_qk = d_qk or embed_dim
        self.d_v = d_v or embed_dim
        self.d_out = d_out or embed_dim
        self.n_head = n_head
        if self.d_qk % n_head != 0 or self.d_v % n_head != 0:
            raise ValueError(f"d_qk={self.d_qk}, d_v={self.d_v} must be divisible by n_head={n_head}")
        self.scale = float(self.d_qk // n_head) ** -0.5

        self.q_proj = nn.Conv2d(embed_dim, self.d_qk, 1)
        self.k_proj = nn.Conv2d(embed_dim, self.d_qk, 1)
        self.v_proj = nn.Conv2d(embed_dim, self.d_v, 1)
        self.out_proj = nn.Conv2d(self.d_v, self.d_out, 1)

    def _attention_fn(self, q, k, v, qk_mask=None, k_mask=None):
        """q,k,v: (B, C, 1, S). Splits per head, never materializes (B, n_head, S, S)."""
        dh_qk = self.d_qk // self.n_head
        dh_v = self.d_v // self.n_head

        mh_q = q.split(dh_qk, dim=1)                          # [H] x (B, dh, 1, S_q)
        mh_k = k.transpose(1, 3).split(dh_qk, dim=3)          # [H] x (B, S_k, 1, dh)
        mh_v = v.split(dh_v, dim=1)                           # [H] x (B, dh, 1, S_k)

        attn_w = [torch.einsum('bchq,bkhc->bkhq', qi, ki) * self.scale
                  for qi, ki in zip(mh_q, mh_k)]              # [H] x (B, S_k, 1, S_q)

        if qk_mask is not None:
            for i in range(self.n_head):
                attn_w[i] = attn_w[i] + qk_mask
        if k_mask is not None:
            for i in range(self.n_head):
                attn_w[i] = attn_w[i] + k_mask

        # softmax over key axis (dim=1 because of the transpose above).
        # Do NOT add .float() upcast — CoreML's fused softmax accumulates fp32 internally.
        attn_w = [aw.softmax(dim=1) for aw in attn_w]

        attn = [torch.einsum('bkhq,bchk->bchq', wi, vi)
                for wi, vi in zip(attn_w, mh_v)]              # [H] x (B, dh, 1, S_q)
        return torch.cat(attn, dim=1)                          # (B, d_v, 1, S_q)

    def forward(self, q, k, v, qk_mask=None, k_mask=None):
        """All of q, k, v are BC1S. For self-attention pass the same tensor three times."""
        return self.out_proj(self._attention_fn(
            self.q_proj(q), self.k_proj(k), self.v_proj(v), qk_mask, k_mask))


# ─────────────────────────────────────────────────────────────────────────────
# Cross-attention — see SKILL.md Optimization 2.2 (cross-attention variant)
# ─────────────────────────────────────────────────────────────────────────────

class CrossAttentionANE(nn.Module):
    """Decoder cross-attention: Q from decoder state (BSC), K/V from encoder output (BSC).

    For production: precompute encoder K/V projections ONCE outside the decoder
    loop — they are identical for every decoder step. See SKILL.md Optimization 2.5.
    """

    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.d_head = n_state // n_head
        self.scale = self.d_head ** -0.5
        self.q_proj = nn.Conv2d(n_state, n_state, 1)
        self.k_proj = nn.Conv2d(n_state, n_state, 1)
        self.v_proj = nn.Conv2d(n_state, n_state, 1)
        self.out_proj = nn.Conv2d(n_state, n_state, 1)

    def forward(self, x: torch.Tensor, xa: torch.Tensor) -> torch.Tensor:
        # x  (decoder): (B, S_q, C)
        # xa (encoder): (B, S_k, C)
        q = self.q_proj(x.transpose(1, 2).unsqueeze(2))    # (B, C, 1, S_q)
        k = self.k_proj(xa.transpose(1, 2).unsqueeze(2))   # (B, C, 1, S_k)
        v = self.v_proj(xa.transpose(1, 2).unsqueeze(2))   # (B, C, 1, S_k)

        mh_q = q.split(self.d_head, dim=1)
        mh_k = k.transpose(1, 3).split(self.d_head, dim=3)
        mh_v = v.split(self.d_head, dim=1)

        attn_w = [torch.einsum('bchq,bkhc->bkhq', qi, ki) * self.scale
                  for qi, ki in zip(mh_q, mh_k)]
        attn_w = [w.softmax(dim=1) for w in attn_w]

        out = torch.cat([torch.einsum('bkhq,bchk->bchq', wi, vi)
                         for wi, vi in zip(attn_w, mh_v)], dim=1)
        out = self.out_proj(out)
        return out.squeeze(2).transpose(1, 2)              # (B, S_q, C)


# ─────────────────────────────────────────────────────────────────────────────
# Pre-Norm residual block — see SKILL.md Optimization 2.3
# ─────────────────────────────────────────────────────────────────────────────

class PreNormResidualSelfAttentionANE(nn.Module):
    """Pre-Norm residual self-attention: x = drop(attn(norm(x))) + x.

    Mirrors Apple's PreNormResidualSelfAttention. Pre-Norm keeps intermediate
    value ranges stable and reduces fp16 overflow risk.
    """

    def __init__(self, embed_dim: int, n_head: int, dropout: float = 0.0):
        super().__init__()
        self.norm = LayerNormANE(embed_dim)
        self.attn = MultiHeadAttentionANE(embed_dim, n_head=n_head)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, qkv, qk_mask=None, k_mask=None):
        # If using LayerNormANE_BSC, qkv comes in BSC and norm stays in BSC.
        # Convert to BC1S only for the attention block.
        normed_bsc = self.norm(qkv) if qkv.dim() == 3 else self.norm(qkv.squeeze(2).transpose(1, 2))
        normed = normed_bsc.transpose(1, 2).unsqueeze(2)
        attn = self.attn(normed, normed, normed, qk_mask=qk_mask, k_mask=k_mask)
        attn_bsc = attn.squeeze(2).transpose(1, 2)
        return self.drop(attn_bsc) + (qkv if qkv.dim() == 3 else qkv.squeeze(2).transpose(1, 2))


# ─────────────────────────────────────────────────────────────────────────────
# FFN — see SKILL.md Optimization 2.4
# ─────────────────────────────────────────────────────────────────────────────

class FFN_ANE(nn.Module):
    """ANE FFN: Conv2d(1x1) → activation → Conv2d(1x1).

    Activation choice:
      - ReLU is the most ANE-friendly option (use for from-scratch training).
      - GELU MUST be retained when porting a pretrained model trained with GELU
        (e.g. Whisper, ViT, BERT) — accuracy loss from a swap usually outweighs
        the marginal ANE gain. CoreML maps GELU to ANE on recent hardware.
    """

    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.0,
                 activation: str = "relu"):
        super().__init__()
        if activation not in ("relu", "gelu"):
            raise ValueError(f"activation must be 'relu' or 'gelu', got {activation!r}")
        self.fc1 = nn.Conv2d(embed_dim, ffn_dim, 1)
        self.act = nn.ReLU() if activation == "relu" else nn.GELU()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc2 = nn.Conv2d(ffn_dim, embed_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, C, 1, S)
        return self.fc2(self.drop(self.act(self.fc1(x))))


# ─────────────────────────────────────────────────────────────────────────────
# State-dict loading — see SKILL.md Optimization 2.1
# ─────────────────────────────────────────────────────────────────────────────

# Default substring whitelist for layers whose .weight should be unsqueezed
# from (out, in) -> (out, in, 1, 1) when migrating Linear -> Conv2d.
DEFAULT_LINEAR_TO_CONV2D_KEYS = (
    ".q_proj", ".k_proj", ".v_proj", ".out_proj",     # attention
    ".fc1", ".fc2", ".lin1", ".lin2",                 # FFN (common naming)
    ".classifier", ".vocab_projector",                # heads
)


def make_linear_to_conv2d_hook(key_substrings=DEFAULT_LINEAR_TO_CONV2D_KEYS):
    """Build a state_dict pre-hook that unsqueezes 2D Linear weights to 4D Conv2d.

    Usage:
        class MyANEModel(UpstreamModel):
            def __init__(self, config):
                super().__init__(config)
                # ... swap nn.Linear -> nn.Conv2d in submodules ...
                self._register_load_state_dict_pre_hook(make_linear_to_conv2d_hook())
    """
    def _hook(state_dict, prefix, *_):
        for k in list(state_dict.keys()):
            if not k.endswith(".weight"):
                continue
            if state_dict[k].dim() != 2:
                continue
            if any(s in k for s in key_substrings):
                state_dict[k] = state_dict[k][:, :, None, None]
    return _hook


# ─────────────────────────────────────────────────────────────────────────────
# KV cache patterns — see SKILL.md Optimization 2.5
# ─────────────────────────────────────────────────────────────────────────────

# Pattern A — disable cache (simplest, used by examples/whisper-ane).
# Just `del kv_cache` at the top of forward and re-process the full prefix
# every step. Convert-stable, no extra I/O.

# Pattern B — explicit past_k/past_v tensor I/O (most portable).
# Sketch:
#
# class CachedAttention(nn.Module):
#     def forward(self, x, past_k, past_v, position_id):
#         # past_k, past_v: (B, C, 1, max_S) pre-allocated buffers
#         new_k = self.k_proj(x)             # (B, C, 1, 1) for one new token
#         new_v = self.v_proj(x)
#         past_k = scatter_update(past_k, new_k, position_id)
#         past_v = scatter_update(past_v, new_v, position_id)
#         attn = self._attention_fn(self.q_proj(x), past_k, past_v,
#                                    qk_mask=causal_mask_up_to(position_id))
#         return self.out_proj(attn), past_k, past_v
#
# Caller threads past_k, past_v through every step and increments position_id.

# Pattern C — stateful CoreML (iOS 18+ / macOS 15+).
# Declare past_k, past_v as ct.StateType in ct.convert; use mb.write_state /
# mb.read_state inside the model. Cache lives on-device between predicts —
# zero per-step I/O overhead.


# ─────────────────────────────────────────────────────────────────────────────
# Cross-attention K/V precomputation — see SKILL.md Optimization 2.5
# ─────────────────────────────────────────────────────────────────────────────

def precompute_cross_kv(cross_attn: CrossAttentionANE, encoder_output_bsc: torch.Tensor):
    """Project encoder output to K/V once, before entering the decoder loop.

    Removes ~15% of per-step decoder cost for encoder-decoder models like Whisper.
    Returns (k, v) in BC1S layout, ready to feed back into a per-step decoder.
    """
    enc_bc1s = encoder_output_bsc.transpose(1, 2).unsqueeze(2)
    k = cross_attn.k_proj(enc_bc1s)
    v = cross_attn.v_proj(enc_bc1s)
    return k, v
