"""RetNet2.0 Multi-Scale Retention and parameter-matched Causal Self-Attention.

Implements RetNet2.0 (Sun et al., 2023 + delta-rule overwritable associative
memory + causal depthwise short convolution) with exact mathematical equivalence
across three execution modes:

  parallel   : O = (Q K^T * D) V                 -- training, O(T^2)
  recurrent  : S_n = g S_{n-1} + k_n^T v_n,
               o_n = q_n S_n                     -- inference, O(1) memory/step
  chunkwise  : parallel inside chunks, recurrent across chunks
                                                 -- long sequences, O(T * chunk)
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor
RetNetState = list[Any]


def rotary_cos_sin(
    offset: int | Tensor, length: int, dim: int, base: float, device, dtype
) -> tuple[Tensor, Tensor]:
    """cos/sin tables for absolute positions offset..offset+length-1.

    Supports both scalar ``offset`` and per-batch ``(B,)`` tensor offsets.
    """
    calc_dtype = torch.float64 if dtype == torch.float64 else torch.float32
    inv_freq = 1.0 / (
        base ** (torch.arange(0, dim, 2, device=device, dtype=calc_dtype) / dim)
    )
    steps = torch.arange(length, device=device, dtype=calc_dtype)
    if isinstance(offset, Tensor):
        pos = offset.to(device=device, dtype=calc_dtype)[:, None, None] + steps
    else:
        pos = steps + offset
    freqs = pos[..., None] * inv_freq
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def apply_rotary(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Rotate pairs (x_i, x_{i+dim/2}). x: (B, H, T, dim)."""
    x1, x2 = x.chunk(2, dim=-1)
    rotated = torch.cat([-x2, x1], dim=-1)
    return x * cos + rotated * sin


def retnet_log_gamma(n_heads: int) -> Tensor:
    """log g_h with g_h = 1 - 2^(-5 - h), the paper's fixed multi-scale decays."""
    h = torch.arange(n_heads, dtype=torch.float64)
    return torch.log1p(-torch.pow(2.0, -5.0 - h))


def clone_retnet_state(state: RetNetState | None) -> RetNetState | None:
    """Deep-clone a RetNet/Transformer recurrent state list."""
    if state is None:
        return None
    cloned: RetNetState = []
    for item in state:
        if isinstance(item, tuple):
            cloned.append(
                tuple(
                    tuple(x.clone() for x in t)
                    if isinstance(t, tuple)
                    else (t.clone() if t is not None else None)
                    for t in item
                )
            )
        elif item is not None:
            cloned.append(item.clone())
        else:
            cloned.append(None)
    return cloned


def index_retnet_state(state: RetNetState | None, idx: int) -> RetNetState | None:
    """Extract batch element ``idx`` (preserving batch dim 1) from state."""
    if state is None:
        return None
    sliced: RetNetState = []
    for item in state:
        if isinstance(item, tuple):
            sliced.append(
                tuple(
                    tuple(x[idx : idx + 1].clone() for x in t)
                    if isinstance(t, tuple)
                    else (t[idx : idx + 1].clone() if t is not None else None)
                    for t in item
                )
            )
        elif item is not None:
            sliced.append(item[idx : idx + 1].clone())
        else:
            sliced.append(None)
    return sliced


def expand_retnet_state(state: RetNetState | None, batch_size: int) -> RetNetState | None:
    """Broadcast a batch-1 state to ``batch_size`` without copying memory."""
    if state is None:
        return None
    expanded: RetNetState = []
    for item in state:
        if isinstance(item, tuple):
            expanded.append(
                tuple(
                    tuple(x.expand(batch_size, *x.shape[1:]) for x in t)
                    if isinstance(t, tuple)
                    else (t.expand(batch_size, *t.shape[1:]) if t is not None else None)
                    for t in item
                )
            )
        elif item is not None:
            expanded.append(item.expand(batch_size, *item.shape[1:]))
        else:
            expanded.append(None)
    return expanded


def retnet_state_bytes(state: RetNetState | None) -> int:
    """Return total memory in bytes occupied by tensors inside ``state``."""
    if state is None:
        return 0
    total = 0

    def _accumulate(obj: Any) -> None:
        nonlocal total
        if isinstance(obj, Tensor):
            total += obj.numel() * obj.element_size()
        elif isinstance(obj, (list, tuple)):
            for child in obj:
                _accumulate(child)

    _accumulate(state)
    return total


class MultiScaleRetention(nn.Module):
    """Multi-scale retention with optional delta-rule memory and causal short conv."""

    MODES = ("parallel", "recurrent", "chunkwise")

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        value_factor: int = 2,
        rope_base: float = 10000.0,
        norm_eps: float = 1e-6,
        memory: str = "none",
        beta_max: float = 1.0,
        short_conv: int = 0,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        if memory not in ("none", "delta"):
            raise ValueError(f"memory must be 'none' or 'delta', got {memory!r}")
        self.memory = memory
        self.beta_max = beta_max
        self.n_heads = n_heads
        self.head_k = d_model // n_heads
        assert self.head_k % 2 == 0, "head dim must be even for RoPE"
        self.head_v = self.head_k * value_factor
        self.rope_base = rope_base
        self.norm_eps = norm_eps
        self.scale = self.head_k**-0.5

        d_v = self.head_v * n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_v, bias=False)
        self.g_proj = nn.Linear(d_model, d_v, bias=False)
        self.o_proj = nn.Linear(d_v, d_model, bias=False)
        if memory == "delta":
            self.b_proj = nn.Linear(d_model, n_heads, bias=True)
        self.short_conv = short_conv
        if short_conv > 0:
            self.conv = nn.Conv1d(
                d_model, d_model, short_conv, groups=d_model, bias=False
            )

        self.register_buffer("log_gamma", retnet_log_gamma(n_heads), persistent=False)

    def _log_gamma(self, dtype: torch.dtype) -> Tensor:
        return self.log_gamma.to(dtype).view(-1, 1, 1)

    def _split_heads(self, x: Tensor, head_dim: int) -> Tensor:
        B, T, _ = x.shape
        return x.view(B, T, self.n_heads, head_dim).transpose(1, 2)

    def _mix(self, x: Tensor, cache: Tensor | None) -> tuple[Tensor, Tensor | None]:
        if self.short_conv == 0:
            return x, None
        K = self.short_conv
        if cache is None:
            cache = x.new_zeros(x.shape[0], K - 1, x.shape[2])
        inp = torch.cat([cache.to(x.dtype), x], dim=1)
        mixed = self.conv(inp.transpose(1, 2)).transpose(1, 2)
        return F.silu(mixed), inp[:, inp.shape[1] - (K - 1) :]

    def _project(self, x: Tensor, offset: int | Tensor, mixed: Tensor):
        q = self._split_heads(self.q_proj(mixed), self.head_k)
        k = self._split_heads(self.k_proj(mixed), self.head_k)
        v = self._split_heads(self.v_proj(mixed), self.head_v)
        cos, sin = rotary_cos_sin(
            offset, x.shape[1], self.head_k, self.rope_base, x.device, x.dtype
        )
        q, k = apply_rotary(q, cos, sin), apply_rotary(k, cos, sin)
        if self.memory == "none":
            return q, k * self.scale, v, None
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        beta = torch.sigmoid(self.b_proj(x)).transpose(1, 2) * self.beta_max
        return q, k, v, beta

    def _output(self, x: Tensor, o: Tensor) -> Tensor:
        o = F.layer_norm(o, (self.head_v,), eps=self.norm_eps)
        B, H, T, dv = o.shape
        o = o.transpose(1, 2).reshape(B, T, H * dv)
        return self.o_proj(F.silu(self.g_proj(x)) * o)

    def _chunk(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        state: Tensor | None,
        need_state: bool,
        beta: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        T = q.shape[-2]
        lg = self._log_gamma(q.dtype)
        idx = torch.arange(T, device=q.device)
        diff = (idx[:, None] - idx[None, :]).to(q.dtype)
        decay = torch.exp(diff.clamp(min=0) * lg).masked_fill(diff < 0, 0.0)
        pos = idx.to(q.dtype)

        if beta is not None:
            v = self._delta_values(k, v, state, beta, decay, diff, lg, pos)

        o = (q @ k.transpose(-1, -2) * decay) @ v

        if state is not None:
            q_decay = torch.exp((pos + 1) * lg.squeeze(-1))
            o = o + (q @ state) * q_decay[..., None]

        new_state = None
        if need_state:
            k_decay = torch.exp((T - 1 - pos) * lg.squeeze(-1))
            new_state = (k * k_decay[..., None]).transpose(-1, -2) @ v
            if state is not None:
                new_state = new_state + state * torch.exp(T * lg)
        return o, new_state

    def _delta_values(self, k, v, state, beta, decay, diff, lg, pos):
        A = (k @ k.transpose(-1, -2)) * decay.masked_fill(diff == 0, 0.0)
        rhs = v
        if state is not None:
            xi = torch.exp((pos + 1) * lg.squeeze(-1))
            rhs = v - xi[..., None] * (k @ state)
        rhs = beta[..., None] * rhs
        M = torch.eye(A.shape[-1], dtype=A.dtype, device=A.device) + beta[..., None] * A
        return torch.linalg.solve_triangular(M, rhs, upper=False, unitriangular=True)

    def _recurrent(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        state: Tensor | None,
        beta: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        B, H, T, _ = q.shape
        if state is None:
            state = q.new_zeros(B, H, self.head_k, self.head_v)
        gamma = torch.exp(self._log_gamma(q.dtype))
        outs = []
        for t in range(T):
            vt = v[:, :, t, None, :]
            if beta is not None:
                stored = k[:, :, t, None, :] @ state
                vt = beta[:, :, t, None, None] * (vt - gamma * stored)
            state = gamma * state + k[:, :, t, :, None] * vt
            outs.append(q[:, :, t, None, :] @ state)
        return torch.cat(outs, dim=2), state

    def forward(
        self,
        x: Tensor,
        state: Any = None,
        offset: int | Tensor = 0,
        mode: str = "parallel",
        chunk_size: int = 64,
        return_state: bool = False,
    ) -> tuple[Tensor, Any]:
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}, got {mode!r}")
        S, cache = state if isinstance(state, tuple) else (state, None)
        mixed, new_cache = self._mix(x, cache)
        q, k, v, beta = self._project(x, offset, mixed)
        state = S

        low = (torch.float16, torch.bfloat16)
        if q.dtype in low or k.dtype in low or v.dtype in low:
            q, k, v = q.float(), k.float(), v.float()
            beta = beta.float() if beta is not None else None
            state = state.float() if state is not None else None
        with torch.autocast(device_type=x.device.type, enabled=False):
            o, new_state = self._retention(
                q, k, v, beta, state, mode, chunk_size, return_state
            )

        if new_state is not None and new_cache is not None:
            new_state = (new_state, new_cache)
        return self._output(x, o), new_state

    def _retention(self, q, k, v, beta, state, mode, chunk_size, return_state):
        if mode == "parallel":
            o, new_state = self._chunk(q, k, v, state, need_state=return_state, beta=beta)
        elif mode == "recurrent":
            o, new_state = self._recurrent(q, k, v, state, beta=beta)
        else:
            outs, new_state = [], state
            for s in range(0, q.shape[2], chunk_size):
                sl = slice(s, s + chunk_size)
                o_c, new_state = self._chunk(
                    q[:, :, sl],
                    k[:, :, sl],
                    v[:, :, sl],
                    new_state,
                    need_state=True,
                    beta=beta[:, :, sl] if beta is not None else None,
                )
                outs.append(o_c)
            o = torch.cat(outs, dim=2)
        return o, new_state


class CausalSelfAttention(nn.Module):
    """Parameter-matched Causal Self-Attention with RoPE, Swish gate, and KV cache.

    Shares identical projection shapes (q_proj, k_proj, v_proj, g_proj, o_proj,
    and optional short_conv) with MultiScaleRetention so RetNet vs Transformer
    ablations match parameter budgets (12 d^2 per block).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        value_factor: int = 2,
        rope_base: float = 10000.0,
        norm_eps: float = 1e-6,
        short_conv: int = 0,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.head_k = d_model // n_heads
        assert self.head_k % 2 == 0, "head dim must be even for RoPE"
        self.head_v = self.head_k * value_factor
        self.rope_base = rope_base
        self.norm_eps = norm_eps
        self.scale = self.head_k**-0.5

        d_v = self.head_v * n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_v, bias=False)
        self.g_proj = nn.Linear(d_model, d_v, bias=False)
        self.o_proj = nn.Linear(d_v, d_model, bias=False)
        self.short_conv = short_conv
        if short_conv > 0:
            self.conv = nn.Conv1d(
                d_model, d_model, short_conv, groups=d_model, bias=False
            )

    def _split_heads(self, x: Tensor, head_dim: int) -> Tensor:
        B, T, _ = x.shape
        return x.view(B, T, self.n_heads, head_dim).transpose(1, 2)

    def _mix(self, x: Tensor, cache: Tensor | None) -> tuple[Tensor, Tensor | None]:
        if self.short_conv == 0:
            return x, None
        K = self.short_conv
        if cache is None:
            cache = x.new_zeros(x.shape[0], K - 1, x.shape[2])
        inp = torch.cat([cache.to(x.dtype), x], dim=1)
        mixed = self.conv(inp.transpose(1, 2)).transpose(1, 2)
        return F.silu(mixed), inp[:, inp.shape[1] - (K - 1) :]

    def forward(
        self,
        x: Tensor,
        state: Any = None,
        offset: int | Tensor = 0,
        mode: str = "parallel",
        chunk_size: int = 64,
        return_state: bool = False,
    ) -> tuple[Tensor, Any]:
        B, T, _ = x.shape
        if self.short_conv > 0 and isinstance(state, tuple) and len(state) == 2 and isinstance(state[0], tuple):
            kv_state, conv_cache = state
        elif self.short_conv == 0 and isinstance(state, tuple):
            kv_state, conv_cache = state, None
        else:
            kv_state, conv_cache = None, None

        mixed, new_conv_cache = self._mix(x, conv_cache)
        q = self._split_heads(self.q_proj(mixed), self.head_k)
        k = self._split_heads(self.k_proj(mixed), self.head_k)
        v = self._split_heads(self.v_proj(mixed), self.head_v)
        cos, sin = rotary_cos_sin(
            offset, T, self.head_k, self.rope_base, x.device, x.dtype
        )
        q, k = apply_rotary(q, cos, sin), apply_rotary(k, cos, sin)

        if kv_state is not None:
            k_prev, v_prev = kv_state
            k = torch.cat([k_prev.to(k.dtype), k], dim=2)
            v = torch.cat([v_prev.to(v.dtype), v], dim=2)

        S_len = k.shape[2]
        if S_len == T:
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=self.scale)
        else:
            q_idx = torch.arange(S_len - T, S_len, device=x.device)[:, None]
            k_idx = torch.arange(S_len, device=x.device)[None, :]
            o = F.scaled_dot_product_attention(
                q, k, v, attn_mask=k_idx <= q_idx, scale=self.scale
            )

        o = F.layer_norm(o, (self.head_v,), eps=self.norm_eps)
        o = o.transpose(1, 2).reshape(B, T, self.n_heads * self.head_v)
        out = self.o_proj(F.silu(self.g_proj(x)) * o)

        keep_state = return_state or mode != "parallel"
        if not keep_state:
            return out, None
        new_kv = (k.detach(), v.detach())
        if new_conv_cache is not None:
            return out, (new_kv, new_conv_cache.detach())
        return out, new_kv
