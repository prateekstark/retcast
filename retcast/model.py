"""RetCast: 1-10M RetNet2.0-JEPA Time-Series Forecasting Models.

Core idea (Yann LeCun's JEPA applied to time-series forecasting on RetNet2.0):
Rather than predicting raw noisy future points alone, RetCast encodes patches
into a time-frequency semantic latent space using an online encoder f_theta and
predicts the future patches' semantic representations z_hat_{N+1:N+K} against a
stop-gradient EMA target encoder f_{theta_bar}, regularized against collapse by
VICReg. A MeaningDecoder with a latent-gated multi-patch linear skip then
projects the predicted future semantics into calibrated point and quantile
forecasts.
"""

from __future__ import annotations

import copy
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .retention import (
    CausalSelfAttention,
    MultiScaleRetention,
    RetNetState,
    retnet_log_gamma,
)

Tensor = torch.Tensor

DEFAULT_QUANTILES: tuple[float, ...] = (
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
)

PRESETS: dict[str, dict[str, int]] = {
    "tiny": {"d_model": 64, "n_layers": 2, "n_heads": 4},
    "2m": {"d_model": 160, "n_layers": 4, "n_heads": 4},
    "5m": {"d_model": 240, "n_layers": 5, "n_heads": 4},
    "9m": {"d_model": 304, "n_layers": 6, "n_heads": 4},
}

VARIANT_ALIASES: dict[str, str] = {
    "2m": "zs_retcast_2m_jepa",
    "zs_2m": "zs_retcast_2m_jepa",
    "5m": "zs_retcast_5m_jepa",
    "zs_5m": "zs_retcast_5m_jepa",
    "9m": "zs_retcast_9m_jepa",
    "zs_9m": "zs_retcast_9m_jepa",
    "indomain_2m": "retcast_2m_jepa",
    "indomain_5m": "retcast_5m_jepa",
    "indomain_9m": "retcast_9m_jepa",
}


@dataclass
class RetCastConfig:
    d_model: int = 160
    n_layers: int = 4
    n_heads: int = 4
    value_factor: int = 2
    ffn_mult: int = 2
    patch_len: int = 16
    max_context_len: int = 512
    max_horizon_len: int = 336
    max_horizon_patches: int = 24
    skip_patches: int = 16
    num_freqs: int = 3
    quantiles: tuple[float, ...] = DEFAULT_QUANTILES
    arch: str = "retnet"  # "retnet" | "transformer"
    memory: str = "delta"  # "none" | "delta"
    beta_max: float = 2.0
    short_conv: int = 4
    dropout: float = 0.0
    norm_eps: float = 1e-5
    rope_base: float = 10000.0
    ema_tau: float = 0.995
    jepa_weight: float = 0.25
    vicreg_weight: float = 0.05
    spectral_weight: float = 0.1
    decoder_mode: str = "universal"  # "universal" | "legacy"

    @classmethod
    def from_preset(cls, preset: str, **overrides: Any) -> RetCastConfig:
        if preset not in PRESETS:
            raise ValueError(
                f"Unknown preset {preset!r}; choose from {list(PRESETS)}"
            )
        kw: dict[str, Any] = dict(PRESETS[preset])
        kw.update(overrides)
        return cls(**kw)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["quantiles"] = list(self.quantiles)
        return d


class RevIN(nn.Module):
    """Mask-aware Reversible Instance Normalization over valid context steps."""

    def __init__(self, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps

    def normalize(
        self, x: Tensor, padding: Tensor | None = None
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        """Normalize ``x`` (B, L) using only valid (padding == 0) elements."""
        if padding is None:
            valid = torch.ones_like(x)
        else:
            valid = (1.0 - padding.to(x.dtype)).clamp(0.0, 1.0)
        count = valid.sum(dim=1, keepdim=True).clamp(min=1.0)
        mu = (x * valid).sum(dim=1, keepdim=True) / count
        var = (((x - mu) * valid) ** 2).sum(dim=1, keepdim=True) / count
        sigma = torch.sqrt(var.clamp(min=0.0) + self.eps).clamp(min=self.eps)
        x_norm = ((x - mu) / sigma) * valid
        return x_norm, (mu, sigma)

    def denormalize(
        self, y_norm: Tensor, stats: tuple[Tensor, Tensor]
    ) -> Tensor:
        """Inverse transform predictions of shape (B, ...) using context stats."""
        mu, sigma = stats
        view_shape = (mu.shape[0],) + (1,) * (y_norm.ndim - 1)
        return y_norm * sigma.view(view_shape) + mu.view(view_shape)


class SpectralPatchEncoder(nn.Module):
    """Time-frequency semantic patch encoder f_theta.

    Extracts raw patch values, morphological descriptors (mean, log-std,
    linear slope, end-to-start delta), rFFT log-magnitude spectrum, and valid
    fraction, projecting into d_model-dimensional semantic embeddings.
    """

    def __init__(self, cfg: RetCastConfig) -> None:
        super().__init__()
        P = cfg.patch_len
        n_spec = P // 2 + 1
        in_dim = P + 4 + n_spec + 1
        d = cfg.d_model
        self.patch_len = P

        self.register_buffer(
            "t_grid", torch.linspace(-1.0, 1.0, P), persistent=False
        )
        self.skip = nn.Linear(in_dim, d, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 2 * d),
            nn.SiLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(2 * d, d),
        )
        self.norm = nn.LayerNorm(d, eps=cfg.norm_eps)

    def forward(
        self, patches: Tensor, patch_pad: Tensor | None = None
    ) -> Tensor:
        """patches: (B, N, P), patch_pad: (B, N, P) with 1=pad, 0=valid."""
        B, N, P = patches.shape
        if patch_pad is None:
            valid = torch.ones_like(patches)
        else:
            valid = (1.0 - patch_pad.to(patches.dtype)).clamp(0.0, 1.0)

        cnt = valid.sum(dim=-1, keepdim=True).clamp(min=1.0)
        raw = patches * valid

        # Morphological statistics per patch
        p_mean = raw.sum(dim=-1, keepdim=True) / cnt
        centered = (patches - p_mean) * valid
        p_var = (centered**2).sum(dim=-1, keepdim=True) / cnt
        p_log_std = torch.log(torch.sqrt(p_var + 1e-5) + 1e-4)

        t = self.t_grid.to(dtype=patches.dtype, device=patches.device).view(1, 1, P)
        t_mean = (t * valid).sum(dim=-1, keepdim=True) / cnt
        t_c = (t - t_mean) * valid
        t_denom = (t_c**2).sum(dim=-1, keepdim=True).clamp(min=1e-4)
        p_slope = (t_c * centered).sum(dim=-1, keepdim=True) / t_denom

        first_idx = valid.argmax(dim=-1, keepdim=True)
        last_idx = (P - 1) - valid.flip(-1).argmax(dim=-1, keepdim=True)
        p_diff = (raw.gather(-1, last_idx) - raw.gather(-1, first_idx)) * (
            cnt >= 2.0
        ).to(patches.dtype)

        # Spectral log-magnitude features (computed in fp32 for FFT stability)
        fft_mag = torch.fft.rfft(raw.float(), dim=-1).abs()
        spec = torch.log1p(fft_mag).to(patches.dtype)

        v_ratio = valid.mean(dim=-1, keepdim=True)
        feat = torch.cat(
            [raw, p_mean, p_log_std, p_slope, p_diff, spec, v_ratio], dim=-1
        )
        z = self.skip(feat) + self.mlp(feat)
        return self.norm(z)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.up = nn.Linear(d_model, hidden, bias=False)
        self.down = nn.Linear(hidden, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.down(F.gelu(self.up(x))))


class RetNetBackboneBlock(nn.Module):
    """Parameter-matched RetNet2.0 or Causal Self-Attention backbone block."""

    def __init__(self, cfg: RetCastConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.d_model, eps=cfg.norm_eps)
        if cfg.arch == "retnet":
            self.retention: nn.Module = MultiScaleRetention(
                cfg.d_model,
                cfg.n_heads,
                value_factor=cfg.value_factor,
                rope_base=cfg.rope_base,
                memory=cfg.memory,
                beta_max=cfg.beta_max,
                short_conv=cfg.short_conv,
            )
        elif cfg.arch == "transformer":
            self.retention = CausalSelfAttention(
                cfg.d_model,
                cfg.n_heads,
                value_factor=cfg.value_factor,
                rope_base=cfg.rope_base,
                short_conv=cfg.short_conv,
            )
        else:
            raise ValueError(
                f"arch must be 'retnet' or 'transformer', got {cfg.arch!r}"
            )
        self.norm2 = nn.LayerNorm(cfg.d_model, eps=cfg.norm_eps)
        self.ffn = FeedForward(
            cfg.d_model, cfg.ffn_mult * cfg.d_model, cfg.dropout
        )
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: Tensor,
        state: Any = None,
        offset: int | Tensor = 0,
        mode: str = "parallel",
        chunk_size: int = 64,
        return_state: bool = False,
    ) -> tuple[Tensor, Any]:
        y, new_state = self.retention(
            self.norm1(x), state, offset, mode, chunk_size, return_state
        )
        x = x + self.dropout(y)
        x = x + self.ffn(self.norm2(x))
        return x, new_state


class LatentHorizonPredictor(nn.Module):
    """JEPA Latent Predictor: predicts what future patches mean in latent space."""

    def __init__(self, cfg: RetCastConfig) -> None:
        super().__init__()
        d = cfg.d_model
        self.d_model = d
        self.n_heads = cfg.n_heads
        self.head_dim = d // cfg.n_heads
        self.register_buffer(
            "log_gamma", retnet_log_gamma(cfg.n_heads), persistent=False
        )

        # 1-step causal predictor (also used for O(1) recurrent latent rollout)
        self.next_latent_mlp = nn.Sequential(
            nn.Linear(d, 2 * d),
            nn.SiLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(2 * d, d),
        )
        self.next_latent_norm = nn.LayerNorm(d, eps=cfg.norm_eps)

        # Multi-horizon parallel latent predictor
        self.pool_proj = nn.Linear(d, d, bias=False)
        self.ctx_proj = nn.Sequential(
            nn.Linear(2 * d, d),
            nn.SiLU(),
            nn.Linear(d, d),
        )
        self.horizon_emb = nn.Embedding(cfg.max_horizon_patches, d)
        self.horizon_mlp = nn.Sequential(
            nn.Linear(2 * d, 2 * d),
            nn.SiLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(2 * d, d),
        )
        # Lightweight causal depthwise mixing across future horizon patches
        self.horizon_conv = nn.Conv1d(
            d, d, kernel_size=3, padding=2, groups=d, bias=True
        )
        self.horizon_out_mlp = nn.Sequential(
            nn.Linear(d, 2 * d),
            nn.SiLU(),
            nn.Linear(2 * d, d),
        )
        self.horizon_norm = nn.LayerNorm(d, eps=cfg.norm_eps)

    def predict_next(self, h: Tensor) -> Tensor:
        """Predict next patch semantic latent z_{t+1} from hidden state h_t."""
        return self.next_latent_norm(h + self.next_latent_mlp(h))

    def _multiscale_pool(self, h_ctx: Tensor) -> Tensor:
        """Exponentially weighted pooling across context patches using RetNet decays."""
        B, N, d = h_ctx.shape
        H = self.n_heads
        dh = self.head_dim
        pos = torch.arange(N, device=h_ctx.device, dtype=h_ctx.dtype)
        dist = (N - 1) - pos  # 0 at last patch, N-1 at oldest
        lg = self.log_gamma.to(dtype=h_ctx.dtype, device=h_ctx.device).view(H, 1)
        weights = torch.exp(dist.view(1, N) * lg)  # (H, N)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        h_heads = h_ctx.view(B, N, H, dh).permute(0, 2, 1, 3)  # (B, H, N, dh)
        pooled = torch.einsum("hn,bhnd->bhd", weights, h_heads).reshape(B, d)
        return self.pool_proj(pooled)

    def forward(
        self,
        h_ctx: Tensor,
        horizon_patches: int,
        freq_emb: Tensor | None = None,
    ) -> Tensor:
        """Predict future semantic embeddings z_hat_{N+1:N+K} of shape (B, K, d)."""
        B, _, d = h_ctx.shape
        h_last = h_ctx[:, -1, :]
        c_pool = self._multiscale_pool(h_ctx)
        ctx_summary = self.ctx_proj(torch.cat([h_last, c_pool], dim=-1))  # (B, d)
        if freq_emb is not None:
            ctx_summary = ctx_summary + freq_emb.squeeze(1)

        k_idx = torch.arange(horizon_patches, device=h_ctx.device)
        e_hor = self.horizon_emb(k_idx).unsqueeze(0).expand(B, -1, -1)  # (B, K, d)
        ctx_exp = ctx_summary.unsqueeze(1).expand(B, horizon_patches, d)

        z_hor = ctx_exp + e_hor + self.horizon_mlp(
            torch.cat([ctx_exp, e_hor], dim=-1)
        )
        # Causal depthwise conv across future steps (truncate future padding)
        z_conv = self.horizon_conv(z_hor.transpose(1, 2))[
            :, :, :horizon_patches
        ].transpose(1, 2)
        z_hor = z_hor + F.silu(z_conv)
        z_hor = z_hor + self.horizon_out_mlp(z_hor)
        return self.horizon_norm(z_hor)


class MeaningDecoder(nn.Module):
    """Maps predicted future semantic latents + universal causal lag primitives to forecasts."""

    SEASONAL_LAGS: tuple[int, ...] = (24, 48, 96, 168)

    def __init__(self, cfg: RetCastConfig) -> None:
        super().__init__()
        d = cfg.d_model
        P = cfg.patch_len
        self.patch_len = P
        self.skip_patches = cfg.skip_patches
        self.num_outputs = 1 + len(cfg.quantiles)
        self.decoder_mode = cfg.decoder_mode

        self.mlp = nn.Sequential(
            nn.Linear(2 * d, 2 * d),
            nn.SiLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(2 * d, P * self.num_outputs),
        )
        self.res_proj = nn.Linear(d, P * self.num_outputs, bias=False)

        skip_in = cfg.skip_patches * P
        skip_out = cfg.max_horizon_patches * P
        self.linear_skip = nn.Linear(skip_in, skip_out)
        if self.decoder_mode == "legacy":
            self.skip_gate = nn.Linear(d, 1)
        elif self.decoder_mode == "universal":
            # 7 causal bases: B0 (local 16-step mean), B1 (64-step mean),
            # B2..B5 (periodic lags 24, 48, 96, 168), B6 (shift-anchored linear skip)
            self.basis_gate = nn.Linear(2 * d + len(self.SEASONAL_LAGS), 7)
            self.res_gate = nn.Linear(d, 1)
        else:
            raise ValueError(
                f"decoder_mode must be 'universal' or 'legacy', got {self.decoder_mode!r}"
            )

    @staticmethod
    def _tile_last(ctx_flat: Tensor, lag: int, target_len: int) -> Tensor:
        """Tile the trailing ``lag`` steps of ``ctx_flat`` across ``target_len`` steps."""
        L = ctx_flat.shape[1]
        s = min(lag, L)
        cycle = ctx_flat[:, L - s :]
        reps = (target_len + s - 1) // s
        return cycle.repeat(1, reps)[:, :target_len]

    @classmethod
    def _lag_autocorrelation(cls, ctx_flat: Tensor) -> Tensor:
        """Compute scale-invariant lag autocorrelation at fundamental periods (24, 48, 96, 168)."""
        B, L = ctx_flat.shape
        rhos = []
        for lag in cls.SEASONAL_LAGS:
            if L >= 2 * lag:
                x1 = ctx_flat[:, L - lag :]
                x2 = ctx_flat[:, L - 2 * lag : L - lag]
                x1_c = x1 - x1.mean(dim=-1, keepdim=True)
                x2_c = x2 - x2.mean(dim=-1, keepdim=True)
                rho = F.cosine_similarity(x1_c, x2_c, dim=-1, eps=1e-6)
            else:
                rho = ctx_flat.new_zeros(B)
            rhos.append(rho)
        return torch.stack(rhos, dim=-1)  # (B, 4)

    def forward(
        self,
        z_fut: Tensor,
        h_last: Tensor,
        ctx_patches_norm: Tensor,
    ) -> Tensor:
        """Returns normalized predictions of shape (B, K, P, 1 + Q)."""
        B, K, d = z_fut.shape
        P = self.patch_len
        H_len = K * P
        h_exp = h_last.unsqueeze(1).expand(B, K, d)
        dec = self.res_proj(z_fut) + self.mlp(torch.cat([z_fut, h_exp], dim=-1))
        dec = dec.view(B, K, P, self.num_outputs)

        N = ctx_patches_norm.shape[1]
        M = self.skip_patches
        if N >= M:
            ctx_tail = ctx_patches_norm[:, N - M :, :]
        else:
            pad = ctx_patches_norm.new_zeros(B, M - N, P)
            ctx_tail = torch.cat([pad, ctx_patches_norm], dim=1)

        if self.decoder_mode == "legacy":
            skip_flat = self.linear_skip(ctx_tail.reshape(B, M * P))
            skip_patches = skip_flat[:, :H_len].view(B, K, P, 1)
            gate = torch.sigmoid(self.skip_gate(z_fut)).unsqueeze(-1)
            return dec + gate * skip_patches

        ctx_flat = ctx_patches_norm.reshape(B, N * P)
        L_flat = ctx_flat.shape[1]

        # B0: local trailing patch mean; B1: medium 64-step trailing mean
        b0_scalar = ctx_flat[:, max(0, L_flat - P) :].mean(dim=1, keepdim=True)
        b0 = b0_scalar.expand(B, H_len)
        b1 = ctx_flat[:, max(0, L_flat - 64) :].mean(dim=1, keepdim=True).expand(
            B, H_len
        )
        # B2..B5: universal seasonal period primitives
        b2 = self._tile_last(ctx_flat, 24, H_len)
        b3 = self._tile_last(ctx_flat, 48, H_len)
        b4 = self._tile_last(ctx_flat, 96, H_len)
        b5 = self._tile_last(ctx_flat, 168, H_len)
        # B6: shift-anchored linear skip invariant to context boundary level drift
        ctx_tail_flat = ctx_tail.reshape(B, M * P) - b0_scalar
        b6 = b0 + self.linear_skip(ctx_tail_flat)[:, :H_len]

        basis_stack = torch.stack([b0, b1, b2, b3, b4, b5, b6], dim=1).view(
            B, 7, K, P
        )
        rho = self._lag_autocorrelation(ctx_flat).unsqueeze(1).expand(B, K, -1)
        gate_in = torch.cat([z_fut, h_exp, rho], dim=-1)
        weights = F.softmax(self.basis_gate(gate_in), dim=-1)  # (B, K, 7)
        basis_mix = (weights.permute(0, 2, 1).unsqueeze(-1) * basis_stack).sum(
            dim=1
        )  # (B, K, P)

        res_scale = torch.sigmoid(self.res_gate(z_fut)).unsqueeze(-1)  # (B, K, 1, 1)
        return dec * res_scale + basis_mix.unsqueeze(-1)


class RetCastJEPA(nn.Module):
    """RetCast: RetNet2.0 + JEPA Time-Series Forecaster."""

    def __init__(self, cfg: RetCastConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or RetCastConfig()
        d = self.cfg.d_model

        self.revin = RevIN(eps=self.cfg.norm_eps)
        self.encoder = SpectralPatchEncoder(self.cfg)
        self.target_encoder = copy.deepcopy(self.encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)

        self.freq_emb = nn.Embedding(self.cfg.num_freqs, d)
        self.blocks = nn.ModuleList(
            [RetNetBackboneBlock(self.cfg) for _ in range(self.cfg.n_layers)]
        )
        self.norm_f = nn.LayerNorm(d, eps=self.cfg.norm_eps)

        self.predictor = LatentHorizonPredictor(self.cfg)
        self.decoder = MeaningDecoder(self.cfg)

        self.register_buffer(
            "quantile_levels",
            torch.tensor(self.cfg.quantiles, dtype=torch.float32),
            persistent=False,
        )

        self.apply(self._init_weights)
        self.target_encoder.load_state_dict(self.encoder.state_dict())
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)

        resid_std = 0.02 / math.sqrt(2 * self.cfg.n_layers)
        for block in self.blocks:
            nn.init.normal_(block.retention.o_proj.weight, std=resid_std)
            nn.init.normal_(block.ffn.down.weight, std=resid_std)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.Embedding, nn.Conv1d)):
            nn.init.normal_(m.weight, std=0.02)
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)

    def num_params(self, include_target_encoder: bool = False) -> int:
        total = sum(p.numel() for p in self.parameters())
        if not include_target_encoder:
            total -= sum(p.numel() for p in self.target_encoder.parameters())
        return total

    @classmethod
    def from_pretrained(
        cls,
        repo_or_path: str | os.PathLike[str],
        variant: str = "zs_2m",
        device: torch.device | str = "cpu",
    ) -> RetCastJEPA:
        """Load a RetCast-JEPA checkpoint from local disk (.pt or safetensors dir) or HF Hub."""
        raw = str(repo_or_path)
        if raw.startswith("hf://"):
            raw = raw[5:]
        resolved = VARIANT_ALIASES.get(variant, variant)

        def _cfg_from_dict(cfg_dict: dict[str, Any], sd: dict[str, Tensor]) -> RetCastConfig:
            d = dict(cfg_dict)
            if "quantiles" in d:
                d["quantiles"] = tuple(d["quantiles"])
            if "decoder_mode" not in d:
                d["decoder_mode"] = (
                    "legacy" if "decoder.skip_gate.weight" in sd else "universal"
                )
            return RetCastConfig(**d)

        local = Path(raw)
        if local.exists():
            if local.is_file():
                ckpt = torch.load(local, map_location="cpu", weights_only=False)
                sd = ckpt["model"]
                cfg = _cfg_from_dict(ckpt["cfg"], sd)
            else:
                from safetensors.torch import load_file

                sub_dir = local / resolved
                if not (sub_dir / "config.json").exists():
                    sub_dir = local / variant if (local / variant / "config.json").exists() else local
                cfg_raw = json.loads((sub_dir / "config.json").read_text())
                cfg_dict = cfg_raw.get("model", cfg_raw)
                sd = load_file(str(sub_dir / "model.safetensors"), device="cpu")
                cfg = _cfg_from_dict(cfg_dict, sd)
        else:
            from huggingface_hub import hf_hub_download
            from safetensors.torch import load_file

            prefix = f"{resolved}/" if resolved and resolved not in (".", "root") else ""
            cfg_path = hf_hub_download(repo_id=raw, filename=f"{prefix}config.json")
            weights_path = hf_hub_download(repo_id=raw, filename=f"{prefix}model.safetensors")
            cfg_raw = json.loads(Path(cfg_path).read_text())
            cfg_dict = cfg_raw.get("model", cfg_raw)
            sd = load_file(weights_path, device="cpu")
            cfg = _cfg_from_dict(cfg_dict, sd)

        model = cls(cfg)
        model.load_state_dict(sd)
        model.to(device)
        model.eval()
        return model

    @torch.no_grad()
    def update_target_encoder(self, tau: float | None = None) -> None:
        alpha = self.cfg.ema_tau if tau is None else tau
        for p_online, p_target in zip(
            self.encoder.parameters(), self.target_encoder.parameters()
        ):
            p_target.data.mul_(alpha).add_(p_online.data, alpha=1.0 - alpha)

    def _patchify(
        self, x: Tensor, padding: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        """Reshape (B, L) into (B, N, P) patches, left-padding if L % P != 0."""
        P = self.cfg.patch_len
        B, L = x.shape
        if padding is None:
            padding = torch.zeros_like(x)
        rem = L % P
        if rem != 0:
            pad_len = P - rem
            x = F.pad(x, (pad_len, 0), value=0.0)
            padding = F.pad(padding, (pad_len, 0), value=1.0)
        N = x.shape[1] // P
        return x.view(B, N, P), padding.view(B, N, P)

    def forward_backbone(
        self,
        z: Tensor,
        state: RetNetState | None = None,
        offset: int | Tensor = 0,
        mode: str = "parallel",
        chunk_size: int = 64,
        return_state: bool = False,
    ) -> tuple[Tensor, RetNetState | None]:
        need_state = return_state or (mode != "parallel")
        x = z
        new_states: RetNetState = []
        states = state if state is not None else [None] * len(self.blocks)
        for block, s in zip(self.blocks, states):
            x, ns = block(
                x,
                state=s,
                offset=offset,
                mode=mode,
                chunk_size=chunk_size,
                return_state=need_state,
            )
            if need_state:
                new_states.append(ns)
        return self.norm_f(x), (new_states if need_state else None)

    def rollout_latent_recurrent(
        self,
        h_last: Tensor,
        state: RetNetState,
        offset: int,
        n_patches: int,
        freq_emb: Tensor | None = None,
    ) -> tuple[Tensor, RetNetState]:
        """Autoregressive latent rollout in O(1) recurrent state memory per step."""
        if h_last.ndim == 2:
            h_curr = h_last.unsqueeze(1)
        else:
            h_curr = h_last[:, -1:, :]
        rollouts = []
        curr_state = state
        for k in range(n_patches):
            z_step = self.predictor.predict_next(h_curr)
            rollouts.append(z_step)
            inp = z_step + freq_emb if freq_emb is not None else z_step
            h_curr, curr_state = self.forward_backbone(
                inp,
                state=curr_state,
                offset=offset + k,
                mode="recurrent",
                return_state=True,
            )
        return torch.cat(rollouts, dim=1), curr_state

    def forward(
        self,
        context: Tensor,
        horizon_len: int,
        freq: Tensor | None = None,
        padding: Tensor | None = None,
        mode: str = "parallel",
        chunk_size: int = 64,
        return_latents: bool = False,
    ) -> dict[str, Tensor]:
        """Forecast ``horizon_len`` steps ahead from ``context`` (B, L)."""
        B = context.shape[0]
        if freq is None:
            freq = torch.zeros(B, dtype=torch.long, device=context.device)
        else:
            freq = freq.view(B).to(device=context.device, dtype=torch.long)

        ctx_norm, stats = self.revin.normalize(context, padding)
        ctx_patches, ctx_pad = self._patchify(ctx_norm, padding)

        z_ctx = self.encoder(ctx_patches, ctx_pad)
        f_emb = self.freq_emb(freq).unsqueeze(1)  # (B, 1, d)
        h_ctx, state = self.forward_backbone(
            z_ctx + f_emb,
            mode=mode,
            chunk_size=chunk_size,
            return_state=(mode != "parallel"),
        )

        P = self.cfg.patch_len
        n_fut_patches = math.ceil(horizon_len / P)
        if n_fut_patches > self.cfg.max_horizon_patches:
            raise ValueError(
                f"Requested {n_fut_patches} patches > max_horizon_patches "
                f"{self.cfg.max_horizon_patches}"
            )

        z_fut = self.predictor(h_ctx, n_fut_patches, freq_emb=f_emb)
        preds_norm = self.decoder(z_fut, h_ctx[:, -1, :], ctx_patches)
        preds = self.revin.denormalize(preds_norm, stats)

        # Flatten patches -> (B, K*P, 1+Q) -> slice exact horizon_len
        preds_flat = preds.reshape(B, n_fut_patches * P, -1)[:, :horizon_len, :]
        preds_norm_flat = preds_norm.reshape(B, n_fut_patches * P, -1)[
            :, :horizon_len, :
        ]
        point_forecast = preds_flat[:, :, 0]

        out: dict[str, Tensor] = {
            "point": point_forecast,
            "quantiles": preds_flat,
            "point_norm": preds_norm_flat[:, :, 0],
            "quantiles_norm": preds_norm_flat,
            "stats_mu": stats[0],
            "stats_sigma": stats[1],
        }
        if return_latents:
            out["z_ctx"] = z_ctx
            out["z_fut"] = z_fut
            out["h_ctx"] = h_ctx
        return out

    def _vicreg_loss(self, z: Tensor) -> Tensor:
        """Variance + Covariance anti-collapse regularization on latent matrix."""
        z_flat = z.reshape(-1, self.cfg.d_model).float()
        if z_flat.shape[0] < 2:
            return z_flat.new_zeros(())
        std_z = torch.sqrt(z_flat.var(dim=0, unbiased=False) + 1e-4)
        var_loss = F.relu(1.0 - std_z).mean()
        z_cent = z_flat - z_flat.mean(dim=0, keepdim=True)
        cov = (z_cent.T @ z_cent) / (z_flat.shape[0] - 1)
        off_diag = cov.pow(2).sum() - cov.diagonal().pow(2).sum()
        cov_loss = off_diag / self.cfg.d_model
        return var_loss + 0.05 * cov_loss

    def compute_loss(
        self,
        context: Tensor,
        future: Tensor,
        freq: Tensor | None = None,
        padding: Tensor | None = None,
        mode: str = "parallel",
        chunk_size: int = 64,
    ) -> tuple[Tensor, dict[str, float]]:
        """Joint Forecasting (Huber + Quantile + Spectral) + JEPA + VICReg loss."""
        B, H_len = future.shape
        if freq is None:
            freq = torch.zeros(B, dtype=torch.long, device=context.device)
        else:
            freq = freq.view(B).to(device=context.device, dtype=torch.long)

        # Normalize context strictly from context stats; normalize future with SAME stats
        ctx_norm, stats = self.revin.normalize(context, padding)
        mu, sigma = stats
        fut_norm = (future - mu) / sigma

        ctx_patches, ctx_pad = self._patchify(ctx_norm, padding)
        P = self.cfg.patch_len
        n_fut_patches = math.ceil(H_len / P)
        fut_pad_rem = n_fut_patches * P - H_len
        if fut_pad_rem > 0:
            fut_padded = F.pad(fut_norm, (0, fut_pad_rem), value=0.0)
            fut_mask = F.pad(
                torch.zeros_like(fut_norm), (0, fut_pad_rem), value=1.0
            )
        else:
            fut_padded = fut_norm
            fut_mask = torch.zeros_like(fut_norm)
        fut_patches = fut_padded.view(B, n_fut_patches, P)
        fut_patch_pad = fut_mask.view(B, n_fut_patches, P)

        # Online encoder on context; stop-gradient EMA target encoder on context & future
        z_ctx = self.encoder(ctx_patches, ctx_pad)
        with torch.no_grad():
            z_ctx_target = self.target_encoder(ctx_patches, ctx_pad)
            z_fut_target = self.target_encoder(fut_patches, fut_patch_pad)

        f_emb = self.freq_emb(freq).unsqueeze(1)
        h_ctx, _ = self.forward_backbone(
            z_ctx + f_emb, mode=mode, chunk_size=chunk_size, return_state=False
        )

        # Predict future meaning z_hat_{N+1:N+K} and 1-step context latents z_hat_{2:N}
        z_fut_pred = self.predictor(h_ctx, n_fut_patches, freq_emb=f_emb)
        preds_norm = self.decoder(z_fut_pred, h_ctx[:, -1, :], ctx_patches)
        preds_norm_flat = preds_norm.reshape(B, n_fut_patches * P, -1)[
            :, :H_len, :
        ]

        point_norm = preds_norm_flat[:, :, 0]
        quant_norm = preds_norm_flat[:, :, 1:]  # (B, H, Q)

        # 1. Forecast losses in normalized space
        point_loss = F.smooth_l1_loss(point_norm, fut_norm, beta=0.5)
        q_levels = self.quantile_levels.to(
            device=quant_norm.device, dtype=quant_norm.dtype
        ).view(1, 1, -1)
        q_err = fut_norm.unsqueeze(-1) - quant_norm
        pinball_loss = torch.maximum(
            q_levels * q_err, (q_levels - 1.0) * q_err
        ).mean()

        # Spectral loss on full-patch predictions (where all P values are valid)
        n_full = H_len // P
        if n_full > 0:
            pred_spec_in = preds_norm[:, :n_full, :, 0].float()
            tgt_spec_in = fut_patches[:, :n_full, :].float()
            pred_mag = torch.log1p(torch.fft.rfft(pred_spec_in, dim=-1).abs())
            tgt_mag = torch.log1p(torch.fft.rfft(tgt_spec_in, dim=-1).abs())
            spectral_loss = F.l1_loss(pred_mag, tgt_mag)
        else:
            spectral_loss = point_norm.new_zeros(())

        forecast_loss = (
            point_loss
            + 0.5 * pinball_loss
            + self.cfg.spectral_weight * spectral_loss
        )

        # 2. JEPA latent predictive loss (predict what the future means)
        cos_fut = F.cosine_similarity(z_fut_pred, z_fut_target, dim=-1)
        jepa_fut = (1.0 - cos_fut).mean() + 0.5 * F.smooth_l1_loss(
            z_fut_pred, z_fut_target
        )

        if ctx_patches.shape[1] >= 2:
            z_ctx_next_pred = self.predictor.predict_next(h_ctx[:, :-1, :])
            z_ctx_next_tgt = z_ctx_target[:, 1:, :]
            cos_ctx = F.cosine_similarity(
                z_ctx_next_pred, z_ctx_next_tgt, dim=-1
            )
            jepa_ctx = (1.0 - cos_ctx).mean() + 0.5 * F.smooth_l1_loss(
                z_ctx_next_pred, z_ctx_next_tgt
            )
        else:
            jepa_ctx = z_fut_pred.new_zeros(())

        jepa_loss = jepa_fut + 0.5 * jepa_ctx

        # 3. VICReg anti-collapse regularization
        vicreg_loss = 0.5 * (
            self._vicreg_loss(z_ctx) + self._vicreg_loss(z_fut_pred)
        )

        total_loss = (
            forecast_loss
            + self.cfg.jepa_weight * jepa_loss
            + self.cfg.vicreg_weight * vicreg_loss
        )

        with torch.no_grad():
            std_ctx = torch.sqrt(
                z_ctx.reshape(-1, self.cfg.d_model)
                .float()
                .var(dim=0, unbiased=False)
                + 1e-4
            ).mean()

        metrics = {
            "loss": float(total_loss.detach().item()),
            "forecast_loss": float(forecast_loss.detach().item()),
            "point_loss": float(point_loss.detach().item()),
            "pinball_loss": float(pinball_loss.detach().item()),
            "spectral_loss": float(spectral_loss.detach().item()),
            "jepa_loss": float(jepa_loss.detach().item()),
            "vicreg_loss": float(vicreg_loss.detach().item()),
            "jepa_cos": float(cos_fut.mean().detach().item()),
            "latent_std": float(std_ctx.detach().item()),
        }
        return total_loss, metrics
