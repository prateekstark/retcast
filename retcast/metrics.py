"""Forecasting accuracy and JEPA latent representation metrics."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

EPS = 1e-7


def mse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return float(np.mean(np.square(y_pred - y_true)))


def mae(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return float(np.mean(np.abs(y_pred - y_true)))


def smape(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    abs_diff = np.abs(y_pred - y_true)
    abs_val = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    valid = abs_val > EPS
    safe_diff = np.where(valid, abs_diff, 0.0)
    safe_val = np.where(valid, abs_val, 1.0)
    return float(np.mean(safe_diff / safe_val))


def wape(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    denom = float(np.sum(np.abs(y_true)))
    if denom < EPS:
        return 0.0
    return float(np.sum(np.abs(y_pred - y_true)) / denom)


def nrmse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    mean_abs = float(np.mean(np.abs(y_true)))
    if mean_abs < EPS:
        return 0.0
    return float(np.sqrt(mse(y_pred, y_true)) / mean_abs)


def compute_all_metrics(y_pred: np.ndarray, y_true: np.ndarray) -> dict[str, float]:
    """Compute TimesFM long-horizon benchmark metrics."""
    return {
        "mse": mse(y_pred, y_true),
        "mae": mae(y_pred, y_true),
        "smape": smape(y_pred, y_true),
        "wape": wape(y_pred, y_true),
        "nrmse": nrmse(y_pred, y_true),
    }


@torch.no_grad()
def effective_rank(z: torch.Tensor) -> float:
    """Shannon entropy effective rank exp(-sum p_i log p_i) of representations."""
    z_flat = z.reshape(-1, z.shape[-1]).float()
    z_cent = z_flat - z_flat.mean(dim=0, keepdim=True)
    s = torch.linalg.svdvals(z_cent)
    s_sum = s.sum()
    if s_sum <= 1e-12:
        return 1.0
    p = s / s_sum
    p = p[p > 1e-12]
    entropy = -(p * torch.log(p)).sum()
    return float(torch.exp(entropy).item())


@torch.no_grad()
def latent_cosine_alignment(
    z_pred: torch.Tensor, z_target: torch.Tensor
) -> float:
    """Mean cosine similarity between predicted and target future patch latents."""
    cos = F.cosine_similarity(z_pred.float(), z_target.float(), dim=-1)
    return float(cos.mean().item())


@torch.no_grad()
def latent_retrieval_r1(
    z_pred: torch.Tensor, z_target: torch.Tensor
) -> float:
    """Top-1 cosine retrieval accuracy matching future horizon summaries."""
    if z_pred.ndim == 3:
        qp = z_pred.float().mean(dim=1)
        kt = z_target.float().mean(dim=1)
    else:
        qp = z_pred.float()
        kt = z_target.float()
    qp = F.normalize(qp, dim=-1)
    kt = F.normalize(kt, dim=-1)
    sim = qp @ kt.T
    targets = torch.arange(sim.shape[0], device=sim.device)
    acc = (sim.argmax(dim=-1) == targets).float().mean()
    return float(acc.item())
