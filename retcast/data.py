"""Leak-free ETT dataset loaders and synthetic procedural pretraining data.

Uses the exact train/val/test boundaries and training-only StandardScaler
protocol from Google TimesFM's long-horizon benchmark evaluation.
"""

from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

ETT_BASE_URL = (
    "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/"
)


@dataclass(frozen=True)
class ETTConfig:
    name: str
    filename: str
    boundaries: tuple[int, int, int]
    freq: str
    freq_id: int
    seasonality: int


ETT_CONFIGS: dict[str, ETTConfig] = {
    "etth1": ETTConfig(
        name="etth1",
        filename="ETTh1.csv",
        boundaries=(8640, 11520, 14400),
        freq="H",
        freq_id=0,
        seasonality=24,
    ),
    "etth2": ETTConfig(
        name="etth2",
        filename="ETTh2.csv",
        boundaries=(8640, 11520, 14400),
        freq="H",
        freq_id=0,
        seasonality=24,
    ),
    "ettm1": ETTConfig(
        name="ettm1",
        filename="ETTm1.csv",
        boundaries=(34560, 46080, 57600),
        freq="15min",
        freq_id=0,
        seasonality=96,
    ),
    "ettm2": ETTConfig(
        name="ettm2",
        filename="ETTm2.csv",
        boundaries=(34560, 46080, 57600),
        freq="15min",
        freq_id=0,
        seasonality=96,
    ),
}


def download_ett_datasets(
    data_dir: str | os.PathLike = "data/ett",
    names: Iterable[str] = ("etth1", "etth2", "ettm1", "ettm2"),
) -> dict[str, Path]:
    """Download ETT CSV files into ``data_dir`` if not already present."""
    out_dir = Path(data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name in names:
        key = name.lower()
        if key not in ETT_CONFIGS:
            raise ValueError(f"Unknown ETT dataset {name!r}")
        cfg = ETT_CONFIGS[key]
        dest = out_dir / cfg.filename
        if not dest.exists() or dest.stat().st_size < 1024:
            url = ETT_BASE_URL + cfg.filename
            tmp = dest.with_suffix(".tmp")
            urllib.request.urlretrieve(url, tmp)
            tmp.replace(dest)
        paths[key] = dest
    return paths


class ETTDataset:
    """Single ETT dataset with strict train-only normalization and split access."""

    def __init__(
        self,
        name: str,
        csv_path: str | os.PathLike | None = None,
        data_dir: str | os.PathLike = "data/ett",
        normalize: bool = True,
    ) -> None:
        key = name.lower()
        if key not in ETT_CONFIGS:
            raise ValueError(f"Unknown ETT dataset {name!r}")
        self.cfg = ETT_CONFIGS[key]
        if csv_path is None:
            paths = download_ett_datasets(data_dir, names=(key,))
            csv_path = paths[key]
        self.csv_path = Path(csv_path)

        df = pd.read_csv(self.csv_path)
        self.ts_cols = [c for c in df.columns if c != "date"]
        raw = df[self.ts_cols].to_numpy(dtype=np.float32)  # (T_total, C)

        b_train, b_val, b_test = self.cfg.boundaries
        if raw.shape[0] < b_test:
            b_test = raw.shape[0]
        self.boundaries = (b_train, b_val, b_test)

        self.raw_data = raw[:b_test]
        self.scaler: StandardScaler | None = None
        if normalize:
            self.scaler = StandardScaler()
            self.scaler.fit(self.raw_data[:b_train])
            self.data = self.scaler.transform(self.raw_data).astype(np.float32)
        else:
            self.data = self.raw_data.copy()

    @property
    def num_series(self) -> int:
        return self.data.shape[1]

    @property
    def train_data(self) -> np.ndarray:
        """Strictly training slice [0 : b_train, C]."""
        return self.data[: self.boundaries[0]]

    def get_eval_windows(
        self,
        split: str = "test",
        context_len: int = 512,
        pred_len: int = 96,
        stride: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """Extract evaluation windows matching TimesFM's long_horizon run_eval.py.

        Returns (past, actuals, freq_id) with shapes:
          past    : (N_windows, context_len)
          actuals : (N_windows, pred_len)
        """
        if stride is None:
            stride = pred_len
        b_train, b_val, b_test = self.boundaries
        if split == "val":
            start, end = b_train, b_val - pred_len + 1
        elif split == "test":
            start, end = b_val, b_test - pred_len + 1
        else:
            raise ValueError(f"split must be 'val' or 'test', got {split!r}")

        if start - context_len < 0:
            raise ValueError(
                f"context_len={context_len} exceeds split start={start}"
            )

        past_list: list[np.ndarray] = []
        actual_list: list[np.ndarray] = []
        n_cols = self.num_series
        for idx in range(start, end, stride):
            ctx = self.data[idx - context_len : idx].T  # (C, context_len)
            fut = self.data[idx : idx + pred_len].T  # (C, pred_len)
            past_list.append(ctx)
            actual_list.append(fut)

        past = np.concatenate(past_list, axis=0).astype(np.float32)
        actuals = np.concatenate(actual_list, axis=0).astype(np.float32)
        assert past.shape[0] == len(range(start, end, stride)) * n_cols
        return past, actuals, self.cfg.freq_id


def generate_synthetic_batch(
    batch_size: int,
    context_len: int = 512,
    horizon_len: int = 336,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate diverse procedural time series (harmonics, trends, AR1, regimes)."""
    if rng is None:
        rng = np.random.default_rng()

    total_len = context_len + horizon_len
    t = np.arange(total_len, dtype=np.float32)[None, :]  # (1, L)

    # Multi-harmonic seasonal components (periods e.g. 12, 24, 48, 96, 168)
    periods = rng.choice(
        np.array([12.0, 24.0, 48.0, 96.0, 168.0], dtype=np.float32),
        size=(batch_size, 3),
        replace=True,
    )
    amps = rng.uniform(0.3, 1.5, size=(batch_size, 3)).astype(np.float32)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=(batch_size, 3)).astype(
        np.float32
    )

    series = np.zeros((batch_size, total_len), dtype=np.float32)
    for k in range(3):
        series += amps[:, k : k + 1] * np.sin(
            2.0 * np.pi * t / periods[:, k : k + 1] + phases[:, k : k + 1]
        )

    # Smooth linear or quadratic drift
    slope = rng.normal(0.0, 0.002, size=(batch_size, 1)).astype(np.float32)
    offset = rng.normal(0.0, 1.0, size=(batch_size, 1)).astype(np.float32)
    series += slope * (t - context_len) + offset

    # AR(1) stochastic innovation process
    phi = rng.uniform(0.6, 0.96, size=(batch_size, 1)).astype(np.float32)
    noise_std = rng.uniform(0.05, 0.25, size=(batch_size, 1)).astype(np.float32)
    eps = rng.normal(0.0, 1.0, size=(batch_size, total_len)).astype(np.float32)
    ar = np.zeros_like(series)
    for step in range(1, total_len):
        ar[:, step] = phi[:, 0] * ar[:, step - 1] + noise_std[:, 0] * eps[:, step]
    series += ar

    # Occasional regime level shift inside context
    shift_mask = rng.random(batch_size) < 0.25
    if np.any(shift_mask):
        for i in np.where(shift_mask)[0]:
            cp = int(rng.integers(32, max(33, context_len - 32)))
            series[i, cp:] += float(rng.normal(0.0, 0.6))

    # Global per-series scale variation
    scale = rng.uniform(0.5, 3.0, size=(batch_size, 1)).astype(np.float32)
    series *= scale

    freq = np.zeros((batch_size,), dtype=np.int64)
    return (
        series[:, :context_len],
        series[:, context_len:],
        freq,
    )


class MultiETTTrainSampler:
    """Samples leak-free training batches across ETT train splits + synthetic mix."""

    def __init__(
        self,
        datasets: list[ETTDataset],
        context_len: int = 512,
        horizon_len: int = 336,
        synthetic_ratio: float = 0.15,
        seed: int = 42,
    ) -> None:
        self.datasets = datasets
        self.context_len = context_len
        self.horizon_len = horizon_len
        self.synthetic_ratio = synthetic_ratio
        self.rng = np.random.default_rng(seed)

        self.series_pool: list[tuple[np.ndarray, int]] = []
        for ds in datasets:
            tr = ds.train_data  # (T_train, C)
            if tr.shape[0] < context_len + horizon_len + 1:
                raise ValueError(
                    f"{ds.cfg.name} train length {tr.shape[0]} too small for "
                    f"context_len={context_len} + horizon_len={horizon_len}"
                )
            for c in range(tr.shape[1]):
                self.series_pool.append((tr[:, c], ds.cfg.freq_id))

    def sample_batch(
        self,
        batch_size: int,
        horizon_len: int | None = None,
        augment: bool = True,
        device: torch.device | str = "cpu",
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        H = self.horizon_len if horizon_len is None else horizon_len
        C = self.context_len
        n_syn = int(round(batch_size * self.synthetic_ratio))
        n_real = batch_size - n_syn

        ctx_list = np.empty((batch_size, C), dtype=np.float32)
        fut_list = np.empty((batch_size, H), dtype=np.float32)
        freq_list = np.empty((batch_size,), dtype=np.int64)

        n_pool = len(self.series_pool)
        for i in range(n_real):
            series, f_id = self.series_pool[int(self.rng.integers(0, n_pool))]
            max_start = len(series) - (C + H)
            s = int(self.rng.integers(0, max_start + 1))
            win = series[s : s + C + H].copy()
            if augment:
                scale = float(self.rng.uniform(0.85, 1.15))
                shift = float(self.rng.normal(0.0, 0.15))
                win = win * scale + shift
                if self.rng.random() < 0.5:
                    win += self.rng.normal(0.0, 0.03, size=win.shape).astype(
                        np.float32
                    )
            ctx_list[i] = win[:C]
            fut_list[i] = win[C:]
            freq_list[i] = f_id

        if n_syn > 0:
            s_ctx, s_fut, s_freq = generate_synthetic_batch(
                n_syn, context_len=C, horizon_len=H, rng=self.rng
            )
            ctx_list[n_real:] = s_ctx
            fut_list[n_real:] = s_fut
            freq_list[n_real:] = s_freq

        # Shuffle real and synthetic within the batch
        perm = self.rng.permutation(batch_size)
        return (
            torch.from_numpy(ctx_list[perm]).to(device),
            torch.from_numpy(fut_list[perm]).to(device),
            torch.from_numpy(freq_list[perm]).to(device),
        )
