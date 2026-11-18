"""Non-ETT multi-domain pretraining corpus for strict zero-shot evaluation.

Loads real-world time series from HuggingFace ``autogluon/chronos_datasets``
and the UCI Solar-AL dataset (zero overlap with ETTh1, ETTh2, ETTm1, ETTm2),
performs train-only (first 80%) standardization per series, applies multi-rate
temporal resampling during sampling, and builds a held-out non-ETT validation
split from the trailing 20% of each series.
"""

from __future__ import annotations

import gzip
import os
import urllib.request
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download

from .data import generate_synthetic_batch

CHRONOS_REPO_ID = "autogluon/chronos_datasets"
SOLAR_AL_URL = (
    "https://raw.githubusercontent.com/laiguokun/"
    "multivariate-time-series-data/master/solar-energy/solar_AL.txt.gz"
)

NON_ETT_CHRONOS_SUBSETS: tuple[str, ...] = (
    "ercot/train-00000-of-00001.parquet",
    "monash_australian_electricity/train-00000-of-00001.parquet",
    "monash_electricity_hourly/train-00000-of-00001.parquet",
    "wind_farms_hourly/train-00000-of-00001.parquet",
    "monash_pedestrian_counts/train-00000-of-00001.parquet",
    "monash_kdd_cup_2018/train-00000-of-00001.parquet",
    "taxi_30min/train-00000-of-00001.parquet",
    "taxi_1h/train-00000-of-00001.parquet",
    "uber_tlc_hourly/train-00000-of-00001.parquet",
    "m4_hourly/train-00000-of-00001.parquet",
    "exchange_rate/train-00000-of-00001.parquet",
    "nn5/train-00000-of-00001.parquet",
)


def _clean_series(arr: np.ndarray) -> np.ndarray | None:
    """Interpolate non-finite values and validate variance."""
    vals = np.asarray(arr, dtype=np.float32).reshape(-1)
    if vals.size == 0:
        return None
    finite = np.isfinite(vals)
    if not np.any(finite):
        return None
    if not np.all(finite):
        idx = np.arange(vals.size)
        vals = np.interp(idx, idx[finite], vals[finite]).astype(np.float32)
    return vals


def download_non_ett_corpus(
    cache_dir: str | os.PathLike = "data/non_ett",
    subsets: Iterable[str] = NON_ETT_CHRONOS_SUBSETS,
    include_solar: bool = True,
) -> dict[str, Path]:
    """Download and cache all non-ETT pretraining files."""
    out_dir = Path(cache_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, Path] = {}

    for rel in subsets:
        if "ett" in rel.lower():
            raise ValueError(f"Refusing ETT dataset in zero-shot corpus: {rel}")
        local = hf_hub_download(
            repo_id=CHRONOS_REPO_ID,
            filename=rel,
            repo_type="dataset",
        )
        name = rel.split("/")[0]
        files[name] = Path(local)

    if include_solar:
        solar_path = out_dir / "solar_AL.txt.gz"
        if not solar_path.exists() or solar_path.stat().st_size < 1024:
            tmp = solar_path.with_suffix(".tmp")
            urllib.request.urlretrieve(SOLAR_AL_URL, tmp)
            tmp.replace(solar_path)
        files["solar_al"] = solar_path

    return files


class NonETTPretrainCorpus:
    """Multi-domain non-ETT training & held-out validation corpus."""

    def __init__(
        self,
        cache_dir: str | os.PathLike = "data/non_ett",
        subsets: Iterable[str] = NON_ETT_CHRONOS_SUBSETS,
        include_solar: bool = True,
        context_len: int = 512,
        horizon_len: int = 336,
        synthetic_ratio: float = 0.25,
        seed: int = 42,
        max_series_per_dataset: int = 300,
    ) -> None:
        self.context_len = context_len
        self.horizon_len = horizon_len
        self.synthetic_ratio = synthetic_ratio
        self.rng = np.random.default_rng(seed)
        self.source_names: list[str] = []

        files = download_non_ett_corpus(
            cache_dir=cache_dir, subsets=subsets, include_solar=include_solar
        )

        # Each entry: (normalized_full_series, train_end_idx, freq_id, source_name)
        self.series_entries: list[tuple[np.ndarray, int, int, str]] = []
        self.dataset_counts: dict[str, int] = {}

        min_len = max(1200, context_len + horizon_len + 160)

        for name, path in files.items():
            assert "ett" not in name.lower(), f"ETT leaked into zero-shot corpus: {name}"
            self.source_names.append(name)
            raw_list: list[np.ndarray] = []

            if path.name.endswith(".parquet"):
                df = pd.read_parquet(path)
                col = "target" if "target" in df.columns else df.columns[-1]
                for item in df[col]:
                    cleaned = _clean_series(item)
                    if cleaned is not None:
                        raw_list.append(cleaned)
            elif path.name == "solar_AL.txt.gz":
                with gzip.open(path, "rt", encoding="utf-8") as f:
                    mat = np.loadtxt(f, delimiter=",", dtype=np.float32)
                for c in range(mat.shape[1]):
                    cleaned = _clean_series(mat[:, c])
                    if cleaned is not None:
                        raw_list.append(cleaned)

            # Deterministically subsample huge collections so no single source dominates
            if len(raw_list) > max_series_per_dataset:
                idx_keep = np.linspace(
                    0, len(raw_list) - 1, max_series_per_dataset, dtype=int
                )
                raw_list = [raw_list[i] for i in idx_keep]

            kept = 0
            for arr in raw_list:
                if arr.size < min_len:
                    continue
                split_idx = int(0.8 * arr.size)
                if split_idx < context_len + horizon_len + 1:
                    continue
                if arr.size - split_idx < horizon_len:
                    continue
                tr = arr[:split_idx]
                mu = float(np.mean(tr))
                sigma = float(np.std(tr))
                if sigma < 1e-5:
                    continue
                norm_arr = ((arr - mu) / sigma).astype(np.float32)
                # Clamp extreme outliers (> 12 std) to prevent gradient spikes
                norm_arr = np.clip(norm_arr, -12.0, 12.0)
                self.series_entries.append((norm_arr, split_idx, 0, name))
                kept += 1
            self.dataset_counts[name] = kept

        if not self.series_entries:
            raise RuntimeError("No valid series loaded for NonETTPretrainCorpus")

        # Group indices by dataset so sampling balances domains instead of series count
        self.by_dataset: dict[str, list[int]] = {}
        for idx, (_, _, _, src) in enumerate(self.series_entries):
            self.by_dataset.setdefault(src, []).append(idx)
        self.active_datasets = [k for k, v in self.by_dataset.items() if len(v) > 0]

    def _resample_window(
        self, tr_slice: np.ndarray, total_len: int
    ) -> np.ndarray:
        """Sample a window from ``tr_slice`` with multi-rate temporal resampling."""
        n = tr_slice.size
        p = float(self.rng.random())

        if p < 0.25 and n >= total_len // 2 + 4:
            # Upsample (2x or 4x) to synthesize 30-min / 15-min periodicities
            factor = 4 if (self.rng.random() < 0.5 and total_len >= 64) else 2
            raw_len = max(32, total_len // factor)
            if n > raw_len:
                s = int(self.rng.integers(0, n - raw_len + 1))
                sub = tr_slice[s : s + raw_len]
                xp = np.linspace(0.0, 1.0, raw_len, dtype=np.float32)
                xq = np.linspace(0.0, 1.0, total_len, dtype=np.float32)
                return np.interp(xq, xp, sub).astype(np.float32)

        if 0.25 <= p < 0.40 and n >= total_len * 2 + 4:
            stride = 3 if (n >= total_len * 3 + 4 and self.rng.random() < 0.4) else 2
            max_s = n - total_len * stride
            s = int(self.rng.integers(0, max_s + 1))
            return tr_slice[s : s + total_len * stride : stride].copy()

        # Original rate (1x)
        max_s = n - total_len
        s = int(self.rng.integers(0, max_s + 1))
        return tr_slice[s : s + total_len].copy()

    def sample_batch(
        self,
        batch_size: int,
        horizon_len: int | None = None,
        augment: bool = True,
        device: torch.device | str = "cpu",
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        H = self.horizon_len if horizon_len is None else horizon_len
        C = self.context_len
        total_len = C + H

        n_syn = int(round(batch_size * self.synthetic_ratio))
        n_real = batch_size - n_syn

        ctx_list = np.empty((batch_size, C), dtype=np.float32)
        fut_list = np.empty((batch_size, H), dtype=np.float32)
        freq_list = np.empty((batch_size,), dtype=np.int64)

        n_ds = len(self.active_datasets)
        for i in range(n_real):
            ds_name = self.active_datasets[int(self.rng.integers(0, n_ds))]
            pool = self.by_dataset[ds_name]
            series_idx = pool[int(self.rng.integers(0, len(pool)))]
            arr, split_idx, freq_id, _ = self.series_entries[series_idx]
            tr = arr[:split_idx]
            win = self._resample_window(tr, total_len)
            if augment:
                scale = float(self.rng.uniform(0.75, 1.25))
                shift = float(self.rng.normal(0.0, 0.2))
                win = win * scale + shift
                if self.rng.random() < 0.5:
                    win += self.rng.normal(0.0, 0.03, size=win.shape).astype(
                        np.float32
                    )
            ctx_list[i] = win[:C]
            fut_list[i] = win[C:]
            freq_list[i] = freq_id

        if n_syn > 0:
            s_ctx, s_fut, s_freq = generate_synthetic_batch(
                n_syn, context_len=C, horizon_len=H, rng=self.rng
            )
            ctx_list[n_real:] = s_ctx
            fut_list[n_real:] = s_fut
            freq_list[n_real:] = s_freq

        perm = self.rng.permutation(batch_size)
        return (
            torch.from_numpy(ctx_list[perm]).to(device),
            torch.from_numpy(fut_list[perm]).to(device),
            torch.from_numpy(freq_list[perm]).to(device),
        )

    def get_val_windows(
        self,
        pred_len: int = 96,
        max_windows: int = 320,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """Extract deterministic validation windows strictly from trailing 20%."""
        C = self.context_len
        past_list: list[np.ndarray] = []
        fut_list: list[np.ndarray] = []
        val_rng = np.random.default_rng(12345 + pred_len)

        per_ds = max(8, max_windows // max(1, len(self.active_datasets)))
        for ds_name in self.active_datasets:
            pool = list(self.by_dataset[ds_name])
            val_rng.shuffle(pool)
            collected = 0
            for s_idx in pool:
                arr, split_idx, _, _ = self.series_entries[s_idx]
                if split_idx < C or arr.size - split_idx < pred_len:
                    continue
                # Pick 1-2 disjoint origins inside [split_idx, len - pred_len]
                for origin in range(split_idx, arr.size - pred_len + 1, pred_len):
                    past_list.append(arr[origin - C : origin])
                    fut_list.append(arr[origin : origin + pred_len])
                    collected += 1
                    if collected >= per_ds:
                        break
                if collected >= per_ds:
                    break

        past = np.stack(past_list[:max_windows], axis=0).astype(np.float32)
        actual = np.stack(fut_list[:max_windows], axis=0).astype(np.float32)
        return past, actual, 0
