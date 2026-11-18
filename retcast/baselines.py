"""No-learning sanity baselines for time-series forecasting."""

from __future__ import annotations

import numpy as np


def seasonal_naive_forecast(
    past: np.ndarray, pred_len: int, seasonality: int = 24
) -> np.ndarray:
    """Repeat the last complete seasonal cycle of length ``seasonality``."""
    if seasonality <= 0:
        raise ValueError(f"seasonality must be positive, got {seasonality}")
    s = min(seasonality, past.shape[1])
    last_season = past[:, -s:]  # (N, s)
    reps = (pred_len + s - 1) // s
    return np.tile(last_season, (1, reps))[:, :pred_len].astype(np.float32)


def last_value_forecast(past: np.ndarray, pred_len: int) -> np.ndarray:
    """Persistence baseline: repeat the final observed value."""
    return np.repeat(past[:, -1:], pred_len, axis=1).astype(np.float32)


def moving_average_forecast(
    past: np.ndarray, pred_len: int, window: int = 24
) -> np.ndarray:
    """Repeat the mean of the last ``window`` context steps."""
    w = min(window, past.shape[1])
    mu = np.mean(past[:, -w:], axis=1, keepdims=True)
    return np.repeat(mu, pred_len, axis=1).astype(np.float32)
