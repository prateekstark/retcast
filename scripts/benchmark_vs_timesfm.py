"""Head-to-head benchmark: RetCast-JEPA (1-10M) vs Google TimesFM-1.0-200M & baselines.

Evaluates all models on the exact held-out test splits of ETTh1, ETTh2, ETTm1,
ETTm2 at horizons {96, 192, 336} with context_len=512 and stride=pred_len,
computes representation metrics for ablations, and benchmarks inference speed,
peak VRAM, and long-context scaling (L=512..16384).
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from retcast import RetCastConfig, RetCastJEPA, retnet_state_bytes
from retcast.baselines import (
    last_value_forecast,
    moving_average_forecast,
    seasonal_naive_forecast,
)
from retcast.data import ETTDataset, download_ett_datasets
from retcast.metrics import (
    compute_all_metrics,
    effective_rank,
    latent_cosine_alignment,
    latent_retrieval_r1,
)
from retcast.timesfm_wrapper import TimesFM200MForecaster

DATASETS = ("etth1", "etth2", "ettm1", "ettm2")
HORIZONS = (96, 192, 336)

MAIN_MODELS = [
    ("retcast_2m_jepa", "RetCast-JEPA 2M (Zero-Shot)", "runs/zs_retcast_2m_jepa/best.pt"),
    ("retcast_5m_jepa", "RetCast-JEPA 5M (Zero-Shot)", "runs/zs_retcast_5m_jepa/best.pt"),
    ("retcast_9m_jepa", "RetCast-JEPA 9M (Zero-Shot)", "runs/zs_retcast_9m_jepa/best.pt"),
]

ABLATION_MODELS = [
    ("retcast_2m_jepa", "RetNet2.0 + JEPA (2M)", "runs/zs_retcast_2m_jepa/best.pt"),
    ("ablation_2m_raw", "RetNet2.0 Raw (No JEPA)", "runs/zs_ablation_2m_raw/best.pt"),
    (
        "ablation_2m_nodelta",
        "Plain RetNet + JEPA",
        "runs/zs_ablation_2m_nodelta/best.pt",
    ),
    (
        "ablation_2m_transformer",
        "Transformer + JEPA",
        "runs/zs_ablation_2m_transformer/best.pt",
    ),
]

INDOMAIN_MODELS = [
    ("indomain_retcast_2m_jepa", "RetCast-JEPA 2M (In-Domain)", "runs/retcast_2m_jepa/best.pt"),
    ("indomain_retcast_5m_jepa", "RetCast-JEPA 5M (In-Domain)", "runs/retcast_5m_jepa/best.pt"),
    ("indomain_retcast_9m_jepa", "RetCast-JEPA 9M (In-Domain)", "runs/retcast_9m_jepa/best.pt"),
]


def _clear_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def _load_retcast_checkpoint(
    ckpt_path: Path, device: torch.device
) -> tuple[RetCastJEPA, dict[str, Any]]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg_dict = dict(ckpt["cfg"])
    if "quantiles" in cfg_dict:
        cfg_dict["quantiles"] = tuple(cfg_dict["quantiles"])
    if "decoder_mode" not in cfg_dict:
        cfg_dict["decoder_mode"] = (
            "legacy" if "decoder.skip_gate.weight" in ckpt["model"] else "universal"
        )
    cfg = RetCastConfig(**cfg_dict)
    model = RetCastJEPA(cfg)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model, ckpt


@torch.no_grad()
def _eval_retcast_task(
    model: RetCastJEPA,
    past: np.ndarray,
    actuals: np.ndarray,
    freq_id: int,
    pred_len: int,
    device: torch.device,
    batch_size: int = 64,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    preds_pt = []
    preds_q = []
    z_pred_list = []
    z_tgt_list = []
    P = model.cfg.patch_len
    K = math.ceil(pred_len / P)
    rem = K * P - pred_len

    for s in range(0, past.shape[0], batch_size):
        p_t = torch.from_numpy(past[s : s + batch_size]).to(device)
        f_t = torch.full((p_t.shape[0],), freq_id, dtype=torch.long, device=device)
        out = model(p_t, horizon_len=pred_len, freq=f_t, return_latents=True)
        preds_pt.append(out["point"].cpu().numpy())
        preds_q.append(out["quantiles"].cpu().numpy())

        fut_t = torch.from_numpy(actuals[s : s + batch_size]).to(device)
        fut_norm = (fut_t - out["stats_mu"]) / out["stats_sigma"]
        if rem > 0:
            fut_pad = torch.nn.functional.pad(fut_norm, (0, rem), value=0.0)
            fut_mask = torch.nn.functional.pad(
                torch.zeros_like(fut_norm), (0, rem), value=1.0
            )
        else:
            fut_pad = fut_norm
            fut_mask = torch.zeros_like(fut_norm)
        z_tgt = model.target_encoder(
            fut_pad.view(-1, K, P), fut_mask.view(-1, K, P)
        )
        z_pred_list.append(out["z_fut"].cpu())
        z_tgt_list.append(z_tgt.cpu())

    pt_np = np.concatenate(preds_pt, axis=0)
    q_np = np.concatenate(preds_q, axis=0)
    zp = torch.cat(z_pred_list, dim=0)
    zt = torch.cat(z_tgt_list, dim=0)
    rep = {
        "jepa_cos": latent_cosine_alignment(zp, zt),
        "effective_rank": effective_rank(zp),
        "retrieval_r1": latent_retrieval_r1(zp[:64], zt[:64])
        if zp.shape[0] >= 8
        else 0.0,
    }
    return pt_np, q_np, rep


def _summarize_model_tasks(tasks: dict[str, dict[str, Any]]) -> dict[str, float]:
    keys = ("mse", "mae", "smape", "wape", "nrmse")
    summary: dict[str, float] = {}
    for k in keys:
        summary[f"mean_{k}"] = float(np.mean([v[k] for v in tasks.values()]))
    if any("eval_time_s" in v for v in tasks.values()):
        summary["total_eval_time_s"] = float(
            sum(v.get("eval_time_s", 0.0) for v in tasks.values())
        )
    return summary


def benchmark_efficiency(
    device: torch.device,
    runs_dir: Path,
    checkpoint_path: Path | None = None,
) -> dict[str, Any]:
    """Measure throughput, latency, VRAM, and L=512..16384 context scaling."""
    eff: dict[str, Any] = {"device": str(device), "models": {}, "context_scaling": {}}

    def _sync() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize()

    # 1. Fixed-window inference comparison (B=16, L=512, H=96)
    B, L, H = 16, 512, 96
    x_np = np.random.default_rng(0).standard_normal((B, L)).astype(np.float32)

    # TimesFM-200M
    _clear_cuda()
    tfm = TimesFM200MForecaster(device=device, checkpoint_path=checkpoint_path)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    # Warmup
    tfm.forecast(x_np[:4], freq_id=0, pred_len=H, batch_size=4)
    _sync()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(3):
        tfm.forecast(x_np, freq_id=0, pred_len=H, batch_size=B)
    _sync()
    dt = (time.perf_counter() - t0) / 3.0
    peak_mb = (
        torch.cuda.max_memory_allocated() / (1024**2)
        if device.type == "cuda"
        else 0.0
    )
    eff["models"]["timesfm_200m"] = {
        "label": "TimesFM-1.0 (200M)",
        "params": tfm.num_params(),
        "checkpoint_mb": round(tfm.checkpoint_bytes() / (1024**2), 2),
        "batch_latency_ms": round(dt * 1000.0, 2),
        "series_per_sec": round(B / dt, 2),
        "peak_vram_mb": round(peak_mb, 2),
    }
    del tfm
    _clear_cuda()

    # RetCast models & Transformer ablation
    bench_entries = [
        ("retcast_2m_jepa", "RetCast-JEPA 2M", runs_dir / "zs_retcast_2m_jepa/best.pt"),
        ("retcast_5m_jepa", "RetCast-JEPA 5M", runs_dir / "zs_retcast_5m_jepa/best.pt"),
        ("retcast_9m_jepa", "RetCast-JEPA 9M", runs_dir / "zs_retcast_9m_jepa/best.pt"),
        (
            "ablation_2m_transformer",
            "Transformer-JEPA 2M",
            runs_dir / "zs_ablation_2m_transformer/best.pt",
        ),
    ]
    x_t = torch.from_numpy(x_np).to(device)
    f_t = torch.zeros(B, dtype=torch.long, device=device)

    for key, label, ckpt_path in bench_entries:
        if not ckpt_path.exists():
            continue
        _clear_cuda()
        model, _ = _load_retcast_checkpoint(ckpt_path, device)
        with torch.no_grad():
            model(x_t[:4], horizon_len=H, freq=f_t[:4])
            _sync()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            for _ in range(10):
                model(x_t, horizon_len=H, freq=f_t)
            _sync()
            dt = (time.perf_counter() - t0) / 10.0
        peak_mb = (
            torch.cuda.max_memory_allocated() / (1024**2)
            if device.type == "cuda"
            else 0.0
        )
        eff["models"][key] = {
            "label": label,
            "params": model.num_params(),
            "checkpoint_mb": round(ckpt_path.stat().st_size / (1024**2), 2),
            "batch_latency_ms": round(dt * 1000.0, 2),
            "series_per_sec": round(B / dt, 2),
            "peak_vram_mb": round(peak_mb, 2),
        }
        del model
        _clear_cuda()

    # 2. Long-context scaling: L = 512 .. 16384 (B=4, H=96)
    ctx_lengths = [512, 1024, 2048, 4096, 8192, 16384]
    scale_configs = [
        (
            "retnet2_chunkwise",
            "RetCast-JEPA 2M (chunkwise)",
            runs_dir / "zs_retcast_2m_jepa/best.pt",
            "chunkwise",
        ),
        (
            "retnet2_parallel",
            "RetCast-JEPA 2M (parallel)",
            runs_dir / "zs_retcast_2m_jepa/best.pt",
            "parallel",
        ),
        (
            "transformer_kv",
            "Transformer-JEPA 2M (KV cache)",
            runs_dir / "zs_ablation_2m_transformer/best.pt",
            "parallel",
        ),
    ]
    B_sc = 4
    for s_key, s_label, ckpt_path, mode in scale_configs:
        if not ckpt_path.exists():
            continue
        rows = []
        model, _ = _load_retcast_checkpoint(ckpt_path, device)
        for ctx_len in ctx_lengths:
            _clear_cuda()
            x_sc = torch.randn(B_sc, ctx_len, device=device)
            f_sc = torch.zeros(B_sc, dtype=torch.long, device=device)
            try:
                with torch.no_grad():
                    # Warmup + measure backbone state size
                    p_sc, pad_sc = model._patchify(x_sc)
                    z_sc = model.encoder(p_sc, pad_sc) + model.freq_emb(f_sc).unsqueeze(1)
                    _, st_warm = model.forward_backbone(
                        z_sc, mode=mode, chunk_size=64, return_state=True
                    )
                    state_kb = retnet_state_bytes(st_warm) / 1024.0
                    del st_warm, p_sc, pad_sc, z_sc

                    _sync()
                    if device.type == "cuda":
                        torch.cuda.reset_peak_memory_stats()
                    t0 = time.perf_counter()
                    iters = 5 if ctx_len <= 4096 else 3
                    for _ in range(iters):
                        model(x_sc, horizon_len=96, freq=f_sc, mode=mode, chunk_size=64)
                    _sync()
                    ms = ((time.perf_counter() - t0) / iters) * 1000.0
                    peak_mb = (
                        torch.cuda.max_memory_allocated() / (1024**2)
                        if device.type == "cuda"
                        else 0.0
                    )
                    rows.append(
                        {
                            "context_len": ctx_len,
                            "num_patches": ctx_len // model.cfg.patch_len,
                            "latency_ms": round(ms, 2),
                            "peak_vram_mb": round(peak_mb, 2),
                            "state_kb": round(state_kb, 2),
                            "oom": False,
                        }
                    )
            except torch.OutOfMemoryError:
                rows.append(
                    {
                        "context_len": ctx_len,
                        "num_patches": ctx_len // model.cfg.patch_len,
                        "latency_ms": None,
                        "peak_vram_mb": None,
                        "state_kb": None,
                        "oom": True,
                    }
                )
            del x_sc, f_sc
            _clear_cuda()
        del model
        _clear_cuda()
        eff["context_scaling"][s_key] = {"label": s_label, "rows": rows}

    return eff


def verify_results(results_dir: Path) -> int:
    """Validate that all benchmark result files exist and pass sanity checks."""
    req_files = [
        results_dir / "benchmark_vs_timesfm.json",
        results_dir / "ablations.json",
        results_dir / "efficiency.json",
    ]
    for f in req_files:
        if not f.exists():
            print(f"[FAIL] Missing required file: {f}")
            return 1

    bench = json.loads((results_dir / "benchmark_vs_timesfm.json").read_text())
    abl = json.loads((results_dir / "ablations.json").read_text())
    eff = json.loads((results_dir / "efficiency.json").read_text())

    expected_models = [
        "timesfm_200m",
        "retcast_2m_jepa",
        "retcast_5m_jepa",
        "retcast_9m_jepa",
        "seasonal_naive",
        "last_value",
        "moving_average",
    ]
    expected_tasks = [f"{d}_H{h}" for d in DATASETS for h in HORIZONS]

    for m in expected_models:
        if m not in bench["models"]:
            print(f"[FAIL] Model {m!r} missing from benchmark_vs_timesfm.json")
            return 1
        tasks = bench["models"][m]["tasks"]
        for t in expected_tasks:
            if t not in tasks:
                print(f"[FAIL] Task {t!r} missing for model {m!r}")
                return 1
            for metric in ("mse", "mae", "smape", "wape", "nrmse"):
                val = tasks[t][metric]
                if not np.isfinite(val):
                    print(f"[FAIL] Non-finite {metric} in {m}/{t}")
                    return 1

    for a in (
        "retcast_2m_jepa",
        "ablation_2m_raw",
        "ablation_2m_nodelta",
        "ablation_2m_transformer",
    ):
        if a not in abl["models"]:
            print(f"[FAIL] Ablation {a!r} missing from ablations.json")
            return 1

    if "timesfm_200m" not in eff["models"] or "retcast_2m_jepa" not in eff["models"]:
        print("[FAIL] Missing efficiency entries")
        return 1

    print("\n=== VERIFICATION PASSED ===")
    print(f"{'Model':<26s} {'Params':>11s} {'Mean MSE':>10s} {'Mean MAE':>10s} {'Mean WAPE':>10s}")
    print("-" * 72)
    for m in expected_models:
        entry = bench["models"][m]
        s = entry["summary"]
        p = entry["params"]
        p_str = f"{p/1e6:.2f}M" if p > 0 else "0"
        print(
            f"{entry['label']:<26s} {p_str:>11s} "
            f"{s['mean_mse']:>10.4f} {s['mean_mae']:>10.4f} {s['mean_wape']:>10.4f}"
        )
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/ett")
    parser.add_argument("--runs_dir", type=str, default="runs")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--timesfm_batch", type=int, default=16)
    parser.add_argument("--retcast_batch", type=int, default=64)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if args.verify:
        sys.exit(verify_results(results_dir))

    results_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = Path(args.runs_dir)
    device = (
        torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )

    paths = download_ett_datasets(args.data_dir, names=DATASETS)
    datasets = {n: ETTDataset(n, csv_path=paths[n]) for n in DATASETS}

    # Pre-extract all 12 test tasks so every model sees identical arrays
    test_windows: dict[str, tuple[np.ndarray, np.ndarray, int, int]] = {}
    for name, ds in datasets.items():
        for h in HORIZONS:
            past, actual, freq_id = ds.get_eval_windows(
                split="test", context_len=512, pred_len=h, stride=h
            )
            test_windows[f"{name}_H{h}"] = (past, actual, freq_id, ds.cfg.seasonality)

    benchmark: dict[str, Any] = {
        "protocol": {
            "mode": "zero_shot_both",
            "context_len": 512,
            "horizons": list(HORIZONS),
            "datasets": list(DATASETS),
            "stride": "pred_len",
            "normalization": "StandardScaler fit strictly on train split [0:boundaries[0]]",
            "retcast_pretraining": "Non-ETT multi-domain corpus (chronos_datasets + solar_al + synthetic; 0 ETT files)",
            "timesfm_repo": TimesFM200MForecaster.REPO_ID,
            "timesfm_point_mode": "median",
        },
        "models": {},
        "indomain_models": {},
        "sample_forecasts": {},
    }

    # Pick one representative test window (window 0, OT channel = col index 6) per dataset at H=192
    for name in DATASETS:
        past, actual, _, _ = test_windows[f"{name}_H192"]
        sample_idx = 6  # OT (Oil Temperature) column in first test window
        benchmark["sample_forecasts"][name] = {
            "context_tail": past[sample_idx, -96:].tolist(),
            "actual": actual[sample_idx].tolist(),
            "horizon": 192,
            "series_name": "OT (window 0)",
        }

    # 1. Sanity baselines
    for b_key, b_label in (
        ("seasonal_naive", "Seasonal Naive"),
        ("last_value", "Last Value (Persistence)"),
        ("moving_average", "Moving Average (24)"),
    ):
        tasks: dict[str, Any] = {}
        for t_key, (past, actual, _, seas) in test_windows.items():
            h = actual.shape[1]
            t0 = time.perf_counter()
            if b_key == "seasonal_naive":
                pred = seasonal_naive_forecast(past, h, seasonality=seas)
            elif b_key == "last_value":
                pred = last_value_forecast(past, h)
            else:
                pred = moving_average_forecast(past, h, window=seas)
            dt = time.perf_counter() - t0
            m = compute_all_metrics(pred, actual)
            m["eval_time_s"] = round(dt, 4)
            m["num_windows"] = int(past.shape[0])
            tasks[t_key] = m
        benchmark["models"][b_key] = {
            "label": b_label,
            "params": 0,
            "tasks": tasks,
            "summary": _summarize_model_tasks(tasks),
        }

    # 2. Google TimesFM-1.0-200M
    print("[bench] Evaluating Google TimesFM-1.0-200M...", flush=True)
    _clear_cuda()
    tfm = TimesFM200MForecaster(device=device)
    tfm_tasks: dict[str, Any] = {}
    for t_key, (past, actual, freq_id, _) in test_windows.items():
        h = actual.shape[1]
        t0 = time.perf_counter()
        pt, full_q = tfm.forecast(
            past,
            freq_id=freq_id,
            pred_len=h,
            batch_size=args.timesfm_batch,
            point_mode="median",
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        m = compute_all_metrics(pt, actual)
        m["eval_time_s"] = round(dt, 3)
        m["num_windows"] = int(past.shape[0])
        tfm_tasks[t_key] = m
        if t_key.endswith("_H192"):
            ds_name = t_key.split("_")[0]
            benchmark["sample_forecasts"][ds_name]["timesfm_200m"] = {
                "point": pt[6].tolist(),
                "q10": full_q[6, :, 1].tolist(),
                "q90": full_q[6, :, 9].tolist(),
            }
        print(
            f"  [timesfm_200m] {t_key:<10s} MSE={m['mse']:.4f} MAE={m['mae']:.4f} ({dt:.2f}s)",
            flush=True,
        )
    benchmark["models"]["timesfm_200m"] = {
        "label": "Google TimesFM 200M",
        "params": tfm.num_params(),
        "checkpoint_mb": round(tfm.checkpoint_bytes() / (1024**2), 2),
        "tasks": tfm_tasks,
        "summary": _summarize_model_tasks(tfm_tasks),
    }
    del tfm
    _clear_cuda()

    # 3. RetCast-JEPA models (2M, 5M, 9M) & Ablations
    ablations: dict[str, Any] = {"models": {}}
    seen_evals: dict[str, dict[str, Any]] = {}

    all_entries = MAIN_MODELS + [x for x in ABLATION_MODELS if x[0] != "retcast_2m_jepa"]
    for m_key, m_label, rel_ckpt in all_entries:
        ckpt_path = Path(rel_ckpt)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        print(f"[bench] Evaluating {m_key} ({ckpt_path})...", flush=True)
        _clear_cuda()
        model, ckpt = _load_retcast_checkpoint(ckpt_path, device)

        tasks = {}
        rep_metrics = {}
        for t_key, (past, actual, freq_id, _) in test_windows.items():
            h = actual.shape[1]
            t0 = time.perf_counter()
            pt, full_q, rep = _eval_retcast_task(
                model,
                past,
                actual,
                freq_id,
                h,
                device,
                batch_size=args.retcast_batch,
            )
            if device.type == "cuda":
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            m = compute_all_metrics(pt, actual)
            m["eval_time_s"] = round(dt, 4)
            m["num_windows"] = int(past.shape[0])
            tasks[t_key] = m
            rep_metrics[t_key] = rep

            if m_key == "retcast_5m_jepa" and t_key.endswith("_H192"):
                ds_name = t_key.split("_")[0]
                benchmark["sample_forecasts"][ds_name]["retcast_5m_jepa"] = {
                    "point": pt[6].tolist(),
                    "q10": full_q[6, :, 1].tolist(),
                    "q90": full_q[6, :, 9].tolist(),
                }
            if m_key == "retcast_2m_jepa" and t_key.endswith("_H192"):
                ds_name = t_key.split("_")[0]
                benchmark["sample_forecasts"][ds_name]["retcast_2m_jepa"] = {
                    "point": pt[6].tolist(),
                    "q10": full_q[6, :, 1].tolist(),
                    "q90": full_q[6, :, 9].tolist(),
                }

        summary = _summarize_model_tasks(tasks)
        summary["mean_jepa_cos"] = float(
            np.mean([v["jepa_cos"] for v in rep_metrics.values()])
        )
        summary["mean_effective_rank"] = float(
            np.mean([v["effective_rank"] for v in rep_metrics.values()])
        )
        summary["mean_retrieval_r1"] = float(
            np.mean([v["retrieval_r1"] for v in rep_metrics.values()])
        )
        entry = {
            "label": m_label,
            "params": model.num_params(),
            "total_params": model.num_params(include_target_encoder=True),
            "best_step": int(ckpt["step"]),
            "best_val_mse": float(ckpt["val_mse"]),
            "cfg": ckpt["cfg"],
            "tasks": tasks,
            "rep_metrics": rep_metrics,
            "summary": summary,
        }
        seen_evals[m_key] = entry
        if any(m_key == x[0] for x in MAIN_MODELS):
            benchmark["models"][m_key] = entry

        print(
            f"  [{m_key}] mean_MSE={summary['mean_mse']:.4f} "
            f"mean_MAE={summary['mean_mae']:.4f} "
            f"cos={summary['mean_jepa_cos']:.3f} erank={summary['mean_effective_rank']:.1f} "
            f"r1={summary['mean_retrieval_r1']:.3f}",
            flush=True,
        )
        del model
        _clear_cuda()

    for a_key, a_label, _ in ABLATION_MODELS:
        entry = dict(seen_evals[a_key])
        entry["label"] = a_label
        ablations["models"][a_key] = entry

    # 3b. Optional secondary evaluation of in-domain checkpoints for comparison
    for m_key, m_label, rel_ckpt in INDOMAIN_MODELS:
        ckpt_path = Path(rel_ckpt)
        if not ckpt_path.exists():
            continue
        print(f"[bench] Evaluating secondary {m_key} ({ckpt_path})...", flush=True)
        _clear_cuda()
        model, ckpt = _load_retcast_checkpoint(ckpt_path, device)
        tasks = {}
        for t_key, (past, actual, freq_id, _) in test_windows.items():
            h = actual.shape[1]
            pt, _, _ = _eval_retcast_task(
                model, past, actual, freq_id, h, device, batch_size=args.retcast_batch
            )
            m = compute_all_metrics(pt, actual)
            m["num_windows"] = int(past.shape[0])
            tasks[t_key] = m
        summary = _summarize_model_tasks(tasks)
        benchmark["indomain_models"][m_key] = {
            "label": m_label,
            "params": model.num_params(),
            "tasks": tasks,
            "summary": summary,
        }
        print(
            f"  [{m_key}] mean_MSE={summary['mean_mse']:.4f} mean_MAE={summary['mean_mae']:.4f}",
            flush=True,
        )
        del model
        _clear_cuda()

    (results_dir / "benchmark_vs_timesfm.json").write_text(
        json.dumps(benchmark, indent=2)
    )
    (results_dir / "ablations.json").write_text(json.dumps(ablations, indent=2))

    # 4. Efficiency & context scaling
    print("[bench] Running efficiency & long-context scaling benchmark...", flush=True)
    eff = benchmark_efficiency(device=device, runs_dir=runs_dir)
    (results_dir / "efficiency.json").write_text(json.dumps(eff, indent=2))

    sys.exit(verify_results(results_dir))


if __name__ == "__main__":
    main()
