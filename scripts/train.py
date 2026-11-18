"""Train RetCast-JEPA models and controlled ablations (zero-shot or in-domain)."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from retcast import RetCastConfig, RetCastJEPA
from retcast.data import ETTDataset, MultiETTTrainSampler, download_ett_datasets
from retcast.metrics import (
    compute_all_metrics,
    effective_rank,
    latent_cosine_alignment,
    latent_retrieval_r1,
)
from retcast.pretrain_data import NonETTPretrainCorpus

DATASET_NAMES = ("etth1", "etth2", "ettm1", "ettm2")
HORIZONS = (96, 192, 336)


def get_lr(step: int, total_steps: int, base_lr: float, warmup_steps: int) -> float:
    if step < warmup_steps:
        return base_lr * float(step + 1) / float(max(1, warmup_steps))
    progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return base_lr * (0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * progress)))


@torch.no_grad()
def _eval_windows_list(
    model: RetCastJEPA,
    eval_tasks: list[tuple[np.ndarray, np.ndarray, int, int]],
    device: torch.device,
    batch_size: int = 64,
) -> dict[str, float]:
    """Evaluate model across a list of (past, actual, freq_id, horizon_len) tuples."""
    was_training = model.training
    model.eval()

    mses, maes, jepa_cos_list, r1_list, erank_list = [], [], [], [], []
    for past_np, actual_np, freq_id, h_len in eval_tasks:
        preds_list = []
        z_pred_list = []
        z_tgt_list = []
        for s in range(0, past_np.shape[0], batch_size):
            p_t = torch.from_numpy(past_np[s : s + batch_size]).to(device)
            f_t = torch.full(
                (p_t.shape[0],), freq_id, dtype=torch.long, device=device
            )
            out = model(p_t, horizon_len=h_len, freq=f_t, return_latents=True)
            preds_list.append(out["point"].cpu().numpy())

            fut_t = torch.from_numpy(actual_np[s : s + batch_size]).to(device)
            fut_norm = (fut_t - out["stats_mu"]) / out["stats_sigma"]
            P = model.cfg.patch_len
            K = math.ceil(h_len / P)
            rem = K * P - h_len
            if rem > 0:
                fut_pad_t = torch.nn.functional.pad(fut_norm, (0, rem), value=0.0)
                fut_mask = torch.nn.functional.pad(
                    torch.zeros_like(fut_norm), (0, rem), value=1.0
                )
            else:
                fut_pad_t = fut_norm
                fut_mask = torch.zeros_like(fut_norm)
            z_tgt = model.target_encoder(
                fut_pad_t.view(-1, K, P), fut_mask.view(-1, K, P)
            )
            z_pred_list.append(out["z_fut"].cpu())
            z_tgt_list.append(z_tgt.cpu())

        preds_np = np.concatenate(preds_list, axis=0)
        m = compute_all_metrics(preds_np, actual_np)
        mses.append(m["mse"])
        maes.append(m["mae"])

        zp = torch.cat(z_pred_list, dim=0)
        zt = torch.cat(z_tgt_list, dim=0)
        jepa_cos_list.append(latent_cosine_alignment(zp, zt))
        if h_len == 96:
            erank_list.append(effective_rank(zp))
            if zp.shape[0] >= 8:
                r1_list.append(latent_retrieval_r1(zp[:64], zt[:64]))

    if was_training:
        model.train()

    return {
        "val_mse": float(np.mean(mses)),
        "val_mae": float(np.mean(maes)),
        "val_jepa_cos": float(np.mean(jepa_cos_list)),
        "val_erank": float(np.mean(erank_list)) if erank_list else 0.0,
        "val_r1": float(np.mean(r1_list)) if r1_list else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument(
        "--mode", type=str, default="zero_shot", choices=["zero_shot", "indomain"]
    )
    parser.add_argument(
        "--preset", type=str, default="2m", choices=["tiny", "2m", "5m", "9m"]
    )
    parser.add_argument(
        "--arch", type=str, default="retnet", choices=["retnet", "transformer"]
    )
    parser.add_argument(
        "--memory", type=str, default="delta", choices=["delta", "none"]
    )
    parser.add_argument("--beta_max", type=float, default=2.0)
    parser.add_argument("--short_conv", type=int, default=4)
    parser.add_argument("--jepa_weight", type=float, default=0.25)
    parser.add_argument("--vicreg_weight", type=float, default=0.05)
    parser.add_argument("--spectral_weight", type=float, default=0.10)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--ema_tau", type=float, default=0.995)
    parser.add_argument("--steps", type=int, default=2400)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1.5e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--eval_interval", type=int, default=300)
    parser.add_argument("--synthetic_ratio", type=float, default=-1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_dir", type=str, default="data/ett")
    parser.add_argument("--non_ett_dir", type=str, default="data/non_ett")
    parser.add_argument("--out_dir", type=str, default="runs")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = (
        torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )

    run_dir = Path(args.out_dir) / args.name
    run_dir.mkdir(parents=True, exist_ok=True)

    syn_ratio = args.synthetic_ratio
    if syn_ratio < 0.0:
        syn_ratio = 0.25 if args.mode == "zero_shot" else 0.15

    if args.mode == "zero_shot":
        corpus = NonETTPretrainCorpus(
            cache_dir=args.non_ett_dir,
            context_len=512,
            horizon_len=336,
            synthetic_ratio=syn_ratio,
            seed=args.seed,
        )
        sampler = corpus
        val_tasks = [
            (*corpus.get_val_windows(pred_len=h, max_windows=320), h)
            for h in HORIZONS
        ]
        assert all("ett" not in s.lower() for s in corpus.source_names)
        print(
            f"[data] zero_shot non-ETT corpus: {len(corpus.series_entries)} series "
            f"across {len(corpus.active_datasets)} domains (0 ETT files)",
            flush=True,
        )
    else:
        paths = download_ett_datasets(args.data_dir, names=DATASET_NAMES)
        datasets = [ETTDataset(n, csv_path=paths[n]) for n in DATASET_NAMES]
        sampler = MultiETTTrainSampler(
            datasets,
            context_len=512,
            horizon_len=336,
            synthetic_ratio=syn_ratio,
            seed=args.seed,
        )
        val_tasks = [
            (*ds.get_eval_windows(split="val", context_len=512, pred_len=h, stride=h), h)
            for ds in datasets
            for h in HORIZONS
        ]

    cfg = RetCastConfig.from_preset(
        args.preset,
        arch=args.arch,
        memory=args.memory,
        beta_max=args.beta_max,
        short_conv=args.short_conv,
        jepa_weight=args.jepa_weight,
        vicreg_weight=args.vicreg_weight,
        spectral_weight=args.spectral_weight,
        dropout=args.dropout,
        ema_tau=args.ema_tau,
        decoder_mode="universal",
    )
    model = RetCastJEPA(cfg).to(device)
    print(
        f"[train] {args.name} | mode={args.mode} preset={args.preset} arch={cfg.arch} "
        f"memory={cfg.memory} conv={cfg.short_conv} jepa_w={cfg.jepa_weight} "
        f"| online_params={model.num_params():,} | device={device}",
        flush=True,
    )

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )

    start_step = 0
    best_val_mse = float("inf")
    latest_ckpt = run_dir / "latest.pt"
    best_ckpt = run_dir / "best.pt"
    metrics_path = run_dir / "metrics.jsonl"

    if args.resume and latest_ckpt.exists():
        ckpt = torch.load(latest_ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        start_step = int(ckpt["step"])
        best_val_mse = float(ckpt.get("best_val_mse", float("inf")))
        print(f"[resume] {args.name} from step={start_step} best_val_mse={best_val_mse:.4f}")

    warmup_steps = max(50, int(0.08 * args.steps))
    if start_step > 0:
        sampler.rng = np.random.default_rng(args.seed + start_step)
    rng = np.random.default_rng(args.seed + 1000 + start_step)
    t0 = time.time()

    running: dict[str, float] = {}
    n_acc = 0
    model.train()
    for step in range(start_step + 1, args.steps + 1):
        lr = get_lr(step - 1, args.steps, args.lr, warmup_steps)
        for pg in opt.param_groups:
            pg["lr"] = lr

        h_len = int(rng.choice([96, 192, 336], p=[0.3, 0.3, 0.4]))
        ctx, fut, freq = sampler.sample_batch(
            args.batch_size, horizon_len=h_len, augment=True, device=device
        )
        loss, m = model.compute_loss(ctx, fut, freq=freq)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        tau = min(
            cfg.ema_tau,
            0.98 + (cfg.ema_tau - 0.98) * (step / max(1, warmup_steps)),
        )
        model.update_target_encoder(tau=tau)

        for k, v in m.items():
            running[k] = running.get(k, 0.0) + v
        n_acc += 1

        if step % args.eval_interval == 0 or step == args.steps:
            elapsed = time.time() - t0
            avg_m = {k: v / max(1, n_acc) for k, v in running.items()}
            running.clear()
            n_acc = 0

            val_m = _eval_windows_list(model, val_tasks, device)
            record = {
                "step": step,
                "mode": args.mode,
                "lr": lr,
                "elapsed_s": round(elapsed, 2),
                **avg_m,
                **val_m,
            }
            with open(metrics_path, "a") as f:
                f.write(json.dumps(record) + "\n")

            is_best = val_m["val_mse"] < best_val_mse
            if is_best:
                best_val_mse = val_m["val_mse"]
                torch.save(
                    {
                        "model": model.state_dict(),
                        "cfg": cfg.to_dict(),
                        "mode": args.mode,
                        "step": step,
                        "val_mse": best_val_mse,
                        "val_metrics": val_m,
                        "online_params": model.num_params(),
                    },
                    best_ckpt,
                )

            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": opt.state_dict(),
                    "cfg": cfg.to_dict(),
                    "mode": args.mode,
                    "step": step,
                    "best_val_mse": best_val_mse,
                    "online_params": model.num_params(),
                },
                latest_ckpt,
            )
            star = " *" if is_best else ""
            print(
                f"[{args.name}] step={step:4d}/{args.steps} "
                f"loss={avg_m['loss']:.4f} fc={avg_m['forecast_loss']:.4f} "
                f"jepa={avg_m['jepa_loss']:.4f}(cos={avg_m['jepa_cos']:.3f}) "
                f"| val_mse={val_m['val_mse']:.4f} val_mae={val_m['val_mae']:.4f} "
                f"erank={val_m['val_erank']:.1f} r1={val_m['val_r1']:.3f} "
                f"| {elapsed:.1f}s{star}",
                flush=True,
            )


if __name__ == "__main__":
    main()
