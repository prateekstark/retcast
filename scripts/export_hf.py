"""Export RetCast-JEPA checkpoints as SafeTensors + Model Card and optionally push to HF Hub."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from retcast import RetCastConfig, RetCastJEPA

GITHUB_URL = "https://github.com/prateekstark/retcast"
DEFAULT_VARIANTS: tuple[str, ...] = (
    "zs_retcast_2m_jepa",
    "zs_retcast_5m_jepa",
    "zs_retcast_9m_jepa",
    "zs_ablation_2m_raw",
    "zs_ablation_2m_nodelta",
    "zs_ablation_2m_transformer",
    "retcast_2m_jepa",
    "retcast_5m_jepa",
    "retcast_9m_jepa",
)


def _load_ckpt_meta(ckpt_path: Path) -> tuple[RetCastJEPA, dict[str, Any]]:
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
    model.eval()
    return model, ckpt


def build_model_card(
    exported: list[dict[str, Any]],
    results_dir: Path,
    repo_id: str,
) -> str:
    bench_path = results_dir / "benchmark_vs_timesfm.json"
    abl_path = results_dir / "ablations.json"
    eff_path = results_dir / "efficiency.json"

    bench_rows = []
    if bench_path.exists():
        bench = json.loads(bench_path.read_text())
        eff_models = (
            json.loads(eff_path.read_text()).get("models", {})
            if eff_path.exists()
            else {}
        )
        for k in (
            "timesfm_200m",
            "retcast_2m_jepa",
            "retcast_5m_jepa",
            "retcast_9m_jepa",
            "seasonal_naive",
        ):
            if k not in bench.get("models", {}):
                continue
            m = bench["models"][k]
            s = m["summary"]
            p = f"{m['params'] / 1e6:.2f}M" if m["params"] > 0 else "0"
            em = eff_models.get(k, {})
            lat = f"{em['batch_latency_ms']:.2f} ms" if "batch_latency_ms" in em else "—"
            vram = f"{em['peak_vram_mb']:.1f} MB" if "peak_vram_mb" in em else "—"
            bench_rows.append(
                f"| {m['label']} | {p} | {s['mean_mse']:.4f} | {s['mean_mae']:.4f} | "
                f"{s['mean_wape']:.4f} | {s['mean_nrmse']:.4f} | {lat} | {vram} |"
            )

    bench_table = (
        "| Model | Params | Mean MSE ↓ | Mean MAE ↓ | Mean WAPE ↓ | Mean NRMSE ↓ | Latency (B=16, L=512) | Peak VRAM |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|\n"
        + "\n".join(bench_rows)
        if bench_rows
        else "See `results/benchmark_vs_timesfm.json`."
    )

    abl_rows = []
    if abl_path.exists():
        abl = json.loads(abl_path.read_text())
        for k in (
            "retcast_2m_jepa",
            "ablation_2m_raw",
            "ablation_2m_nodelta",
            "ablation_2m_transformer",
        ):
            if k not in abl.get("models", {}):
                continue
            m = abl["models"][k]
            s = m["summary"]
            abl_rows.append(
                f"| {m['label']} | {m['params']:,} | {s['mean_mse']:.4f} | {s['mean_mae']:.4f} | "
                f"{s['mean_jepa_cos']:.3f} | {s['mean_effective_rank']:.1f} | {s['mean_retrieval_r1']:.3f} |"
            )

    abl_table = (
        "| Variant (2M Zero-Shot) | Params | Mean MSE ↓ | Mean MAE ↓ | JEPA Cosine ↑ | Eff. Rank ↑ | Retrieval R@1 ↑ |\n"
        "|---|---:|---:|---:|---:|---:|---:|\n"
        + "\n".join(abl_rows)
        if abl_rows
        else "See `results/ablations.json`."
    )

    variant_rows = "\n".join(
        f"| `{v['variant']}` | {v['mode']} | {v['arch']} (`{v['memory']}`) | {v['online_params']:,} | {v['step']:,} | {v['size_mb']:.2f} MB |"
        for v in exported
    )

    return f"""---
license: mit
library_name: pytorch
pipeline_tag: time-series-forecasting
tags:
  - retnet
  - retnet2
  - jepa
  - delta-rule
  - time-series
  - forecasting
  - zero-shot
  - vicreg
datasets:
  - autogluon/chronos_datasets
---

# RetCast-JEPA (2M–9M RetNet2.0 + JEPA Time-Series Forecasters)

**Zero-shot time-series foundation models (2.05M, 5.11M, and 9.19M parameters) built from scratch with RetNet2.0 (delta-rule overwritable memory + causal depthwise short convolution + multi-scale retention) and Yann LeCun's JEPA ("predict what the future means, not its raw values").**

Code, training scripts, tests, and full benchmark logs live at **[{GITHUB_URL}]({GITHUB_URL})**.

---

## Zero-Shot vs. Zero-Shot Benchmark (12 Held-Out ETT Tasks)

Pretrained strictly on `NonETTPretrainCorpus` (**1,345 real series across 9 non-ETT domains** from `autogluon/chronos_datasets` + UCI `solar_AL` + multi-rate temporal resampling + 25% synthetic trajectories; **0 ETT files seen during pretraining**), and evaluated head-to-head against Google's `timesfm-1.0-200m-pytorch` on `ETTh1`, `ETTh2`, `ETTm1`, and `ETTm2` at horizons $H \\in \\{{96, 192, 336\\}}$ (`context_len=512`, `stride=pred_len`):

{bench_table}

![Zero-shot comparison vs Google TimesFM 200M](assets/timesfm_vs_retcast.png)

### Controlled 2M Zero-Shot Ablations

{abl_table}

---

## Available Variants in This Repository

| Variant | Regime | Backbone | Online Params | Step | SafeTensors Size |
|---|---|---|---:|---:|---:|
{variant_rows}

Convenient aliases supported by `RetCastJEPA.from_pretrained`:
- `"2m"` / `"zs_2m"` → `"zs_retcast_2m_jepa"`
- `"5m"` / `"zs_5m"` → `"zs_retcast_5m_jepa"`
- `"9m"` / `"zs_9m"` → `"zs_retcast_9m_jepa"`
- `"indomain_2m"` / `"indomain_5m"` / `"indomain_9m"` → `"retcast_2m_jepa"` / `"retcast_5m_jepa"` / `"retcast_9m_jepa"`

---

## Quickstart

Clone [{GITHUB_URL}]({GITHUB_URL}) (or install with `pip install -e .`), then load any variant directly from HuggingFace Hub:

```python
import torch
from retcast import RetCastJEPA

# Load zero-shot RetCast-JEPA 2M directly from HuggingFace Hub
model = RetCastJEPA.from_pretrained("{repo_id}", variant="2m", device="cuda")

# Univariate context batch: shape (B, L)
context = torch.randn(4, 512, device="cuda")
with torch.no_grad():
    out = model(context, horizon_len=192, mode="chunkwise")

point = out["point"]          # (4, 192)
quantiles = out["quantiles"]  # (4, 192, 10) -> [point, q10, q20, ..., q90]
```

### O(1)-Memory Recurrent Latent Rollout

```python
with torch.no_grad():
    ctx_norm, stats = model.revin.normalize(context)
    patches, pad = model._patchify(ctx_norm)
    z_ctx = model.encoder(patches, pad) + model.freq_emb(torch.zeros(4, dtype=torch.long, device="cuda")).unsqueeze(1)
    h_ctx, state = model.forward_backbone(z_ctx, mode="chunkwise", return_state=True)
    z_rollout, next_state = model.rollout_latent_recurrent(h_ctx, state, offset=patches.shape[1], n_patches=6)
```

---

## Why RetNet2.0 + JEPA?

1. **Parallel training + chunkwise prefill + $O(1)$ recurrent state (`830 KB` fixed at $L=16,384$ vs `30,750 KB` Transformer KV cache)**: **1.85× faster** and **2.72× lower VRAM** than a parameter-matched Transformer at $L=16,384$.
2. **Delta-rule overwritable memory ($\\beta_{{\\max}}=2.0$) + causal `short_conv=4`**: Erases stale regime projections before writing new observations and enables negative eigenvalues in $(-1, 1)$ for cycle/phase tracking, reducing zero-shot MSE by **-7.0%** over plain RetNet (`0.3706` vs `0.3985`) and **-11.9%** over Causal Transformer (`0.3706` vs `0.4209`).
3. **JEPA + VICReg**: Predicts stop-gradient EMA time-frequency patch semantics rather than raw point noise alone, yielding **0.809–0.842** future latent cosine alignment and **5.4× higher** future-horizon retrieval R@1.

---

## License

Code and weights are released under the **MIT License**. Pretraining datasets (`autogluon/chronos_datasets`, UCI `solar_AL`) retain their respective upstream licenses.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_dir", type=str, default="runs")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--assets_dir", type=str, default="assets")
    parser.add_argument("--out", type=str, default="export/retcast-jepa")
    parser.add_argument("--variants", nargs="*", default=list(DEFAULT_VARIANTS))
    parser.add_argument("--push", type=str, default=None, help="HF repo id, e.g. starkprateek1/retcast-jepa")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    from safetensors.torch import save_file

    runs_dir = Path(args.runs_dir)
    results_dir = Path(args.results_dir)
    assets_dir = Path(args.assets_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    exported: list[dict[str, Any]] = []
    for variant in args.variants:
        ckpt_path = runs_dir / variant / "best.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

        model, ckpt = _load_ckpt_meta(ckpt_path)
        sd = {k: v.detach().cpu().float().contiguous() for k, v in model.state_dict().items()}

        v_dir = out_dir / variant
        v_dir.mkdir(parents=True, exist_ok=True)
        weights_file = v_dir / "model.safetensors"
        save_file(
            sd,
            str(weights_file),
            metadata={"format": "pt", "variant": variant, "step": str(ckpt["step"])},
        )

        mode = ckpt.get("mode", "zero_shot" if variant.startswith("zs_") else "in_domain")
        cfg_payload = {
            "architecture": "RetCastJEPA",
            "variant": variant,
            "mode": mode,
            "step": int(ckpt["step"]),
            "val_mse": float(ckpt.get("val_mse", 0.0)),
            "online_params": model.num_params(),
            "total_params": model.num_params(include_target_encoder=True),
            "model": model.cfg.to_dict(),
            "code": GITHUB_URL,
        }
        (v_dir / "config.json").write_text(json.dumps(cfg_payload, indent=2))
        size_mb = weights_file.stat().st_size / (1024**2)
        exported.append(
            {
                "variant": variant,
                "mode": mode,
                "arch": model.cfg.arch,
                "memory": model.cfg.memory,
                "online_params": model.num_params(),
                "step": int(ckpt["step"]),
                "size_mb": size_mb,
            }
        )
        print(f"[export] {variant:<28s} -> {weights_file} ({size_mb:.2f} MB, step={ckpt['step']})")

    # Copy benchmark results & assets for the HuggingFace repo
    if results_dir.exists():
        dst_res = out_dir / "results"
        dst_res.mkdir(parents=True, exist_ok=True)
        for p in sorted(results_dir.glob("*.json")):
            shutil.copy2(p, dst_res / p.name)
    if assets_dir.exists():
        dst_assets = out_dir / "assets"
        dst_assets.mkdir(parents=True, exist_ok=True)
        for p in sorted(assets_dir.glob("*.png")):
            shutil.copy2(p, dst_assets / p.name)

    license_file = ROOT / "LICENSE"
    if license_file.exists():
        shutil.copy2(license_file, out_dir / "LICENSE")

    repo_id = args.push or "starkprateek1/retcast-jepa"
    (out_dir / "README.md").write_text(build_model_card(exported, results_dir, repo_id))
    print(f"[export] wrote model card -> {out_dir / 'README.md'}")

    if args.push:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(
            repo_id=args.push,
            repo_type="model",
            private=args.private,
            exist_ok=True,
        )
        api.upload_folder(
            folder_path=str(out_dir),
            repo_id=args.push,
            repo_type="model",
            commit_message="Upload RetCast-JEPA 2M/5M/9M zero-shot & ablation SafeTensors checkpoints",
        )
        print(f"[hf] uploaded -> https://huggingface.co/{args.push}")


if __name__ == "__main__":
    main()
