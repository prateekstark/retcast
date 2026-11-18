"""Generate light & dark publication figures from benchmark JSON results."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


THEMES = {
    "light": {
        "bg": "#ffffff",
        "axes_bg": "#f9fafb",
        "fg": "#111827",
        "muted": "#4b5563",
        "grid": "#e5e7eb",
        "border": "#d1d5db",
        "retcast_2m": "#2563eb",
        "retcast_5m": "#059669",
        "retcast_9m": "#7c3aed",
        "timesfm": "#dc2626",
        "naive": "#6b7280",
        "ablation_raw": "#f59e0b",
        "ablation_nodelta": "#0891b2",
        "ablation_tr": "#db2777",
    },
    "dark": {
        "bg": "#0d1117",
        "axes_bg": "#161b22",
        "fg": "#f0f6fc",
        "muted": "#8b949e",
        "grid": "#30363d",
        "border": "#30363d",
        "retcast_2m": "#60a5fa",
        "retcast_5m": "#34d399",
        "retcast_9m": "#a78bfa",
        "timesfm": "#f87171",
        "naive": "#9ca3af",
        "ablation_raw": "#fbbf24",
        "ablation_nodelta": "#22d3ee",
        "ablation_tr": "#f472b6",
    },
}


def _style_ax(ax: plt.Axes, theme: dict[str, str]) -> None:
    ax.set_facecolor(theme["axes_bg"])
    ax.tick_params(colors=theme["fg"], labelsize=9)
    ax.xaxis.label.set_color(theme["fg"])
    ax.yaxis.label.set_color(theme["fg"])
    ax.title.set_color(theme["fg"])
    for spine in ax.spines.values():
        spine.set_color(theme["border"])
    ax.grid(True, linestyle="--", linewidth=0.6, color=theme["grid"], alpha=0.8)
    ax.set_axisbelow(True)


def _save_both(fig_fn, out_dir: Path, stem: str, **kwargs) -> None:
    for mode, theme in THEMES.items():
        suffix = "" if mode == "light" else "-dark"
        fig = fig_fn(theme, **kwargs)
        fig.patch.set_facecolor(theme["bg"])
        out_path = out_dir / f"{stem}{suffix}.png"
        fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=theme["bg"])
        plt.close(fig)


def plot_timesfm_vs_retcast(theme: dict[str, str], bench: dict) -> plt.Figure:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.2))
    _style_ax(ax1, theme)
    _style_ax(ax2, theme)

    # Panel A: Grouped bar chart by dataset (mean MSE across horizons 96, 192, 336)
    datasets = ("etth1", "etth2", "ettm1", "ettm2")
    ds_labels = ("ETTh1", "ETTh2", "ETTm1", "ETTm2")
    horizons = (96, 192, 336)

    model_specs = [
        ("seasonal_naive", "Seasonal Naive (0M)", theme["naive"]),
        ("timesfm_200m", "TimesFM (200M)", theme["timesfm"]),
        ("retcast_2m_jepa", "RetCast-JEPA (2M)", theme["retcast_2m"]),
        ("retcast_5m_jepa", "RetCast-JEPA (5M)", theme["retcast_5m"]),
        ("retcast_9m_jepa", "RetCast-JEPA (9M)", theme["retcast_9m"]),
    ]

    x = np.arange(len(datasets))
    width = 0.15
    for idx, (m_key, m_label, color) in enumerate(model_specs):
        tasks = bench["models"][m_key]["tasks"]
        vals = [
            np.mean([tasks[f"{d}_H{h}"]["mse"] for h in horizons])
            for d in datasets
        ]
        offset = (idx - (len(model_specs) - 1) / 2.0) * width
        bars = ax1.bar(x + offset, vals, width * 0.92, label=m_label, color=color, alpha=0.92)
        for b, v in zip(bars, vals):
            ax1.text(
                b.get_x() + b.get_width() / 2.0,
                v + 0.01,
                f"{v:.2f}",
                ha="center",
                va="bottom",
                fontsize=6.5,
                color=theme["fg"],
                rotation=90,
            )

    ax1.set_xticks(x)
    ax1.set_xticklabels(ds_labels, fontsize=10, fontweight="bold")
    ax1.set_ylabel("Mean Test MSE (Horizons 96, 192, 336)", fontsize=10)
    ax1.set_title("Zero-Shot Per-Dataset Test MSE vs Google TimesFM 200M", fontsize=11, fontweight="bold", pad=10)
    ymax = ax1.get_ylim()[1]
    ax1.set_ylim(0, ymax * 1.18)
    leg = ax1.legend(frameon=True, fontsize=8, loc="upper right")
    leg.get_frame().set_facecolor(theme["bg"])
    leg.get_frame().set_edgecolor(theme["border"])
    for t in leg.get_texts():
        t.set_color(theme["fg"])

    # Panel B: Parameter Count vs Mean Test MSE Pareto plot
    pareto_specs = [
        ("retcast_2m_jepa", "RetCast-JEPA 2M", theme["retcast_2m"], "o"),
        ("retcast_5m_jepa", "RetCast-JEPA 5M", theme["retcast_5m"], "s"),
        ("retcast_9m_jepa", "RetCast-JEPA 9M", theme["retcast_9m"], "D"),
        ("timesfm_200m", "Google TimesFM 200M", theme["timesfm"], "^"),
    ]
    pts_x, pts_y = [], []
    label_offsets = {
        "retcast_2m_jepa": (1.45, +0.024, "left"),
        "retcast_5m_jepa": (3.20, -0.032, "left"),
        "retcast_9m_jepa": (11.5, +0.020, "left"),
        "timesfm_200m": (245.0, +0.022, "right"),
    }
    for m_key, m_label, color, marker in pareto_specs:
        entry = bench["models"][m_key]
        p_m = entry["params"] / 1e6
        mse_val = entry["summary"]["mean_mse"]
        mae_val = entry["summary"]["mean_mae"]
        pts_x.append(p_m)
        pts_y.append(mse_val)
        ax2.scatter(
            [p_m],
            [mse_val],
            s=170,
            color=color,
            marker=marker,
            edgecolors=theme["fg"],
            linewidths=1.0,
            zorder=4,
            label=f"{m_label} ({p_m:.1f}M)",
        )
        lx, dy, ha = label_offsets[m_key]
        ly = mse_val + dy
        ax2.annotate(
            f"{m_label}\nMSE={mse_val:.3f}, MAE={mae_val:.3f}",
            xy=(p_m, mse_val),
            xytext=(lx, ly),
            ha=ha,
            fontsize=8,
            color=theme["fg"],
            fontweight="bold",
            arrowprops=dict(arrowstyle="-", color=theme["muted"], lw=0.7, shrinkA=2, shrinkB=5),
        )

    # Naive reference line
    naive_mse = bench["models"]["seasonal_naive"]["summary"]["mean_mse"]
    ax2.axhline(naive_mse, color=theme["naive"], linestyle=":", linewidth=1.4, label=f"Seasonal Naive (MSE={naive_mse:.3f})")

    ax2.set_xscale("log")
    ax2.set_xlim(1.2, 400.0)
    y_lo = min(pts_y) * 0.85
    y_hi = max(max(pts_y), naive_mse) * 1.15
    ax2.set_ylim(y_lo, y_hi)
    ax2.set_xlabel("Parameters (Millions, log scale)", fontsize=10)
    ax2.set_ylabel("Grand Mean Test MSE (12 ETT Tasks)", fontsize=10)
    ax2.set_title("Zero-Shot Accuracy vs Model Size: 22x–99x Smaller than TimesFM", fontsize=11, fontweight="bold", pad=10)
    leg2 = ax2.legend(frameon=True, fontsize=8, loc="upper right")
    leg2.get_frame().set_facecolor(theme["bg"])
    leg2.get_frame().set_edgecolor(theme["border"])
    for t in leg2.get_texts():
        t.set_color(theme["fg"])

    fig.tight_layout()
    return fig


def plot_jepa_ablation(theme: dict[str, str], abl: dict) -> plt.Figure:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 5.0))
    _style_ax(ax1, theme)
    _style_ax(ax2, theme)

    specs = [
        ("retcast_2m_jepa", "RetNet2.0\n+ JEPA", theme["retcast_2m"]),
        ("ablation_2m_raw", "RetNet2.0\nRaw (No JEPA)", theme["ablation_raw"]),
        ("ablation_2m_nodelta", "Plain RetNet\n+ JEPA", theme["ablation_nodelta"]),
        ("ablation_2m_transformer", "Transformer\n+ JEPA", theme["ablation_tr"]),
    ]

    x = np.arange(len(specs))
    mses = [abl["models"][k]["summary"]["mean_mse"] for k, _, _ in specs]
    maes = [abl["models"][k]["summary"]["mean_mae"] for k, _, _ in specs]
    colors = [c for _, _, c in specs]

    w = 0.35
    b1 = ax1.bar(x - w / 2, mses, w, label="Mean Test MSE", color=colors, alpha=0.92)
    b2 = ax1.bar(x + w / 2, maes, w, label="Mean Test MAE", color=colors, alpha=0.52, hatch="//", edgecolor=theme["fg"], linewidth=0.6)

    for b, v in list(zip(b1, mses)) + list(zip(b2, maes)):
        ax1.text(
            b.get_x() + b.get_width() / 2.0,
            v + 0.008,
            f"{v:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color=theme["fg"],
            fontweight="bold",
        )

    ax1.set_xticks(x)
    ax1.set_xticklabels([lbl for _, lbl, _ in specs], fontsize=9)
    ax1.set_ylabel("Forecasting Error (lower is better)", fontsize=10)
    ax1.set_ylim(0, max(mses + maes) * 1.2)
    ax1.set_title("2M Zero-Shot Controlled Ablations: Forecast Accuracy", fontsize=11, fontweight="bold", pad=10)
    leg1 = ax1.legend(frameon=True, fontsize=8.5, loc="upper right")
    leg1.get_frame().set_facecolor(theme["bg"])
    leg1.get_frame().set_edgecolor(theme["border"])
    for t in leg1.get_texts():
        t.set_color(theme["fg"])

    # Panel B: Representation Metrics (Effective Rank & Future Retrieval R@1)
    eranks = [abl["models"][k]["summary"]["mean_effective_rank"] for k, _, _ in specs]
    r1s = [abl["models"][k]["summary"]["mean_retrieval_r1"] for k, _, _ in specs]
    coss = [abl["models"][k]["summary"]["mean_jepa_cos"] for k, _, _ in specs]

    ax2_r = ax2.twinx()
    b_er = ax2.bar(x - w / 2, eranks, w, label="Effective Rank", color=colors, alpha=0.88)
    b_r1 = ax2_r.bar(x + w / 2, r1s, w, label="Future Latent R@1", color=colors, alpha=0.45, hatch="..", edgecolor=theme["fg"], linewidth=0.6)

    for b, v in zip(b_er, eranks):
        ax2.text(
            b.get_x() + b.get_width() / 2.0,
            v + 1.0,
            f"{v:.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color=theme["fg"],
            fontweight="bold",
        )
    for b, v, c_val in zip(b_r1, r1s, coss):
        ax2_r.text(
            b.get_x() + b.get_width() / 2.0,
            v + 0.02,
            f"{v:.2f}\n(cos={c_val:.2f})",
            ha="center",
            va="bottom",
            fontsize=7,
            color=theme["fg"],
        )

    ax2.set_xticks(x)
    ax2.set_xticklabels([lbl for _, lbl, _ in specs], fontsize=9)
    ax2.set_ylabel("Latent Effective Rank (max 160)", fontsize=10)
    ax2_r.set_ylabel("Future Latent Retrieval R@1", fontsize=10, color=theme["fg"])
    ax2_r.tick_params(colors=theme["fg"], labelsize=9)
    ax2_r.spines["right"].set_color(theme["border"])
    ax2.set_ylim(0, max(eranks) * 1.36)
    ax2_r.set_ylim(0, 1.18)
    ax2.set_title("Zero-Shot JEPA Latent Space Geometry & Future Alignment", fontsize=11, fontweight="bold", pad=10)
    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax2_r.get_legend_handles_labels()
    leg2 = ax2.legend(h1 + h2, l1 + l2, frameon=True, fontsize=8.5, loc="upper left")
    leg2.get_frame().set_facecolor(theme["bg"])
    leg2.get_frame().set_edgecolor(theme["border"])
    for t in leg2.get_texts():
        t.set_color(theme["fg"])

    fig.tight_layout()
    return fig


def plot_efficiency_scaling(theme: dict[str, str], eff: dict) -> plt.Figure:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 5.0))
    _style_ax(ax1, theme)
    _style_ax(ax2, theme)

    scaling = eff["context_scaling"]
    series_specs = [
        ("retnet2_chunkwise", "RetCast-JEPA 2M (chunkwise)", theme["retcast_5m"], "o", "-"),
        ("retnet2_parallel", "RetCast-JEPA 2M (parallel)", theme["retcast_2m"], "s", "--"),
        ("transformer_kv", "Transformer-JEPA 2M (KV cache)", theme["ablation_tr"], "^", "-"),
    ]

    for s_key, s_label, color, marker, ls in series_specs:
        if s_key not in scaling:
            continue
        rows = [r for r in scaling[s_key]["rows"] if not r["oom"]]
        xs = [r["context_len"] for r in rows]
        lats = [r["latency_ms"] for r in rows]
        states = [r["state_kb"] for r in rows]
        ax1.plot(xs, lats, marker=marker, linestyle=ls, linewidth=2.0, markersize=7, color=color, label=s_label)
        ax2.plot(xs, states, marker=marker, linestyle=ls, linewidth=2.0, markersize=7, color=color, label=s_label)

    ax1.set_xscale("log", base=2)
    ax1.set_xlabel("Context Length (Time Steps)", fontsize=10)
    ax1.set_ylabel("Inference Latency (ms / batch of 4)", fontsize=10)
    ax1.set_title("Context-Length Latency Scaling (GTX 940MX)", fontsize=11, fontweight="bold", pad=10)
    leg1 = ax1.legend(frameon=True, fontsize=8.5, loc="upper left")
    leg1.get_frame().set_facecolor(theme["bg"])
    leg1.get_frame().set_edgecolor(theme["border"])
    for t in leg1.get_texts():
        t.set_color(theme["fg"])

    ax2.set_xscale("log", base=2)
    ax2.set_xlabel("Context Length (Time Steps)", fontsize=10)
    ax2.set_ylabel("Persistent State / KV Cache (KB)", fontsize=10)
    ax2.set_title("O(1) RetNet2.0 State vs O(L) Transformer KV Cache", fontsize=11, fontweight="bold", pad=10)
    leg2 = ax2.legend(frameon=True, fontsize=8.5, loc="upper left")
    leg2.get_frame().set_facecolor(theme["bg"])
    leg2.get_frame().set_edgecolor(theme["border"])
    for t in leg2.get_texts():
        t.set_color(theme["fg"])

    # Annotate TimesFM vs RetCast throughput box inside Panel A bottom-right
    if "timesfm_200m" in eff["models"] and "retcast_2m_jepa" in eff["models"]:
        tfm = eff["models"]["timesfm_200m"]
        rc2 = eff["models"]["retcast_2m_jepa"]
        rc5 = eff["models"].get("retcast_5m_jepa", rc2)
        speedup = tfm["batch_latency_ms"] / max(0.01, rc2["batch_latency_ms"])
        txt = (
            f"L=512, B=16 Throughput:\n"
            f"TimesFM 200M: {tfm['batch_latency_ms']:.1f} ms ({tfm['series_per_sec']:.0f} ser/s)\n"
            f"RetCast 2M:   {rc2['batch_latency_ms']:.1f} ms ({rc2['series_per_sec']:.0f} ser/s, {speedup:.1f}x)\n"
            f"RetCast 5M:   {rc5['batch_latency_ms']:.1f} ms ({rc5['series_per_sec']:.0f} ser/s)"
        )
        ax1.text(
            0.03,
            0.70,
            txt,
            transform=ax1.transAxes,
            ha="left",
            va="top",
            fontsize=8.0,
            family="monospace",
            color=theme["fg"],
            bbox=dict(boxstyle="round,pad=0.4", facecolor=theme["bg"], edgecolor=theme["border"], alpha=0.92),
        )

    fig.tight_layout()
    return fig


def plot_forecast_examples(theme: dict[str, str], bench: dict) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.0))
    samples = bench["sample_forecasts"]
    datasets = [("etth1", "ETTh1"), ("etth2", "ETTh2"), ("ettm1", "ETTm1"), ("ettm2", "ETTm2")]

    for ax, (ds_key, ds_title) in zip(axes.flat, datasets):
        _style_ax(ax, theme)
        entry = samples[ds_key]
        ctx = np.array(entry["context_tail"])
        act = np.array(entry["actual"])
        t_ctx = np.arange(-len(ctx), 0)
        t_fut = np.arange(0, len(act))

        ax.plot(t_ctx, ctx, color=theme["muted"], linewidth=1.3, label="Context (Last 96)")
        ax.plot(t_fut, act, color=theme["fg"], linewidth=1.6, label="Ground Truth (OT)")

        if "timesfm_200m" in entry:
            tf = entry["timesfm_200m"]
            ax.plot(t_fut, tf["point"], color=theme["timesfm"], linewidth=1.5, linestyle="--", label="TimesFM 200M")
            ax.fill_between(t_fut, tf["q10"], tf["q90"], color=theme["timesfm"], alpha=0.12)

        rc_key = "retcast_5m_jepa" if "retcast_5m_jepa" in entry else "retcast_2m_jepa"
        rc_lbl = "RetCast-JEPA 5M" if rc_key == "retcast_5m_jepa" else "RetCast-JEPA 2M"
        if rc_key in entry:
            rc = entry[rc_key]
            ax.plot(t_fut, rc["point"], color=theme["retcast_5m"], linewidth=1.7, label=rc_lbl)
            ax.fill_between(t_fut, rc["q10"], rc["q90"], color=theme["retcast_5m"], alpha=0.18)

        ax.axvline(0, color=theme["border"], linestyle=":", linewidth=1.2)
        ax.set_title(f"{ds_title} Test Window 0 (Horizon = 192, 10–90% Quantiles)", fontsize=10, fontweight="bold")
        ax.set_xlabel("Relative Step", fontsize=9)
        ax.set_ylabel("Normalized Value", fontsize=9)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc="upper center", ncol=4, frameon=True, fontsize=9, bbox_to_anchor=(0.5, 1.02))
    leg.get_frame().set_facecolor(theme["bg"])
    leg.get_frame().set_edgecolor(theme["border"])
    for t in leg.get_texts():
        t.set_color(theme["fg"])

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--assets_dir", type=str, default="assets")
    args = parser.parse_args()

    res_dir = Path(args.results_dir)
    out_dir = Path(args.assets_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bench = json.loads((res_dir / "benchmark_vs_timesfm.json").read_text())
    abl = json.loads((res_dir / "ablations.json").read_text())
    eff = json.loads((res_dir / "efficiency.json").read_text())

    _save_both(plot_timesfm_vs_retcast, out_dir, "timesfm_vs_retcast", bench=bench)
    _save_both(plot_jepa_ablation, out_dir, "jepa_ablation", abl=abl)
    _save_both(plot_efficiency_scaling, out_dir, "efficiency_scaling", eff=eff)
    _save_both(plot_forecast_examples, out_dir, "forecast_examples", bench=bench)

    for p in sorted(out_dir.glob("*.png")):
        print(f"[fig] {p} ({p.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
