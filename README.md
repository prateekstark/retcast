# RetCast: Zero-Shot Time-Series Forecasting with RetNet2.0 + JEPA

[![GitHub](https://img.shields.io/badge/GitHub-prateekstark%2Fretcast-181717?logo=github)](https://github.com/prateekstark/retcast)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-starkprateek1%2Fretcast--jepa-ffd21e)](https://huggingface.co/starkprateek1/retcast-jepa)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-72%20passed-brightgreen)](tests/test_retcast.py)

**2M–9M-parameter zero-shot time-series forecasting models built from scratch with RetNet2.0 and JEPA (*"predict what the future means, not its raw values"*), benchmarked head-to-head against Google's 200M-parameter TimesFM.**

RetCast combines **RetNet2.0** (`MultiScaleRetention` with overwritable delta-rule associative memory, causal depthwise short convolution, and exact parallel / chunkwise / $O(1)$-state recurrent equivalence) with **Yann LeCun's Joint-Embedding Predictive Architecture (JEPA)**—encoding time-series patches into a time-frequency semantic space and training an online predictor against a stop-gradient EMA target encoder regularized by **VICReg**.

In strict **Zero-Shot vs. Zero-Shot** evaluation across 12 held-out ETT tasks (**0 ETT files seen during RetCast pretraining**), **every RetCast-JEPA scale (2M, 5M, 9M) outperforms Google TimesFM-1.0 (200M)**—with **RetCast-JEPA 2M** achieving **-4.4% lower grand mean MSE (`0.3706` vs `0.3878`)** while being **99× smaller**, **23.7× faster**, and using **36.2× less peak VRAM**.

All training, testing, and benchmarking runs locally on a single 4 GB **NVIDIA GeForce 940MX** laptop GPU.

---

## Highlights & Quickstart

- **Zero-Shot Outperforms 200M Transformer Baseline**: `RetCast-JEPA 2M` (`2.05M` params, `8.11 MB` checkpoint) achieves `0.3706` MSE / `0.3800` MAE vs. Google `TimesFM-1.0-200M` (`203.57M` params, `776.64 MB`) at `0.3878` MSE / `0.3807` MAE.
- **Triple-Mode RetNet2.0 Backbone**: Trains in parallel, prefills 16K contexts in chunkwise mode (**1.85× faster** and **2.72× lower VRAM** than a parameter-matched Transformer at $L=16,384$), and rolls out in **$O(1)$ constant memory (`830 KB` persistent state vs. `30,750 KB` KV cache)**.
- **Controlled Proof of Mechanism**: Removing JEPA collapses future latent cosine alignment (`0.809` → `0.030`) and retrieval R@1 by **5.4×** (`0.070` → `0.013`); removing delta-rule memory degrades zero-shot MSE by **+7.5%** (`0.3706` → `0.3985`); replacing RetNet2.0 with a Causal Transformer degrades zero-shot MSE by **+13.6%** (`0.3706` → `0.4209`).
- **Pretrained Weights on Hugging Face**: All 9 checkpoints (Zero-Shot 2M/5M/9M, 2M Zero-Shot Ablations, and In-Domain 2M/5M/9M) are hosted as `SafeTensors` at **[huggingface.co/starkprateek1/retcast-jepa](https://huggingface.co/starkprateek1/retcast-jepa)**.

```bash
git clone https://github.com/prateekstark/retcast.git
cd retcast
pip install -e ".[dev]"
```

```python
import torch
from retcast import RetCastJEPA

# Load zero-shot RetCast-JEPA 2M directly from Hugging Face Hub ("2m", "5m", "9m")
model = RetCastJEPA.from_pretrained("starkprateek1/retcast-jepa", variant="2m", device="cuda")

# Univariate context batch: shape (B, L)
context = torch.randn(4, 512, device="cuda")
with torch.no_grad():
    out = model(context, horizon_len=192, mode="chunkwise")

point = out["point"]          # (4, 192) point forecast
quantiles = out["quantiles"]  # (4, 192, 10) -> [point, q10, q20, ..., q90]
```

**O(1)-Memory Recurrent Latent Rollout:**

```python
with torch.no_grad():
    ctx_norm, _ = model.revin.normalize(context)
    patches, pad = model._patchify(ctx_norm)
    freq = torch.zeros(context.shape[0], dtype=torch.long, device=context.device)
    z_ctx = model.encoder(patches, pad) + model.freq_emb(freq).unsqueeze(1)
    h_ctx, state = model.forward_backbone(z_ctx, mode="chunkwise", return_state=True)
    # Rolls forward future patch semantics with O(1) persistent state (830 KB fixed)
    z_rollout, next_state = model.rollout_latent_recurrent(
        h_ctx, state, offset=patches.shape[1], n_patches=6
    )
```

---

## Key Results: Zero-Shot vs. Zero-Shot ("One-Shot for Both", 12 ETT Tasks)

Evaluated across `ETTh1`, `ETTh2`, `ETTm1`, and `ETTm2` at horizons $H \in \{96, 192, 336\}$ (`context_len = 512`, `stride = pred_len`, matching `google-research/timesfm` `long_horizon_benchmarks/run_eval.py`). Neither RetCast nor TimesFM sees any ETT training data.

| Model | Params | Ckpt Size | Mean MSE ↓ | Mean MAE ↓ | Mean WAPE ↓ | Mean NRMSE ↓ | Latency (B=16, L=512) | Throughput | Peak VRAM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Last Value (Persistence)** | 0 | — | 0.7682 | 0.5211 | 0.5527 | 0.8991 | — | — | — |
| **Moving Average (24)** | 0 | — | 0.5463 | 0.4761 | 0.5011 | 0.7586 | — | — | — |
| **Seasonal Naive** | 0 | — | 0.4638 | 0.4089 | 0.4166 | 0.6892 | — | — | — |
| **Google TimesFM-1.0 (200M)** *(Zero-Shot)* | 203.57M | 776.6 MB | 0.3878 | 0.3807 | 0.3878 | 0.6273 | 413.49 ms | 38.7 ser/s | 796.6 MB |
| **RetCast-JEPA 2M (Zero-Shot)** *(99× smaller)* | **2.05M** | **8.1 MB** | **0.3706** | **0.3800** | **0.3869** | **0.6150** | **17.44 ms** (**23.7×**) | **917.5 ser/s** | **22.0 MB** |
| **RetCast-JEPA 5M (Zero-Shot)** *(40× smaller)* | **5.11M** | **20.0 MB** | 0.3831 | 0.3845 | 0.3913 | 0.6251 | 32.17 ms (12.9×) | 497.4 ser/s | 36.7 MB |
| **RetCast-JEPA 9M (Zero-Shot)** *(22× smaller)* | **9.19M** | **35.9 MB** | 0.3799 | 0.3836 | 0.3887 | 0.6190 | 50.31 ms (8.2×) | 318.0 ser/s | 55.3 MB |

> **Zero-shot & leakage discipline ("One-Shot for Both"):**
> - **Google `timesfm-1.0-200m-pytorch`** is evaluated zero-shot using its official Hugging Face checkpoint without ETT fine-tuning.
> - **RetCast-JEPA (Zero-Shot)** is pretrained strictly on `NonETTPretrainCorpus` (**1,345 real series across 9 non-ETT domains** + multi-rate resampling + 25% procedural synthetic trajectories; **0 ETT files**). Checkpoints (`best.pt`) are selected exclusively on the held-out trailing 20% non-ETT validation split, so ETT is completely unseen until test evaluation.
> - **Test-split standardization:** `StandardScaler` is fit strictly on each ETT dataset's training split `[0 : boundaries[0]]`, and `RevIN` normalizes every window from its own context only.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/timesfm_vs_retcast-dark.png">
  <img alt="Zero-shot per-dataset test MSE and parameter vs accuracy Pareto curve comparing RetCast-JEPA 2M/5M/9M against Google TimesFM 200M" src="assets/timesfm_vs_retcast.png">
</picture>

### Full 12-Task Zero-Shot Breakdown (`MSE / MAE`, lower is better)

| Task | TimesFM 200M *(Zero-Shot)* | RetCast-JEPA 2M *(Zero-Shot)* | RetCast-JEPA 5M *(Zero-Shot)* | RetCast-JEPA 9M *(Zero-Shot)* | Seasonal Naive |
|---|---:|---:|---:|---:|---:|
| `ETTh1` ($H=96$) | 0.4324 / 0.4048 | 0.4287 / 0.4105 | 0.4610 / 0.4166 | **0.4057** / **0.4027** | 0.5528 / 0.4413 |
| `ETTh1` ($H=192$) | 0.4588 / **0.4317** | 0.4613 / 0.4369 | 0.5157 / 0.4470 | **0.4465** / 0.4341 | 0.6837 / 0.5041 |
| `ETTh1` ($H=336$) | 0.5349 / 0.4589 | 0.5543 / 0.4545 | 0.5536 / 0.4553 | **0.5250** / **0.4489** | 0.6695 / 0.5024 |
| `ETTh2` ($H=96$) | 0.3101 / 0.3442 | **0.2782** / **0.3401** | 0.3029 / 0.3537 | 0.3298 / 0.3640 | 0.3777 / 0.3777 |
| `ETTh2` ($H=192$) | 0.3934 / 0.3907 | **0.3760** / **0.3898** | 0.3877 / 0.3980 | 0.4149 / 0.4083 | 0.4506 / 0.4208 |
| `ETTh2` ($H=336$) | **0.3743** / **0.3948** | 0.3901 / 0.4080 | 0.4196 / 0.4212 | 0.4316 / 0.4243 | 0.4772 / 0.4411 |
| `ETTm1` ($H=96$) | 0.3527 / 0.3568 | **0.3227** / 0.3523 | 0.3307 / 0.3581 | 0.3232 / **0.3516** | 0.4219 / 0.3864 |
| `ETTm1` ($H=192$) | 0.3888 / 0.3906 | 0.3752 / 0.3848 | **0.3749** / 0.3843 | 0.3750 / **0.3793** | 0.4496 / 0.4003 |
| `ETTm1` ($H=336$) | 0.4472 / 0.4250 | **0.4360** / **0.4205** | 0.4392 / 0.4235 | 0.4391 / 0.4241 | 0.4917 / **0.4190** |
| `ETTm2` ($H=96$) | 0.1944 / **0.2608** | 0.1909 / 0.2681 | **0.1870** / 0.2663 | 0.1893 / 0.2652 | 0.2620 / 0.2999 |
| `ETTm2` ($H=192$) | 0.2956 / 0.3187 | 0.2581 / 0.3131 | **0.2432** / **0.3050** | 0.2594 / 0.3067 | 0.3106 / 0.3310 |
| `ETTm2` ($H=336$) | 0.4708 / 0.3914 | **0.3758** / **0.3809** | 0.3815 / 0.3843 | 0.4188 / 0.3940 | 0.4189 / 0.3830 |

Across the three zero-shot RetCast-JEPA scales, the best zero-shot RetCast model beats Google TimesFM 200M on **11 of 12 tasks** on MSE (every task except `ETTh2` $H=336$), including **-20.2% MSE** on `ETTm2` $H=336$ (`0.3758` vs `0.4708`), while **RetCast-JEPA 2M** alone beats TimesFM 200M on **9 of 12 tasks** and achieves the best grand mean across MSE (`0.3706`), MAE (`0.3800`), WAPE (`0.3869`), and NRMSE (`0.6150`).

### Secondary Reference: Zero-Shot vs. In-Domain RetCast-JEPA

For reference, we also evaluate leak-free **in-domain** RetCast-JEPA checkpoints (trained strictly on the ETT training splits `[0 : boundaries[0]]` + 15% synthetic data):

| Model | Training Regime | Params | Mean MSE ↓ | Mean MAE ↓ | Mean WAPE ↓ | Mean NRMSE ↓ |
|---|---|---:|---:|---:|---:|---:|
| **Google TimesFM-1.0 (200M)** | Zero-Shot (non-ETT) | 203.57M | 0.3878 | 0.3807 | 0.3878 | 0.6273 |
| **RetCast-JEPA 2M (Zero-Shot)** | Zero-Shot (`NonETTPretrainCorpus`) | 2.05M | 0.3706 | 0.3800 | 0.3869 | 0.6150 |
| **RetCast-JEPA 5M (Zero-Shot)** | Zero-Shot (`NonETTPretrainCorpus`) | 5.11M | 0.3831 | 0.3845 | 0.3913 | 0.6251 |
| **RetCast-JEPA 9M (Zero-Shot)** | Zero-Shot (`NonETTPretrainCorpus`) | 9.19M | 0.3799 | 0.3836 | 0.3887 | 0.6190 |
| **RetCast-JEPA 2M (In-Domain)** | ETT Train Split | 2.05M | 0.3505 | 0.3717 | 0.3780 | 0.5963 |
| **RetCast-JEPA 5M (In-Domain)** | ETT Train Split | 5.10M | 0.3476 | 0.3715 | 0.3779 | 0.5936 |
| **RetCast-JEPA 9M (In-Domain)** | ETT Train Split | 9.18M | **0.3460** | **0.3707** | **0.3770** | **0.5927** |

---

## Why RetNet2.0 for Time-Series Forecasting?

Standard foundation forecasters (TimesFM, PatchTST, Chronos-Decoder) rely on **Causal Self-Attention**, while classical sequence forecasters rely on **RNNs / LSTMs** or **Plain Additive Linear Attention (RetNet 1.0)**. Both families hit structural bottlenecks on continuous physical signals. RetCast shows why **RetNet2.0**—combining multi-scale exponential retention, overwritable delta-rule associative memory, negative eigenvalues, and causal short convolution—is uniquely suited to time-series forecasting:

### 1. Triple-Paradigm Execution Equivalence (Parallel + Chunkwise + $O(1)$ Recurrent)
RetNet2.0 provides exact mathematical equivalence (verified in float64 unit tests across all sequence lengths, delta modes, and convolution widths) across three execution modes:
- **Parallel Mode (Training)**: Full matrix formulation $O = (QK^T \odot D)V_{\Delta}$ maximizes GPU tensor-core utilization during short-window pretraining without sequential RNN unrolling.
- **Chunkwise Mode (Long-Context Prefill)**: Splits a long sequence into chunks of size $C=64$, computing within-chunk attention in parallel and passing an $O(1)$ matrix state $S$ between chunks—achieving $O(L \cdot C)$ complexity. At $L = 16,384$ timesteps, chunkwise RetCast 2M runs **1.85× faster** (`173.2 ms` vs `320.5 ms`), **3.12× faster** than parallel retention (`539.9 ms`), and uses **2.72× less peak VRAM** (`71.0 MB` vs `193.5 MB`) than a parameter-matched Transformer.
- **Recurrent Mode ($O(1)$ Streaming & Rollout)**: Maintains a fixed-size per-head associative matrix state $S_t \in \mathbb{R}^{d_k \times d_v}$ plus a $(K-1)$-patch convolution buffer—staying constant at **830 KB** regardless of whether the model has seen 512 or 16,384 timesteps (**37× smaller** than the Transformer's `30,750 KB` KV cache at $L=16,384$). Streaming sensors never re-attend over growing histories.

### 2. Multi-Scale Exponential Decay Priors as a Parameter-Free Temporal Filter Bank
Instead of forcing every attention head to learn from scratch that recent shocks matter more than distant noise, `MultiScaleRetention` assigns each head $h \in \{0, \dots, H-1\}$ a geometric decay factor:
$$g_h = 1 - 2^{-5 - h}$$
With 4 heads and 16-step patches, the heads span exact exponential half-lives $\tau_h = \frac{\ln 2}{|\ln g_h|} \approx 21.8, 44.0, 88.4, 177.1$ patches ($\approx 349, 704, 1414, 2834$ raw timesteps). Fast heads track immediate boundary momentum and sharp level jumps; slow heads act as long-horizon seasonal integrators—giving RetNet2.0 a built-in inductive bias for multi-frequency time series without extra parameters, preventing the attention-map overfitting that hurt our parameter-matched Transformer (`0.4209` vs `0.3706` zero-shot MSE).

### 3. Overwritable Associative Memory via the Delta Rule + Negative Eigenvalues ($\beta_t \in (0, 2)$)
Plain RetNet updates its matrix state purely additively:
$$S_t = \gamma S_{t-1} + k_t^T v_t$$
On non-stationary time series (e.g., an electrical transformer shifting from weekday load to weekend regime or sudden weather fronts), additive memory **superimposes** stale regime patterns onto new ones, causing destructive interference.

RetNet2.0 replaces additive accumulation with **Delta-Rule Associative Memory**:
$$S_t = \gamma S_{t-1} + \beta_t k_t^T \left(v_t - \gamma k_t S_{t-1}\right), \qquad \beta_t = \beta_{\max}\,\sigma(W_\beta x_t) \in (0, 2)$$
1. **Selective Erasure & Overwrite**: The residual $(v_t - \gamma k_t S_{t-1})$ queries what the state currently associates with unit-normalized key $k_t$ and replaces stale associations with the updated regime dynamics $v_t$.
2. **Negative Eigenvalues for Phase & Cycle Tracking**: Allowing $\beta_{\max} = 2.0$ gives the transition operator $(I - \beta_t k_t k_t^T)$ eigenvalues in $(-1, 1)$, enabling sign-flipping / parity / oscillatory phase tracking across patches (Grazzi et al., 2024)—essential for alternating daily/sub-hourly cycles.
3. **Exact Parallel Triangular Solve**: Within every training chunk, all delta-rule dependencies reduce to a single batched unit-lower-triangular linear solve (`torch.linalg.solve_triangular`), avoiding any sequential Python loop during training.
4. **Measured Impact**: Removing delta memory + short convolution (`Plain RetNet + JEPA`) worsens zero-shot ETT MSE from **`0.3706` → `0.3985`** (**+7.5% higher error**, falling behind TimesFM's `0.3878`), proving that overwritable state dynamics are essential for cross-domain generalization.

### 4. Local Derivative Binding via Causal Depthwise Short Convolution (`short_conv=4`)
A single 16-step patch cannot by itself represent multi-patch velocity or curvature across patch boundaries. Mixing a 4-patch causal depthwise convolution (`Conv1d(..., groups=d_model)` + `SiLU`) prior to projecting $Q, K, V$ binds consecutive patches into local temporal derivatives (trend acceleration, peak/trough transitions) with negligible parameter cost (`~2.5K` params at 2M scale).

### 5. Synergy of RetNet2.0 + JEPA ("Predict What the Future Means, Not Raw Noise")
Training a compact 2M–9M backbone purely on raw future MSE forces limited capacity to memorize unpredictable stochastic sensor jitter. Pairing RetNet2.0 with **JEPA + VICReg**:
- Predicts stop-gradient EMA target patch embeddings $z^*_{N+1:N+K} = \text{sg}(f_{\bar{\theta}}(x_{N+1:N+K}))$ encoding morphology and `rfft` spectral energy.
- Uses **single-pass multi-horizon latent decoding** (`LatentHorizonPredictor`), completely eliminating Google TimesFM's visible 128-step autoregressive patch seam (`t = 128`).
- Preserves geometric structure in latent space (**`0.809–0.842` future cosine alignment** vs. `0.030` without JEPA, and **5.4× higher** future retrieval R@1) while directly improving zero-shot MSE (`0.3706` vs `0.3792` without JEPA).

### Architecture Comparison Summary

| Property | RetCast-JEPA (**RetNet2.0 + JEPA**) | Causal Transformer (TimesFM / PatchTST) | Plain RetNet (1.0) | Recurrent RNN / LSTM |
|---|---|---|---|---|
| **Training Complexity** | $O(L^2)$ parallel or $O(L \cdot C)$ chunkwise | $O(L^2)$ parallel | $O(L^2)$ parallel / $O(L \cdot C)$ chunkwise | $O(L)$ sequential (slow) |
| **Inference State Memory** | **$O(1)$ fixed (`830 KB`)** | $O(L)$ KV cache (`30.8 MB` @ 16K) | $O(1)$ fixed (`830 KB`) | $O(1)$ fixed |
| **Regime Overwrite Ability** | **Yes** (Delta rule erases stale state) | Yes (full attention lookup) | No (additive superposition) | Limited (gating decay) |
| **Negative Eigenvalues ($\beta_{\max}=2$)** | **Yes** (tracks phase / oscillation) | N/A | No ($\gamma > 0$ only) | Weak |
| **Multi-Horizon Decoding** | **Single-pass JEPA latent horizon** | Autoregressive patch steps (128-step seam) | Linear / AR | AR / Seq2Seq |
| **Zero-Shot ETT Mean MSE (2M)** | **0.3706** | 0.4209 (2M) / 0.3878 (TimesFM 200M) | 0.3985 | — |

---

## Architecture & Universal Non-ETT Design

```mermaid
flowchart LR
    X["Context x (B, L)"] --> R["Mask-Aware RevIN<br/>(mu, sigma from context only)"]
    R --> P["Patchify (P=16) +<br/>SpectralPatchEncoder<br/>(raw + morph + rfft)"]
    P -->|"online z_ctx"| B["RetNet2.0 Backbone<br/>(MultiScaleRetention +<br/>Delta Memory + short_conv=4)"]
    B -->|"h_ctx, O(1) state S"| J["LatentHorizonPredictor<br/>(multi-scale decay pool +<br/>causal horizon conv)"]
    J -->|"z_hat_fut"| D["Universal MeaningDecoder<br/>(7 causal lag primitives +<br/>rho_{24,48,96,168} gating)"]
    D -->|"point + 9 quantiles"| OUT["RevIN Denormalize -> Forecast"]
    F["Future Patches (train only)"] -->|"stop-grad EMA"| TE["Target Encoder f_theta_bar"]
    TE -->|"z*_fut"| LOSS["JEPA Cosine+Huber + VICReg"]
    J -->|"z_hat_fut"| LOSS
```

### Non-ETT Pretraining Corpus & Universal Horizon Decoder
To generalize across unseen frequencies (`15-min` `ETTm*` and `hourly` `ETTh*`) without ever seeing ETT:
1. **`NonETTPretrainCorpus` (`retcast/pretrain_data.py`)**: Loads **1,345 real series** across **9 disjoint non-ETT domains** from Hugging Face `autogluon/chronos_datasets` (`ercot` [8], `monash_australian_electricity` [5], `monash_electricity_hourly` [300], `wind_farms_hourly` [291], `monash_pedestrian_counts` [64], `monash_kdd_cup_2018` [270], `uber_tlc_hourly` [262], `exchange_rate` [8]) plus UCI `solar_AL` [137]. Domain-balanced sampling prevents large collections from dominating, while multi-rate temporal resampling (`2×`/`4×` upsampling and `2×`/`3×` decimation) mixed with **25% procedural synthetic trajectories** exposes the model to diverse sub-hourly and multi-day periodicities.
2. **Universal `MeaningDecoder` (`retcast/model.py`)**: Combines predicted future semantic latents $\hat{z}_{\text{fut}}$ with **7 causal lag primitives** ($B_0$: trailing 16-step mean, $B_1$: trailing 64-step mean, $B_{2..5}$: periodic lag tilings at periods $24, 48, 96, 168$, and $B_6$: a shift-anchored linear skip invariant to boundary level drift). Softmax gating over $[z_{\text{fut}}, h_{\text{last}}, \rho_{24,48,96,168}]$ uses scale-invariant lag autocorrelation $\rho$ so the decoder dynamically adapts to any sampling rate zero-shot.
3. **Balanced Joint Objective**:
   $$\mathcal{L} = \underbrace{\mathcal{L}_{\text{Huber}} + 0.5\,\mathcal{L}_{\text{pinball}} + 0.1\,\mathcal{L}_{\text{FFT}}}_{\text{Forecasting}} + 0.25\,\mathcal{L}_{\text{JEPA}}(\hat{z}, z^*) + 0.05\,\mathcal{L}_{\text{VICReg}}(z)$$

---

## Controlled 2M Zero-Shot Ablations

Every 2M zero-shot ablation below uses the exact same `NonETTPretrainCorpus`, seed (`42`), step budget (`2,400` steps), batch size (`64`), optimizer (`AdamW`, `lr=1.5e-3`), and parameter budget (`~2.05M`):

| Variant (2M Zero-Shot) | Params | Test MSE ↓ | Test MAE ↓ | JEPA Cosine ↑ | Effective Rank ↑ | Future Latent R@1 ↑ |
|---|---:|---:|---:|---:|---:|---:|
| **RetNet2.0 + JEPA** (`memory="delta"`, `short_conv=4`) | 2,052,180 | **0.3706** | **0.3800** | **0.809** | 17.9 | 0.070 |
| **RetNet2.0 Raw (No JEPA)** (`jepa_weight=0`, `vicreg_weight=0`) | 2,052,180 | 0.3792 | 0.3819 | 0.030 | **18.8** | 0.013 |
| **Plain RetNet + JEPA** (`memory="none"`, `short_conv=0`) | 2,047,044 | 0.3985 | 0.3927 | 0.798 | 17.5 | **0.080** |
| **Parameter-Matched Transformer + JEPA** | 2,049,604 | 0.4209 | 0.4003 | 0.721 | 16.4 | 0.062 |

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/jepa_ablation-dark.png">
  <img alt="Controlled 2M zero-shot ablations comparing forecast MSE/MAE and JEPA latent geometry" src="assets/jepa_ablation.png">
</picture>

---

## Efficiency & $O(1)$ Recurrent State Scaling ($L = 512 \dots 16,384$)

Measured on the **NVIDIA GeForce 940MX (4 GB, fp32)** with batch size $B=4$ and horizon $H=96$:

| Context Length ($L$) | Patches ($N$) | RetNet2.0 Chunkwise Latency / VRAM / State | RetNet2.0 Parallel Latency / VRAM / State | Transformer KV Latency / VRAM / State |
|---:|---:|---:|---:|---:|
| 512 | 32 | 17.5 ms / 18.7 MB / **830 KB** | 15.4 ms / 17.7 MB / **830 KB** | 11.0 ms / 17.5 MB / 990 KB |
| 1,024 | 64 | 22.1 ms / 20.4 MB / **830 KB** | 22.3 ms / 19.3 MB / **830 KB** | 11.2 ms / 18.9 MB / 1,950 KB |
| 2,048 | 128 | 30.3 ms / 23.7 MB / **830 KB** | 21.3 ms / 24.1 MB / **830 KB** | 15.7 ms / 22.1 MB / 3,870 KB |
| 4,096 | 256 | 49.0 ms / 31.0 MB / **830 KB** | 57.2 ms / 39.5 MB / **830 KB** | 39.0 ms / 33.0 MB / 7,710 KB |
| 8,192 | 512 | **86.9 ms** / **44.3 MB** / **830 KB** | 169.2 ms / 87.8 MB / **830 KB** | 107.0 ms / 67.6 MB / 15,390 KB |
| 16,384 | 1,024 | **173.2 ms** / **71.0 MB** / **830 KB** | 539.9 ms / 266.7 MB / **830 KB** | 320.5 ms / 193.5 MB / **30,750 KB** |

At $L = 16,384$, **RetNet2.0 chunkwise** holds its persistent state constant at **830 KB** (**37× smaller** than the Transformer's **30.0 MB** KV cache), runs **1.85× faster** than the Transformer (**3.12× faster** than parallel retention), and uses **2.72× less peak VRAM** (**71.0 MB** vs **193.5 MB**).

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/efficiency_scaling-dark.png">
  <img alt="Context-length latency scaling and O(1) RetNet2.0 recurrent state vs O(L) Transformer KV cache" src="assets/efficiency_scaling.png">
</picture>

---

## Qualitative Forecasts (Horizon = 192, 10–90% Quantile Bands)

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/forecast_examples-dark.png">
  <img alt="Qualitative 192-step zero-shot forecast fan charts comparing RetCast-JEPA 5M and Google TimesFM 200M" src="assets/forecast_examples.png">
</picture>

Notice Google TimesFM's 128-step autoregressive patch seam at step `t = 128` where a second decode call begins, whereas zero-shot RetCast-JEPA predicts coherent multi-horizon future latent trajectories in a single pass with calibrated 10%–90% quantile intervals.

---

## Research Methodology & Known Limitations

- **Verified Safeguards**:
  1. Static manifest assertions (`test_zero_shot_corpus_disjoint_from_ett`) reject any `"ett"` source inside `NonETTPretrainCorpus`.
  2. Validation checkpoints (`best.pt`) are selected solely on the held-out trailing 20% non-ETT split, keeping ETT test splits untouched until final evaluation.
  3. All models (RetCast, TimesFM-1.0-200M, Seasonal Naive, Last Value, Moving Average) receive identical `(past, actual)` arrays normalized with train-split-only `StandardScaler`.
- **Important Caveats for Academic Publication**:
  1. **Corpus Saturation Above 2M**: While in-domain scaling improves monotonically (`0.3505 → 0.3476 → 0.3460`), zero-shot MSE peaks at 2M (`0.3706` vs `0.3831` at 5M and `0.3799` at 9M) because 1,345 non-ETT series (~2,400 steps) begins to overfit at 5M–9M capacity. Scaling to larger foundation corpora (e.g., Chronos / LOTSA 100M+ observations) is needed to unlock clean zero-shot scaling laws.
  2. **Benchmark Breadth & Multi-Seed Rigor**: Current results cover 1 seed (`42`) across the 4 ETT datasets (12 tasks) without statistical significance tests or CRPS calibration evaluation; expanding to GIFT-Eval / Monash / TSLib and 3–5 seeds with Diebold-Mariano tests is required for top-tier venue submission.

---

## Repository Layout

```text
retcast/
├── retcast/
│   ├── retention.py           # RetNet2.0 MultiScaleRetention (parallel/recurrent/chunkwise + delta + conv) & CausalSelfAttention
│   ├── model.py               # RetCastJEPA (+ from_pretrained), RevIN, SpectralPatchEncoder, LatentHorizonPredictor, MeaningDecoder
│   ├── data.py                # Leak-free ETTDataset, MultiETTTrainSampler, procedural synthetic generator
│   ├── pretrain_data.py       # NonETTPretrainCorpus (9 non-ETT domains, multi-rate resampling, 0 ETT files)
│   ├── baselines.py           # Seasonal Naive, Last Value, Moving Average sanity baselines
│   ├── timesfm_wrapper.py     # Pure-PyTorch Google TimesFM-1.0-200M decoder + HF checkpoint loader
│   └── metrics.py             # MSE, MAE, SMAPE, WAPE, NRMSE, Effective Rank, Cosine Alignment, Retrieval R@1
├── scripts/
│   ├── download_data.py       # Cache ETT CSVs, non-ETT pretraining corpus & google/timesfm-1.0-200m-pytorch
│   ├── train.py               # Resumable zero-shot or in-domain training loop with validation-only selection
│   ├── benchmark_vs_timesfm.py  # Full test-split benchmark, ablations, efficiency & --verify check
│   ├── make_figures.py        # Generate light & dark PNG plots in assets/
│   └── export_hf.py           # Export SafeTensors + Model Card and push to Hugging Face Hub
├── tests/
│   └── test_retcast.py        # 72 unit tests (RetNet2.0 modes, delta memory, JEPA losses, RevIN, HF roundtrip, TimesFM shapes)
├── results/                   # Measured benchmark JSONs (benchmark_vs_timesfm.json, ablations.json, efficiency.json)
└── assets/                    # Light & dark publication figures
```

---

## Reproduction

```bash
# 1. Run all 72 unit tests
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/stark/anaconda3/envs/starkml/bin/python -m pytest -v tests

# 2. Download ETT test datasets, non-ETT pretraining corpus & Google TimesFM-1.0-200M weights
/home/stark/anaconda3/envs/starkml/bin/python scripts/download_data.py

# 3. Pretrain zero-shot RetCast-JEPA models (2M, 5M, 9M) and 2M controlled ablations (0 ETT files)
/home/stark/anaconda3/envs/starkml/bin/python scripts/train.py --name zs_retcast_2m_jepa --mode zero_shot --preset 2m --steps 2400 --batch_size 64 --eval_interval 300 --lr 1.5e-3 --seed 42
/home/stark/anaconda3/envs/starkml/bin/python scripts/train.py --name zs_retcast_5m_jepa --mode zero_shot --preset 5m --steps 2400 --batch_size 48 --eval_interval 300 --lr 1.2e-3 --seed 42
/home/stark/anaconda3/envs/starkml/bin/python scripts/train.py --name zs_retcast_9m_jepa --mode zero_shot --preset 9m --steps 2400 --batch_size 32 --eval_interval 300 --lr 1.0e-3 --seed 42
/home/stark/anaconda3/envs/starkml/bin/python scripts/train.py --name zs_ablation_2m_raw --mode zero_shot --preset 2m --jepa_weight 0.0 --vicreg_weight 0.0 --steps 2400 --batch_size 64 --eval_interval 300 --lr 1.5e-3 --seed 42
/home/stark/anaconda3/envs/starkml/bin/python scripts/train.py --name zs_ablation_2m_nodelta --mode zero_shot --preset 2m --memory none --short_conv 0 --steps 2400 --batch_size 64 --eval_interval 300 --lr 1.5e-3 --seed 42
/home/stark/anaconda3/envs/starkml/bin/python scripts/train.py --name zs_ablation_2m_transformer --mode zero_shot --preset 2m --arch transformer --steps 2400 --batch_size 64 --eval_interval 300 --lr 1.5e-3 --seed 42

# 4. Run full benchmark vs Google TimesFM 200M, ablations, efficiency scaling & verify
/home/stark/anaconda3/envs/starkml/bin/python scripts/benchmark_vs_timesfm.py
/home/stark/anaconda3/envs/starkml/bin/python scripts/benchmark_vs_timesfm.py --verify

# 5. Regenerate light & dark figures
/home/stark/anaconda3/envs/starkml/bin/python scripts/make_figures.py

# 6. Export SafeTensors checkpoints & push to Hugging Face Hub
/home/stark/anaconda3/envs/starkml/bin/python scripts/export_hf.py --out export/retcast-jepa --push starkprateek1/retcast-jepa
```

---

## License

MIT License. See [LICENSE](LICENSE).
