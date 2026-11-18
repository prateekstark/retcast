# RetCast-JEPA: Research Methodology Audit & Paper Scaling Roadmap

This document provides a rigorous self-audit of the **RetCast-JEPA** research methodology, evaluates publication readiness, and lays out a concrete 5-axis scaling blueprint for turning RetCast-JEPA into a top-tier conference or journal paper.

---

## 1. Executive Assessment: Can We Write a Paper?

**Yes.** The current results demonstrate a compelling core scientific claim: a **2.05M-parameter RetNet2.0 + JEPA model (`RetCast-JEPA 2M`)**, pretrained exclusively on non-ETT domains (`0` ETT files seen), outperforms Google's **203.57M-parameter `TimesFM-1.0-200M`** zero-shot across 12 held-out ETT tasks (**`0.3706` vs `0.3878` MSE**, **99× fewer parameters**, **23.7× faster inference**, and **36.2× lower peak VRAM**), while controlled parameter- and step-matched ablations isolate the causal contributions of both **JEPA** and **RetNet2.0 delta-rule memory**.

### Publication Readiness by Venue Tier

| Venue Tier | Current Readiness | Required Additions Before Submission |
|---|---|---|
| **ArXiv Technical Report / Workshop Paper** *(e.g., NeurIPS / ICML Time-Series Workshops)* | **Ready after minor ablations** | Add 3–5 training seeds for error bars, ablate `MeaningDecoder` without hardcoded periodic lag bases `(24, 48, 96, 168)`, and evaluate on 4–6 additional non-ETT TSLib/Monash datasets. |
| **TMLR (Transactions on Machine Learning Research)** | **Strong fit after mid-scale validation** | Same as above + probabilistic evaluation (CRPS / weighted quantile loss), statistical significance tests (Diebold-Mariano / Wilcoxon), and resolving the `2M vs 5M/9M` zero-shot scaling inversion. |
| **Top-Tier Main Track** *(ICLR / ICML / NeurIPS / KDD)* | **Requires 5-axis scaling** | Scale pretraining corpus to `100K–1M+` series (`LOTSA` / full `Chronos`), sweep `2M → 10M → 50M → 150M` with fixed effective batch size, benchmark on full **GIFT-Eval / TSLib / Monash** against Chronos-Bolt, Moirai, TimesFM-2.0, PatchTST, and iTransformer, and show long-context *accuracy* scaling ($L = 512 \to 8,192$). |

### Core Scientific Contributions

1. **RetNet2.0 as a Structural Inductive Bias for Non-Stationary Time Series**:
   - Exact equivalence across parallel training, chunkwise long-context prefill (**1.85× faster** and **2.72× lower VRAM** than a parameter-matched Causal Transformer at $L=16,384$), and **$O(1)$ constant-memory recurrent rollout** (`830 KB` persistent state vs `30,750 KB` Transformer KV cache, **37× smaller**).
   - Fixed geometric multi-scale decay priors ($g_h = 1 - 2^{-5-h}$) acting as a parameter-free multi-horizon filter bank (half-lives $\approx 21.8, 44.0, 88.4, 177.1$ patches).
   - Delta-rule overwritable associative memory ($S_t = \gamma S_{t-1} + \beta_t k_t^T (v_t - \gamma k_t S_{t-1})$, $\beta_{\max} = 2.0$, solved within chunks via `torch.linalg.solve_triangular`) + causal `short_conv=4`, preventing state superposition under regime shifts and achieving **-7.0% lower zero-shot MSE** than Plain RetNet (`0.3706` vs `0.3985`) and **-11.9% lower zero-shot MSE** than a parameter-matched Causal Transformer (`0.3706` vs `0.4209`).
2. **Joint-Embedding Predictive Architecture (JEPA) in Time-Frequency Patch Space**:
   - Predicting stop-gradient EMA target patch representations (`rfft` log-magnitude + morphological descriptors) regularized by **VICReg** prevents compact backbones from spending capacity memorizing unpredictable high-frequency noise.
   - Single-pass multi-horizon prediction (`LatentHorizonPredictor`) eliminates Google TimesFM's visible 128-step autoregressive patch seam while preserving latent geometry (**`0.809–0.842` future cosine alignment** vs `0.030` without JEPA, and **5.4× higher** future retrieval R@1).
3. **Pareto-Optimal Zero-Shot Efficiency**:
   - Demonstrates that careful temporal inductive biases + latent-space supervision achieve competitive or superior zero-shot accuracy at **2M parameters (`8.11 MB` SafeTensors)**—fitting comfortably inside a 4 GB laptop GPU (`NVIDIA GeForce 940MX`) for both training and inference.

---

## 2. Rigorous Research Methodology Audit

### 2.1 Verified Safeguards & Leak-Free Guarantees

| Aspect | Current Implementation | Verification Evidence |
|---|---|---|
| **Zero-Shot Corpus Disjointness** | `NonETTPretrainCorpus` (`retcast/pretrain_data.py`) loads strictly from `autogluon/chronos_datasets` and UCI `solar_AL`, explicitly raising `ValueError` if `"ett"` appears in any subset path or dataset name. | Enforced statically and dynamically in `tests/test_retcast.py::test_zero_shot_corpus_disjoint_from_ett`. `0` ETT files are seen during zero-shot pretraining. |
| **Non-ETT Checkpoint Selection** | `scripts/train.py` (`--mode zero_shot`) selects `best.pt` strictly using `NonETTPretrainCorpus.get_val_windows()` from the trailing 20% of non-ETT series (while pretraining samples strictly from the leading 80%). | ETT validation and test splits remain completely unseen until final benchmark evaluation. |
| **StandardScaler & RevIN Leakage Prevention** | `ETTDataset` (`retcast/data.py`) fits `StandardScaler` exclusively on `[0 : boundaries[0]]` (`8640` for `ETTh*`, `34560` for `ETTm*`). `RevIN` (`retcast/model.py`) computes $(\mu, \sigma)$ strictly across valid context timesteps, and `compute_loss` normalizes future targets using the *same* context statistics. | Verified by `tests/test_retcast.py::test_ett_splits_and_scaler_no_leakage` and `test_revin_invertibility_and_masking`. |
| **Identical Evaluation Harness** | `scripts/benchmark_vs_timesfm.py` extracts each `(past, actual)` array once (`context_len=512`, `stride=pred_len`) and passes the exact same numpy arrays to `TimesFM200MForecaster`, `RetCastJEPA`, `Seasonal Naive`, `Last Value`, and `Moving Average`. | Verified reproducible end-to-end via `scripts/benchmark_vs_timesfm.py --verify` (`=== VERIFICATION PASSED ===`). |
| **Controlled 2M Ablations** | All four 2M zero-shot runs (`zs_retcast_2m_jepa`, `zs_ablation_2m_raw`, `zs_ablation_2m_nodelta`, `zs_ablation_2m_transformer`) share identical `NonETTPretrainCorpus`, `seed=42`, `steps=2400`, `batch_size=64`, `AdamW (lr=1.5e-3)`, and ~`2.05M` params (`2,047,044`–`2,052,180`). | Verified in `results/ablations.json`. |
| **Triple-Mode Mathematical Equivalence** | `MultiScaleRetention` parallel, chunkwise, and $O(1)$ recurrent modes are tested for exact numerical agreement in `float64` across arbitrary lengths, `memory="none"/"delta"`, `beta_max=1.0/2.0`, and `short_conv=0/4`. | `72 passed` in `tests/test_retcast.py`. |

### 2.2 Critical Reviewer Stress-Tests & Methodological Gaps to Address

An experienced peer reviewer will raise five critical questions about the current experimental setup. Addressing each one directly turns a strong prototype into an airtight paper:

1. **Calendar Lag Primitive Bias (`SEASONAL_LAGS = (24, 48, 96, 168)`)**:
   - *Finding:* `MeaningDecoder` (`retcast/model.py`) includes fixed periodic lag-tiling primitives `SEASONAL_LAGS = (24, 48, 96, 168)` gated by lag autocorrelation $\rho_{24,48,96,168}$, and `generate_synthetic_batch` (`retcast/data.py`) samples harmonic periods from `[12, 24, 48, 96, 168]`. Because `ETTh1/2` have `seasonality=24` and `ETTm1/2` have `seasonality=96`, reviewers will question how much of the zero-shot gain stems from RetNet2.0 + JEPA vs. inductive periodic bases tailored to hourly/15-minute calendar frequencies.
   - *Required Fix:*
     - Run a **decoder ablation** disabling `b2..b5` (or replacing fixed integer lags with learned frequency/basis routing or pure `z_fut` decoding) across both ETT and non-24/96 datasets.
     - Evaluate zero-shot on series whose fundamental periods fall outside `{24, 48, 96, 168}` (e.g., Weather 10-minute `period=144`, M4 Monthly `12`, Tourism Quarterly `4`, Exchange daily `5`/`7`, ILI weekly `52`).

2. **Corpus Filtering & Zero-Shot Scaling Inversion (`2M: 0.3706` vs `5M: 0.3831`, `9M: 0.3799`)**:
   - *Finding:* In-domain ETT training scaled monotonically with model size (`2M: 0.3505` → `5M: 0.3476` → `9M: 0.3460`), whereas zero-shot MSE was best at `2M` (`0.3706`) and slightly worse at `5M` (`0.3831`) and `9M` (`0.3799`), even though non-ETT validation MSE improved (`0.5212` → `0.5148` at `9M`). Two factors explain this:
     1. **Corpus size saturation**: In `NonETTPretrainCorpus` (`retcast/pretrain_data.py`), `min_len = max(1200, context_len + horizon_len + 160)` filtered out 4 of the 13 configured subsets completely (`taxi_30min=0`, `taxi_1h=0`, `m4_hourly=0`, `nn5=0`), leaving `1,345` series across 9 domains (`max_series_per_dataset=300`). Larger 5M/9M models began memorizing source-specific patterns in those 1,345 series over 2,400 steps.
     2. **Unmatched batch sizes across scales**: Due to 4 GB VRAM limits, `2M` trained at `batch_size=64, lr=1.5e-3`, `5M` at `batch_size=48, lr=1.2e-3`, and `9M` at `batch_size=32, lr=1.0e-3` without gradient accumulation, confounding capacity scaling with optimization variance.
   - *Required Fix:* Lower `min_len` using RetCast's existing `padding` mask support so `m4_hourly`, `taxi`, `nn5`, and full `chronos_datasets`/`lotsa_data` are included, and keep effective batch size constant via gradient accumulation across all model scales.

3. **Benchmark Breadth (ETT-Only Evaluation)**:
   - *Finding:* All 12 current test tasks come from the 4 ETT electricity-transformer datasets in univariate channel-independent mode. ETT alone is insufficient for ICLR/ICML/NeurIPS main-track acceptance.
   - *Required Fix:* Expand `benchmark_vs_timesfm.py` to full **TSLib** (`Weather`, `Electricity/ECL`, `Traffic`, `ILI`, `Exchange`), **Monash**, and **GIFT-Eval** zero-shot benchmarks, plus contemporary baselines (`TimesFM-2.0`, `Chronos-Bolt`, `Moirai`, `PatchTST`, `iTransformer`, `DLinear`).

4. **Multi-Seed Variance, Statistical Significance, & Probabilistic Calibration**:
   - *Finding:* Current experiments use a single seed (`seed=42`) and report deterministic point accuracy (`MSE`, `MAE`, `SMAPE`, `WAPE`, `NRMSE`) plus representation geometry, without evaluating the calibration of the 9 predicted quantiles `(0.1..0.9)`.
   - *Required Fix:* Run 3–5 random seeds (`mean ± std`), compute paired **Diebold-Mariano** / **Wilcoxon signed-rank** $p$-values against TimesFM and ablations, and report **CRPS / Weighted Quantile Loss (WQL)** and empirical 80%/90% prediction-interval coverage.

5. **Long-Context *Accuracy* Scaling (Not Just Latency/VRAM)**:
   - *Finding:* `results/efficiency.json` establishes RetNet2.0's $O(1)$ memory and latency advantage up to $L = 16,384$, but forecasting accuracy is evaluated strictly at $L = 512$ (matching TimesFM-1.0's maximum context window).
   - *Required Fix:* Measure accuracy across $L \in \{512, 1024, 2048, 4096, 8192\}$ on long-history datasets (`ETTm*`, `Weather`, `Traffic`, `Electricity`) to prove that RetNet2.0's $O(1)$ recurrent/chunkwise state translates longer context directly into lower forecast error where `TimesFM-1.0` hits its 512-step limit.

---

## 3. Five-Axis Scaling Roadmap for a Top-Tier Paper

To elevate RetCast-JEPA from a single-GPU proof-of-concept into a general time-series foundation model paper:

### Axis 1: Pretraining Corpus Scaling (`1.3K` → `100K–1M+` Series, `10B+` Points)
- Remove the `min_len = 1200` filter by utilizing `RevIN` + `SpectralPatchEncoder` left-padding masks (`padding` is already supported throughout `RetCastJEPA.forward` and `compute_loss`).
- Ingest full `autogluon/chronos_datasets` + `Salesforce/lotsa_data` (strictly holding out all benchmark test sets / GIFT-Eval leak-free splits).
- Replace static synthetic harmonics with kernel-sampled Gaussian Process / KernelSynth mixtures (RBF, Periodic, Linear, Rational Quadratic, step-change regimes) mixed at 10–15%.

### Axis 2: Model Scaling Laws (`2M → 10M → 50M → 150M`)
- Standardize effective batch size (e.g., `256` sequences per optimizer step via gradient accumulation) and cosine learning-rate scaling across `2M`, `10M`, `50M`, and `150M` variants on A100/H100 hardware.
- Plot iso-FLOP and parameter-vs-loss Pareto frontiers comparing **RetNet2.0 + JEPA** against **Causal Transformer + JEPA** and **RetNet2.0 Raw (No JEPA)** across scales—testing whether JEPA + delta memory shifts the entire scaling curve downward rather than only helping at 2M.

### Axis 3: Multi-Resolution Patching & Arbitrary Context/Horizon Lengths
- Current architecture uses a single `patch_len = 16`. High-frequency 5-min/15-min series benefit from finer patches ($P = 8$) while low-frequency daily/weekly series benefit from coarser patches ($P = 32$) or multi-scale patch fusion.
- Combine chunkwise prefill (`chunk_size = 64`) with curriculum context lengths $L \in [256, 8192]$ during pretraining so the model actively uses slow decay heads ($\gamma_3 = 1 - 2^{-8}$, half-life $\approx 2,834$ steps at $P=16$).

### Axis 4: Cross-Variate Delta Retention & Continuous-Time Decay
- **Multivariate Dependencies**: Interleave channel tokens or add lightweight cross-variate delta-retention layers so multivariate sensors (e.g., `HUFL`/`HULL`/`OT` in ETT or regional load zones in Electricity) share state dynamically instead of forecasting strictly channel-independently.
- **Irregular Sampling**: Parameterize decay with elapsed timestamp gaps $\gamma_h^{\Delta t_i}$ (`exp(log_gamma * dt_i)`) to generalize natively to irregular event streams and varying sampling rates without manual frequency IDs.

### Axis 5: Probabilistic / Flow-Matching Decoder Conditioned on JEPA Latents
- Upgrade the deterministic linear quantile head (`MeaningDecoder`) to a lightweight **conditional flow-matching or diffusion / spline quantile head** conditioned on $\hat{z}_{\text{fut}}$, enabling calibrated multimodal trajectories and exact CRPS / WQL benchmarking alongside MSE/MAE.

---

## 4. Proposed Paper Structure & Experiment Checklist

### Working Title Options
- *"RetCast-JEPA: Sub-10M Zero-Shot Time-Series Forecasting via Delta-Rule Retention and Joint-Embedding Predictive Architectures"*
- *"Predict What the Future Means: Scaling Efficient Time-Series Foundation Models with RetNet2.0 and JEPA"*

### Section Outline
1. **Introduction**: Compute/KV-cache wall of 200M+ Transformer forecasters; why raw-point MSE wastes compact capacity on stochastic noise; summary of the 2.05M vs 203.57M zero-shot Pareto breakthrough.
2. **Method (`RetCast-JEPA`)**:
   - Mask-aware `RevIN` & `SpectralPatchEncoder` (time + morphology + `rfft` log-magnitude).
   - `RetNet2.0` temporal backbone (triple execution equivalence, geometric decays $g_h = 1 - 2^{-5-h}$, delta-rule overwrite + triangular solve, `short_conv=4`).
   - `LatentHorizonPredictor` (multi-scale exponential pooling, causal horizon mixing, stop-gradient EMA target encoder, VICReg).
   - Horizon decoding & joint forecasting + semantic alignment objective.
3. **Zero-Shot Foundation Evaluation**:
   - Main benchmark table across **GIFT-Eval / TSLib / Monash** (MSE, MAE, CRPS/WQL; 3–5 seeds; TimesFM-1.0/2.0, Chronos-Bolt, Moirai, PatchTST, iTransformer).
   - Long-context accuracy & systems scaling ($L = 512 \dots 16,384$: MSE, latency, peak VRAM, persistent state size).
4. **Controlled Ablations & Latent Geometry Analysis**:
   - JEPA vs No-JEPA (MSE/MAE + Effective Rank + Cosine Alignment + Retrieval R@1).
   - RetNet2.0 vs Plain RetNet vs Parameter-Matched Causal Transformer.
   - Decoder lag-primitive ablation (`with vs without SEASONAL_LAGS`).
   - Qualitative horizon rollouts (showing single-pass continuity vs autoregressive patch seams).
5. **Scaling Laws (`2M → 150M`) & Conclusion**: Iso-parameter and iso-FLOP Pareto curves, limitations, and edge-deployment implications.
