# BAH-2026 PS-14 — ML Time-Series Forecasting: State of the Art & Model Selection

**Problem:** Multi-horizon forecasting of GOES >2 MeV electron flux at geostationary orbit (GEO) at **30–45 min (nowcast), 6 h, and 12 h** ahead, from solar-wind drivers (Wind speed, IMF, density) + flux history. The target spans many orders of magnitude (model in **log10**), is **strongly autocorrelated**, has **severe class imbalance** (rare "harsh"/extreme enhancement events), a **strong diurnal (local-time) cycle**, and **missing data / spikes**.

This document surveys classical baselines, deep sequence models, transformers, time-series foundation models, multi-horizon strategies, uncertainty quantification, class-imbalance handling, evaluation metrics, libraries, and feature engineering, then recommends a **primary model + 2 alternatives**, a **metric suite**, an **evaluation/CV protocol**, and a **library stack**.

> **TL;DR recommendation:** **Temporal Fusion Transformer (TFT)** as the primary model — direct multi-horizon, native quantile (probabilistic) outputs, interpretable variable selection, and explicit support for *known-future* inputs (sin/cos local time). Augment with a **dual head**: a quantile-regression head for log-flux and a binary classification head for threshold exceedance (>10³ pfu "harsh" alert). Backups: **(A) N-HiTS / TiDE** (fast, strong, simple direct multi-horizon) and **(B) a fine-tuned/covariate-augmented time-series foundation model** (TimesFM+Cov or Chronos-2). Library stack: **PyTorch Lightning + pytorch-forecasting** (TFT, N-HiTS) and **neuralforecast/darts** for fast alternatives, with **LightGBM** for the strong tabular baseline. Always benchmark against **persistence + diurnal-climatology + NOAA-REFM-style linear filter**, evaluated per-horizon with a strict **chronological split + walk-forward CV**.

---

## 0. Why this problem is special (domain grounding)

Several published studies forecast exactly this target and directly inform model choice:

- **Transformer > LSTM at every horizon for GEO >2 MeV flux.** A modeling study at multiple prediction time scales (5-min, 1 h, 3 h, 6 h, 12 h, 1 day) found transformer networks consistently beat LSTM across all horizons, with Prediction Efficiency (PE) degrading monotonically with horizon: e.g. transformer PE ≈ 0.94 (1 h), 0.83 (6 h), 0.75 (12 h), 0.66 (1 day) vs LSTM PE ≈ 0.92 / 0.77 / 0.55 / 0.49. Best inputs combine flux history with solar wind (Vsw, N, Pd) + geomagnetic indices (Dst, AE, Kp) + position (MLT, Lm). [SWSC 2024](https://www.swsc-journal.org/articles/swsc/full_html/2024/01/swsc230072/swsc230072.html)
- **LSTM daily fluence models** reach PE ≈ 0.80 (day+1), 0.66 (day+2), 0.52 (day+3); the most important inputs were flux history, Vsw, and Kp. [Remote Sensing 2023, MDPI](https://www.mdpi.com/2072-4292/15/10/2538)
- **Time-series foundation model works here.** "TimesFM+Cov" (Google TimesFM + ridge regression on covariates for the residual) achieved average R² ≈ 0.90 across L-shells for MeV outer-belt electrons at 6 h, needing only ~6 months of fine-tuning data vs 11 years to train from scratch — but it underestimates storm peaks and lags storm onset by up to ~24 h. [arXiv 2605.15752](https://arxiv.org/html/2605.15752)
- **Three deep-learning strategies** specifically improve ≥2 MeV fluence prediction at GEO. [Sun et al., Space Weather 2025](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2025SW004519)
- **Operational baseline to beat:** NOAA's **Relativistic Electron Forecast Model (REFM)** predicts >2 MeV 24 h fluence from a *linear prediction filter on solar-wind speed*. Recent operational PE: **0.38 (1-day), −0.49 (2-day), −0.74 (3-day)** — i.e. beyond ~1 day it is worse than climatology. This is both the benchmark and the opportunity. [NOAA SWPC REFM](https://www.swpc.noaa.gov/products/relativistic-electron-forecast-model); [REFM statistics](https://services.swpc.noaa.gov/text/relativistic-electron-fluence-statistics.txt)
- **Diurnal structure is physical and predictable:** flux peaks near local noon, minimum near midnight, because a GEO spacecraft samples different drift shells vs local time (noon ~L=5 near the belt peak, midnight ~L=7). Local-time variation is well-described as a Kp-dependent Gaussian → **time-of-day must be a known-future covariate**, and **differencing removes diurnal + 27-day cycles**. [Su et al. 2014](https://agupubs.onlinelibrary.wiley.com/doi/10.1002/2014SW001069); [differencing to remove spurious correlations](https://ui.adsabs.harvard.edu/abs/2022mlph.conf...40S/abstract)
- **Operational alert threshold:** NOAA warns when >2 MeV flux exceeds **10³ pfu (cm² s sr)⁻¹** for ≥3 consecutive 5-min periods → defines the "harsh"/event class for the classification head. [NOAA SWPC](https://www.swpc.noaa.gov/products/relativistic-electron-forecast-model)

**Implication:** an attention/transformer-class model with explicit covariates, quantile outputs, and an event-classification head is well-matched; foundation models are a viable few-shot route; and any model must be measured against persistence/climatology/REFM with proper event metrics.

---

## 1. Classical & strong baselines

**Why baselines matter:** for a highly autocorrelated target, **persistence is deceptively strong at short horizons** (the nowcast). Reporting RMSE alone is misleading; the field uses *skill scores relative to a reference* (Prediction Efficiency = 1 − MSE/Var, equivalent to R² when the reference is the mean). A model is only useful if it **beats persistence and diurnal-climatology** at each horizon, and ideally beats REFM at 1 day. [PE/skill convention — SWSC 2024](https://www.swsc-journal.org/articles/swsc/full_html/2024/01/swsc230072/swsc230072.html)

| Baseline | What it is | Strength here | Weakness |
|---|---|---|---|
| **Persistence** | ŷ(t+h)=y(t) | Very hard to beat at 30–45 min due to autocorrelation; the reference for skill | Useless across storm onsets; degrades fast with h |
| **Climatology / diurnal-climatology** | mean (or per-local-time / per-Kp mean) of log-flux | Captures strong diurnal + seasonal cycle; reference for longer h | No event response |
| **ARIMA/SARIMAX** | linear AR + seasonal + exogenous | Cheap, interpretable; SARIMAX can ingest solar-wind exogenously | Single-series, linear, weak on multi-order-of-magnitude spikes & long memory |
| **NOAA-REFM-style linear filter** | linear prediction filter on Vsw | The literal operational benchmark for 24 h fluence | Linear; PE collapses past 1 day |
| **Ridge / linear w/ lags** | regression on lagged + coupling features | Fast, robust, great sanity check; basis of TimesFM+Cov residual trick | Linear only |
| **Gradient-boosted trees (LightGBM/XGBoost)** | tabular regression on lag + rolling + coupling features | **Strongest non-DL baseline**; handles nonlinearity, missingness, feature importances; 40%+ RMSE cuts over naive baselines reported; LightGBM ~20× faster than XGBoost | Needs one model per horizon (direct); no native sequence inductive bias; weak extrapolation to unseen extremes | 

References: [XGBoost/LightGBM for time series](https://medium.com/@geokam/time-series-forecasting-with-xgboost-and-lightgbm-predicting-energy-consumption-460b675a9cee); [LightGBM speed/accuracy](https://365datascience.com/tutorials/python-tutorials/xgboost-lgbm/); [40%+ RMSE reduction example](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12273526/).

**Recommendation:** implement **persistence, diurnal-climatology, REFM-style linear filter, and a LightGBM direct-multi-horizon model** as the baseline tier. The LightGBM model is also a credible *deliverable-grade* solution and a feature-importance oracle for the DL models.

---

## 2. Deep sequence models (RNN / TCN)

| Model | Pros | Cons | Multi-horizon? | Uncertainty? | Recommended? |
|---|---|---|---|---|---|
| **LSTM** | Proven on this exact task (PE 0.80–0.92 short h); handles long memory via gates | Slow, sequential, weaker than transformers here; recursive long-horizon error | Recursive or multi-output head | Via MC-dropout/ensemble | Baseline DL; not primary |
| **GRU** | Lighter LSTM, similar accuracy | Same limits | Same | Same | Lightweight alt |
| **BiLSTM** | Sees full context in encoder | Bidirectional only valid on *history* window, not future | Encoder–decoder | Same | Encoder option |
| **seq2seq + attention** | Encoder-decoder is the natural multi-horizon decoder; attention aligns drivers→response | More params, careful teacher-forcing | **Direct multi-step decoder** | + quantile decoder | Strong; TFT is its evolution |
| **TCN** | Dilated causal convs → long receptive field, parallel/fast, stable gradients | Fixed receptive field; less adaptive than attention | Multi-output head | + quantile/ensemble | Fast strong baseline |

LSTM and transformer comparisons for this target are documented above ([SWSC 2024](https://www.swsc-journal.org/articles/swsc/full_html/2024/01/swsc230072/swsc230072.html), [MDPI 2023](https://www.mdpi.com/2072-4292/15/10/2538)). The arXiv electron-belt study used LSTM, Conv1D, and a transformer-encoder as supervised baselines ([arXiv 2605.15752](https://arxiv.org/html/2605.15752)). **These are the right DL baselines, but the literature already shows attention beats them — so the primary model should be attention-based.**

---

## 3. Transformers & modern architectures

### 3a. Long-sequence efficient transformers
| Model | Key idea | Notes for this task |
|---|---|---|
| **Informer** (AAAI'21) | ProbSparse attention O(L log L), generative decoder one-shot multi-horizon | Good for long context; generative decoder = direct multi-horizon. [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/17325) |
| **Autoformer** (NeurIPS'21) | Series decomposition + Auto-Correlation | Decomposition suits diurnal/seasonal flux. |
| **FEDformer** (ICML'22) | Frequency-enhanced (Fourier/wavelet) blocks, linear complexity | Periodic structure (diurnal, 27-day) → frequency domain helps. |
| **Crossformer** (ICLR'23) | Cross-dimension attention (DSW embedding) | Models cross-variable (flux↔solar-wind) dependency. |

References: [Informer/Autoformer/FEDformer/Crossformer survey context](https://arxiv.org/pdf/2411.05793).

### 3b. Patch / inverted / channel transformers
- **PatchTST** — patches the series into subseries tokens; channel-independent; strong long-horizon SOTA, low memory. Consistently top-tier on long-horizon benchmarks. [Time-Series-Library leaderboard](https://github.com/thuml/Time-Series-Library)
- **iTransformer** — *inverts* dims (each variable is a token), excels at multivariate cross-series correlation; #2 on look-back-96 LTSF leaderboard (2024). [leaderboard](https://github.com/thuml/Time-Series-Library)
- **Crossformer** — explicit cross-dimension dependency (above).

### 3c. Multi-horizon / interpretable / linear models
- **Temporal Fusion Transformer (TFT)** — **the recommended primary**. Combines (i) *Variable Selection Networks* per timestep (interpretable feature importance — directly answers PS-14's "identify important solar-wind variables"), (ii) Gated Residual Networks + gating to skip unused components, (iii) an LSTM encoder-decoder for local processing, (iv) *interpretable multi-head attention* for long-range dependencies, (v) static covariate encoders, and crucially (vi) **explicit channels for static covariates, known-future inputs, and observed-past inputs**, with (vii) **quantile (pinball) outputs → native probabilistic multi-horizon**. Beats DeepAR by 36–69% in the original benchmarks; designed precisely for multi-horizon + mixed inputs. [Lim et al. 2021, IJF](https://www.sciencedirect.com/science/article/pii/S0169207021000637); [arXiv 1912.09363](https://arxiv.org/abs/1912.09363); [pytorch-forecasting TFT docs](https://pytorch-forecasting.readthedocs.io/en/v1.0.0/api/pytorch_forecasting.models.temporal_fusion_transformer.TemporalFusionTransformer.html)
- **N-BEATS** — deep stack of fully-connected basis-expansion blocks; interpretable trend/seasonality; pure direct multi-horizon. [arXiv](https://arxiv.org/abs/2201.12886)
- **N-HiTS** — N-BEATS + multi-rate sampling + hierarchical interpolation; ~20% better and ~50× faster on long horizons; excellent, simple, fast direct multi-horizon. [arXiv 2201.12886](https://arxiv.org/abs/2201.12886)
- **TiDE** — MLP encoder-decoder ("time-series dense encoder"); transformer-level accuracy at MLP speed; handles covariates; strong long-horizon. [Darts/neuralforecast]
- **DLinear / NLinear** — one-layer linear (with decomposition / normalization) that *beat many transformers* on standard LTSF benchmarks — must be in the baseline set as a "is deep learning even needed?" check. [Zeng et al.]; [leaderboard](https://github.com/thuml/Time-Series-Library)
- **TimesNet** — reshapes 1D series into 2D by period (FFT-found) to model intra/inter-period variation; strong general-purpose. [Time-Series-Library](https://github.com/thuml/Time-Series-Library)
- **TimeMixer / TimeXer** — current top of the THU long-horizon leaderboards (2024); TimeXer #1, iTransformer #2, TimeMixer #3 on look-back-96. [leaderboard](https://github.com/thuml/Time-Series-Library)

| Modern model | Multi-horizon | Native uncertainty | Known-future covariates | Interpretable | Recommended for PS-14 |
|---|---|---|---|---|---|
| **TFT** | ✅ direct | ✅ quantiles | ✅ (core feature) | ✅ var-selection + attention | **PRIMARY** |
| N-HiTS | ✅ direct | via ensemble/quantile loss | partial (futr_exog) | ✅ decomposition | **ALT A** |
| TiDE | ✅ direct | ensemble/quantile | ✅ | partial | ALT A (tie) |
| PatchTST | ✅ direct | ensemble/quantile | limited (channel-indep) | weak | strong baseline |
| iTransformer | ✅ direct | ensemble | limited | cross-var | multivariate baseline |
| DLinear/NLinear | ✅ direct | ensemble | no | weak | sanity baseline |
| TimesNet | ✅ direct | ensemble | limited | period maps | general baseline |
| Informer/Autoformer/FEDformer | ✅ | quantile (DeepAR-style) | partial | decomposition | optional |

---

## 4. Time-series foundation models (zero/few-shot)

Foundation models are pretrained on ~10¹¹ observations and forecast new series zero-shot or with light fine-tuning — attractive because PS-14 has *only* ~11 years of one signal and severe class imbalance, so transfer/few-shot can help.

| Model | Org | Arch | Covariates | License | Fit for PS-14 |
|---|---|---|---|---|---|
| **TimesFM** | Google | decoder-only (autoregressive) | via "+Cov" residual trick | open (Apache-2.0) | **Proven on this exact target (R²≈0.90 @6h with +Cov, 6-mo fine-tune).** [arXiv 2605.15752](https://arxiv.org/html/2605.15752) |
| **Chronos / Chronos-Bolt** | Amazon | T5 enc-dec; Bolt patches→**direct multi-step quantiles** | Bolt: covariate regressors via AutoGluon | **Apache-2.0** | Bolt is 250× faster, gives quantiles directly; easy fine-tune. [Chronos-Bolt](https://huggingface.co/amazon/chronos-bolt-base); [GitHub](https://github.com/amazon-science/chronos-forecasting) |
| **Chronos-2** | Amazon | universal; group-attention ICL | **past + known-future covariates, multivariate** (zero-shot) | open | Newest SOTA; supports covariates natively — strong few-shot candidate. [arXiv 2510.15821](https://arxiv.org/abs/2510.15821); [Amazon Science](https://www.amazon.science/blog/introducing-chronos-2-from-univariate-to-universal-forecasting) |
| **Moirai / Moirai-MoE** | Salesforce | masked-encoder, any-variate attention | multivariate (flattened in v1) | open | Universal multivariate; good baseline. [TDS](https://towardsdatascience.com/moirai-time-series-foundation-models-for-universal-forecasting-dc93f74b330f/) |
| **Lag-Llama** | ServiceNow | Llama backbone + lag features | univariate; LoRA/PEFT-friendly | open | Easiest to PEFT-fine-tune on proprietary series. |
| **TimeGPT** | Nixtla | proprietary transformer (API) | exog supported | **closed/commercial API** | Strong & fast but API dependency — risky for an offline ISRO deliverable. |
| **Tiny Time Mixers (TTM)** | IBM Granite | MLP-Mixer, <1–5M params | **exogenous mixer (multivariate)** | **Apache-2.0** | Runs on a laptop/1 GPU; few-shot with 5% data; great lightweight option. [HF r2](https://huggingface.co/ibm-granite/granite-timeseries-ttm-r2); [IBM](https://developer.ibm.com/tutorials/awb-foundation-model-time-series-forecasting/) |

**Feasibility/licensing:** TimesFM, Chronos/Chronos-Bolt/Chronos-2, Moirai, Lag-Llama, and TTM are **open (mostly Apache-2.0)** → safe to fine-tune and ship offline. **TimeGPT is a commercial API** → avoid as the deliverable core. For PS-14 the most credible foundation-model route is **TimesFM+Cov (already validated on MeV electrons) or Chronos-2/Chronos-Bolt (native covariates + quantiles + Apache-2.0)**, used **few-shot fine-tuned**, not pure zero-shot, because the diurnal + storm dynamics are domain-specific. Caveat from the literature: foundation models **underestimate storm peaks and lag onset** ([arXiv 2605.15752](https://arxiv.org/html/2605.15752)) — exactly the rare "harsh" events PS-14 cares about — so pair them with the event-classification head / weighted loss.

---

## 5. Multi-horizon strategy — emit 30–45 min, 6 h, 12 h simultaneously

| Strategy | How | Pros | Cons |
|---|---|---|---|
| **Recursive (iterated)** | one-step model fed its own predictions | simple, coherent | **errors accumulate geometrically**; sensitive to drift; bad across storm onsets |
| **Direct (per-horizon)** | separate model/head per horizon | **no error accumulation**; each horizon tuned | ignores inter-step coherence; more models (for trees) |
| **Joint multi-output (MIMO)** | one model emits the whole horizon vector in one pass | no accumulation **and** preserves inter-step dependency; one model | fixed horizon set; heavier model |

Evidence: recursive accumulates error and benefits from frequent retraining; direct avoids accumulation; **MIMO/joint multi-output produces all steps in one forward pass, avoiding accumulation while preserving inter-step dependencies**. [strategy comparison](https://towardsdatascience.com/6-methods-for-multi-step-forecasting-823cbde4127a/); [recursive vs direct](https://letsdatascience.com/blog/multi-step-time-series-forecasting-recursive-direct-and-hybrid-strategies); [bias-variance decomposition](https://arxiv.org/html/2511.11461).

**Recommendation: DIRECT / JOINT multi-horizon (MIMO).** Define the decoder horizon to cover 12 h at the native cadence (e.g. 5-min GOES → 144 steps) and **read off the 30–45 min, 6 h, 12 h targets from the single multi-step output**. This is exactly how TFT, N-HiTS, TiDE, PatchTST, and Chronos-Bolt operate. Rationale: (1) avoids the recursive error blow-up that ruins 6–12 h forecasts (and that sinks REFM past 1 day); (2) one training run yields all horizons; (3) preserves the temporal shape of an enhancement; (4) the literature's best GEO results came from direct/horizon-specific transformer models.

---

## 6. Probabilistic / uncertainty quantification

**Why it matters operationally:** satellite operators need *risk*, not a point estimate — e.g. P(flux > 10³ pfu in next 6–12 h). Quantiles/intervals enable cost-loss decisions and calibrated alerts.

| Method | Pros | Cons | Fit |
|---|---|---|---|
| **Quantile regression (pinball loss)** | direct prediction intervals, single model, native in TFT/N-HiTS/Chronos-Bolt | quantiles can cross; needs multiple outputs | **Primary** — use τ = {0.1, 0.5, 0.9} (or 0.05–0.95) |
| **Deep ensembles** | best-calibrated, captures epistemic uncertainty, simple | N× training cost | Use 3–5 seeds for the final model |
| **MC dropout** | cheap epistemic estimate at inference | underestimates tails; approximate | optional add-on |
| **Conformal prediction (CQR, copula-CP)** | **distribution-free coverage guarantee**, model-agnostic, wraps any model | needs calibration set; base intervals matter | **Recommended wrapper** for guaranteed coverage on top of quantiles |
| **Gaussian Processes** | principled posterior | O(n³), poor at 11-yr 5-min scale | not for full series |

References: [conformal + quantile + ensembles + MC-dropout overview](https://arxiv.org/pdf/2212.03281); [Conformalized Quantile Regression / CQR](https://daniel-bethell.co.uk/posts/conformal-prediction-guide/); [neural conformal control for TS](https://arxiv.org/pdf/2412.18144).

**Recommendation:** train with **pinball/quantile loss** (TFT native), report **P10/P50/P90**, optionally form a **deep ensemble (3–5 seeds)**, and wrap with **Conformalized Quantile Regression** on a held-out calibration block for guaranteed coverage. Validate with **reliability diagrams** and **PICP / pinball / CRPS**.

---

## 7. Class imbalance & extremes (rare "harsh" events)

Extreme enhancement events (>10³ pfu) are rare → plain MSE under-weights them and the model regresses to the diurnal mean; foundation models specifically **underestimate storm peaks** ([arXiv 2605.15752](https://arxiv.org/html/2605.15752)).

| Technique | Idea | Notes |
|---|---|---|
| **Weighted / inverse-frequency loss** | up-weight rare high-flux samples in MSE | "inverse weighting shifts the forecast distribution toward the extreme tail" [GMD 2023](https://gmd.copernicus.org/articles/16/251/2023/) |
| **Focal-style loss (regression focal / focal-R)** | down-weight easy samples, focus on hard/rare | used for rare SEP events [SEPNET, arXiv 2512.12786](https://arxiv.org/pdf/2512.12786) |
| **Output-weighted loss (OWL)** | weight by rarity of the *output value* | designed for extreme-event precursors [arXiv 2112.00825](https://arxiv.org/pdf/2112.00825) |
| **Oversampling / event-balanced batches** | sample storm windows more often | beware leakage/overfitting; combine with augmentation |
| **Dual regression + classification heads** | predict log-flux **and** P(exceed threshold) jointly | multi-task: shared encoder, regression head (quantiles) + classification head (BCE/focal) for >10³ pfu; "symmetric dual-head decoder: regression head + classification head for ordered intensity thresholds" [MAG-Net](https://arxiv.org/pdf/2604.02818); exceedance via probabilistic classifier [TDS](https://towardsdatascience.com/an-introduction-to-exceedance-probability-forecasting-4c96c0e7772c/); multi-task reg+clf [Sci Rep](https://www.nature.com/articles/s41598-026-43551-3) |
| **Extreme Value Theory (POT/GPD)** | model the tail above a high threshold with Generalized Pareto | Pickands–Balkema–de Haan: exceedances → GPD; principled tail extrapolation beyond observed extremes [arXiv 1509.01051](https://arxiv.org/pdf/1509.01051) |

**Recommendation:** use a **multi-task dual-head architecture** — a quantile-regression head for log10 flux **and** a binary classification head for P(flux > 10³ pfu) at each horizon — trained with a **combined weighted-quantile + focal-BCE loss**. This directly serves PS-14's "predict the *harsh* radiation fluxes" mandate (operators want the exceedance probability) and mitigates peak underestimation. Optionally fit a **GPD/EVT tail** on residual extremes for the most severe-event probabilities.

---

## 8. Evaluation metrics & protocol

### 8a. Continuous (per horizon, in log10 space)
- **RMSE, MAE** on log10(flux).
- **Prediction Efficiency (PE) = 1 − MSE/Var(obs)** (the field's standard skill score; = R² vs the mean). [SWSC 2024](https://www.swsc-journal.org/articles/swsc/full_html/2024/01/swsc230072/swsc230072.html)
- **Skill score vs reference** = 1 − MSE_model/MSE_ref, with ref = persistence (short h) and diurnal-climatology / REFM (long h). A model must show **positive skill over persistence and climatology**.
- **Linear correlation (LC), R², bias** (mean error). [SWSC 2024](https://www.swsc-journal.org/articles/swsc/full_html/2024/01/swsc230072/swsc230072.html)

### 8b. Event / threshold (>10³ pfu "harsh" alert) — from the 2×2 contingency table
- **POD** (hit rate / recall), **FAR** (false-alarm ratio), **POFD** (false-positive rate), **CSI** (critical success index), **HSS** (Heidke), **TSS** (True Skill = POD − POFD), **F1**, **ROC-AUC**. Space-weather verification emphasizes **F1, POD, TSS, HSS**. [SWPC verification glossary](https://www.swpc.noaa.gov/sites/default/files/images/u30/Forecast%20Verification%20Glossary.pdf); [solar-flare verification 1998–2024](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2025SW004546); [SEPNET metrics](https://arxiv.org/pdf/2512.12786)
- **Probabilistic event:** **reliability (calibration) diagram**, **Brier score / Brier skill score**, ROC. [forecast verification](https://www.cawcr.gov.au/projects/verification/)

### 8c. Probabilistic continuous
- **Pinball loss** (per quantile), **CRPS**, **PICP / interval coverage**, reliability diagrams. [UQ overview](https://arxiv.org/pdf/2212.03281)

### 8d. Reporting rule
Report **every metric per horizon separately** (30–45 min, 6 h, 12 h) — PE degrades strongly with horizon, so a single aggregate hides failure modes ([SWSC 2024](https://www.swsc-journal.org/articles/swsc/full_html/2024/01/swsc230072/swsc230072.html)).

### 8e. Train/val/test split & cross-validation (LEAKAGE-CRITICAL)
- **Strict chronological split** (e.g. ~7 yr train / 2 yr val / 2 yr test), never random — random splits leak future info through autocorrelation and inflate scores. [temporal split](https://apxml.com/courses/time-series-analysis-forecasting/chapter-6-model-evaluation-selection/train-test-split-time-series); [hidden leakage in LSTM eval](https://arxiv.org/pdf/2512.06932)
- **Walk-forward (expanding/rolling) CV** for hyperparameters: train on a block, validate on the next, slide forward — mirrors operational retraining. [walk-forward CV](https://machinelearningmastery.com/5-ways-to-use-cross-validation-to-improve-time-series-models/)
- **Purge + embargo** between train and test to remove windows whose input/label horizons overlap (here ≥12 h gap). [purged CV](https://en.wikipedia.org/wiki/Purged_cross-validation)
- **Fit scalers/feature stats on train only**; ensure both train and test span quiet *and* storm periods (blocked CV mixes seasons). [blocked CV](https://medium.com/@pacosun/respect-the-order-cross-validation-in-time-series-7d12beab79a1)
- Hold out a **separate calibration block** for conformal prediction.
- For the final demo, also validate against **ISRO GRASP/GSAT** flux at Indian longitude (1–2 yr) as an independent cross-instrument test.

---

## 9. Practical libraries

| Library | Has | Strengths | Use |
|---|---|---|---|
| **PyTorch + Lightning** | base | full control, scaling, callbacks | foundation runtime |
| **pytorch-forecasting** | **TFT, N-BEATS, N-HiTS, DeepAR, TiDE**, QuantileLoss, TimeSeriesDataSet (static/known-future/past covariates), interpretation utils | **Best TFT + multi-horizon + quantile + interpretability** ergonomics | **PRIMARY (TFT, N-HiTS)** [docs](https://pytorch-forecasting.readthedocs.io/); [GitHub](https://github.com/sktime/pytorch-forecasting) |
| **neuralforecast (Nixtla)** | N-HiTS, N-BEATS, TFT, PatchTST, iTransformer, TiDE, TimesNet, TSMixer | fast, consistent API, great long-horizon coverage | fast model bake-off [long-horizon transformers](https://nixtlaverse.nixtla.io/neuralforecast/docs/tutorials/longhorizon_transformers.html) |
| **darts** | classical + RNN/TCN/NBEATS/NHiTS/TFT + foundation + backtesting | sklearn-like `.fit/.predict`, **built-in backtesting & metrics**, probabilistic | baselines + backtesting + quick comparisons [models](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.html) |
| **GluonTS** | DeepAR, probabilistic models, Chronos/Moirai integration | probabilistic-first | probabilistic baselines |
| **AutoGluon-TimeSeries** | Chronos/Chronos-Bolt fine-tune + covariate regressors + ensembling | one-line foundation-model fine-tuning | foundation-model route |
| **sktime** | unified TS API, reductions, CV splitters | pipelines & temporal CV | CV scaffolding/baselines |
| **tsai** | InceptionTime, TST, fastai-based | classification/regression on windows | extra DL baselines |
| **LightGBM/XGBoost** | GBDT | the strong tabular baseline | baseline tier |

**Recommendation — standardize on:** **PyTorch Lightning + pytorch-forecasting** for the primary TFT (and N-HiTS) because it uniquely combines TFT, the TimeSeriesDataSet covariate abstraction (static / known-future / past), QuantileLoss, and built-in interpretation. Add **neuralforecast/darts** for fast alternatives + backtesting, **AutoGluon-TS** for the foundation-model route, **LightGBM** for the tabular baseline, and **sktime/darts** for CV splitters and metrics.

---

## 10. Feature engineering

**Target:** `log10(>2 MeV flux)`; consider **differencing** to remove the diurnal + 27-day cycles and trends ([differencing](https://ui.adsabs.harvard.edu/abs/2022mlph.conf...40S/abstract)). Handle missing data via interpolation + a **missingness mask feature**, and despike before logging.

**Lag & rolling features (observed-past inputs):**
- Lagged flux (e.g. t−1…t−N at native cadence; multi-scale lags 1 h/6 h/24 h).
- Rolling mean/std/min/max/slope of flux over 1 h/6 h/24 h windows.
- Rolling stats of solar wind (Vsw, N, Pdyn, |B|, Bz).
- **Rates / differences** Δflux, dBz/dt (storm-onset signal).

**Known-future inputs (calendar — available for all horizons → feed to TFT decoder):**
- **sin/cos of time-of-day** (diurnal cycle), **sin/cos day-of-year** (seasonal), local-time / MLT encodings. The diurnal cycle is physical and predictable, so these belong in the *future-known* channel. [Su et al. 2014](https://agupubs.onlinelibrary.wiley.com/doi/10.1002/2014SW001069)

**Solar-wind coupling functions (physics-informed drivers):**
- **vBs** (half-wave-rectified dawn-dusk E-field; 0 for Bz>0) — best simple driver of Dst. [Newell 2007](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2006JA012015)
- **Newell coupling** dΦ/dt = v^(4/3)·B_T^(2/3)·sin^(8/3)(θc/2) — best universal solar-wind→magnetosphere coupling. [Newell 2007](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2006JA012015)
- **Epsilon / Akasofu** ε ∝ v·B²·sin⁴(θc/2) — energy input. [Newell 2007](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2006JA012015)
- IMF **clock angle** θc, dynamic pressure Pd = ρv², magnetopause standoff distance R₀.
- Geomagnetic indices if allowed (Kp, Dst/SYM-H, AE) — repeatedly the most predictive ([MDPI 2023](https://www.mdpi.com/2072-4292/15/10/2538), [SWSC 2024](https://www.swsc-journal.org/articles/swsc/full_html/2024/01/swsc230072/swsc230072.html)).

**Static covariates:** satellite ID / longitude (75°W vs 135°W vs GSAT) — feed to TFT static encoder.

Let **TFT's variable-selection network rank these** → satisfies PS-14's requirement to "identify important solar-wind variables."

---

## 11. Final recommendation

### PRIMARY model — Temporal Fusion Transformer (TFT), dual-head, direct multi-horizon, quantile
- **Why:** purpose-built for multi-horizon + mixed inputs; native **quantile (probabilistic)** outputs; **interpretable variable selection** (answers the "important drivers" objective); explicit **known-future** channel for the diurnal sin/cos covariates; attention-class models already shown to beat LSTM on this exact target. [Lim et al. 2021](https://www.sciencedirect.com/science/article/pii/S0169207021000637); [SWSC 2024](https://www.swsc-journal.org/articles/swsc/full_html/2024/01/swsc230072/swsc230072.html)
- **Heads:** quantile-regression head (P10/P50/P90 of log-flux) + binary classification head (P(>10³ pfu)) per horizon; combined weighted-quantile + focal-BCE loss for the rare events.
- **Strategy:** direct/joint multi-horizon decoder covering 12 h; read off 30–45 min / 6 h / 12 h.
- **Library:** pytorch-forecasting + Lightning.

### ALTERNATIVE A — N-HiTS (or TiDE)
- Fast, simple, strong direct multi-horizon; ~50× faster than transformers, ~20% better than N-BEATS on long horizons; easy quantile loss & ensembling. Excellent secondary model / ablation and a robust fallback if TFT overfits the limited data. [N-HiTS](https://arxiv.org/abs/2201.12886) (via neuralforecast/pytorch-forecasting).

### ALTERNATIVE B — Fine-tuned time-series foundation model (TimesFM+Cov or Chronos-2 / Chronos-Bolt)
- Few-shot route needing only ~months of data; **TimesFM+Cov already validated on MeV electrons (R²≈0.90 @6h)**; Chronos-2/Bolt give native covariates + quantiles under Apache-2.0. Pair with the event-classification head to fix the known **storm-peak underestimation / onset lag**. [arXiv 2605.15752](https://arxiv.org/html/2605.15752); [Chronos-2](https://arxiv.org/abs/2510.15821)

### Baselines to always report (skill reference)
Persistence, diurnal-climatology, REFM-style linear filter, **LightGBM** (direct multi-horizon), and LSTM/GRU — each per horizon.

### Recommended metric suite
- **Continuous (per horizon, log10):** RMSE, MAE, **PE/skill score vs persistence & climatology**, R², LC, bias.
- **Event (>10³ pfu, per horizon):** POD, FAR, CSI, **HSS, TSS**, F1, ROC-AUC; **Brier/BSS** + **reliability diagram** for the probabilistic head.
- **Probabilistic continuous:** pinball loss, **CRPS**, PICP / coverage, reliability.

### Recommended library stack
**PyTorch Lightning + pytorch-forecasting** (TFT, N-HiTS) · **neuralforecast / darts** (fast alternatives + backtesting + metrics) · **AutoGluon-TS** (foundation-model fine-tuning) · **LightGBM** (tabular baseline) · **sktime/darts** (temporal CV splitters).

### Evaluation/CV protocol
**Strict chronological split** (~7/2/2 yr) → **walk-forward (expanding) CV** for tuning → **purge + ≥12 h embargo** between train/test → **scalers fit on train only** → ensure quiet+storm coverage in each split → **separate calibration block** for conformal intervals → independent **GRASP/GSAT** cross-validation for the demo. **Never random-split.**

---

## References (URLs)

**Domain (GEO electron flux / space weather):**
1. SWSC 2024 — LSTM & transformer for ≥2 MeV GEO flux at multiple horizons: https://www.swsc-journal.org/articles/swsc/full_html/2024/01/swsc230072/swsc230072.html
2. Remote Sensing 2023 (MDPI) — LSTM 3-day ≥2 MeV daily fluence: https://www.mdpi.com/2072-4292/15/10/2538
3. Space Weather 2025 — three deep-learning strategies for ≥2 MeV fluence: https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2025SW004519
4. arXiv 2605.15752 — ML + TimesFM foundation model for MeV outer-belt electrons: https://arxiv.org/html/2605.15752
5. NOAA SWPC REFM (operational baseline): https://www.swpc.noaa.gov/products/relativistic-electron-forecast-model
6. REFM fluence statistics (PE values): https://services.swpc.noaa.gov/text/relativistic-electron-fluence-statistics.txt
7. Su et al. 2014 — local-time/Kp specification of >2 MeV flux: https://agupubs.onlinelibrary.wiley.com/doi/10.1002/2014SW001069
8. Differencing to remove spurious correlations (diurnal/27-day): https://ui.adsabs.harvard.edu/abs/2022mlph.conf...40S/abstract
9. Newell et al. 2007 — universal solar-wind–magnetosphere coupling (Newell/vBs/epsilon): https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2006JA012015

**Models / architectures:**
10. Temporal Fusion Transformer (Lim et al. 2021, IJF): https://www.sciencedirect.com/science/article/pii/S0169207021000637
11. TFT arXiv: https://arxiv.org/abs/1912.09363
12. N-HiTS arXiv: https://arxiv.org/abs/2201.12886
13. Informer (AAAI'21): https://ojs.aaai.org/index.php/AAAI/article/view/17325
14. THU Time-Series-Library (PatchTST/iTransformer/TimesNet/DLinear + leaderboard): https://github.com/thuml/Time-Series-Library
15. Deep-learning TS survey (architectural diversity): https://arxiv.org/pdf/2411.05793
16. Multi-Horizon Quantile Recurrent Forecaster: https://arxiv.org/pdf/1711.11053

**Foundation models:**
17. Chronos / Chronos-Bolt (GitHub): https://github.com/amazon-science/chronos-forecasting
18. Chronos-Bolt-base (HF, Apache-2.0): https://huggingface.co/amazon/chronos-bolt-base
19. Chronos-2 arXiv: https://arxiv.org/abs/2510.15821
20. Chronos-2 (Amazon Science): https://www.amazon.science/blog/introducing-chronos-2-from-univariate-to-universal-forecasting
21. Moirai (Salesforce, TDS): https://towardsdatascience.com/moirai-time-series-foundation-models-for-universal-forecasting-dc93f74b330f/
22. IBM Granite Tiny Time Mixers (HF r2): https://huggingface.co/ibm-granite/granite-timeseries-ttm-r2
23. IBM Granite TS tutorial: https://developer.ibm.com/tutorials/awb-foundation-model-time-series-forecasting/

**Multi-horizon strategy:**
24. 6 methods for multi-step forecasting (direct/recursive/MIMO): https://towardsdatascience.com/6-methods-for-multi-step-forecasting-823cbde4127a/
25. Recursive vs direct vs hybrid: https://letsdatascience.com/blog/multi-step-time-series-forecasting-recursive-direct-and-hybrid-strategies
26. Epistemic error decomposition (recursive vs direct): https://arxiv.org/html/2511.11461

**Uncertainty:**
27. Copula conformal prediction for multi-step TS (UQ survey): https://arxiv.org/pdf/2212.03281
28. Conformal prediction guide (CQR): https://daniel-bethell.co.uk/posts/conformal-prediction-guide/
29. Neural conformal control for TS: https://arxiv.org/pdf/2412.18144

**Class imbalance / extremes:**
30. GMD 2023 — imbalanced-regression loss for extreme wind: https://gmd.copernicus.org/articles/16/251/2023/
31. SEPNET multi-task DL (focal loss, rare events): https://arxiv.org/pdf/2512.12786
32. Output-weighted/relative-entropy loss for extremes: https://arxiv.org/pdf/2112.00825
33. Dual-head regression+classification (MAG-Net): https://arxiv.org/pdf/2604.02818
34. Exceedance probability forecasting: https://towardsdatascience.com/an-introduction-to-exceedance-probability-forecasting-4c96c0e7772c/
35. Multi-task reg+clf joint feature selection: https://www.nature.com/articles/s41598-026-43551-3
36. EVT POT/GPD for time series: https://arxiv.org/pdf/1509.01051

**Metrics / verification / CV:**
37. SWPC Forecast Verification Glossary (POD/FAR/CSI/HSS/TSS): https://www.swpc.noaa.gov/sites/default/files/images/u30/Forecast%20Verification%20Glossary.pdf
38. NOAA solar-flare forecast verification 1998–2024: https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2025SW004546
39. WMO/CAWCR Forecast Verification methods: https://www.cawcr.gov.au/projects/verification/
40. Time-series train/test split: https://apxml.com/courses/time-series-analysis-forecasting/chapter-6-model-evaluation-selection/train-test-split-time-series
41. Hidden leakage in LSTM evaluation: https://arxiv.org/pdf/2512.06932
42. Walk-forward CV best practices: https://machinelearningmastery.com/5-ways-to-use-cross-validation-to-improve-time-series-models/
43. Purged cross-validation: https://en.wikipedia.org/wiki/Purged_cross-validation
44. Blocked/ordered CV: https://medium.com/@pacosun/respect-the-order-cross-validation-in-time-series-7d12beab79a1

**Libraries:**
45. pytorch-forecasting docs (TFT, QuantileLoss, TimeSeriesDataSet): https://pytorch-forecasting.readthedocs.io/
46. pytorch-forecasting GitHub: https://github.com/sktime/pytorch-forecasting
47. neuralforecast long-horizon transformers: https://nixtlaverse.nixtla.io/neuralforecast/docs/tutorials/longhorizon_transformers.html
48. darts forecasting models: https://unit8co.github.io/darts/generated_api/darts.models.forecasting.html
49. XGBoost/LightGBM for TS: https://medium.com/@geokam/time-series-forecasting-with-xgboost-and-lightgbm-predicting-energy-consumption-460b675a9cee
50. LightGBM/XGBoost TS tutorial: https://365datascience.com/tutorials/python-tutorials/xgboost-lgbm/

*Compiled 2026-06-20 for BAH-2026 PS-14.*
