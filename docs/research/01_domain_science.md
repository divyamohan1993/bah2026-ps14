# Domain Science Brief — Forecasting the >2 MeV Energetic Electron Radiation Environment at Geostationary Orbit

**BAH-2026 Problem Statement 14: "Forecasting Energetic Particle Radiation Environment for ISRO's Geostationary Satellites"**

Target: forecast >2 MeV ("killer"/relativistic) **electron** flux at geostationary orbit (GEO, L≈6.6) at horizons of **30–45 min (nowcast), 6 h, and 12 h**, using GOES electron flux and Wind/L1 solar-wind data.

Prepared as a space-weather domain-science reference for model design. Date: 2026-06-20.

---

## 0. Executive Summary (read this first)

- **Solar wind speed (Vsw) is the single dominant external driver** of >2 MeV electron flux at GEO. The flux response **peaks ~1–2 days after** a Vsw increase (Paulikas & Blake 1979; Reeves et al. 2011; Wing et al. 2016 found the information transfer Vsw→J_e peaks at **+2 days**; Wang et al. 2024 SHAP found **Vsw lagged by 1 day** is the most important single feature). This 1–2 day lag is the physical reason multi-hour-to-day forecasting is feasible.
- **L1 (Wind) gives a ~20–90 min lead** (mean ≈ 47 min) for the solar wind to reach the magnetosphere. This is the physical basis of the **30–45 min nowcast** horizon: at the nowcast horizon you are largely propagating already-observed L1 conditions plus persistence of the slowly varying belt.
- **Acceleration**: inward radial diffusion (ULF/Pc5 waves, driven by Vsw) + local acceleration by whistler-mode chorus waves acting on a substorm-injected **seed population (~100–300 keV)**. Both require **hours to ~1–2 days**.
- **Loss**: magnetopause shadowing (L>5, dynamic-pressure/Bz driven, **minutes–hours**), outward radial diffusion, and EMIC-/chorus-wave precipitation. Sudden "dropouts" can drop flux 1–3 orders of magnitude in hours.
- **Hazard**: deep-dielectric / internal charging. NOAA alert threshold = **>2 MeV integral flux ≥ 1000 pfu** [particles cm⁻² s⁻¹ sr⁻¹] sustained; multi-day **fluence** (~10⁹ pfu/day) is the key cumulative damage metric.
- **Target representation**: **log10(flux)** (flux spans >3 orders of magnitude); predict **multi-horizon log-flux regression** plus a parallel **threshold-exceedance classification** (≥1000 pfu) for operational alerting.
- **Benchmark against**: NOAA **REFM** (Vsw linear filter, day-ahead fluence), Sheffield **SNB³GEO** (NARMAX, Vsw+N+Bz, 24 h, PE≈0.78), **Boynton NARMAX** family, **PreMevE 2.0** (LANL), **ORIENT-R** (Chu 2021 NN, corr 0.95), and the **LSTM/Transformer multi-horizon** GEO study (Wei/Zhang 2024) which directly reports 1 h / 6 h / 12 h / 1-day skill.

---

## 1. Physics of the >2 MeV ("killer") electron population at GEO

### 1.1 Where GEO sits
GEO is at **L ≈ 6.6**, in the **heart-to-outer-edge of the outer Van Allen radiation belt** (outer belt typically L≈3–7). >2 MeV electrons here are **relativistic** (Lorentz factor γ ≳ 5; v ≳ 0.98c). Their intensity varies by **>3 orders of magnitude** on timescales from minutes (dropouts) to the solar cycle (Reeves et al. 2011; DREAM literature). This enormous dynamic range is the central modelling challenge and the reason a log target is mandatory.

### 1.2 Acceleration mechanisms (build-up; timescale hours → ~2 days)
The modern consensus is that **two processes act together**, with their relative importance varying by event, L-shell and pitch angle:

1. **Inward radial diffusion driven by ULF (Pc5) waves.** ULF waves violate the third adiabatic invariant, transporting electrons inward to lower L and (conserving the first invariant) energizing them. ULF/Pc5 power correlates strongly with **Vsw** (Kelvin–Helmholtz shear at the magnetopause + solar-wind dynamic-pressure buffeting), which is the mechanistic link behind the Vsw–flux correlation. Dominant at lower L-shells and higher equatorial pitch angles; enhanced D_LL during storms. (Mathie & Mann; Rae/Ozeke; Nature s41598-025-23908-w; ScienceDirect S1364682603002098.)
2. **Local acceleration by whistler-mode chorus waves.** Chorus (generated outside the plasmapause by anisotropic 10–100 keV electrons injected during substorms) stochastically accelerates a **seed population (~100–300 keV)** up to multi-MeV energies via gyro-resonance. A growing local **phase-space-density peak at L≈4.5–5.5** is the observational fingerprint. Can energize 100–300 keV → >1 MeV in **a few hours** when chorus amplitudes are ~10–20 nT (Thorne; Reeves; Frontiers fspas.2023.1168636; Mithaiwala 2005).

**The seed/source population is essential and sequential** ("two-step"): (i) substorm injection of a **source** population (tens of keV) that drives chorus wave growth, then (ii) acceleration of the **seed** (hundreds of keV) to MeV energies (Jaynes; Boyd). A storm without an adequate seed population often fails to produce a >2 MeV enhancement. **Practical implication: lower-energy GOES channels (tens–hundreds of keV) are leading indicators of the >2 MeV channel.**

### 1.3 Loss mechanisms (depletion; timescale minutes → hours)
- **Magnetopause shadowing + outward radial diffusion.** When the magnetopause is compressed inside GEO (high dynamic pressure, strong southward Bz, CME/shock arrival), electrons on drift paths that intersect the magnetopause are lost to the magnetosheath ("shadowing"), preferentially near **90° pitch angle** and at **L > 5** (i.e., GEO is directly exposed). Combined with outward radial diffusion this produces rapid **dropouts**. (RG 307999943; Frontiers fspas.2025.1694836.)
- **EMIC-wave precipitation.** Electromagnetic ion cyclotron waves resonate with **>1 MeV** electrons (especially in high-density regions at the plasmapause/plumes), scattering them into the atmosphere; dominant loss at **L* < 4** and at low pitch angles; can drive 1.5–3 MeV dropouts. (Nakamura 2019; Cervantes 2020.)
- **Chorus/hiss precipitation.** Plasmaspheric hiss slowly erodes the slot/inner edge; chorus also scatters lower-energy (<500 keV) electrons on the duskside.
- Net: "**dropouts** are a combination of magnetopause shadowing at L*>5 and EMIC loss at L*<4." During the main phase of storms (Dst minimum) flux first drops, then re-builds during recovery — the classic **Dst-flux "two-phase" behaviour** (Reeves' "depletion then enhancement").

### 1.4 Satellite hazard — deep-dielectric / internal charging
- >2 MeV electrons **penetrate spacecraft shielding** and bury charge inside dielectric materials (cables, circuit boards, coax). When the internal E-field exceeds breakdown, an **electrostatic discharge (ESD)** can corrupt or destroy electronics.
- Unlike surface charging (driven by ~keV plasma, fast), deep-dielectric charging is **cumulative**: dielectrics retain charge for **days to months**, so the **multi-day fluence** — not just instantaneous flux — governs risk (CiteseerX deep-dielectric review).
- Documented anomalies tied to relativistic-electron environments: **Galaxy 15** (5 Apr 2010), **Telstar 401**, and many others (Saiz et al. 2018; Lohmeyer; Baker). This is exactly why ISRO/operators need forecasts: to schedule load-shedding, defer manoeuvres, and increase monitoring before high-fluence intervals.

---

## 2. Drivers / predictive features and their physical lag times

The literature is remarkably consistent: **Vsw dominates; geomagnetic indices (AE/AL, Kp) and the seed population add skill; Bz, density and dynamic pressure act both as acceleration enablers and as loss triggers.** Quantitative findings:

| Driver | Physical role for >2 MeV @ GEO | Typical lag to >2 MeV response | Sign / correlation | Key source(s) |
|---|---|---|---|---|
| **Solar wind speed Vsw** | #1 driver; powers ULF radial diffusion & chorus (via convection/substorms) | **+1 to +2 days** (peak); Wing 2016: TE peak at **+2 d**; Wang 2024 SHAP: **+1 d** strongest | Strong positive; r≈0.5–0.6 (log flux); "triangle" distribution (high Vsw → wide flux range) | Paulikas & Blake 1979; Reeves 2011; Wing 2016; Wang 2024 |
| **AE / AL index** (substorm/auroral) | Proxy for substorm injection of seed/source population & chorus generation | **~0 to +1–2 days** | Positive; co-ranks with/above Vsw in some analyses | Ganushkina/Bhaskar 2024 (30-yr); Chu 2021 (ORIENT-R: AL top feature) |
| **Seed electrons (tens–hundreds keV, e.g. GOES >0.8 MeV, 40–475 keV)** | Direct precursor population that is accelerated to MeV | **Hours → ~1 day** (faster than Vsw) | Strong positive; near-term leading indicator | Jaynes; Boyd; Boynton 2016/2019 |
| **IMF Bz (southward)** | Enhances dayside reconnection/convection → injections & ULF; but strong southward + shock → magnetopause erosion/loss | **Acceleration:** ~hours–day; **loss:** minutes–hours | Mixed: southward enables build-up, but extreme southward/pressure → dropout | Sakaguchi 2013; Wang 2024; Boynton 2016 |
| **Solar wind density N** | Low N favours enhancement; high N (+ pressure) → magnetopause compression/shadowing loss; controls plasmaspheric density (EMIC) | minutes–hours (loss); contributes to multi-day balance | Generally **negative** for high-N pulses; nontrivial | Balikhin/SNB³GEO; Wang 2024; Kataoka |
| **Dynamic pressure Pdyn (∝ N·Vsw²)** | Magnetopause compression → shadowing; shock injection (sudden commencement can also inject) | **minutes–hours** | Dual: prompt loss via shadowing; can trigger injections | Sakaguchi 2013 (Vsw+Bz+Pdyn best); Boynton |
| **ULF (Pc5) wave power** | The actual radial-diffusion engine (mediates Vsw→flux) | hours–~2 days | Strong positive; clearest in declining phase | Mathie & Mann; Rae/Ozeke; S1364682603002098 |
| **Kp** | Coarse 3-hourly global activity proxy | ~0 to +1–2 days | Positive | Koons & Gorney 1991; many |
| **Dst / SYM-H** | Ring-current/storm-phase proxy; main-phase depletion then recovery enhancement | Flux often **anticorrelated at main phase**, then rebuilds in recovery (days) | Nonlinear / two-phase | Reeves; Chu 2021 (SYM-H top feature) |
| **Persistence (flux yesterday / autoregressive term)** | The belt has long memory (days); strongest single predictor at short horizons | lag-1 day / recent history | Very strong positive | All NARMAX/AR/ML models |

**Solar-wind structure context.** **CIRs / high-speed streams (HSS)** — dominant in the **solar-cycle declining phase** — are *more efficient* than CMEs at producing prolonged multi-MeV enhancements at GEO (sustained Vsw + recurrent ULF/chorus), often to >7 MeV. **CME-driven storms** give brief, intense, denser events with strong Dst, larger main-phase dropouts, but the largest *outer-belt/GEO* enhancements often follow *moderate CIR-driven* storms during recovery (Pandya 2019; Miyoshi/Kataoka; NTRS 20110023417).

**Information-theory ranking (Wing et al. 2016, 1.8–3.5 MeV @ GEO):** Vsw is "by far the most important parameter," IMF |B| a clear second, with Vsw→J_e transfer entropy peaking at **+2 days**. **30-year cross-correlation/transfer-entropy study (2024 AdSpR, LANL 1990–2018):** "AE index and solar wind velocity are the dominant driving parameters." **ORIENT-R (Chu 2021):** most important features = **AL, Vsw, solar-wind density, SYM-H**.

---

## 3. L1-to-GEO solar-wind propagation lead time (basis of the 30–45 min nowcast)

- Wind orbits the **L1 Lagrange point ~1.5 million km (~235 R_E) sunward** of Earth. Solar-wind features observed at L1 take time to convect to the magnetosphere, giving an intrinsic **forecast lead**.
- **Propagation delay ≈ 20–90 min**, **mean ≈ 47 min** (Cameron & Jackel-type ML timing study, SWSC 2021): "~20 min for extreme ICMEs (~1000 km/s) and ~90 min for slow shocks (~300 km/s)." Equivalent statement: warning time ~1.5 h at 300 km/s down to ~25 min at 1000 km/s.
- Components: L1→bow shock (dominant), magnetosheath transit (~1 min), magnetopause→ground.
- **Implication for PS-14:** the **30–45 min nowcast** is precisely the regime where the most recent L1 (Wind) measurements are still "in flight" toward the magnetosphere. A nowcast model can exploit (a) propagated L1 conditions and (b) persistence/short-memory of the slowly varying >2 MeV belt. (Note: for the >2 MeV population specifically, the *acceleration* response to that solar wind is slow (1–2 days); so the 30–45 min skill comes mostly from belt persistence + immediate loss signals like dynamic-pressure-driven shadowing, while the 6 h/12 h/day horizons increasingly tap the Vsw-driven acceleration lag.)

---

## 4. Local-time (MLT) / diurnal & seasonal variation at GEO

- **Diurnal variation ≈ one order of magnitude** in the GOES >2 MeV flux: **maximum near local noon, minimum near local midnight** (Rodriguez/SWPC; Kamp 2024). Cause: GEO sits at fixed geocentric distance but the magnetosphere is asymmetric — on the dayside, compressed field + Shabansky/off-equatorial drift raise the locally measured trapped flux; on the nightside, field-line stretching and a more tail-like configuration lower it. **Operationally, a GOES at a given longitude samples a different MLT every hour, imprinting a strong diurnal cycle that must be modelled (include MLT/sin-cos of local time as a feature).**
- **Dawn–dusk asymmetry**: occurrence of >2 MeV flux-increase events is **higher in the dusk sector**; lower-energy (100–300 keV) trapped flux ~2× higher at dawn for L=3–7 when activity is low (reversing at L>7). Mechanisms: duskward-drifting injected electrons are precipitated by chorus (<500 keV) / EMIC (>1 MeV) or lost to the magnetopause at L>5; duskside diamagnetic field depression. (arXiv 1701.04701; ResearchGate 362627258.)
- **Magnetopause-compression effect**: when Pdyn pushes the magnetopause toward/inside GEO (especially **local noon**), GOES can see abrupt flux changes and even cross the last-closed-drift-shell / magnetopause — a direct loss signature and a confound for the dayside maximum.
- **Semiannual / seasonal variation**: clear **maxima near the equinoxes, minima near solstices**, with up to **~2 orders of magnitude** difference in 0.3–4.2 MeV flux at L>3.5. Driven primarily by the **Russell–McPherron effect** (equinoctial geometry increases the effective southward IMF and thus solar-wind coupling). (ANGEO 39/413/2021; S027311772100185X.) **Implication: include day-of-year / equinox phase as a slowly varying feature.**

---

## 5. Existing operational & published forecast models (benchmarks)

> Note on metrics: **PE = Prediction Efficiency** = 1 − MSE/Var(obs) (i.e., R²-like; 1 = perfect, 0 = climatology). **LC / r** = linear (Pearson) correlation, usually on **log10 flux**. "Uncertainty factor" = 10^RMSE(log10 flux).

### 5.1 NOAA SWPC **REFM** (Relativistic Electron Forecast Model) — *the operational baseline*
- **Type**: linear prediction filter (Baker/Rodriguez lineage). **Input**: real-time **Vsw at L1** (ACE/DSCOVR) + GOES >2 MeV electrons. **Output**: **daily fluence** of >2 MeV electrons at GEO for **+1, +2, +3 days** (a fluence forecast, not instantaneous flux). Purpose: anticipate damaging relativistic-electron events to supplement alerts.
- **Skill**: solid day-ahead fluence skill but **simplest input set** (Vsw only) limits it; outperformed by SNB³GEO at +1 day. URL: https://www.swpc.noaa.gov/products/relativistic-electron-forecast-model

### 5.2 Sheffield **SNB³GEO** (Balikhin et al.) — *operational since 2012*
- **Type**: Multi-Input Single-Output **NARMAX** (nonlinear autoregressive moving-average with exogenous inputs; FROLS structure selection). **Inputs**: **Vsw, density N, IMF Bz** at L1 (+ autoregressive flux). **Output**: **24 h-ahead** >2 MeV flux at GEO.
- **Skill**: **PE ≈ 0.775** for the >2 MeV flux; **more accurate than REFM at 1-day** (Balikhin et al. 2016). URLs: https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1002/2015SW001303 ; perf. eval https://www.sciencedirect.com/science/article/abs/pii/S0273117722010766

### 5.3 **Boynton et al.** NARMAX family
- **Type**: MISO NARMAX for **>0.8 MeV and >2 MeV** (and later 40 keV–475 keV local-time-resolved channels). **Inputs**: **Vsw, density, pressure, fraction of time IMF southward, and an IMF-based solar-wind–magnetosphere coupling function**. **Output**: 1-day-ahead flux.
- **Key physics finding**: **higher energies have longer lead/lag** ("higher-energy electrons take time to be accelerated… possible to forecast further into the future than lower energies") — directly supports energy-dependent horizon design. URLs: https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2016SW001506 ; open PDF https://deepblue.lib.umich.edu/bitstream/handle/2027.42/134930/swe20385.pdf

### 5.4 **PreMevE / PreMevE 2.0 / PreMevE-MEO** (LANL: Chen, Li, Pires de Lima, Feng)
- **Type**: two-submodel supervised ML (onset classifier + flux regressor). **Inputs**: **NOAA POES precipitating electrons (LEO), L1 Vsw, and one LANL/GEO satellite's MeV flux**. **Output**: MeV (~1 MeV) electron flux **across the outer belt by L-shell**, **+1 and +2 day**.
- **Skill**: **PE ≈ 0.87 (1-day), ≈ 0.82 (2-day)**; event-onset success **~70%** at 2 days. Notable result: **linear regression often ties or beats MLP/CNN/LSTM** → the trapped↔precipitating relationship is largely linear. PreMevE-MEO extends to ultra-relativistic electrons using **GPS** particle data. URLs: arXiv https://arxiv.org/abs/1911.01315 ; https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2019SW002399 ; MEO https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2024SW003975

### 5.5 **DREAM / DREAM3D** (LANL: Reeves, Tu, Chen) — physics-based + assimilation
- **Type**: physics-based **3D Fokker–Planck diffusion** (radial + pitch-angle + momentum + mixed) with **CRRES-statistical chorus/hiss wave databases**, plus **Kalman-filter data assimilation** of multi-satellite data (the assimilative DREAM). Not a pure solar-wind regressor but the reference **physical** model: it demonstrated enhancements need **radial diffusion + local chorus heating**. Useful as a physical sanity-check / hybrid component, not as a fast operational regressor. URLs: https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2011SW000729 ; DREAM3D https://ui.adsabs.harvard.edu/abs/2013JGRA..118.6197T/abstract

### 5.6 **ORIENT-R** (Chu et al. 2021) — neural-network outer-belt model
- **Type**: feed-forward NN. **Inputs**: **solar-wind speed, density, dynamic pressure, IMF, Dst, Kp** (most important features: **AL, Vsw, density, SYM-H**). **Output**: **>1.8 MeV** electron flux across the outer belt (incl. GEO), from solar wind + indices **only** (no initial/boundary conditions).
- **Skill**: out-of-sample **correlation ≈ 0.95**, **uncertainty factor ≈ 2** (RMSE ≈ 0.3 in log10). Reproduces transport, acceleration, decay and **dropouts** from storm to solar-cycle scales. URLs: https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2021SW002808 ; arXiv https://arxiv.org/abs/2109.10532

### 5.7 **Sakaguchi & Miyoshi** (2013, 2015) — Kalman-filter multivariate AR
- **Type**: multivariate **autoregressive model + Kalman filter** (adaptive). **Inputs**: **Vsw, IMF Bz, dynamic pressure** (combination of all three best). **Output**: daily >2 MeV flux at GEO (2013); later **16 L-resolved models** across the outer belt using Van Allen Probes 2.3 MeV + GOES-15 >2 MeV (2015). URLs: https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/swe.20020 ; https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2015SW001254

### 5.8 **LSTM / Transformer multi-horizon GEO model** (Wei/Zhang et al., *J. Space Weather Space Clim.* 2024) — *most directly comparable to PS-14*
- **Type**: LSTM vs Transformer-encoder, **5-min resolution**, target = **log10(≥2 MeV flux)**. **17 candidate inputs** (Vsw, N, Pdyn, |B|, Bx/By/Bz, E, T, plasma β, Kp, AE, SYM-H, Dst, magnetopause standoff R₀, L_m, MLT). Tested horizons **1 h, 3 h, 6 h, 12 h, 1 day** — *exactly PS-14's range*.
- **Skill (Transformer, 2005 test year):** PE = **0.940 (1 h), 0.886 (3 h), 0.828 (6 h), 0.747 (12 h), 0.660 (1 day)**; LSTM lower at long horizons (0.554 @ 12 h). Best feature subsets by horizon: **(Flux, N) @ 6 h; (Flux, N, Dst, L_m) @ 12 h; (Flux, Pdyn, AE) @ 1 day**; near-term dominated by **MLT/L_m (geometry)**. Optimal input history window ≈ **4 days**. URL: https://www.swsc-journal.org/articles/swsc/full_html/2024/01/swsc230072/swsc230072.html

### 5.9 Other recent ML benchmarks worth noting
- **Li et al. 2026** "How well can solar wind parameters predict outer-belt electron flux?": dual-module per-channel models (235–909 keV, nowcast→24 h). **Solar-wind-only PE ≈ 0.79–0.85**; **+ geomagnetic indices → 0.81–0.90**; degrades in extreme storms. URL: https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2026SW004986
- **Sun et al. 2025** "Enhancing ≥2 MeV electron *fluence* predictions in GEO through three deep-learning strategies." URL: https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2025SW004519
- **2026 foundation-model study** (TimesFM hybrid vs LSTM/Conv1D/Transformer) for ~1 MeV outer-belt flux: hybrid **R² ≈ 0.90 @ 6 h**, **0.82 @ 24 h**, **0.48 @ 72 h**; +15% over LSTM. URL: https://arxiv.org/html/2605.15752
- **Wang et al. 2024** Deep-SHAP feature attribution (the cleanest published feature-importance ranking). URL: https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/2024SW003880
- **Pires de Lima 2020 / Tang 2022 / Shin 2016 / Koons & Gorney 1991** — historical NN lineage.

**Benchmark summary table**

| Model | Type | Key inputs | Horizon | Reported skill |
|---|---|---|---|---|
| REFM (NOAA) | Linear filter | Vsw | 1–3 d fluence | Operational baseline |
| SNB³GEO | NARMAX | Vsw, N, Bz | 24 h flux | **PE ≈ 0.775** |
| Boynton | NARMAX | Vsw, N, P, IMF-south frac, coupling fn | 24 h | high; energy-dependent lead |
| PreMevE 2.0 | Supervised ML | POES precip + L1 Vsw + GEO flux | 1–2 d | **PE 0.87 / 0.82**; onset ~70% |
| ORIENT-R | Feed-fwd NN | AL, Vsw, N, SYM-H (+P, IMF, Dst, Kp) | nowcast/diagnostic | **r ≈ 0.95**, unc. ×2 |
| Sakaguchi | AR + Kalman | Vsw, Bz, Pdyn | 1 d | reduced error w/ all 3 |
| LSTM/Transformer (2024) | DL | Flux + N/Pdyn/AE/Dst/MLT/L_m… | **1 h–1 d** | **PE 0.94→0.66 (1 h→1 d)** |
| DREAM3D | Physics + assimilation | wave DBs, Kp, multi-sat | (assimilative) | physical reference |

---

## 6. "Harsh"/hazardous flux thresholds & event definition

- **NOAA SWPC operational alert**: issued when **>2 MeV integral electron flux ≥ 1000 pfu**, where **1 pfu = 1 particle cm⁻² s⁻¹ sr⁻¹**, **sustained** (typically ≥ 3 consecutive 5-min readings). This **1000 pfu** is the canonical "elevated/hazardous" threshold for deep-dielectric-charging risk. URL: https://www.swpc.noaa.gov/products/goes-electron-flux
- **Daily fluence**: cumulative-damage metric; **~10⁹ electrons cm⁻² day⁻¹ (sr⁻¹)** order-of-magnitude is used by SWPC/REFM as a significant daily-fluence level; deep-dielectric risk scales with multi-day accumulated fluence because dielectrics hold charge for days–months.
- **Climatological context for "harsh"** (Pierrard-type / climatology PMC9541471): high daily-averaged 2-MeV flux defined as **> 3.5×10⁵ e cm⁻² sr⁻¹ MeV⁻¹ s⁻¹** (differential), occurring **~15% of days**; the most dangerous **10-day** integrated periods (**> 8×10¹¹ e cm⁻² sr⁻¹ MeV⁻¹**) occur **~4% of the time**, last **~10 ± 4 days**, and follow strong time-integrated activity (∫aa_H > ~1400–1900 nT·h) by 0–2 days.
- **Relativistic-electron-enhancement (REE) event threshold** used in ML papers: e.g., **≥ ~1157 e cm⁻² s⁻¹ sr⁻¹** (Wei/Zhang 2024) — close to NOAA's 1000 pfu.
- **Operational "alert" definition** to adopt for PS-14 classification target: **>2 MeV flux crossing 1000 pfu** (and/or predicted **daily fluence > 10⁹**), with optional graded levels (e.g., 10³, 10⁴ pfu) mirroring SWPC's electron alert tiers.

---

## 7. Recommended target representation

1. **Use log10(flux).** Flux spans **>3 orders of magnitude**; a log target (a) makes the error distribution roughly homoscedastic/Gaussian, (b) stabilises NN training, (c) matches how thresholds and damage scale, and (d) is what essentially every successful model uses (Chu 2021, Wei 2024, Sakaguchi, SNB³GEO). Report errors as **RMSE in log10** and the intuitive **"uncertainty factor" = 10^RMSE**.
2. **Multi-horizon targets.** Predict log10 flux at **t+{30–45 min, 6 h, 12 h}** (and optionally +1 day to align with REFM/SNB³GEO baselines). Because **higher-horizon skill increasingly comes from the Vsw acceleration lag** while **nowcast skill comes from persistence + immediate loss**, consider **separate models/heads per horizon** (as Li 2026 and Wei 2024 do) rather than one shared model.
3. **Predict the operational quantities the user actually needs**: at minimum the **5-min instantaneous >2 MeV flux**; also provide **forward daily fluence** (sum of predicted flux) since that is the charging-relevant metric and the REFM output.
4. **Add a parallel threshold-exceedance classification** (will flux exceed **1000 pfu** within the horizon? / will daily fluence exceed 10⁹?). Rationale: operators act on **alerts**, the classes are imbalanced (events are rare → ~4–15% of time), and a calibrated probability is more decision-useful than a point regression near a hard threshold. Train as multitask (shared encoder, regression + classification heads) or post-process the regression with a calibrated threshold; evaluate with **POD/FAR/HSS/TSS and ROC-AUC**, not just RMSE.
5. **Always report skill vs two baselines**: **persistence** (flux now) and **climatology** (PE definition). The 27-day-recurrence/persistence baseline is strong for this slowly varying quantity, so beating it is the real bar.

---

## 8. Ranked, physically-justified input feature list (with recommended lag windows)

Ranking synthesises Wing 2016, Wang 2024 (SHAP), Chu 2021 (ORIENT-R), Boynton 2016, Sakaguchi 2013, Ganushkina 2024, and Wei/Zhang 2024.

| Rank | Feature | Physical justification | Recommended lag / window | Source @ PS-14 |
|---|---|---|---|---|
| **1** | **Autoregressive >2 MeV flux** (current + recent) | Belt has multi-day memory; persistence is the strongest short-horizon predictor | **lags 0 → ~4 days** (rolling history); emphasise last 1–2 d | GOES SEISS/MPS-HI (or EPEAD) |
| **2** | **Solar wind speed Vsw** | #1 external driver (ULF radial diffusion + chorus via convection/substorms) | **lags +1 to +2 days**, plus running mean over prior 2–3 days; also instantaneous (propagated) for nowcast loss/compression | Wind/SWE (L1), or OMNI |
| **3** | **AE / AL** (or proxy) | Substorm injection of source/seed pops; chorus generation | **lags 0 → +2 days** | OMNI / ground (or Kp proxy) |
| **4** | **Seed/source population: GOES >0.8 MeV and sub-MeV (40–475 keV) flux** | Direct precursor accelerated to >2 MeV; faster response than Vsw | **lags hours → ~1 day** | GOES SEISS/MPS-HI lower channels |
| **5** | **IMF Bz** (and/or coupling function, e.g. VBz_south, ε, Newell dΦ/dt) | Southward → coupling/injection/ULF (build-up); extreme southward+shock → loss | **acceleration lag ~hours–1 d**; **loss lag minutes–hours** (use propagated value) | Wind/MFI (L1), OMNI |
| **6** | **Solar wind density N** | Low-N favours enhancement; high-N pulses → shadowing loss; sets plasmaspheric density (EMIC) | **instantaneous–hours** (propagated) | Wind/SWE (L1) |
| **7** | **Dynamic pressure Pdyn (∝ N·Vsw²)** | Magnetopause compression → prompt shadowing loss; can also inject | **minutes–hours** (propagated, esp. nowcast) | derived from Wind |
| **8** | **Kp** | Coarse global activity; complements AE | **0 → +1–2 days** | OMNI/GFZ |
| **9** | **Dst / SYM-H** | Storm phase (main-phase depletion vs recovery enhancement); ring-current proxy | history over storm (days) | OMNI |
| **10** | **MLT of the GOES sensor + magnetopause standoff R₀** | Captures ~1-order-of-magnitude diurnal/noon–midnight variation & dayside compression at GEO | instantaneous (cyclic sin/cos features) | derived (ephemeris + Shue model from Wind) |
| **11** | **ULF/Pc5 index** (if available) | The actual radial-diffusion engine; can substitute for Vsw lag | hours–2 days | ground magnetometer index (optional) |
| **12** | **Seasonal phase (day-of-year, equinox/RM-effect term)** | Semiannual ×~2 (Russell–McPherron) | slow (cyclic) | calendar |

**Practical feature-engineering notes**
- **Propagate L1 (Wind) data to the magnetopause/GEO** using a flat or vector convection delay (mean ~47 min; speed-dependent ~20–90 min) before use — especially important for the **nowcast** where Pdyn/Bz drive immediate loss.
- Provide **both instantaneous and time-lagged/running-mean** versions of Vsw, Bz, N, Pdyn; the **~2-day running integral of Vsw (or ∫AE, ∫aa)** is a powerful single feature (climatology paper).
- **Window length ~4 days** of input history is supported empirically (Wei/Zhang 2024).
- Beware **data gaps and instrument cross-calibration** (GOES-13/15 EPEAD vs GOES-16+ SEISS/MPS-HI; ACE vs DSCOVR vs Wind at L1) — harmonise channels and flag gaps.

---

## 9. Key takeaways for the modelling pipeline (PS-14)

1. **Multi-horizon, multi-head model** on **log10 >2 MeV flux** at 5-min cadence; separate treatment for 30–45 min / 6 h / 12 h.
2. **Core inputs**: autoregressive flux + GOES seed channels + **Vsw (lagged 1–2 d & running mean)** + AE/Kp + propagated **Bz, N, Pdyn** + **MLT/seasonal** cyclic features.
3. **Propagate Wind/L1 → GEO** (~30–60 min) to physically justify and power the nowcast.
4. **Add a calibrated exceedance classifier at 1000 pfu** (and daily-fluence > 10⁹) for operational alerting; score with HSS/TSS/POD/FAR/ROC-AUC.
5. **Benchmark against persistence + climatology**, and against **REFM / SNB³GEO (PE≈0.78) / PreMevE (PE≈0.82–0.87) / the 2024 LSTM-Transformer (PE 0.83 @6h, 0.75 @12h)**.
6. Expect **degraded skill in extreme storms / dropouts** (data scarcity + magnetopause shadowing); consider hybrid physics-ML or extra loss-feature engineering there.

---

## References (URLs)

**Physics — acceleration / loss / seed population**
- Rapid local acceleration by chorus (Thorne et al. 2013): https://www.researchgate.net/publication/259603430_Rapid_Local_Acceleration_of_Relativistic_Radiation-Belt_Electrons_by_Magnetospheric_Chorus
- Testing key acceleration processes (Frontiers 2023): https://www.frontiersin.org/journals/astronomy-and-space-sciences/articles/10.3389/fspas.2023.1168636/full
- Radial vs local diffusion (Nature Sci. Rep. 2025): https://www.nature.com/articles/s41598-025-23908-w
- ULF-wave-driven radial diffusion (PMC4703845): https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4703845/
- Substorm injections sufficient for MeV (Mithaiwala 2005): https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2004JA010511
- Source/seed electrons & wave–particle interactions: https://www.sciencedirect.com/science/article/abs/pii/S1364682620302133
- Magnetopause shadowing characterization: https://www.researchgate.net/publication/307999943
- Rapid losses across the magnetopause (Frontiers 2025): https://www.frontiersin.org/journals/astronomy-and-space-sciences/articles/10.3389/fspas.2025.1694836/full
- EMIC rising-tone precipitation (Nakamura 2019): https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2019JA026772
- EMIC + shadowing via data assimilation (Cervantes 2020): https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2020JA028208

**Drivers / lags / feature importance**
- Reeves et al. 2011 (Paulikas & Blake revisited): https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2010JA015735
- Wing et al. 2016 (information theory; Vsw #1, +2 d): https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2016JA022711 ; OSTI: https://www.osti.gov/pages/biblio/1402657
- Wang et al. 2024 (Deep-SHAP; Vsw +1 d): https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/2024SW003880
- Ganushkina/Bhaskar 2024 (30-yr ranking; AE & Vsw dominant): https://ui.adsabs.harvard.edu/abs/2024AdSpR..73.5145G/abstract
- ULF–Vsw–flux correlation (solar-cycle dependence): https://www.sciencedirect.com/science/article/abs/pii/S1364682603002098
- CIR vs CME radiation-belt response (Pandya 2019): https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2019JA026771
- HSS & radiation-belt electrons (NTRS): https://ntrs.nasa.gov/citations/20110023417
- Drivers for GEO 2–200 keV electrons (Kamp 2024): https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2024SW003984

**L1 → Earth propagation**
- Timing of L1→Earth delay via ML (SWSC 2021; mean ~47 min, 20–90 min): https://www.swsc-journal.org/articles/swsc/full_html/2021/01/swsc200105/swsc200105.html
- PRIME probabilistic L1 propagation (Frontiers 2023): https://www.frontiersin.org/journals/astronomy-and-space-sciences/articles/10.3389/fspas.2023.1250779/full

**MLT / diurnal / seasonal**
- Semiannual variation of relativistic electrons (ANGEO 2021): https://angeo.copernicus.org/articles/39/413/2021/
- Russell–McPherron role: https://www.sciencedirect.com/science/article/abs/pii/S1364682608003593
- Dawn–dusk asymmetry review (arXiv): https://arxiv.org/pdf/1701.04701
- Geosynchronous magnetopause/LCDS vs MLT: https://www.researchgate.net/publication/362627258

**Models / benchmarks**
- NOAA REFM: https://www.swpc.noaa.gov/products/relativistic-electron-forecast-model
- REFM vs SNB³GEO (Balikhin 2016): https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1002/2015SW001303 ; PMC: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4995643/
- SNB³GEO performance eval (2022): https://www.sciencedirect.com/science/article/abs/pii/S0273117722010766
- Boynton et al. 2016 (energy-dependent models): https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2016SW001506 ; PDF: https://deepblue.lib.umich.edu/bitstream/handle/2027.42/134930/swe20385.pdf
- Boynton local-time 40 keV models 2019: https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2018SW002128
- PreMevE 2.0 (Pires de Lima 2020): https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2019SW002399 ; arXiv: https://arxiv.org/abs/1911.01315
- PreMevE-MEO (Feng 2024): https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2024SW003975
- DREAM (Reeves 2012): https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2011SW000729
- DREAM3D (Tu 2013): https://ui.adsabs.harvard.edu/abs/2013JGRA..118.6197T/abstract
- ORIENT-R (Chu 2021): https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2021SW002808 ; arXiv: https://arxiv.org/abs/2109.10532
- Sakaguchi 2013 (Kalman/AR @ GEO): https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/swe.20020
- Sakaguchi 2015 (L-resolved): https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2015SW001254
- LSTM vs Transformer multi-horizon @ GEO (Wei/Zhang 2024): https://www.swsc-journal.org/articles/swsc/full_html/2024/01/swsc230072/swsc230072.html
- Li et al. 2026 (solar-wind-only predictability): https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2026SW004986
- Sun et al. 2025 (fluence deep learning): https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2025SW004519
- Foundation-model (TimesFM) outer-belt 2026: https://arxiv.org/html/2605.15752
- Stacking-ensemble short-time prediction (Tang 2022): https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2021SW002969

**Thresholds / hazard / data**
- NOAA GOES electron flux product (1000 pfu alert): https://www.swpc.noaa.gov/products/goes-electron-flux
- Climatology of long-duration high 2-MeV periods (PMC9541471): https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9541471/
- Deep-dielectric charging overview (CiteseerX): https://citeseerx.ist.psu.edu/document?doi=67fb5d4233bd9c7249985b55c04b5fce0513fe0d
- Galaxy 15 / Telstar 401 charging conditions (Saiz 2018): https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2018SW001974
- GOES-R SEISS / MPS-HI data (>2 MeV channel): https://www.ncei.noaa.gov/products/goes-r-space-environment-in-situ ; ReadMe: https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/goes16/l2/docs/GOES-R_SEISS_L2_MPS-HI.ReadMe.pdf

---
*Compiled from ~16 web searches and multiple full-text/abstract fetches. Some Wiley/Elsevier full texts were paywalled; in those cases figures are taken from open abstracts, author/OSTI/arXiv copies, or peer-reviewed secondary summaries and are flagged where approximate.*
