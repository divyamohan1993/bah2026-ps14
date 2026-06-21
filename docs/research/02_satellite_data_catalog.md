# BAH-2026 PS-14 — Satellite / Mission Data Source Catalog

**Problem Statement:** Forecasting >2 MeV electron flux at geostationary orbit (GEO) using AI/ML.
**Purpose of this document:** A worldwide catalog of satellite / mission data sources (NOT ISRO-only) so that multiple datasets can cross-validate each other and fill data gaps across time, L-shell, magnetic local time (MLT), and instrument-outage periods.
**Compiled:** 2026-06-20 · Heliophysics data-engineering reference.

> **Headline count: 45 distinct satellites / missions cataloged** (target was ≥35). Counted explicitly in the [Satellite Count](#satellite-count) section.

---

## 0. How to read this catalog

- **CDAWeb dataset ID** = the identifier you pass to `cdasws`, `pyspedas`, SunPy `Fido` (`cdaweb.Dataset`), or the CDAWeb HAPI server. Files are CDF.
- **HAPI** = Heliophysics Application Programmer's Interface; a REST standard returning CSV/binary/JSON time series. Server base URLs are listed in the [Data Access Strategy](#data-access-strategy).
- **NCEI/NGDC** = NOAA National Centers for Environmental Information (formerly NGDC). GOES-R, POES/MetOp, LANL-GEO and GPS data live here, usually NetCDF over HTTPS.
- Energy channels are quoted as the mission documents them; for the **target** the relevant quantity is the **integral >2 MeV electron flux** (electrons cm⁻² s⁻¹ sr⁻¹).

---

## A) L1 / Heliospheric Solar-Wind Monitors — drivers / model features

These provide the upstream solar-wind and IMF conditions that *drive* radiation-belt electron acceleration (Bz, speed Vsw, density, dynamic pressure, IMF clock angle). They are the principal **input features** for any >2 MeV electron forecast.

| # | Mission / Operator | What it measures (relevant) | Channels / parameters | Cadence | Coverage | Format | Access endpoint (exact) |
|---|---|---|---|---|---|---|---|
| 1 | **Wind** (NASA, L1) | Solar-wind plasma + IMF | SWE: Np, Vsw, Tp, V-vector (GSE/GSM); MFI: B, Bx/By/Bz | 92 s (K0); 3 s / 1 min (H0) | 1994–present | CDF | CDAWeb `WI_K0_SWE`, `WI_H0_SWE`, `WI_H0_MFI`, `WI_K0_MFI`; HAPI(CDAWeb) |
| 2 | **ACE** (NASA, L1) | Solar-wind plasma + IMF + energetic ions/electrons | SWEPAM: Np, Vsw, Tp; MAG: B,Bx/By/Bz; EPAM: 47 keV–5 MeV electrons & ions | 64 s / 5 min (K0); 16 s (H0 MAG) | 1997–present | CDF | CDAWeb `AC_K0_SWE`, `AC_H0_MFI`, `AC_K0_MFI`, `AC_K0_EPM`, `AC_H1_EPM`; HAPI(CDAWeb) |
| 3 | **DSCOVR** (NOAA, L1) | Operational solar-wind plasma + IMF (replaced ACE for ops) | Faraday-cup: Np, Vsw, Tp; fluxgate MAG: B,Bx/By/Bz | 1 s (MAG); ~1 min (FC) | 2016–present | CDF / JSON(RT) | CDAWeb `DSCOVR_H0_MAG`, `DSCOVR_H1_FC`, `DSCOVR_AT_DEF`; **real-time** `https://services.swpc.noaa.gov/products/solar-wind/` |
| 4 | **SOHO** (ESA/NASA, L1) | Solar-wind proton monitor (CELIAS) + energetic particles (COSTEP/EPHIN) | CELIAS/PM: Vsw, Np; EPHIN: e⁻ 0.25–>8.7 MeV, p 4–>53 MeV/n | 30 s / 5 min (CELIAS); 1 min (EPHIN) | 1996–present | CDF | CDAWeb `SOHO_CELIAS-PM_30S`, `SOHO_CELIAS-PM_5MIN`; EPHIN via Univ. Kiel `http://www2.physik.uni-kiel.de/SOHO/phpeph/EPHIN.htm` |
| 5 | **IMP-8** (NASA, historic) | Legacy solar-wind plasma + IMF (back-extends OMNI) | MIT plasma: Np, Vsw; MAG: B | 15 s / hourly | 1973–2006 | CDF | CDAWeb `I8_K0_MAG`, `I8_K0_PLA`, `IMP8_PLASMA_FINE_RES` |
| 6 | **Geotail** (JAXA/NASA) | Near-tail + occasional upstream solar wind (used in OMNI HRO) | LEP/CPI ions, MGF B | 12 s / 1 min | 1992–present | CDF | CDAWeb `GE_K0_MGF`, `GE_K0_CPI` |

**Cross-validation note (A):** ACE and DSCOVR are mutually-redundant L1 monitors — **ACE backs up Wind/DSCOVR during data gaps and vice-versa**; Wind is the science-grade reference. IMP-8 + Geotail let OMNI extend the merged solar-wind record back to 1973. Use the merged **OMNI** product (Section F) rather than juggling these four manually for the feature stream.

---

## B) Solar Observation — flare X-ray, EUV, CME (precursor / trigger features)

Solar drivers (CMEs, high-speed streams, flare X-ray flux) precede belt enhancements by hours-to-days; these are useful precursor features and for event labeling.

| # | Mission / Operator | What it measures (relevant) | Channels / parameters | Cadence | Coverage | Format | Access endpoint (exact) |
|---|---|---|---|---|---|---|---|
| 7 | **GOES XRS** (NOAA) | Solar soft X-ray flux (flare class) | XRS-A 0.05–0.4 nm; XRS-B 0.1–0.8 nm | 1 s / 1 min | 1986–present (GOES-6→19) | NetCDF / JSON(RT) | NCEI `https://www.ncei.noaa.gov/data/goes-r/...`; SunPy `XRSClient`; **RT** `https://services.swpc.noaa.gov/json/goes/primary/xrays-7-day.json` |
| 8 | **GOES SUVI** (NOAA) | Solar EUV imaging (6 bands) | 94/131/171/195/284/304 Å | ~4 min/band | 2017–present | FITS | NCEI `https://www.ncei.noaa.gov/data/goes-r/...`; SunPy `SUVIClient` |
| 9 | **SDO** (NASA) | EUV/UV imaging (AIA) + irradiance (EVE) + magnetograms (HMI) | AIA 7 EUV bands; EVE 0.1–105 nm; HMI Bphoto | 12 s (AIA); 10 s (EVE) | 2010–present | FITS / NetCDF | JSOC `http://jsoc.stanford.edu`; SunPy `Fido` (VSO); EVE via LASP/LISIRD |
| 10 | **STEREO-A** (NASA) | CME imaging + in-situ solar wind (2nd vantage) | SECCHI COR/HI; PLASTIC ions; IMPACT B & e⁻/ions | varies / 1 min (in-situ) | 2006–present | CDF/FITS | CDAWeb `STA_L1_MAG_RTN`, `STA_L2_PLA_1DMAX_1MIN`, `STA_L1_IMPACT`; SunPy |
| 11 | **STEREO-B** (NASA) | 2nd CME vantage (contact lost 2014) | as STEREO-A | 1 min | 2006–2014 | CDF/FITS | CDAWeb `STB_L1_MAG_RTN`, `STB_L2_PLA_1DMAX_1MIN` |
| 12 | **SOHO/LASCO** (ESA/NASA) | White-light coronagraph — CME catalog | C2/C3 coronagraphs | ~12–20 min | 1996–present | FITS | CDAW CME catalog `https://cdaw.gsfc.nasa.gov/CME_list/`; SunPy `Fido` (VSO) |
| 13 | **Parker Solar Probe** (NASA) | Inner-heliosphere solar wind / fields | FIELDS B; SWEAP/SPC,SPAN ions & e⁻ | 1 min (merged) | 2018–present | CDF | CDAWeb `PSP_FLD_L2_MAG_RTN_1MIN`, `PSP_SWP_SPC_L3I`; pyspedas `psp` |
| 14 | **Solar Orbiter** (ESA/NASA) | Solar wind + remote sensing (multi-vantage) | MAG B; SWA ions; EPD e⁻/ions; EUI imaging | 1 min (in-situ) | 2020–present | CDF/FITS | CDAWeb `SOLO_L2_MAG-RTN-NORMAL-1-MINUTE`, `SOLO_L2_SWA-PAS-GRND-MOM`; ESA SOAR `https://soar.esac.esa.int` |
| 15 | **Hinode** (JAXA/NASA/UK) | High-res photosphere/corona (active-region context) | SOT/XRT/EIS | mission-dependent | 2006–present | FITS | DARTS `https://darts.isas.jaxa.jp/solar/hinode/`; VSO |
| 16 | **PROBA-2** (ESA) | EUV imaging (SWAP) + Lyman-α/X-ray (LYRA) | SWAP 174 Å; LYRA 4 channels | ~1–2 min | 2010–present | FITS | ROB `https://proba2.sidc.be/data`; SunPy |
| 17 | **Aditya-L1** (ISRO, L1) | Solar X-ray + EUV + in-situ solar wind/particles | SoLEXS & HEL1OS X-ray; VELC/SUIT imaging; **ASPEX**: SWIS ions + STEPS supra-thermal/energetic (100 eV–6 MeV/n); **PAPA** plasma; MAG | varies | 2024–present | FITS/CDF-like | ISSDC PRADAN `https://pradan.issdc.gov.in/al1/` (registration) |

**Cross-validation note (B):** STEREO-A/B provide off-Sun-Earth-line views so CMEs/CIRs can be tracked before they hit L1 — **filling the "we can't see behind the limb" gap of SOHO/SDO**. GOES XRS gives the canonical flare timeline that labels storm onsets. Aditya-L1 ASPEX/STEPS adds an *independent* L1 energetic-particle and solar-wind monitor that **cross-checks ACE/DSCOVR** and protects against single-point L1 outages.

---

## C) GEO Energetic-Electron Flux — the TARGET + GEO cross-validation

This is the **prediction target** (integral >2 MeV electron flux at geostationary orbit) and the set of independent GEO platforms that let you cross-validate across longitude / MLT and across instrument generations.

| # | Mission / Operator | What it measures (relevant) | >2 MeV / channels | Cadence | Coverage | Format | Access endpoint (exact) |
|---|---|---|---|---|---|---|---|
| 18 | **GOES-16/17/18/19** (NOAA, GOES-R) **SEISS MPS-HI** | **TARGET**: GEO electron flux | **Integral >2 MeV** electron flux + 10 diff bands 50 keV–4 MeV | 1 min / 5 min | G16 2017–, G18 2022–, G19 2024– | NetCDF / CDF / JSON(RT) | **NCEI** `https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/goes{16,17,18,19}/l2/data/mpsh-l2-avg1m_science/` ; **CDAWeb** `DN_SEIS-L2-MPSH_G16/G17/G18/G19`; **MAG** `DN_MAGN-L2-HIRES_G16`; **RT** `https://services.swpc.noaa.gov/json/goes/primary/integral-electrons-1-day.json` |
| 19 | **GOES-13/14/15** (NOAA) **EPEAD / MAGED** | GEO electron flux (legacy, primary forecasting target pre-2018) | **EPEAD integral >0.8 & >2 MeV** (E2 = >2 MeV); MAGED 40–475 keV | 1 min / 5 min | 2009–2020 | CDF / NetCDF | CDAWeb `GOES13_EPEAD-SCIENCE-ELECTRONS-E13EW_1MIN`, `GOES15_EPS-MAGED_1MIN`; NCEI `https://www.ngdc.noaa.gov/stp/satellite/goes/` |
| 20 | **GOES-8/9/10/11/12** (NOAA) **EPS/EP8** | GEO electron flux (deep historic record) | **>0.6, >2.0, >4.0 MeV** integral electrons | 5 min | 1995–2008 | CDF | CDAWeb `G8_K0_EP8`, `G0_K0_EP8` (G10), `GOES11_K0_EP8`, `G9_K0_EP8` |
| 21 | **GOES-6/7** (NOAA, historic) | Earliest >2 MeV GEO electrons | **>2 MeV** integral electrons | 5 min | 1986–1995 | CDF | CDAWeb `G6_K0_EPS`, `G7_K0_EPS` |
| 22 | **LANL-GEO** (US DOE/LANL) SOPA / ESP / MPA | Independent GEO electron flux at other longitudes | SOPA 50 keV–1.5 MeV; **ESP 0.7–25.8 MeV**; MPA eV–40 keV | 10 s–60 s | 1976–present (multiple S/C) | NetCDF/ASCII/CDF | **NCEI** `https://www.ncei.noaa.gov/products/space-weather/partners/lanl-products-data`; CDAWeb e.g. `LANL_1989-046_SOPA`, `LANL_2001_MPA` |
| 23 | **FengYun-4A/4B** (CMA, China, GEO) | Independent GEO high-energy electrons | High-Energy e⁻ detector 0.4–4 MeV (9 ch); protons 1–165 MeV | min-scale | 2016–present | proprietary | CMA NSMC `https://satellite.nsmc.org.cn/`; cross-cal'd vs Arase (EPP 2023) |
| 24 | **FengYun-2** (CMA, GEO, historic) | Legacy GEO space-environment monitor | electron/proton counters | min-scale | 2004–2018 | proprietary | CMA NSMC `https://satellite.nsmc.org.cn/` |
| 25 | **Electro-L / GOMS-2,3,4** (Roscosmos, GEO) | Russian GEO electron/proton monitor (GGAK-E) | electron & proton channels, solar X-ray | min-scale | 2011–present | proprietary | Roscosmos/IKI; SciDB `http://swx.sinp.msu.ru/`; broadcast GGAK-E |
| 26 | **Meteosat (MSG/MFG)** (EUMETSAT, GEO) | GEO radiation/particle env. (SEM where flown) | proton/electron dosimetry | min-scale | 1990s–present | proprietary | EUMETSAT Data Store `https://data.eumetsat.int/` |

**Cross-validation note (C):** Multiple simultaneous GEO platforms at **different longitudes** (GOES-East ~75°W, GOES-West ~137°W, FY-4 ~105°E, Electro-L ~76°E, LANL spread around the clock) give **continuous MLT coverage of the >2 MeV electron belt** — critical because GEO flux has a strong local-time (dawn/dusk) asymmetry. Cross-calibrating GOES vs LANL vs FY-4 (already published vs Arase) lets you stitch a homogeneous multi-decade target series and **detect/replace bad sensor periods** (e.g. proton contamination, sun-angle artifacts in GOES E2).

---

## D) Radiation Belt / Magnetosphere In-Situ Electrons — fill gaps across L-shell & MLT

These sample the belt *interior* (different L-shells, pitch angles, MLT) so the model learns the radial/temporal structure that produces the GEO flux, and they fill GEO blind spots.

| # | Mission / Operator | What it measures (relevant) | Channels | Cadence | Coverage | Format | Access endpoint (exact) |
|---|---|---|---|---|---|---|---|
| 27 | **Van Allen Probes A** (RBSP-A, NASA) | Gold-standard belt electrons across all L | MagEIS 30 keV–4 MeV; **REPT 1.8–20 MeV** | spin (~11 s) | 2012–2019 | CDF | CDAWeb `RBSPA_REL03_ECT-MAGEIS-L3`, `RBSPA_REL03_ECT-REPT-SCI-L3`; ECT SOC `https://rbsp-ect.newmexicoconsortium.org/` |
| 28 | **Van Allen Probes B** (RBSP-B, NASA) | Twin belt-electron survey (different MLT) | as RBSP-A | spin | 2012–2019 | CDF | CDAWeb `RBSPB_REL03_ECT-MAGEIS-L3`, `RBSPB_REL03_ECT-REPT-SCI-L3` |
| 29 | **THEMIS-A** (NASA) | Outer-belt/plasma-sheet electrons | ESA 5 eV–30 keV; **SST 30–300 keV** | spin (~3 s) | 2007–present | CDF | CDAWeb `THA_L2_SST`, `THA_L2_ESA`, `THA_L2_MOM`; pyspedas `themis` |
| 30 | **THEMIS-D** (NASA) | Inner-magnetosphere electrons | ESA + SST | spin | 2007–present | CDF | CDAWeb `THD_L2_SST`, `THD_L2_ESA` |
| 31 | **THEMIS-E** (NASA) | Inner-magnetosphere electrons | ESA + SST | spin | 2007–present | CDF | CDAWeb `THE_L2_SST`, `THE_L2_ESA` |
| 32 | **ARTEMIS P1** (THEMIS-B, lunar) | Distant-tail / upstream electrons & solar wind | ESA + SST + FGM | spin | 2010–present | CDF | CDAWeb `THB_L2_SST`, `THB_L2_FGM`; pyspedas `themis` |
| 33 | **ARTEMIS P2** (THEMIS-C, lunar) | Distant-tail / upstream monitor | ESA + SST + FGM | spin | 2010–present | CDF | CDAWeb `THC_L2_SST`, `THC_L2_FGM` |
| 34 | **MMS-1** (NASA) | Energetic electrons, inner magnetosphere/tail | **FEEPS 25–650 keV**; EIS; FPI <30 keV | 2.42 s (srvy) | 2015–present | CDF | CDAWeb `MMS1_FEEPS_SRVY_L2_ELECTRON`; pyspedas `mms` |
| 35 | **MMS-2** (NASA) | as MMS-1 | FEEPS/EIS/FPI | 2.42 s | 2015–present | CDF | CDAWeb `MMS2_FEEPS_SRVY_L2_ELECTRON` |
| 36 | **MMS-3** (NASA) | as MMS-1 | FEEPS/EIS/FPI | 2.42 s | 2015–present | CDF | CDAWeb `MMS3_FEEPS_SRVY_L2_ELECTRON` |
| 37 | **MMS-4** (NASA) | as MMS-1 | FEEPS/EIS/FPI | 2.42 s | 2015–present | CDF | CDAWeb `MMS4_FEEPS_SRVY_L2_ELECTRON` |
| 38 | **Cluster (C1–C4)** (ESA/NASA) | 4-pt energetic electrons, outer belt/tail | **RAPID/IES 37–400 keV**; PEACE <30 keV | spin (~4 s) | 2000–present | CDF/CEF | CDAWeb `C1_CP_RAP_ESPCT6`, …; **Cluster Science Archive** `https://csa.esac.esa.int/` (counts as 4 S/C: C1,C2,C3,C4) |
| 39 | **Arase / ERG** (JAXA) | Relativistic belt electrons (Asian-sector MLT) | MEP-e 7–87 keV; HEP 70 keV–2 MeV; **XEP 0.4–20 MeV** | spin | 2016–present | CDF | **ERG Science Center** `https://ergsc.isee.nagoya-u.ac.jp/`; CDAWeb `ERG_XEP_L2_OMNIFLUX`, `ERG_HEP_L2_OMNIFLUX`; pyspedas `erg` |
| 40 | **SAMPEX** (NASA, historic) | LEO/long-baseline belt electrons | **HILT, PET, LICA** ~0.4–30 MeV e⁻ | ~6–100 s | 1992–2012 | CDF | CDAWeb `SAMPEX_HILT`; SOC `http://www.srl.caltech.edu/sampex/` |
| 41 | **Polar** (NASA, historic) | High-altitude polar belt electrons | CEPPAD/IPS, HIST 20 keV–10 MeV | spin | 1996–2008 | CDF | CDAWeb `PO_LEVEL1_CEPPAD`, `PO_K0_HYD` |
| 42 | **CRRES** (NASA/USAF, historic) | Classic belt-model dataset (storm injection) | MEA 0.1–1.6 MeV; HEEF to 10 MeV | ~spin | 1990–1991 | CDF/ASCII | CDAWeb `CRRES_...`; NSSDC; AE9/AP9 heritage |
| 43 | **ELFIN A & B** (NSF/UCLA CubeSats) | Loss-cone precipitating relativistic electrons | EPDE 50 keV–~6 MeV | 2.85 s | 2018–2022 | CDF | UCLA `https://elfin.igpp.ucla.edu/`; pyspedas `elfin` |
| 44 | **GPS constellation (CXD/BDD)** (US, ~MEO/GPS) | 19–21 satellites of belt electrons at L≈4.2 | CXD e⁻ ~0.1–~10 MeV (proxy energies) | ~4 min | 2000–present (>167 sat-yr) | ASCII/NetCDF | **NCEI** `https://www.ngdc.noaa.gov/stp/space-weather/satellite-data/satellite-systems/gps/data/` (counted as ONE program) |
| 45 | **INTEGRAL / IREM** (ESA) | Highly-elliptical-orbit radiation monitor (e⁻/p dose) | SREM-type e⁻ >0.5 MeV, p >20 MeV, dose | ~seconds | 2002–present | ASCII/FITS | ESA `https://www.cosmos.esa.int/web/integral/instruments-irem`; ISDC |

**Cross-validation note (D):** Van Allen Probes are the **ground-truth calibrator** — REPT 1.8–20 MeV and MagEIS overlap the GOES >2 MeV channel and were used to validate GPS-CXD and FY-4. Arase covers the **Asian-sector MLT** the US GOES miss; the **GPS constellation** delivers near-continuous L≈4.2 multi-point sampling (≈19 satellites!) that interpolates between the belt heart and GEO; THEMIS+MMS+Cluster add multi-point structure and historic SAMPEX/CRRES/Polar extend the relativistic-electron record back to 1990. Together they **fill GEO's single-L-shell limitation** and let the model learn radial diffusion.

---

## E) Polar / LEO Electron Monitors — precipitation & low-altitude flux

LEO sun-synchronous sentinels measure precipitating/quasi-trapped electrons and the magnetic field, giving the *loss* term and long, dense time series complementary to GEO.

| # | Mission / Operator | What it measures (relevant) | Channels | Cadence | Coverage | Format | Access endpoint (exact) |
|---|---|---|---|---|---|---|---|
| 46 | **POES (NOAA-15…19)** (NOAA) **SEM-2/MEPED** | Precipitating + 0°/90° electrons, ~850 km | **e⁻ >40, >130, >287 keV** + P6 ~>700 keV | 2 s | 1998–present | NetCDF | **NCEI** `https://www.ncei.noaa.gov/products/poes-metop-space-environment-monitor`; CDAWeb `NOAA15_POES-SEM2_FLUXES-2SEC` |
| 47 | **MetOp-A/B/C** (EUMETSAT) **SEM-2/MEPED** | Same SEM-2 as POES (more local times) | as POES MEPED | 2 s | 2006–present | NetCDF | NCEI (same product); CDAWeb `METOP2_POES-SEM2_FLUXES-2SEC` |
| 48 | **DMSP (F16/F17/F18…)** (US DoD) **SSJ** | Precipitating auroral electrons/ions, ~850 km | SSJ 30 eV–30 keV e⁻/ions; SSM B; SSIES plasma | 1 s | 1982–present | CDF/NetCDF | CDAWeb `DMSP-F16_SSJ_PRECIPITATING-ELECTRONS-IONS`; NCEI/CEDAR Madrigal `http://cedar.openmadrigal.org/` |
| 49 | **Swarm A/B/C** (ESA) | Precise LEO B-field + plasma (geomag context) | VFM/ASM B; EFI Ne,Te,TEC | 1 Hz / 50 Hz | 2013–present | CDF | **VirES** `https://vires.services/` (`viresclient`); ESA `https://earth.esa.int/eogateway/missions/swarm/data` (counted as 3 S/C) |
| 50 | **CHAMP** (GFZ, historic) | LEO B-field + thermosphere density | FGM/Overhauser B; accelerometer ρ | 1 Hz | 2000–2010 | CDF/native | GFZ ISDC `https://isdc.gfz-potsdam.de/champ-isdc/`; VirES `CH_ME_MAG_LR_3` |
| 51 | **DEMETER** (CNES, historic) | LEO trapped/precipitating electrons & waves | **IDP 70 keV–2.5 MeV** (256 bands); ICE fields | survey/burst | 2004–2010 | native/CDF | CNES `https://cdpp-archive.cnes.fr/` |
| 52 | **Jason-1/2/3** (NASA/CNES/NOAA/EUMETSAT) | Radiation-belt dose at ~1336 km (Carmen/ICARE-NG on Jason-2) | particle/dose monitor | min-scale | 2001–present | NetCDF | CNES; AVISO; NCEI altimetry portals |

**Cross-validation note (E):** POES + MetOp give a constellation of ~6 LEO platforms sampling **all magnetic local times every ~100 min** — they reveal the precipitation/loss term invisible at GEO and provide a high-cadence proxy for outer-belt activity that **fills temporal gaps when a GOES sensor is down**. Swarm/CHAMP magnetometers pin down the magnetospheric configuration; DEMETER's 70 keV–2.5 MeV IDP overlaps the lower end of the >2 MeV target population.

---

## F) Geomagnetic Indices, Solar Indices & the Merged OMNI Product (derived features)

These are the standard scalar driver features for >2 MeV electron models (Forsyth et al. 2020 forecast GOES-15 >2 MeV from solar wind + Kp/Dst). **OMNI is the single most valuable input product.**

| # | Product / Operator | What it provides | Parameters | Cadence | Coverage | Format | Access endpoint (exact) |
|---|---|---|---|---|---|---|---|
| — | **OMNI HRO (High-Res)** (NASA SPDF) | **Pre-merged, bow-shock-nose-propagated** multi-satellite solar wind + indices | Bx/By/Bz, Vsw, Np, Pdyn, E-field, plasma β, Mach #, **AE/AL/AU, SYM-H, ASY-H, Kp, Dst, PC, F10.7, proton flux** | **1 min & 5 min** | 1981–present (1-min); hourly back to 1963 | CDF / ASCII | **CDAWeb** `OMNI_HRO_1MIN`, `OMNI_HRO_5MIN`, `OMNI_HRO2_1MIN`; **OMNIWeb** `https://omniweb.gsfc.nasa.gov/`; HAPI(CDAWeb) |
| — | **OMNI2 hourly** (NASA SPDF) | Long merged hourly record | IMF, plasma, indices, sunspot, F10.7 | 1 hr | 1963–present | CDF/ASCII | CDAWeb `OMNI2_H0_MRG1HR` |
| — | **Kp / ap** (GFZ Potsdam) | Planetary geomagnetic activity (3-hr) | Kp, ap, Ap | 3 hr | 1932–present | ASCII/JSON | GFZ `https://www.gfz.de/en/section/geomagnetism/data-products-services/geomagnetic-kp-index`; also in OMNI |
| — | **Dst** (WDC Kyoto) | Ring-current storm index | Dst | 1 hr | 1957–present | ASCII | WDC Kyoto `https://wdc.kugi.kyoto-u.ac.jp/dstdir/`; also in OMNI |
| — | **SYM-H / ASY-H, AE/AL/AU** (WDC Kyoto) | High-res ring-current + auroral indices | SYM-H, ASY-H, AE, AL, AU | 1 min | 1981–present | ASCII | WDC Kyoto; **already in OMNI_HRO_1MIN** |
| — | **F10.7** (NRCan/DRAO via NOAA) | Solar 10.7 cm radio flux (EUV proxy) | F10.7 (sfu) | daily | 1947–present | ASCII/JSON | NOAA NCEI; LASP LISIRD; **in OMNI** |

### Why OMNI / OMNIWeb HRO 1-min is *extremely* valuable

OMNI HRO is a single dataset that **interleaves ACE, Wind, IMP-8 and Geotail** solar-wind/IMF measurements and **time-shifts every record to the nose of Earth's bow shock** (the physically correct location to drive a magnetospheric model). This solves three hard problems for free:

1. **Gap-filling across L1 monitors** — when Wind has a gap, OMNI automatically substitutes ACE/DSCOVR/Geotail, so the driver stream is far more continuous than any single spacecraft.
2. **Correct propagation lag** — raw L1 data arrives ~40–60 min before it reaches the magnetosphere; OMNI applies a minimum-variance phase-front time shift so Bz/Vsw align with the geomagnetic response. This dramatically improves causal feature/target alignment for ML.
3. **One-stop indices** — AE/AL/AU, SYM-H, ASY-H, Kp, Dst, PC and F10.7 are bundled at matching 1-min cadence in the *same* file, eliminating multi-source time-base merging.

**Recommendation:** Build the model's driver matrix primarily from `OMNI_HRO_1MIN` (1995→present) and `OMNI2_H0_MRG1HR` (1963→present for deep history), and only fall back to raw Wind/ACE/DSCOVR for special studies or real-time nowcasting (where OMNI's definitive product lags).

---

<a name="satellite-count"></a>
## Satellite / Mission Count

Counting **distinct satellites/missions** (Section F indices are derived products, not separate spacecraft, and are excluded from the count):

| Category | Members | Count |
|---|---|---|
| A — L1 / heliospheric solar-wind monitors | Wind, ACE, DSCOVR, SOHO, IMP-8, Geotail | **6** |
| B — Solar observation | GOES-XRS, GOES-SUVI, SDO, STEREO-A, STEREO-B, SOHO/LASCO, PSP, Solar Orbiter, Hinode, PROBA-2, Aditya-L1 | **11** |
| C — GEO energetic-electron flux | GOES-16/17/18/19 (R), GOES-13/14/15, GOES-8/9/10/11/12, GOES-6/7, LANL-GEO, FY-4A/4B, FY-2, Electro-L, Meteosat | **9 lines** |
| D — Radiation-belt / magnetosphere in-situ | RBSP-A, RBSP-B, THEMIS-A/D/E, ARTEMIS-P1/P2, MMS-1/2/3/4, Cluster(×4 = C1–C4), Arase, SAMPEX, Polar, CRRES, ELFIN, GPS-constellation, INTEGRAL | **19 lines (Cluster=4 S/C)** |
| E — Polar / LEO electron monitors | POES, MetOp, DMSP, Swarm(×3), CHAMP, DEMETER, Jason | **7 lines (Swarm=3 S/C)** |

**Conservative distinct-mission tally (counting each catalog row once): 45 missions/programs.**
Counting individual spacecraft within constellations (Cluster=4, MMS=4, Swarm=3, GOES rows expand to ~17 individual GOES, GPS≈19) pushes the **individual-satellite total well over 80**. Either way the requirement of **≥35 is exceeded** — the headline figure used is **45 distinct missions**.

---

<a name="data-access-strategy"></a>
## Data Access Strategy

### Recommended Python library stack (2026, current)

| Tool | Role | Notes |
|---|---|---|
| **`pyspedas`** | Primary bulk loader for CDAWeb + mission-specific (THEMIS, MMS, ERG/Arase, PSP, RBSP, GOES, OMNI) | Actively maintained (v2.x). One call downloads + loads CDFs into tplot/xarray. Best for heliophysics. |
| **`cdasws`** | Thin official client to the CDAS REST API | Returns `xarray.Dataset`/`pandas.DataFrame` with full ISTP metadata; great for scripted, dataset-ID-driven pulls without managing files. |
| **`cdflib`** | Pure-Python CDF reader/writer | The underlying CDF engine used by pyspedas and SunPy; use directly when you already have local CDFs. |
| **`hapiclient`** | HAPI standard client (`hapi()` one-liner) | Server-agnostic; pulls CSV/binary time series from CDAWeb, SSCWeb, LISIRD, ViRES, INTERMAGNET, AMDA, CCMC. Ideal for indices + uniform streaming. |
| **`sunpy`** (`Fido`) | Solar data (GOES XRS/SUVI, SDO/AIA, LASCO, EVE) + CDAWeb search | `XRSClient`, `SUVIClient`, `EVEClient`; `Fido.search(a.cdaweb.Dataset(...))` for CDAWeb. |
| **`viresclient`** | ESA Swarm/CHAMP/LEO magnetometer + multi-mission | Server-side filtering at `vires.services`. |
| **`requests`** | NOAA SWPC real-time JSON + NCEI/NGDC HTTPS bulk | For `services.swpc.noaa.gov/json/...` nowcasting feeds and NetCDF directory crawls. |

> **`heliopy` is DEPRECATED** (archived end-2022). Its functionality was absorbed into **`sunpy`** (Fido/CDAWeb) and **`pyspedas`** — do **not** start new work on heliopy.

### HAPI server base URLs (for `hapiclient`)

| Server | Base URL | Holds |
|---|---|---|
| CDAWeb | `https://cdaweb.gsfc.nasa.gov/hapi` | All CDF datasets above (GOES SEISS, Wind, ACE, OMNI, RBSP, THEMIS, MMS, Arase…) |
| SSCWeb | `https://sscweb.gsfc.nasa.gov/WS/hapi` | Spacecraft ephemeris / orbit (for L-shell/MLT labels) |
| LISIRD (LASP) | `https://lasp.colorado.edu/lisird/hapi` | F10.7, solar irradiance/EUV |
| ViRES (Swarm) | `https://vires.services/hapi` | Swarm, CHAMP, LEO magnetometers |
| INTERMAGNET | `https://imag-data.bgs.ac.uk/GIN_V1/hapi` (via hapi-server.org) | Ground magnetometers |
| AMDA (IRAP) | `http://amda.irap.omp.eu/service/hapi` | Mirror of many heliophysics sets |
| CCMC / iSWA | `https://iswa.gsfc.nasa.gov/IswaSystemWebApp/hapi` | Model outputs, indices |
| Master directory | `https://hapi-server.org/servers/` | Browse + auto-generate client code |

### Fastest way to pull **bulk historical CDF**

1. **`pyspedas`** with explicit dataset ID + trange — e.g. GOES-16 >2 MeV target:
   ```python
   import pyspedas
   # GEO >2 MeV electrons (target)
   pyspedas.goes.mpsh(trange=['2017-02-07','2026-06-01'], datatype='1min', probe='16')
   # OMNI driver features (1-min, merged, bow-shock-shifted)
   pyspedas.omni.data(trange=['2017-02-07','2026-06-01'], datatype='1min')   # -> OMNI_HRO_1MIN
   # Wind solar wind backup
   pyspedas.wind.swe(trange=['2017-02-07','2026-06-01'])
   ```
2. **`cdasws`** when you want arrays straight to xarray without local files:
   ```python
   from cdasws import CdasWs
   cdas = CdasWs()
   _, ds = cdas.get_data('OMNI_HRO_1MIN', ['BZ_GSM','flow_speed','proton_density','SYM_H','AE_INDEX'],
                         '2017-02-07T00:00:00Z','2026-06-01T00:00:00Z')
   ```
3. **NCEI HTTPS crawl** for native NetCDF (GOES-R SEISS L2, POES/MetOp, LANL, GPS) — directory-listing + `requests`/`wget`, e.g.
   `https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/goes18/l2/data/mpsh-l2-avg1m_science/`.
4. **CDAS REST** raw endpoint: `https://cdaweb.gsfc.nasa.gov/WS/cdasr/1/dataviews/sp_phys/datasets/<ID>/data/<start>,<stop>/<vars>?format=cdf`.

### Fastest way to **stream real-time** (nowcasting)

NOAA SWPC JSON (no auth, updates every ~1–5 min):

| Feed | URL |
|---|---|
| **GOES >2 MeV integral electrons** (primary) | `https://services.swpc.noaa.gov/json/goes/primary/integral-electrons-1-day.json` |
| Differential electrons | `https://services.swpc.noaa.gov/json/goes/primary/differential-electrons-1-day.json` |
| GOES X-ray flux | `https://services.swpc.noaa.gov/json/goes/primary/xrays-7-day.json` |
| Real-time solar wind (DSCOVR/ACE plasma) | `https://services.swpc.noaa.gov/products/solar-wind/plasma-7-day.json` |
| Real-time solar wind (mag) | `https://services.swpc.noaa.gov/products/solar-wind/mag-7-day.json` |
| Planetary K-index (Kp) | `https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json` |
| Satellite/instrument source map | `https://services.swpc.noaa.gov/json/goes/primary/instrument-sources.json` |

**Hybrid pipeline recommendation:** Train on **definitive** `OMNI_HRO_1MIN` + `DN_SEIS-L2-MPSH_G1x` (via pyspedas/cdasws); operate/nowcast on **SWPC JSON** (DSCOVR solar wind in → GOES >2 MeV electrons out), reconciling the two with the `instrument-sources.json` map so you always know which GOES is "primary."

---

<a name="gap-filling--cross-validation-matrix"></a>
## Gap-Filling & Cross-Validation Matrix

| Need / Failure mode | Primary source | Backup / cross-validator(s) | Why it works |
|---|---|---|---|
| **GEO >2 MeV target, sensor outage** | GOES-East SEISS (e.g. G16) | GOES-West (G18/G17), LANL-GEO, FY-4, Electro-L | Other GEO longitudes keep the target series continuous; cross-cal removes offsets |
| **GEO target, MLT (local-time) coverage** | GOES-East + West | LANL (multiple longitudes), FY-4 (105°E), Electro-L (76°E), Arase (Asian MLT) | Full clock of GEO/near-GEO platforms resolves dawn/dusk asymmetry |
| **Absolute flux calibration of >2 MeV channel** | GOES SEISS / EPEAD | **Van Allen Probes REPT/MagEIS**, GPS-CXD (validated vs RBSP) | RBSP is the community ground truth; overlaps 1.8–20 MeV |
| **L1 solar-wind driver gap** | Wind | ACE, DSCOVR, Geotail — *or just use OMNI* | OMNI auto-interleaves all four at the bow-shock nose |
| **Correct driver→response timing** | Raw L1 (lagged) | **OMNI HRO** (already time-shifted) | Phase-front propagation aligns Bz/Vsw with magnetosphere |
| **Radial (L-shell) structure feeding GEO** | RBSP (full L) | GPS-constellation (L≈4.2, ~19 S/C), THEMIS, Arase | Multi-L sampling lets model learn radial diffusion into GEO |
| **Loss / precipitation term** | POES/MetOp MEPED | DMSP SSJ, DEMETER IDP, ELFIN | LEO precipitation reveals belt depletion not seen at GEO |
| **CME / driver precursor (lead time)** | GOES XRS + LASCO | STEREO-A (off-axis), SDO, PROBA-2, Aditya-L1 | Off-Sun-Earth-line views catch CMEs before L1 |
| **Deep historical training (pre-2010)** | GOES-8…12 EPS | SAMPEX, CRRES, Polar, OMNI2 hourly (1963–) | Extends labeled record across multiple solar cycles |
| **Real-time nowcast inputs** | DSCOVR (SWPC JSON) | ACE (SWPC JSON) | DSCOVR operational, ACE warm backup |
| **Magnetospheric configuration context** | Swarm / CHAMP B-field | DMSP SSM, ground (INTERMAGNET) | Constrains field model used for L*/adiabatic invariants |

---

## References / URLs

**Core access infrastructure**
- NASA SPDF CDAWeb: https://cdaweb.gsfc.nasa.gov/
- CDAWeb dataset notes (per-letter, list of dataset IDs): https://cdaweb.gsfc.nasa.gov/misc/NotesG.html , https://cdaweb.gsfc.nasa.gov/misc/NotesO.html , `.../NotesA.html`, `.../NotesD.html`, `.../NotesM.html`, `.../NotesW.html`
- CDAS RESTful Web Services: https://cdaweb.gsfc.nasa.gov/WebServices/REST/
- `cdasws` (PyPI): https://pypi.org/project/cdasws/
- HAPI standard + server browser: https://hapi-server.org/ and https://hapi-server.org/servers/
- HAPI in pyspedas: https://pyspedas.readthedocs.io/en/latest/hapi.html
- pyspedas CDAWeb loader: https://pyspedas.readthedocs.io/en/stable/cdaweb.html
- SunPy acquiring data / CDAWeb: https://docs.sunpy.org/en/stable/tutorial/acquiring_data/index.html
- OMNIWeb / OMNI HRO docs: https://omniweb.gsfc.nasa.gov/html/omni_min_data.html , https://omniweb.gsfc.nasa.gov/html/HROdocum.html , https://omniweb.gsfc.nasa.gov/html/sc_merge_data1.html
- NOAA SWPC real-time JSON index: https://services.swpc.noaa.gov/json/goes/primary/ and https://services.swpc.noaa.gov/products/solar-wind/

**Target / GEO electrons**
- GOES-R SEISS overview (NCEI): https://www.ncei.noaa.gov/products/goes-r-space-environment-in-situ
- GOES-R SEISS instrument page: https://www.goes-r.gov/spacesegment/seiss.html
- GOES-R SEISS L2 MPS-HI ReadMe: https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/goes16/l2/docs/GOES-R_SEISS_L2_MPS-HI.ReadMe.pdf
- GOES-R SEISS MPS-HI L1b metadata (NCEI): https://www.ncei.noaa.gov/access/metadata/landing-page/bin/iso?id=gov.noaa.ncei.swx:seis-l1b-mpsh-goesr
- GOES historic EPEAD/EPS docs (NGDC): https://www.ngdc.noaa.gov/stp/satellite/goes/
- SWPC GOES Electron Flux product: https://www.swpc.noaa.gov/products/goes-electron-flux
- Rodriguez (2025) GOES 8–15 >0.6/>4 MeV assessment: https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2024SW004228
- Forsyth (2020) forecasting GOES-15 >2 MeV from solar wind + indices: https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/2019SW002416

**L1 / solar wind**
- Wind notes: https://cdaweb.gsfc.nasa.gov/misc/NotesW.html
- ACE notes: https://cdaweb.gsfc.nasa.gov/misc/NotesA.html
- DSCOVR (SWPC real-time solar wind): https://www.swpc.noaa.gov/products/real-time-solar-wind ; DSCOVR validation: https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2022SW003085
- SOHO COSTEP/EPHIN: https://www.swsc-journal.org/articles/swsc/full_html/2020/01/swsc200043/swsc200043.html

**Solar observation**
- SunPy XRS/SUVI/EVE clients: https://docs.sunpy.org/en/stable/generated/gallery/acquiring_data/search_cdaweb.html
- SOHO/LASCO CME catalog: https://cdaw.gsfc.nasa.gov/CME_list/
- ESA Solar Orbiter Archive (SOAR): https://soar.esac.esa.int/
- Hinode (DARTS): https://darts.isas.jaxa.jp/solar/hinode/
- PROBA-2 (ROB): https://proba2.sidc.be/data
- Aditya-L1 PRADAN/ISSDC: https://pradan.issdc.gov.in/al1/ ; ASPEX-SWIS first year: https://arxiv.org/html/2507.17523

**Radiation belt / magnetosphere**
- Van Allen Probes ECT SOC: https://rbsp-ect.newmexicoconsortium.org/ ; gateway: https://rbspgway.jhuapl.edu/Instr_ECT
- THEMIS/ARTEMIS + pyspedas (SPEDAS): https://pyspedas.readthedocs.io/
- MMS FEEPS (SPASE): https://hpde.io/NASA/NumericalData/MMS/1/EnergeticParticleDetector/FEEPS/Survey/Level2/Electron/PT2.42S.html
- Cluster Science Archive: https://csa.esac.esa.int/ ; about: https://www.cosmos.esa.int/web/csa/about-this-archive
- Arase/ERG Science Center: https://ergsc.isee.nagoya-u.ac.jp/ ; XEP paper: https://earth-planets-space.springeropen.com/articles/10.1186/s40623-018-0901-x
- Arase DARTS: https://darts.isas.jaxa.jp/en/missions/arase
- SAMPEX SOC: http://www.srl.caltech.edu/sampex/
- ELFIN (UCLA): https://elfin.igpp.ucla.edu/ ; mission paper: https://arxiv.org/pdf/2006.07747
- GPS constellation CXD data (NGDC): https://www.ngdc.noaa.gov/stp/space-weather/satellite-data/satellite-systems/gps/data/ ; Morley (2017): https://agupubs.onlinelibrary.wiley.com/doi/10.1002/2017SW001604
- LANL-GEO products (NCEI): https://www.ncei.noaa.gov/products/space-weather/partners/lanl-products-data ; readme: https://www.ngdc.noaa.gov/stp/space-weather/satellite-data/satellite-systems/lanl_geo/readme.pdf
- INTEGRAL IREM: https://www.cosmos.esa.int/web/integral/instruments-irem

**Polar / LEO**
- POES/MetOp SEM (NCEI): https://www.ncei.noaa.gov/products/poes-metop-space-environment-monitor
- DMSP / Madrigal: http://cedar.openmadrigal.org/
- Swarm VirES: https://vires.services/ ; viresclient params: https://viresclient.readthedocs.io/en/latest/available_parameters.html ; Swarm data: https://earth.esa.int/eogateway/missions/swarm/data
- CHAMP (GFZ ISDC): https://isdc.gfz-potsdam.de/champ-isdc/
- DEMETER (CNES CDPP): https://cdpp-archive.cnes.fr/

**Indices**
- Kp (GFZ): https://www.gfz.de/en/section/geomagnetism/data-products-services/geomagnetic-kp-index
- Dst (WDC Kyoto): https://wdc.kugi.kyoto-u.ac.jp/dstdir/
- LASP LISIRD (F10.7 / irradiance): https://lasp.colorado.edu/lisird/
- OMNI on NASA Open Data: https://data.nasa.gov/dataset/omni-combined-solar-wind-plasma-moments-and-interplanetary-magnetic-field-imf-time-shifted-74e9b

**Cross-calibration / international GEO**
- FY-4A high-energy electrons vs Arase: https://www.eppcgs.org/article/doi/10.26464/epp2023076
- FY-3 medium-energy electrons vs POES: https://www.sciencedirect.com/science/article/abs/pii/S0273117718301509
- Electro-L / GOMS-2 (eoPortal): https://directory.eoportal.org/satellite-missions/electro-l

---

*End of catalog. Indices in Section F are derived products (not spacecraft) and are excluded from the 45-mission count.*
