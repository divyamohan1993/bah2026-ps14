# PS-14 real-data (CDAWeb) validation demo

**Status:** SUCCESS

## Fetch log

- Run at 2026-06-21T00:01:13.857279+00:00 UTC
- Requested window: 2017-09-01 .. 2017-12-01
- OMNI fetch OK: OMNI_HRO_1MIN -> 131041 rows, 8 variables (['flow_speed', 'proton_density', 'Pressure', 'BZ_GSM', 'F', 'AE_INDEX', 'AL_INDEX', 'SYM_H']).
- GOES try DN_SEIS-L2-MPSH_G18:AvgIntElectronFlux FAILED: RuntimeError: cdasws returned no data for 'DN_SEIS-L2-MPSH_G18' (2017-09-01T00:00:00Z..2017-12-01T00:00:00Z); status={'http': {'status_code': 400}, 'cdas': {'status': [], 'message': [], 'warning': [], 'error': []}}.
- GOES try DN_SEIS-L2-MPSH_G18:IntElectronFlux FAILED: RuntimeError: cdasws returned no data for 'DN_SEIS-L2-MPSH_G18' (2017-09-01T00:00:00Z..2017-12-01T00:00:00Z); status={'http': {'status_code': 400}, 'cdas': {'status': [], 'message': [], 'warning': [], 'error': []}}.
- GOES try DN_SEIS-L2-MPSH_G18:flux FAILED: RuntimeError: cdasws returned no data for 'DN_SEIS-L2-MPSH_G18' (2017-09-01T00:00:00Z..2017-12-01T00:00:00Z); status={'http': {'status_code': 400}, 'cdas': {'status': [], 'message': [], 'warning': [], 'error': []}}.
- GOES try DN_SEIS-L2-MPSH_G16:AvgIntElectronFlux FAILED: RuntimeError: cdasws returned no data for 'DN_SEIS-L2-MPSH_G16' (2017-09-01T00:00:00Z..2017-12-01T00:00:00Z); status={'http': {'status_code': 400}, 'cdas': {'status': [], 'message': [], 'warning': [], 'error': []}}.
- GOES try DN_SEIS-L2-MPSH_G16:IntElectronFlux FAILED: RuntimeError: cdasws returned no data for 'DN_SEIS-L2-MPSH_G16' (2017-09-01T00:00:00Z..2017-12-01T00:00:00Z); status={'http': {'status_code': 400}, 'cdas': {'status': [], 'message': [], 'warning': [], 'error': []}}.
- GOES try DN_SEIS-L2-MPSH_G16:flux FAILED: RuntimeError: cdasws returned no data for 'DN_SEIS-L2-MPSH_G16' (2017-09-01T00:00:00Z..2017-12-01T00:00:00Z); status={'http': {'status_code': 400}, 'cdas': {'status': [], 'message': [], 'warning': [], 'error': []}}.
- GOES try DN_SEIS-L2-MPSH_G17:AvgIntElectronFlux FAILED: RuntimeError: cdasws returned no data for 'DN_SEIS-L2-MPSH_G17' (2017-09-01T00:00:00Z..2017-12-01T00:00:00Z); status={'http': {'status_code': 400}, 'cdas': {'status': [], 'message': [], 'warning': [], 'error': []}}.
- GOES try DN_SEIS-L2-MPSH_G17:IntElectronFlux FAILED: RuntimeError: cdasws returned no data for 'DN_SEIS-L2-MPSH_G17' (2017-09-01T00:00:00Z..2017-12-01T00:00:00Z); status={'http': {'status_code': 400}, 'cdas': {'status': [], 'message': [], 'warning': [], 'error': []}}.
- GOES try DN_SEIS-L2-MPSH_G17:flux FAILED: RuntimeError: cdasws returned no data for 'DN_SEIS-L2-MPSH_G17' (2017-09-01T00:00:00Z..2017-12-01T00:00:00Z); status={'http': {'status_code': 400}, 'cdas': {'status': [], 'message': [], 'warning': [], 'error': []}}.
- GOES try GOES13_EPEAD-SCIENCE-ELECTRONS-E13EW_1MIN:E2W_UNCOR_FLUX FAILED: RuntimeError: cdasws returned no data for 'GOES13_EPEAD-SCIENCE-ELECTRONS-E13EW_1MIN' (2017-09-01T00:00:00Z..2017-12-01T00:00:00Z); status={'http': {'status_code': 400}, 'cdas': {'status': [], 'message': [], 'warning': [], 'error': []}}.
- GOES fetch OK: GOES13_EPEAD-SCIENCE-ELECTRONS-E13EW_1MIN:E2W_COR_FLUX -> 105024 non-NaN rows.
- Real-data windows: X=(4000, 72, 33), merged rows=26209, GOES dataset=GOES13_EPEAD-SCIENCE-ELECTRONS-E13EW_1MIN.

## Persistence baseline metrics on REAL data (per horizon, log10 space)

| horizon | rmse | mae | pe | skill_vs_persistence | roc_auc | hss |
|---|---|---|---|---|---|---|
| nowcast | 0.08459 | 0.06279 | 0.9347 | 0 | nan | nan |
| 6h | 0.4449 | 0.3881 | -1.052 | 0 | nan | nan |
| 12h | 0.5618 | 0.5071 | -2.343 | 0 | nan | nan |

Regression metrics in log10(flux) space; event metrics at the 1000 pfu threshold.

