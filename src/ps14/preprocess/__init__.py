"""Preprocessing: clean -> resample -> align/merge -> transform.

Pipeline order (R5 §4): fill/valid masking -> Hampel/MAD despike -> gap detection
(interpolate short gaps, flag imputed, leave long gaps NaN) -> resample to uniform
5-min -> log10 with positive floor -> L1->GEO time-alignment/merge -> chronological
scaling fit on TRAIN only. Produces the canonical merged dataframe (CONTRACTS.md §2).
"""

from __future__ import annotations

__all__: list[str] = []
