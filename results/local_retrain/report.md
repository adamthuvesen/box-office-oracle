# Local retrain evaluation — 1980-2026 TMDB dataset

**FROZEN CONFIRMATION RUN** (`--confirm`): eval years include the held-out 2024-2025. These numbers are final for this retrain; do not iterate against them.

**Confirmation discipline:** iterate against eval years <= 2023 (default mode); 2024-2025 are held out and touched only for a final `--confirm` run. Once confirmed, the numbers are frozen — further iteration against 2024-2025 would turn the confirmation set into a validation set.

Frame: `data/generated/training/train_frame_1980_2026.parquet` (6080 rows, 323 missing budgets kept as NaN). CV: expanding window, eval years 2015-2025, production `TimeSeriesCrossValidator` + `BoxOfficeXGBoostModel`, feature contract frozen at v8. Leakage-free: each fold fits a fresh `FeaturePreprocessorHigh` on train-years rows only.

## Headline: recent window (2023-2025)

Pooled over 2023-2025 (360 movies): **median APE 57.0%**, **mean log-R² 0.605**.

| Year | n | Model R² (log) | Baseline R² (log) | Gain (log) | Model ρ | Baseline ρ | Model R² ($) | Median APE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2023 | 117 | 0.584 | 0.370 | +0.215 | 0.772 | 0.651 | 0.219 | 56.8% |
| 2024 | 119 | 0.587 | 0.407 | +0.180 | 0.725 | 0.648 | 0.338 | 62.0% |
| 2025 | 124 | 0.644 | 0.447 | +0.197 | 0.771 | 0.704 | 0.525 | 52.7% |

## Diagnostic: full per-year table (2015-2025)

Pooled over 2015-2025 (1404 movies): median APE 56.0%, mean log-R² 0.568.

| Year | n | Model R² (log) | Baseline R² (log) | Gain (log) | Model ρ | Baseline ρ | Model R² ($) | Median APE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2015 | 158 | 0.617 | 0.390 | +0.227 | 0.756 | 0.695 | 0.506 | 48.0% |
| 2016 | 181 | 0.609 | 0.490 | +0.119 | 0.765 | 0.734 | 0.711 | 52.4% |
| 2017 | 164 | 0.626 | 0.422 | +0.204 | 0.755 | 0.648 | 0.601 | 55.6% |
| 2018 | 164 | 0.634 | 0.407 | +0.227 | 0.766 | 0.693 | 0.656 | 51.0% |
| 2019 | 132 | 0.602 | 0.417 | +0.185 | 0.727 | 0.647 | 0.629 | 51.4% |
| 2020 | 54 | 0.354 | 0.087 | +0.267 | 0.651 | 0.555 | 0.385 | 75.1% |
| 2021 | 90 | 0.417 | 0.253 | +0.165 | 0.609 | 0.560 | 0.271 | 62.8% |
| 2022 | 101 | 0.573 | 0.399 | +0.174 | 0.743 | 0.673 | 0.259 | 65.5% |
| 2023 | 117 | 0.584 | 0.370 | +0.215 | 0.772 | 0.651 | 0.219 | 56.8% |
| 2024 | 119 | 0.587 | 0.407 | +0.180 | 0.725 | 0.648 | 0.338 | 62.0% |
| 2025 | 124 | 0.644 | 0.447 | +0.197 | 0.771 | 0.704 | 0.525 | 52.7% |

## Delta vs the committed old-model table (results/per_year_table.md)

Positive ΔR²/Δρ and negative ΔAPE mean the new run is better. The eval population changed (old ~2.7k-row snapshot vs the new $5M+ frame), so this table confounds model and population — judge on the overlap view. The old table also predates the leakage fix.

| Year | n old | n new | R² log old | R² log new | ΔR² log | ρ old | ρ new | Δρ | APE old | APE new | ΔAPE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2015 | 110 | 158 | 0.671 | 0.617 | -0.053 | 0.781 | 0.756 | -0.024 | 40.7% | 48.0% | +7.3% |
| 2016 | 119 | 181 | 0.634 | 0.609 | -0.025 | 0.790 | 0.765 | -0.025 | 40.4% | 52.4% | +12.0% |
| 2017 | 108 | 164 | 0.597 | 0.626 | +0.028 | 0.744 | 0.755 | +0.011 | 46.4% | 55.6% | +9.2% |
| 2018 | 111 | 164 | 0.640 | 0.634 | -0.006 | 0.765 | 0.766 | +0.000 | 38.9% | 51.0% | +12.2% |
| 2019 | 104 | 132 | 0.650 | 0.602 | -0.047 | 0.777 | 0.727 | -0.050 | 45.2% | 51.4% | +6.2% |
| 2020 | 56 | 54 | -0.498 | 0.354 | +0.852 | 0.648 | 0.651 | +0.003 | 244.4% | 75.1% | -169.3% |
| 2021 | 92 | 90 | 0.450 | 0.417 | -0.033 | 0.588 | 0.609 | +0.021 | 63.0% | 62.8% | -0.2% |
| 2022 | 96 | 101 | 0.540 | 0.573 | +0.033 | 0.670 | 0.743 | +0.073 | 66.1% | 65.5% | -0.5% |
| 2023 | 117 | 117 | 0.599 | 0.584 | -0.015 | 0.763 | 0.772 | +0.009 | 62.6% | 56.8% | -5.8% |

## Overlap view (eval restricted to movies in the old snapshot)

Overlap key: (release_year, runtime, production_budget) — the old snapshot has no imdb_id or title, so this match is approximate. 1988 of 6080 new rows matched 2762 old rows.

| Year | n old | n overlap | R² log old | R² log overlap | ρ old | ρ overlap | APE old | APE overlap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2015 | 110 | 90 | 0.671 | 0.483 | 0.781 | 0.806 | 40.7% | 37.8% |
| 2016 | 119 | 96 | 0.634 | 0.396 | 0.790 | 0.732 | 40.4% | 42.5% |
| 2017 | 108 | 79 | 0.597 | 0.202 | 0.744 | 0.681 | 46.4% | 50.6% |
| 2018 | 111 | 91 | 0.640 | 0.504 | 0.765 | 0.699 | 38.9% | 46.4% |
| 2019 | 104 | 70 | 0.650 | 0.464 | 0.777 | 0.777 | 45.2% | 47.2% |
| 2020 | 56 | 32 | -0.498 | 0.281 | 0.648 | 0.506 | 244.4% | 68.6% |
| 2021 | 92 | 58 | 0.450 | 0.401 | 0.588 | 0.580 | 63.0% | 56.4% |
| 2022 | 96 | 59 | 0.540 | 0.514 | 0.670 | 0.718 | 66.1% | 68.6% |
| 2023 | 117 | 57 | 0.599 | 0.418 | 0.763 | 0.683 | 62.6% | 63.5% |

## Pre-leakage-fix history: variant comparison

Numbers below predate the frequency-feature leakage fix (preprocessor fit on the full frame before CV) and are inflated: over 2015-2025, mean per-year log-R² dropped from 0.579 (pre-fix) to 0.568 (leakage-free), -0.011.

| Variant | n | missing budgets | Mean CV MAE (log) | Per-year R² (log) range | Pooled median APE |
|---|---:|---:|---:|---:|---:|
| headline_nan_passthrough | 6080 | 323 | 0.7290 ± 0.0496 | 0.364 – 0.669 | 55.3% |
| drop_missing_budget | 5757 | 0 | 0.7431 ± 0.0489 | 0.312 – 0.635 | 54.8% |
| missing_budget_flag_11th_col | 6080 | 323 | 0.7304 ± 0.0548 | 0.367 – 0.670 | 54.4% |
| gross_50m_plus | 2640 | 18 | 0.4712 ± 0.0561 | 0.102 – 0.667 | 39.2% |
| min_year_1990 | 5185 | 254 | 0.7355 ± 0.0529 | 0.327 – 0.650 | 55.5% |

## Dropped rows

Dropped rows: 82 (reasons overlap; counts are per rule)
- `gross_not_final_future_year`: 58
- `runtime_under_60_non_feature`: 19
- `gross_over_100m_with_no_documented_budget`: 7

Flagged by the spec's 50x gross/budget rule but KEPT after a hand check (12 legitimate low-budget sleeper hits):
- E.T. the Extra-Terrestrial (1982): budget $10,500,000, gross $797,307,407
- Crocodile Dundee (1986): budget $5,000,000, gross $328,203,506
- Four Weddings and a Funeral (1994): budget $4,400,000, gross $245,700,832
- The Full Monty (1997): budget $3,500,000, gross $257,850,122
- The Blair Witch Project (1999): budget $60,000, gross $248,639,099
- My Big Fat Greek Wedding (2002): budget $5,000,000, gross $368,744,044
- Saw (2004): budget $1,200,000, gross $104,045,735
- Paranormal Activity (2007): budget $15,000, gross $193,355,800
- Paranormal Activity 2 (2010): budget $3,000,000, gross $177,512,032
- Insidious (2011): budget $1,500,000, gross $100,106,454
- The Devil Inside (2012): budget $1,000,000, gross $101,800,000
- Get Out (2017): budget $4,500,000, gross $255,407,969

## Caveats

- The overlap join is approximate (composite key, no stable id).
- Per-fold preprocessor refits make CV slower than the pre-fix runs; the numbers are not comparable to pre-fix reports (see the history section for the quantified gap).
