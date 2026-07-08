# Local retrain evaluation — 1980-2026 TMDB dataset

Iteration mode: eval years 2015-2023 only. 2024-2025 stay held out until a final `--confirm` run.

**Confirmation discipline:** iterate against eval years <= 2023 (default mode); 2024-2025 are held out and touched only for a final `--confirm` run. Once confirmed, the numbers are frozen — further iteration against 2024-2025 would turn the confirmation set into a validation set.

Frame: `data/generated/training/train_frame_1980_2026.parquet` (6073 rows, 324 missing budgets kept as NaN). CV: expanding window, eval years 2015-2023, production `TimeSeriesCrossValidator` + `BoxOfficeXGBoostModel`, feature contract frozen at v9. Leakage-free: each fold fits a fresh `FeaturePreprocessorHigh` on train-years rows only.

## Headline: recent window (2023-2023)

Pooled over 2023-2023 (116 movies): **median APE 52.0%**, **mean log-R² 0.628**.

| Year | n | Model R² (log) | Baseline R² (log) | Gain (log) | Model ρ | Baseline ρ | Model R² ($) | Median APE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2023 | 116 | 0.628 | 0.405 | +0.223 | 0.790 | 0.655 | 0.256 | 52.0% |

## Diagnostic: full per-year table (2015-2023)

Pooled over 2015-2023 (1156 movies): median APE 50.8%, mean log-R² 0.602.

| Year | n | Model R² (log) | Baseline R² (log) | Gain (log) | Model ρ | Baseline ρ | Model R² ($) | Median APE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2015 | 158 | 0.677 | 0.393 | +0.284 | 0.796 | 0.695 | 0.667 | 44.8% |
| 2016 | 181 | 0.628 | 0.497 | +0.131 | 0.778 | 0.734 | 0.649 | 51.4% |
| 2017 | 164 | 0.648 | 0.427 | +0.220 | 0.759 | 0.648 | 0.670 | 55.1% |
| 2018 | 163 | 0.667 | 0.437 | +0.229 | 0.788 | 0.694 | 0.706 | 43.5% |
| 2019 | 132 | 0.642 | 0.420 | +0.222 | 0.752 | 0.647 | 0.722 | 45.8% |
| 2020 | 53 | 0.439 | 0.176 | +0.263 | 0.686 | 0.558 | 0.465 | 56.9% |
| 2021 | 89 | 0.504 | 0.285 | +0.220 | 0.676 | 0.553 | 0.409 | 56.6% |
| 2022 | 100 | 0.583 | 0.426 | +0.158 | 0.725 | 0.670 | 0.438 | 61.7% |
| 2023 | 116 | 0.628 | 0.405 | +0.223 | 0.790 | 0.655 | 0.256 | 52.0% |

## Delta vs the committed old-model table (results/per_year_table.md)

Positive ΔR²/Δρ and negative ΔAPE mean the new run is better. The eval population changed (old ~2.7k-row snapshot vs the new $5M+ frame), so this table confounds model and population — judge on the overlap view. The old table also predates the leakage fix.

| Year | n old | n new | R² log old | R² log new | ΔR² log | ρ old | ρ new | Δρ | APE old | APE new | ΔAPE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2015 | 110 | 158 | 0.671 | 0.677 | +0.006 | 0.781 | 0.796 | +0.016 | 40.7% | 44.8% | +4.0% |
| 2016 | 119 | 181 | 0.634 | 0.628 | -0.006 | 0.790 | 0.778 | -0.012 | 40.4% | 51.4% | +11.0% |
| 2017 | 108 | 164 | 0.597 | 0.648 | +0.050 | 0.744 | 0.759 | +0.016 | 46.4% | 55.1% | +8.7% |
| 2018 | 111 | 163 | 0.640 | 0.667 | +0.027 | 0.765 | 0.788 | +0.023 | 38.9% | 43.5% | +4.6% |
| 2019 | 104 | 132 | 0.650 | 0.642 | -0.007 | 0.777 | 0.752 | -0.025 | 45.2% | 45.8% | +0.5% |
| 2020 | 56 | 53 | -0.498 | 0.439 | +0.937 | 0.648 | 0.686 | +0.039 | 244.4% | 56.9% | -187.5% |
| 2021 | 92 | 89 | 0.450 | 0.504 | +0.054 | 0.588 | 0.676 | +0.088 | 63.0% | 56.6% | -6.4% |
| 2022 | 96 | 100 | 0.540 | 0.583 | +0.043 | 0.670 | 0.725 | +0.055 | 66.1% | 61.7% | -4.4% |
| 2023 | 117 | 116 | 0.599 | 0.628 | +0.029 | 0.763 | 0.790 | +0.027 | 62.6% | 52.0% | -10.6% |

## Overlap view (eval restricted to movies in the old snapshot)

Overlap key: (release_year, runtime, production_budget) — the old snapshot has no imdb_id or title, so this match is approximate. 1988 of 6073 new rows matched 2762 old rows.

| Year | n old | n overlap | R² log old | R² log overlap | ρ old | ρ overlap | APE old | APE overlap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2015 | 110 | 90 | 0.671 | 0.598 | 0.781 | 0.843 | 40.7% | 38.0% |
| 2016 | 119 | 96 | 0.634 | 0.395 | 0.790 | 0.725 | 40.4% | 37.8% |
| 2017 | 108 | 79 | 0.597 | 0.250 | 0.744 | 0.702 | 46.4% | 49.5% |
| 2018 | 111 | 91 | 0.640 | 0.546 | 0.765 | 0.740 | 38.9% | 40.7% |
| 2019 | 104 | 70 | 0.650 | 0.522 | 0.777 | 0.798 | 45.2% | 41.2% |
| 2020 | 56 | 32 | -0.498 | 0.384 | 0.648 | 0.514 | 244.4% | 51.1% |
| 2021 | 92 | 58 | 0.450 | 0.538 | 0.588 | 0.684 | 63.0% | 53.3% |
| 2022 | 96 | 59 | 0.540 | 0.523 | 0.670 | 0.701 | 66.1% | 57.3% |
| 2023 | 117 | 57 | 0.599 | 0.505 | 0.763 | 0.745 | 62.6% | 53.3% |

## Pre-leakage-fix history: variant comparison

Numbers below predate the frequency-feature leakage fix (preprocessor fit on the full frame before CV) and are inflated: over 2015-2023, mean per-year log-R² dropped from 0.568 (pre-fix) to 0.602 (leakage-free), +0.034.

| Variant | n | missing budgets | Mean CV MAE (log) | Per-year R² (log) range | Pooled median APE |
|---|---:|---:|---:|---:|---:|
| headline_nan_passthrough | 6080 | 323 | 0.7290 ± 0.0496 | 0.364 – 0.669 | 55.3% |
| drop_missing_budget | 5757 | 0 | 0.7431 ± 0.0489 | 0.312 – 0.635 | 54.8% |
| missing_budget_flag_11th_col | 6080 | 323 | 0.7304 ± 0.0548 | 0.367 – 0.670 | 54.4% |
| gross_50m_plus | 2640 | 18 | 0.4712 ± 0.0561 | 0.102 – 0.667 | 39.2% |
| min_year_1990 | 5185 | 254 | 0.7355 ± 0.0529 | 0.327 – 0.650 | 55.5% |

## Dropped rows

Dropped rows: 75 (reasons overlap; counts are per rule)
- `gross_not_final_future_year`: 55
- `runtime_under_60_non_feature`: 10
- `no_reliable_worldwide_gross`: 10
- `gross_over_100m_with_no_documented_budget`: 3

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
