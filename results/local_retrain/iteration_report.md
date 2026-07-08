# Local retrain evaluation — 1980-2026 TMDB dataset

Iteration mode: eval years 2015-2023 only. 2024-2025 stay held out until a final `--confirm` run.

**Confirmation discipline:** iterate against eval years <= 2023 (default mode); 2024-2025 are held out and touched only for a final `--confirm` run. Once confirmed, the numbers are frozen — further iteration against 2024-2025 would turn the confirmation set into a validation set.

Frame: `data/generated/training/train_frame_1980_2026.parquet` (6077 rows, 324 missing budgets kept as NaN). CV: expanding window, eval years 2015-2023, production `TimeSeriesCrossValidator` + `BoxOfficeXGBoostModel`, feature contract frozen at v9. Leakage-free: each fold fits a fresh `FeaturePreprocessorHigh` on train-years rows only.

## Headline: recent window (2023-2023)

Pooled over 2023-2023 (116 movies): **median APE 52.5%**, **mean log-R² 0.621**.

| Year | n | Model R² (log) | Baseline R² (log) | Gain (log) | Model ρ | Baseline ρ | Model R² ($) | Median APE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2023 | 116 | 0.621 | 0.405 | +0.216 | 0.786 | 0.655 | 0.252 | 52.5% |

## Diagnostic: full per-year table (2015-2023)

Pooled over 2015-2023 (1159 movies): median APE 51.8%, mean log-R² 0.603.

| Year | n | Model R² (log) | Baseline R² (log) | Gain (log) | Model ρ | Baseline ρ | Model R² ($) | Median APE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2015 | 158 | 0.676 | 0.393 | +0.283 | 0.797 | 0.695 | 0.660 | 46.2% |
| 2016 | 181 | 0.634 | 0.497 | +0.137 | 0.781 | 0.734 | 0.669 | 50.5% |
| 2017 | 165 | 0.655 | 0.432 | +0.224 | 0.767 | 0.654 | 0.643 | 54.5% |
| 2018 | 163 | 0.659 | 0.438 | +0.221 | 0.779 | 0.694 | 0.722 | 46.5% |
| 2019 | 134 | 0.655 | 0.435 | +0.220 | 0.764 | 0.662 | 0.653 | 45.1% |
| 2020 | 53 | 0.437 | 0.174 | +0.263 | 0.684 | 0.558 | 0.457 | 60.5% |
| 2021 | 89 | 0.508 | 0.284 | +0.224 | 0.683 | 0.553 | 0.407 | 58.7% |
| 2022 | 100 | 0.583 | 0.425 | +0.157 | 0.725 | 0.670 | 0.422 | 62.3% |
| 2023 | 116 | 0.621 | 0.405 | +0.216 | 0.786 | 0.655 | 0.252 | 52.5% |

## Delta vs the committed old-model table (results/per_year_table.md)

Positive ΔR²/Δρ and negative ΔAPE mean the new run is better. The eval population changed (old ~2.7k-row snapshot vs the new $5M+ frame), so this table confounds model and population — judge on the overlap view. The old table also predates the leakage fix.

| Year | n old | n new | R² log old | R² log new | ΔR² log | ρ old | ρ new | Δρ | APE old | APE new | ΔAPE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2015 | 110 | 158 | 0.671 | 0.676 | +0.006 | 0.781 | 0.797 | +0.017 | 40.7% | 46.2% | +5.5% |
| 2016 | 119 | 181 | 0.634 | 0.634 | +0.000 | 0.790 | 0.781 | -0.010 | 40.4% | 50.5% | +10.1% |
| 2017 | 108 | 165 | 0.597 | 0.655 | +0.058 | 0.744 | 0.767 | +0.024 | 46.4% | 54.5% | +8.1% |
| 2018 | 111 | 163 | 0.640 | 0.659 | +0.019 | 0.765 | 0.779 | +0.014 | 38.9% | 46.5% | +7.6% |
| 2019 | 104 | 134 | 0.650 | 0.655 | +0.006 | 0.777 | 0.764 | -0.013 | 45.2% | 45.1% | -0.1% |
| 2020 | 56 | 53 | -0.498 | 0.437 | +0.935 | 0.648 | 0.684 | +0.036 | 244.4% | 60.5% | -183.9% |
| 2021 | 92 | 89 | 0.450 | 0.508 | +0.058 | 0.588 | 0.683 | +0.095 | 63.0% | 58.7% | -4.2% |
| 2022 | 96 | 100 | 0.540 | 0.583 | +0.043 | 0.670 | 0.725 | +0.056 | 66.1% | 62.3% | -3.8% |
| 2023 | 117 | 116 | 0.599 | 0.621 | +0.022 | 0.763 | 0.786 | +0.023 | 62.6% | 52.5% | -10.2% |

## Overlap view (eval restricted to movies in the old snapshot)

Overlap key: (release_year, runtime, production_budget) — the old snapshot has no imdb_id or title, so this match is approximate. 1991 of 6077 new rows matched 2762 old rows.

| Year | n old | n overlap | R² log old | R² log overlap | ρ old | ρ overlap | APE old | APE overlap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2015 | 110 | 90 | 0.671 | 0.574 | 0.781 | 0.846 | 40.7% | 36.2% |
| 2016 | 119 | 96 | 0.634 | 0.405 | 0.790 | 0.729 | 40.4% | 41.6% |
| 2017 | 108 | 79 | 0.597 | 0.250 | 0.744 | 0.714 | 46.4% | 48.6% |
| 2018 | 111 | 91 | 0.640 | 0.512 | 0.765 | 0.714 | 38.9% | 43.4% |
| 2019 | 104 | 72 | 0.650 | 0.552 | 0.777 | 0.818 | 45.2% | 42.6% |
| 2020 | 56 | 32 | -0.498 | 0.378 | 0.648 | 0.502 | 244.4% | 53.9% |
| 2021 | 92 | 58 | 0.450 | 0.546 | 0.588 | 0.684 | 63.0% | 54.7% |
| 2022 | 96 | 59 | 0.540 | 0.516 | 0.670 | 0.688 | 66.1% | 62.6% |
| 2023 | 117 | 57 | 0.599 | 0.492 | 0.763 | 0.732 | 62.6% | 54.7% |

## Pre-leakage-fix history: variant comparison

Numbers below predate the frequency-feature leakage fix (preprocessor fit on the full frame before CV) and are inflated: over 2015-2023, mean per-year log-R² dropped from 0.568 (pre-fix) to 0.603 (leakage-free), +0.035.

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
