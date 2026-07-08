# v9 vs v8 — iteration comparison (eval years 2015-2023)

Feature contract v9 adds three pre-release IP/franchise features to the 10
v8 features: `IP_TIER` (ordinal 1-5 from the time-safe IP classifier),
`PRIOR_FRANCHISE_GROSS_LOG` (log1p of the same TMDB collection's
strictly-earlier worldwide gross), `IS_FRANCHISE_FOLLOWUP`. Adopted from the
E variant of `results/ip_experiment/report.md` (now superseded).

**Eval discipline:** both runs are iteration mode (2015-2023). 2024-2025 are
a spent confirmation set — v8's frozen confirmation
(`results/local_retrain/report.md`) stands untouched; v9 is adopted on
<=2023 evidence and 2026 actuals will confirm it. `--confirm` now also
requires `--i-know-this-burns-the-holdout`.

The frame changed between the runs (source cleanup: 6080 -> 6073 kept rows),
so per-year n differs slightly; the deltas confound feature and population
by that small amount.

## Headline (2015-2023)

| Run | Frame rows | Eval n | Mean CV MAE (log) | Mean per-year R² (log) | Pooled median APE |
|---|---:|---:|---:|---:|---:|
| v8 iteration | 6080 | 1161 | 0.7403 ± 0.0496 | 0.5575 | 55.7% |
| v9 iteration | 6073 | 1156 | 0.6958 ± 0.0488 | 0.6018 | 50.8% |

Close to the ip_experiment E variant on the old frame (R² 0.5999, pooled
median APE 51.7%), as expected.

## Per-year: v8 vs v9

| Year | n v8 | n v9 | R² log v8 | R² log v9 | ΔR² log | APE v8 | APE v9 | ΔAPE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2015 | 158 | 158 | 0.617 | 0.677 | +0.060 | 48.0% | 44.8% | -3.3% |
| 2016 | 181 | 181 | 0.609 | 0.628 | +0.019 | 52.4% | 51.4% | -1.0% |
| 2017 | 164 | 164 | 0.626 | 0.648 | +0.022 | 55.6% | 55.1% | -0.5% |
| 2018 | 164 | 163 | 0.634 | 0.667 | +0.033 | 51.0% | 43.5% | -7.5% |
| 2019 | 132 | 132 | 0.602 | 0.642 | +0.040 | 51.4% | 45.8% | -5.6% |
| 2020 | 54 | 53 | 0.354 | 0.439 | +0.085 | 75.1% | 56.9% | -18.2% |
| 2021 | 90 | 89 | 0.417 | 0.504 | +0.087 | 62.8% | 56.6% | -6.2% |
| 2022 | 101 | 100 | 0.573 | 0.583 | +0.010 | 65.5% | 61.7% | -3.9% |
| 2023 | 117 | 116 | 0.584 | 0.628 | +0.044 | 56.8% | 52.0% | -4.8% |

Generated from `iteration_results.json` (v9, this run) and the v8 iteration
run archived before this retrain.
