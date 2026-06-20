# Drop-COVID Challenger Comparison

Offline comparison run against the ignored `analysis/datasets_high` snapshot with
`scripts/run_backtest.py`'s leakage controls: 2,762 raw rows, 6 rows dropped for
the `production_budget = 0.4 * worldwide_gross` imputation signature, 2,756 rows
scored. Both variants use the same expanding yearly folds and baseline builder.

Replacement bar used here: improve validation on the model's training objective
(`log1p(worldwide_gross)` R2) and be non-worse on the 2023 final holdout across
log R2, dollar R2, and dollar MAE. Median APE is reported as a calibration caveat.

| Split | Metric | Current depth-6 + COVID | Depth-3 drop-COVID | Delta |
|---|---:|---:|---:|---:|
| Validation 2015-2022 | OOF R2 (log) | 0.742558 | 0.750061 | +0.007502 |
| Validation 2015-2022 | OOF R2 ($) | 0.745366 | 0.711094 | -0.034272 |
| Validation 2015-2022 | OOF MAE ($M) | 77.669 | 79.324 | +1.654 |
| Validation 2015-2022 | Median APE | 35.6% | 34.7% | -0.9 pp |
| Validation excl. 2020 | OOF R2 (log) | 0.775997 | 0.782605 | +0.006608 |
| Validation excl. 2020 | OOF R2 ($) | 0.747463 | 0.713429 | -0.034033 |
| Validation excl. 2020 | OOF MAE ($M) | 78.594 | 80.254 | +1.660 |
| Validation excl. 2020 | Median APE | 34.4% | 32.8% | -1.6 pp |
| Final 2023 | OOF R2 (log) | 0.699691 | 0.711468 | +0.011777 |
| Final 2023 | OOF R2 ($) | 0.467123 | 0.614490 | +0.147366 |
| Final 2023 | OOF MAE ($M) | 84.167 | 75.802 | -8.365 |
| Final 2023 | Median APE | 49.6% | 52.4% | +2.8 pp |

Decision: replace the default with `xgb-depth3-drop-covid`. The challenger is
better on validation log R2, the same metric optimized by training and led in
the README, and better on the 2023 final holdout across log R2, dollar R2, and
dollar MAE. The validation dollar-space regression and 2023 median APE regression
are known caveats rather than silent wins.
