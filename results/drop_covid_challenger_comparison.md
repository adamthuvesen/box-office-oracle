# Model Replacement Evidence

Snapshot: ignored `analysis/datasets_high`; 2,762 rows; 6 budget-imputation rows
dropped; 2,756 rows scored. Same yearly folds, same baseline builder.

Replacement bar: validation log R2 improves; 2023 log R2, dollar R2, and dollar
MAE are non-worse. Median APE is reported but is not the bar.

| Split | Metric | Previous default | New default | Delta |
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

Decision: replace the default with `xgb-depth3-drop-covid`.
