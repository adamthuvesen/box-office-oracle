# IP-feature experiment (iterate mode, eval years 2015-2023)

> **Superseded (2026-07-08):** the E-variant features were adopted as feature
> contract v9 (`IP_TIER`, `PRIOR_FRANCHISE_GROSS_LOG`, `IS_FRANCHISE_FOLLOWUP`
> appended to `SELECTED_FEATURES`). Production franchise history is
> collection-keyed only (`box_office/franchise_history.py`); the
> umbrella-ip_name grouping used by variant C's franchise key in this report
> is retired. See `results/local_retrain/` for the adopted v9 numbers.

Experiment only — the v8 feature contract, saved artifacts, and
`results/local_retrain/` are untouched. All variants ran through the
identical leakage-fixed CV path (`TimeSeriesCrossValidator` with a
per-fold `FeaturePreprocessorHigh`); experimental columns are appended
after the 10 engineered v8 features. 2024-2025 were never evaluated
(spent confirmation set).

## Comparison

| Variant | n | Mean CV MAE (log) | Mean per-year R² (log) | Pooled median APE |
|---|---:|---:|---:|---:|
| baseline | 6080 | 0.7403 ± 0.0496 | 0.5575 | 55.7% |
| naive_ip_tier (LEAKY) | 6080 | 0.6827 ± 0.0530 | 0.6177 | 50.7% |
| time_safe_ip | 6080 | 0.7050 ± 0.0467 | 0.5930 | 52.8% |
| time_safe_plus_tier_prior | 6080 | 0.7054 ± 0.0491 | 0.5932 | 52.4% |

## Per-year: baseline

| Year | n | MAE (log) | R² (log) | Spearman | Median APE |
|---:|---:|---:|---:|---:|---:|
| 2015 | 158 | 0.6730 | 0.6173 | 0.7561 | 48.0% |
| 2016 | 181 | 0.7049 | 0.6095 | 0.7651 | 52.4% |
| 2017 | 164 | 0.7296 | 0.6256 | 0.7546 | 55.6% |
| 2018 | 164 | 0.6905 | 0.6339 | 0.7656 | 51.0% |
| 2019 | 132 | 0.6967 | 0.6024 | 0.7267 | 51.4% |
| 2020 | 54 | 0.7949 | 0.3540 | 0.6512 | 75.1% |
| 2021 | 90 | 0.7935 | 0.4174 | 0.6090 | 62.8% |
| 2022 | 101 | 0.8139 | 0.5731 | 0.7429 | 65.5% |
| 2023 | 117 | 0.7658 | 0.5845 | 0.7724 | 56.8% |

## Per-year: naive_ip_tier

| Year | n | MAE (log) | R² (log) | Spearman | Median APE |
|---:|---:|---:|---:|---:|---:|
| 2015 | 158 | 0.5982 | 0.6860 | 0.7981 | 43.4% |
| 2016 | 181 | 0.6647 | 0.6504 | 0.7859 | 52.1% |
| 2017 | 164 | 0.6520 | 0.6793 | 0.7800 | 45.3% |
| 2018 | 164 | 0.6447 | 0.6961 | 0.8001 | 49.1% |
| 2019 | 132 | 0.6473 | 0.6541 | 0.7656 | 46.1% |
| 2020 | 54 | 0.7181 | 0.4192 | 0.6665 | 62.1% |
| 2021 | 90 | 0.7152 | 0.5263 | 0.6671 | 52.4% |
| 2022 | 101 | 0.7856 | 0.6050 | 0.7384 | 62.3% |
| 2023 | 117 | 0.7183 | 0.6428 | 0.7943 | 55.1% |

## Per-year: time_safe_ip

| Year | n | MAE (log) | R² (log) | Spearman | Median APE |
|---:|---:|---:|---:|---:|---:|
| 2015 | 158 | 0.6285 | 0.6626 | 0.7849 | 47.0% |
| 2016 | 181 | 0.6746 | 0.6390 | 0.7829 | 48.5% |
| 2017 | 164 | 0.7017 | 0.6435 | 0.7584 | 55.9% |
| 2018 | 164 | 0.6867 | 0.6436 | 0.7635 | 49.6% |
| 2019 | 132 | 0.6606 | 0.6430 | 0.7533 | 48.9% |
| 2020 | 54 | 0.7470 | 0.3964 | 0.6807 | 64.2% |
| 2021 | 90 | 0.7062 | 0.5182 | 0.6916 | 52.7% |
| 2022 | 101 | 0.7823 | 0.5902 | 0.7381 | 59.9% |
| 2023 | 117 | 0.7577 | 0.6006 | 0.7748 | 55.7% |

## Per-year: time_safe_plus_tier_prior

| Year | n | MAE (log) | R² (log) | Spearman | Median APE |
|---:|---:|---:|---:|---:|---:|
| 2015 | 158 | 0.6224 | 0.6661 | 0.7905 | 46.9% |
| 2016 | 181 | 0.6793 | 0.6392 | 0.7805 | 53.6% |
| 2017 | 164 | 0.7040 | 0.6403 | 0.7573 | 55.4% |
| 2018 | 164 | 0.6798 | 0.6489 | 0.7679 | 49.1% |
| 2019 | 132 | 0.6593 | 0.6414 | 0.7491 | 45.7% |
| 2020 | 54 | 0.7295 | 0.4156 | 0.6680 | 62.8% |
| 2021 | 90 | 0.7267 | 0.5043 | 0.6692 | 56.6% |
| 2022 | 101 | 0.7959 | 0.5805 | 0.7282 | 61.4% |
| 2023 | 117 | 0.7514 | 0.6023 | 0.7808 | 51.0% |

## Leakage in variant B (naive_ip_tier)

`ip_tier` derives partly from TOTAL collection gross
(data/ip_rules.yml `tier_thresholds.collection_gross`), which includes
each movie's own gross and its future sequels' gross — hindsight the
model cannot have at release time. Variant C rebuilds the same signal
using only franchise history strictly before each movie's release
date, so the B-vs-C gap measures the leak inflation:

- B mean R² (log): 0.6177; C: 0.5930; gap: +0.0247
- B mean CV MAE (log): 0.6827; C: 0.7050; gap: -0.0224

## Limitations

- First films of a franchise get 0/0/0 in C/D — correct and honest:
  their IP awareness from books/games/toys is real but not measurable
  from our data without hindsight.
- Variant D's BRAND_NONFILM_TIER only covers umbrella brands in
  ip_rules.yml with a non-film origin; the brand list itself was
  curated in 2026, so a residual hindsight caveat remains (we know
  today which toy/game/book brands got movies).
- Franchise coverage: 1679 of 6080 rows have a franchise key; 874 are follow-ups; 185 carry a non-film brand tier.

## Recommendation

- **REJECT C (time_safe_ip)**: R² gain +0.0355, MAE gain +0.0353 vs baseline std 0.0496 (accept requires beating baseline on R² AND MAE by more than one std).
- **REJECT D (time_safe_plus_tier_prior)**: R² gain +0.0357, MAE gain +0.0349 vs baseline std 0.0496 (accept requires beating baseline on R² AND MAE by more than one std).

---

# Restructured time-safe ip_tier run (2026-07-08, eval years 2015-2023)

Re-run after restructuring the tier system: `ip_tier` is now pre-sold
magnitude at release (as-of-date `tier_by_period` brand rules,
prior-franchise gross strictly before release, `source_works` rules).
The total-collection-gross thresholds are abolished, so the old leaky
variant B no longer exists. Same leakage-fixed CV path as above;
2024-2025 were never evaluated.

Variants: A `baseline` (10 v8 features), C `time_safe_ip` (baseline +
PRIOR_FRANCHISE_GROSS_LOG + PRIOR_FRANCHISE_FILM_COUNT +
IS_FRANCHISE_FOLLOWUP), E `time_safe_tier` (baseline + new ordinal
IP_TIER_TIME_SAFE + IS_FRANCHISE_FOLLOWUP + PRIOR_FRANCHISE_GROSS_LOG).

## Comparison

| Variant | n | Mean CV MAE (log) | Mean per-year R² (log) | Pooled median APE |
|---|---:|---:|---:|---:|
| baseline | 6080 | 0.7403 ± 0.0496 | 0.5575 | 55.7% |
| time_safe_ip | 6080 | 0.7054 ± 0.0455 | 0.5904 | 52.1% |
| time_safe_tier | 6080 | 0.6988 ± 0.0465 | 0.5999 | 51.7% |

## Per-year: baseline

| Year | n | MAE (log) | R² (log) | Spearman | Median APE |
|---:|---:|---:|---:|---:|---:|
| 2015 | 158 | 0.6730 | 0.6173 | 0.7561 | 48.0% |
| 2016 | 181 | 0.7049 | 0.6095 | 0.7651 | 52.4% |
| 2017 | 164 | 0.7296 | 0.6256 | 0.7546 | 55.6% |
| 2018 | 164 | 0.6905 | 0.6339 | 0.7656 | 51.0% |
| 2019 | 132 | 0.6967 | 0.6024 | 0.7267 | 51.4% |
| 2020 | 54 | 0.7949 | 0.3540 | 0.6512 | 75.1% |
| 2021 | 90 | 0.7935 | 0.4174 | 0.6090 | 62.8% |
| 2022 | 101 | 0.8139 | 0.5731 | 0.7429 | 65.5% |
| 2023 | 117 | 0.7658 | 0.5845 | 0.7724 | 56.8% |

## Per-year: time_safe_ip

| Year | n | MAE (log) | R² (log) | Spearman | Median APE |
|---:|---:|---:|---:|---:|---:|
| 2015 | 158 | 0.6285 | 0.6626 | 0.7849 | 47.0% |
| 2016 | 181 | 0.6795 | 0.6342 | 0.7786 | 48.5% |
| 2017 | 164 | 0.7013 | 0.6410 | 0.7577 | 53.7% |
| 2018 | 164 | 0.6856 | 0.6424 | 0.7647 | 49.9% |
| 2019 | 132 | 0.6613 | 0.6419 | 0.7518 | 48.6% |
| 2020 | 54 | 0.7458 | 0.3879 | 0.6726 | 62.3% |
| 2021 | 90 | 0.7118 | 0.5084 | 0.6793 | 51.4% |
| 2022 | 101 | 0.7805 | 0.5932 | 0.7416 | 59.5% |
| 2023 | 117 | 0.7539 | 0.6016 | 0.7761 | 54.1% |

## Per-year: time_safe_tier

| Year | n | MAE (log) | R² (log) | Spearman | Median APE |
|---:|---:|---:|---:|---:|---:|
| 2015 | 158 | 0.6160 | 0.6802 | 0.8001 | 46.7% |
| 2016 | 181 | 0.6772 | 0.6351 | 0.7781 | 50.3% |
| 2017 | 164 | 0.6899 | 0.6494 | 0.7627 | 53.6% |
| 2018 | 164 | 0.6783 | 0.6511 | 0.7680 | 48.8% |
| 2019 | 132 | 0.6648 | 0.6430 | 0.7528 | 46.7% |
| 2020 | 54 | 0.7265 | 0.4234 | 0.6801 | 56.1% |
| 2021 | 90 | 0.7088 | 0.5229 | 0.6785 | 51.6% |
| 2022 | 101 | 0.7877 | 0.5851 | 0.7227 | 59.2% |
| 2023 | 117 | 0.7404 | 0.6092 | 0.7746 | 55.8% |

## Diagnostics

- Franchise coverage: 1686 of 6080 rows have a franchise key; 877 are follow-ups.
- New ip_tier counts in the training frame: {1: 197, 2: 155, 3: 360, 4: 1252, 5: 4116}.

## Recommendation

- **REJECT C (time_safe_ip)**: R² gain +0.0328, MAE gain +0.0349 vs baseline std 0.0496 (accept requires beating baseline on R² AND MAE by more than one std).
- **REJECT E (time_safe_tier)**: R² gain +0.0424, MAE gain +0.0415 vs baseline std 0.0496 (accept requires beating baseline on R² AND MAE by more than one std).

---

# Restructured time-safe ip_tier run (2026-07-08, eval years 2015-2023)

Re-run after restructuring the tier system: `ip_tier` is now pre-sold
magnitude at release (as-of-date `tier_by_period` brand rules,
prior-franchise gross strictly before release, `source_works` rules).
The total-collection-gross thresholds are abolished, so the old leaky
variant B no longer exists. Same leakage-fixed CV path as above;
2024-2025 were never evaluated.

Variants: A `baseline` (10 v8 features), C `time_safe_ip` (baseline +
PRIOR_FRANCHISE_GROSS_LOG + PRIOR_FRANCHISE_FILM_COUNT +
IS_FRANCHISE_FOLLOWUP), E `time_safe_tier` (baseline + new ordinal
IP_TIER_TIME_SAFE + IS_FRANCHISE_FOLLOWUP + PRIOR_FRANCHISE_GROSS_LOG).

## Comparison

| Variant | n | Mean CV MAE (log) | Mean per-year R² (log) | Pooled median APE |
|---|---:|---:|---:|---:|
| baseline | 6080 | 0.7403 ± 0.0496 | 0.5575 | 55.7% |
| time_safe_ip | 6080 | 0.7054 ± 0.0455 | 0.5904 | 52.1% |
| time_safe_tier | 6080 | 0.7003 ± 0.0452 | 0.5985 | 51.3% |

## Per-year: baseline

| Year | n | MAE (log) | R² (log) | Spearman | Median APE |
|---:|---:|---:|---:|---:|---:|
| 2015 | 158 | 0.6730 | 0.6173 | 0.7561 | 48.0% |
| 2016 | 181 | 0.7049 | 0.6095 | 0.7651 | 52.4% |
| 2017 | 164 | 0.7296 | 0.6256 | 0.7546 | 55.6% |
| 2018 | 164 | 0.6905 | 0.6339 | 0.7656 | 51.0% |
| 2019 | 132 | 0.6967 | 0.6024 | 0.7267 | 51.4% |
| 2020 | 54 | 0.7949 | 0.3540 | 0.6512 | 75.1% |
| 2021 | 90 | 0.7935 | 0.4174 | 0.6090 | 62.8% |
| 2022 | 101 | 0.8139 | 0.5731 | 0.7429 | 65.5% |
| 2023 | 117 | 0.7658 | 0.5845 | 0.7724 | 56.8% |

## Per-year: time_safe_ip

| Year | n | MAE (log) | R² (log) | Spearman | Median APE |
|---:|---:|---:|---:|---:|---:|
| 2015 | 158 | 0.6285 | 0.6626 | 0.7849 | 47.0% |
| 2016 | 181 | 0.6795 | 0.6342 | 0.7786 | 48.5% |
| 2017 | 164 | 0.7013 | 0.6410 | 0.7577 | 53.7% |
| 2018 | 164 | 0.6856 | 0.6424 | 0.7647 | 49.9% |
| 2019 | 132 | 0.6613 | 0.6419 | 0.7518 | 48.6% |
| 2020 | 54 | 0.7458 | 0.3879 | 0.6726 | 62.3% |
| 2021 | 90 | 0.7118 | 0.5084 | 0.6793 | 51.4% |
| 2022 | 101 | 0.7805 | 0.5932 | 0.7416 | 59.5% |
| 2023 | 117 | 0.7539 | 0.6016 | 0.7761 | 54.1% |

## Per-year: time_safe_tier

| Year | n | MAE (log) | R² (log) | Spearman | Median APE |
|---:|---:|---:|---:|---:|---:|
| 2015 | 158 | 0.6230 | 0.6727 | 0.7934 | 46.4% |
| 2016 | 181 | 0.6794 | 0.6347 | 0.7774 | 52.5% |
| 2017 | 164 | 0.6895 | 0.6459 | 0.7586 | 52.7% |
| 2018 | 164 | 0.6807 | 0.6495 | 0.7661 | 48.2% |
| 2019 | 132 | 0.6615 | 0.6426 | 0.7499 | 44.7% |
| 2020 | 54 | 0.7267 | 0.4229 | 0.6804 | 56.2% |
| 2021 | 90 | 0.7147 | 0.5203 | 0.6821 | 52.8% |
| 2022 | 101 | 0.7867 | 0.5885 | 0.7286 | 58.6% |
| 2023 | 117 | 0.7400 | 0.6089 | 0.7750 | 54.8% |

## Diagnostics

- Franchise coverage: 1686 of 6080 rows have a franchise key; 877 are follow-ups.
- New ip_tier counts in the training frame: {1: 168, 2: 176, 3: 368, 4: 1253, 5: 4115}.

## Recommendation

- **REJECT C (time_safe_ip)**: R² gain +0.0328, MAE gain +0.0349 vs baseline std 0.0496 (accept requires beating baseline on R² AND MAE by more than one std).
- **REJECT E (time_safe_tier)**: R² gain +0.0409, MAE gain +0.0400 vs baseline std 0.0496 (accept requires beating baseline on R² AND MAE by more than one std).
