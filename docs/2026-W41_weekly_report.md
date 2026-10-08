# Numerai Weekly Report — 2026-W41 | Live Era 1228

**Model:** tailspin | **Built:** 2026-10-08 | **Era window:** 1086 – 1227 | **Live submission era:** 1228

---

## Feature Changes vs Previous Week

> Previous build: **2026-09-26** — era window 1084 – 1225

| | Count |
| --- | --- |
| Total features (current) | 120 |
| Added this week | 27 |
| Removed this week | 27 |
| Retained | 93 |

**Added** (27 features)

| Group | Count | Sample Features |
| --- | --- | --- |
| faith | 12 | `feature_cantonal_factorable_mordvin`, `feature_divisible_lengthening_brummagem`, `feature_divisive_welsh_mutualisation`, `feature_icy_jake_mix`, +8 more |
| quantum | 15 | `feature_anile_unlocated_forerunner`, `feature_awakening_polychrome_utu`, `feature_besetting_gargantuan_strip`, `feature_censurable_romance_cannery`, +11 more |

**Removed** (27 features)

| Group | Count | Sample Features |
| --- | --- | --- |
| faith | 9 | `feature_cased_nicene_lymphoma`, `feature_crumb_archegonial_quayside`, `feature_faustian_rescued_heterotopia`, `feature_monobasic_glairiest_quist`, +5 more |
| quantum | 16 | `feature_combust_barbaric_storyboard`, `feature_diarrheic_clustered_sherbet`, `feature_geodic_couth_zibet`, `feature_guileful_decrescendo_pyrite`, +12 more |
| strength | 1 | `feature_choreic_sterilized_lagune` |
| wisdom | 1 | `feature_unguessed_abroach_wingman` |

**Retained** (93 features)

| Group | Count | Sample Features |
| --- | --- | --- |
| extra | 2 | `feature_different_wilier_burweed`, `feature_tonal_illuminating_porgy` |
| faith | 6 | `feature_dreamiest_turki_greet`, `feature_independent_unliving_ballon`, `feature_ropy_eurythmical_genera`, `feature_spun_consolute_pyuria`, +2 more |
| intelligence | 6 | `feature_capreolate_philharmonic_mazzard`, `feature_flawier_oversized_sophism`, `feature_giddied_smooth_circumvallation`, `feature_melismatic_daily_freak`, +2 more |
| quantum | 73 | `feature_acceleratory_purloined_balaklava`, `feature_advisory_environmental_canister`, `feature_anecdotical_psephological_preventive`, `feature_antiquated_slanting_zeugma`, +69 more |
| strength | 1 | `feature_debonnaire_opulent_stayer` |
| wisdom | 5 | `feature_circulative_devolution_cittern`, `feature_heliconian_vociferant_cheechako`, `feature_sulphuric_unremitting_grammaticism`, `feature_unbeknown_volcanic_cyrano`, +1 more |


---

## Target Analysis

**Current target:** `target_ender_60`

This model is trained on `target_ender_60` as established by the v5.2 feature analysis. This target
was selected because it provides the best generalization for MMC in walk-forward testing.

> A dynamic target recommendation system is planned for a future update. Until then,
> `target_ender_60` remains the fixed default.

---

## Top Statistics

**Model Snapshot**

| Metric | Value |
| --- | --- |
| Live training target | `target_ender_60` |
| Validation target | `target_ender_20` |
| MMC benchmark | `v53_lgbm_ender20` |
| Training: CORR mean | 0.03107 |
| Training: MMC mean | 0.00643 |
| Training Sharpe | 2.486 |
| Validation: CORR mean | 0.01479 |
| Validation: MMC mean | 0.00225 |
| Validation Sharpe | 1.876 |

---

## Live Prediction QA


### Visualization

![Live prediction QA plot](../artifacts/live_prediction_distribution_train_1086_1227.png)

The chart combines the raw histogram, sorted prediction curve, benchmark exposure scatter, and percentile-ranked distribution for the current live batch.


**Distribution Check**

| Metric | Value |
| --- | --- |
| Verdict | PASS |
| Ready for submission | yes |
| Rows scored | 6902 |
| Prediction std | 0.00560 |
| Prediction p99-p01 spread | 0.02548 |
| Duplicate fraction | 0.00000 |
| Benchmark corr | 0.19365 |

| Check | Status | Details |
| --- | --- | --- |
| row_count | PASS | Scored 6,902 live rows. |
| dispersion | PASS | Prediction std is 0.005599. |
| tail_spread | PASS | Prediction p99-p01 spread is 0.025479. |
| duplicates | PASS | Duplicate prediction fraction is 0.000%. |
| benchmark_corr | PASS | abs corr(pred, v53_lgbm_ender20) is 0.194. |
| live_data_freshness | PASS | Live parquet is 0.0 days old. |

| Artifact | Path |
| --- | --- |
| Distribution plot | `../artifacts/live_prediction_distribution_train_1086_1227.png` |
| Scored CSV | `../artifacts/live_predictions_train_1086_1227.csv` |
| Summary JSON | `../artifacts/live_prediction_distribution_train_1086_1227_summary.json` |

---

## Artifact Details

| Metric | Value |
| --- | --- |
| Built date | 2026-10-08 |
| Model type | XGBoost (GPU) |
| Best iteration | 1037 |
| Wall clock time | 149.8s |
| Pickle size | 0.95 MB |

---

## Training Configuration

| Parameter | Value |
| --- | --- |
| Target | `target_ender_60` |
| Era window | 1086 – 1227 |
| Era count | 142 |
| Lookback eras | 142 |
| Trailing eras (feature ranking) | 20 |
| Top-K features selected | 120 |
| Feature pool size | 1506 |
| Fit eras | 132 |
| Early stopping eras | 10 |
| Best iteration | 1037 |
| Benchmark neutralization | 0.1 vs `v53_lgbm_ender20` |

---

_Generated by the weekly pipeline on 2026-10-08 09:14._
