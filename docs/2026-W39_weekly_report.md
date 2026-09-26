# Numerai Weekly Report — 2026-W39 | Live Era 1226

**Model:** tailspin | **Built:** 2026-09-26 | **Era window:** 1084 – 1225 | **Live submission era:** 1226

---

## Feature Changes vs Previous Week

> Previous build: **2026-09-20** — era window 1083 – 1224

| | Count |
| --- | --- |
| Total features (current) | 120 |
| Added this week | 13 |
| Removed this week | 13 |
| Retained | 107 |

**Added** (13 features)

| Group | Count | Sample Features |
| --- | --- | --- |
| faith | 1 | `feature_spun_consolute_pyuria` |
| quantum | 9 | `feature_antispasmodic_overpriced_gill`, `feature_askew_asserting_overactivity`, `feature_flutier_shrinkable_cavalryman`, `feature_gaulish_lycanthropic_legislating`, +5 more |
| wisdom | 3 | `feature_sulphuric_unremitting_grammaticism`, `feature_unbeknown_volcanic_cyrano`, `feature_unpicked_wet_drambuie` |

**Removed** (13 features)

| Group | Count | Sample Features |
| --- | --- | --- |
| extra | 3 | `feature_imminent_unobserved_lengthening`, `feature_readier_reversed_accusal`, `feature_unchecked_parented_ngultrum` |
| faith | 3 | `feature_kirtled_gobelin_aerie`, `feature_phasic_unabridged_lactose`, `feature_snod_accostable_bagman` |
| intelligence | 1 | `feature_esemplastic_droopier_scad` |
| quantum | 6 | `feature_clarino_isocratic_advertising`, `feature_dizygotic_humming_mademoiselle`, `feature_headfirst_archducal_stockinette`, `feature_hindward_amoeboid_belladonna`, +2 more |

**Retained** (107 features)

| Group | Count | Sample Features |
| --- | --- | --- |
| extra | 2 | `feature_different_wilier_burweed`, `feature_tonal_illuminating_porgy` |
| faith | 14 | `feature_cased_nicene_lymphoma`, `feature_crumb_archegonial_quayside`, `feature_dreamiest_turki_greet`, `feature_faustian_rescued_heterotopia`, +10 more |
| intelligence | 6 | `feature_capreolate_philharmonic_mazzard`, `feature_flawier_oversized_sophism`, `feature_giddied_smooth_circumvallation`, `feature_melismatic_daily_freak`, +2 more |
| quantum | 80 | `feature_acceleratory_purloined_balaklava`, `feature_advisory_environmental_canister`, `feature_anecdotical_psephological_preventive`, `feature_antiquated_slanting_zeugma`, +76 more |
| strength | 2 | `feature_choreic_sterilized_lagune`, `feature_debonnaire_opulent_stayer` |
| wisdom | 3 | `feature_circulative_devolution_cittern`, `feature_heliconian_vociferant_cheechako`, `feature_unguessed_abroach_wingman` |


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
| Training: CORR mean | 0.02968 |
| Training: MMC mean | 0.00602 |
| Training Sharpe | 2.322 |
| Validation: CORR mean | 0.01675 |
| Validation: MMC mean | 0.00307 |
| Validation Sharpe | 1.309 |

---

## Live Prediction QA


### Visualization

![Live prediction QA plot](../artifacts/live_prediction_distribution_train_1084_1225.png)

The chart combines the raw histogram, sorted prediction curve, benchmark exposure scatter, and percentile-ranked distribution for the current live batch.


**Distribution Check**

| Metric | Value |
| --- | --- |
| Verdict | PASS |
| Ready for submission | yes |
| Rows scored | 6923 |
| Prediction std | 0.00551 |
| Prediction p99-p01 spread | 0.02488 |
| Duplicate fraction | 0.00000 |
| Benchmark corr | 0.20778 |

| Check | Status | Details |
| --- | --- | --- |
| row_count | PASS | Scored 6,923 live rows. |
| dispersion | PASS | Prediction std is 0.005512. |
| tail_spread | PASS | Prediction p99-p01 spread is 0.024885. |
| duplicates | PASS | Duplicate prediction fraction is 0.000%. |
| benchmark_corr | PASS | abs corr(pred, v53_lgbm_ender20) is 0.208. |
| live_data_freshness | PASS | Live parquet is 0.0 days old. |

| Artifact | Path |
| --- | --- |
| Distribution plot | `../artifacts/live_prediction_distribution_train_1084_1225.png` |
| Scored CSV | `../artifacts/live_predictions_train_1084_1225.csv` |
| Summary JSON | `../artifacts/live_prediction_distribution_train_1084_1225_summary.json` |

---

## Artifact Details

| Metric | Value |
| --- | --- |
| Built date | 2026-09-26 |
| Model type | XGBoost (GPU) |
| Best iteration | 1024 |
| Wall clock time | 119.7s |
| Pickle size | 0.94 MB |

---

## Training Configuration

| Parameter | Value |
| --- | --- |
| Target | `target_ender_60` |
| Era window | 1084 – 1225 |
| Era count | 142 |
| Lookback eras | 142 |
| Trailing eras (feature ranking) | 20 |
| Top-K features selected | 120 |
| Feature pool size | 1506 |
| Fit eras | 132 |
| Early stopping eras | 10 |
| Best iteration | 1024 |
| Benchmark neutralization | 0.1 vs `v53_lgbm_ender20` |

---

_Generated by numerai-weekly MCP on 2026-09-26 14:59._
