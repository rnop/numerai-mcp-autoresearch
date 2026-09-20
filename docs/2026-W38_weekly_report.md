# Numerai Weekly Report — 2026-W38 | Live Era 1225

**Model:** tailspin | **Built:** 2026-09-20 | **Era window:** 1083 – 1224 | **Live submission era:** 1225

---

## Feature Changes vs Previous Week

> Previous build: **2026-09-12** — era window 1082 – 1223

| | Count |
| --- | --- |
| Total features (current) | 120 |
| Added this week | 11 |
| Removed this week | 11 |
| Retained | 109 |

**Added** (11 features)

| Group | Count | Sample Features |
| --- | --- | --- |
| faith | 4 | `feature_crumb_archegonial_quayside`, `feature_dreamiest_turki_greet`, `feature_shaped_pinnatiped_activating`, `feature_throatier_saxatile_spermatorrhoea` |
| quantum | 7 | `feature_coccal_unemotional_hyponasty`, `feature_exsiccative_canalicular_mono`, `feature_hatted_exaggerative_gamba`, `feature_primatial_thirsty_anbury`, +3 more |

**Removed** (11 features)

| Group | Count | Sample Features |
| --- | --- | --- |
| extra | 1 | `feature_stalworth_rotund_inflammability` |
| faith | 2 | `feature_infuscate_taming_totalitarianism`, `feature_unbending_expandable_slew` |
| quantum | 7 | `feature_hierological_unbrotherly_kroo`, `feature_jaundiced_polar_joyance`, `feature_orphan_epidermic_spade`, `feature_sultanic_jaculatory_cot`, +3 more |
| wisdom | 1 | `feature_sulphuric_unremitting_grammaticism` |

**Retained** (109 features)

| Group | Count | Sample Features |
| --- | --- | --- |
| extra | 5 | `feature_different_wilier_burweed`, `feature_imminent_unobserved_lengthening`, `feature_readier_reversed_accusal`, `feature_tonal_illuminating_porgy`, +1 more |
| faith | 13 | `feature_cased_nicene_lymphoma`, `feature_faustian_rescued_heterotopia`, `feature_independent_unliving_ballon`, `feature_kirtled_gobelin_aerie`, +9 more |
| intelligence | 7 | `feature_capreolate_philharmonic_mazzard`, `feature_esemplastic_droopier_scad`, `feature_flawier_oversized_sophism`, `feature_giddied_smooth_circumvallation`, +3 more |
| quantum | 79 | `feature_acceleratory_purloined_balaklava`, `feature_advisory_environmental_canister`, `feature_anecdotical_psephological_preventive`, `feature_antiquated_slanting_zeugma`, +75 more |
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
| Training: CORR mean | 0.03163 |
| Training: MMC mean | 0.00648 |
| Training Sharpe | 2.623 |
| Validation: CORR mean | 0.01846 |
| Validation: MMC mean | 0.00398 |
| Validation Sharpe | 1.483 |

---

## Live Prediction QA


### Visualization

![Live prediction QA plot](../artifacts/live_prediction_distribution_train_1083_1224.png)

The chart combines the raw histogram, sorted prediction curve, benchmark exposure scatter, and percentile-ranked distribution for the current live batch.


**Distribution Check**

| Metric | Value |
| --- | --- |
| Verdict | PASS |
| Ready for submission | yes |
| Rows scored | 6832 |
| Prediction std | 0.00627 |
| Prediction p99-p01 spread | 0.02825 |
| Duplicate fraction | 0.00000 |
| Benchmark corr | 0.20926 |

| Check | Status | Details |
| --- | --- | --- |
| row_count | PASS | Scored 6,832 live rows. |
| dispersion | PASS | Prediction std is 0.006268. |
| tail_spread | PASS | Prediction p99-p01 spread is 0.028254. |
| duplicates | PASS | Duplicate prediction fraction is 0.000%. |
| benchmark_corr | PASS | abs corr(pred, v53_lgbm_ender20) is 0.209. |
| live_data_freshness | PASS | Live parquet is 0.0 days old. |

| Artifact | Path |
| --- | --- |
| Distribution plot | `../artifacts/live_prediction_distribution_train_1083_1224.png` |
| Scored CSV | `../artifacts/live_predictions_train_1083_1224.csv` |
| Summary JSON | `../artifacts/live_prediction_distribution_train_1083_1224_summary.json` |

---

## Artifact Details

| Metric | Value |
| --- | --- |
| Built date | 2026-09-20 |
| Model type | XGBoost (GPU) |
| Best iteration | 1313 |
| Wall clock time | 116.9s |
| Pickle size | 1.2 MB |

---

## Training Configuration

| Parameter | Value |
| --- | --- |
| Target | `target_ender_60` |
| Era window | 1083 – 1224 |
| Era count | 142 |
| Lookback eras | 142 |
| Trailing eras (feature ranking) | 20 |
| Top-K features selected | 120 |
| Feature pool size | 1506 |
| Fit eras | 132 |
| Early stopping eras | 10 |
| Best iteration | 1313 |
| Benchmark neutralization | 0.1 vs `v53_lgbm_ender20` |

---

_Generated by numerai-weekly MCP on 2026-09-20 15:49._
