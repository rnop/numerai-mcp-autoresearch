# Numerai Weekly Report — 2026-W32 | Live Era 1223

**Model:** tailspin | **Built:** 2026-08-09 | **Era window:** 1077 – 1218 | **Live submission era:** 1223

---

## Feature Changes vs Previous Week

> Previous build: **2026-08-01** — era window 1084 – 1225

| | Count |
| --- | --- |
| Total features (current) | 120 |
| Added this week | 67 |
| Removed this week | 67 |
| Retained | 53 |

**Added** (67 features)

| Group | Count | Sample Features |
| --- | --- | --- |
| extra | 2 | `feature_imminent_unobserved_lengthening`, `feature_readier_reversed_accusal` |
| faith | 14 | `feature_demiurgic_hedgiest_plaque`, `feature_donnard_anile_barcarole`, `feature_exhalant_meteorological_excavator`, `feature_heathenish_phonotypic_internuncio`, +10 more |
| intelligence | 3 | `feature_esemplastic_droopier_scad`, `feature_splanchnic_notional_pint`, `feature_uncertificated_mat_evisceration` |
| quantum | 46 | `feature_agraphic_semifinished_withholder`, `feature_alaskan_equivocal_althea`, `feature_anecdotical_psephological_preventive`, `feature_bare_irrelievable_collimation`, +42 more |
| strength | 1 | `feature_discrete_bicuspidate_bricole` |
| wisdom | 1 | `feature_tongued_tricarpellary_inge` |

**Removed** (67 features)

| Group | Count | Sample Features |
| --- | --- | --- |
| extra | 1 | `feature_millennial_uncanonical_sunna` |
| faith | 22 | `feature_cased_nicene_lymphoma`, `feature_consecrate_untaxed_alexia`, `feature_constitutive_spanking_granulation`, `feature_counterbalanced_edified_idolism`, +18 more |
| quantum | 38 | `feature_advisory_environmental_canister`, `feature_antispasmodic_overpriced_gill`, `feature_askew_asserting_overactivity`, `feature_attributable_nationalistic_ascertainment`, +34 more |
| strength | 3 | `feature_choreic_sterilized_lagune`, `feature_debonnaire_opulent_stayer`, `feature_hurling_elastomeric_nanny` |
| wisdom | 3 | `feature_circulative_devolution_cittern`, `feature_heliconian_vociferant_cheechako`, `feature_radiosensitive_actable_radish` |

**Retained** (53 features)

| Group | Count | Sample Features |
| --- | --- | --- |
| extra | 1 | `feature_tonal_illuminating_porgy` |
| faith | 9 | `feature_aerodynamical_exhibitive_keyword`, `feature_attachable_martinique_beg`, `feature_crumb_archegonial_quayside`, `feature_exuvial_curdier_surfperch`, +5 more |
| intelligence | 4 | `feature_capreolate_philharmonic_mazzard`, `feature_flawier_oversized_sophism`, `feature_melismatic_daily_freak`, `feature_substitutive_lacerated_souchong` |
| quantum | 39 | `feature_acceleratory_purloined_balaklava`, `feature_antiquated_slanting_zeugma`, `feature_combust_barbaric_storyboard`, `feature_complicate_uninflamed_beautification`, +35 more |


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
| Training: CORR mean | 0.03071 |
| Training: MMC mean | 0.00607 |
| Training Sharpe | 2.359 |
| Validation: CORR mean | 0.02703 |
| Validation: MMC mean | 0.00629 |
| Validation Sharpe | 2.487 |

---

## Live Prediction QA


### Visualization

![Live prediction QA plot](../artifacts/live_prediction_distribution_train_1077_1218.png)

The chart combines the raw histogram, sorted prediction curve, benchmark exposure scatter, and percentile-ranked distribution for the current live batch.


**Distribution Check**

| Metric | Value |
| --- | --- |
| Verdict | PASS |
| Ready for submission | yes |
| Rows scored | 7142 |
| Prediction std | 0.00611 |
| Prediction p99-p01 spread | 0.02729 |
| Duplicate fraction | 0.00000 |
| Benchmark corr | 0.21045 |

| Check | Status | Details |
| --- | --- | --- |
| row_count | PASS | Scored 7,142 live rows. |
| dispersion | PASS | Prediction std is 0.006107. |
| tail_spread | PASS | Prediction p99-p01 spread is 0.027287. |
| duplicates | PASS | Duplicate prediction fraction is 0.000%. |
| benchmark_corr | PASS | abs corr(pred, v53_lgbm_ender20) is 0.210. |

| Artifact | Path |
| --- | --- |
| Distribution plot | `../artifacts/live_prediction_distribution_train_1077_1218.png` |
| Scored CSV | `../artifacts/live_predictions_train_1077_1218.csv` |
| Summary JSON | `../artifacts/live_prediction_distribution_train_1077_1218_summary.json` |

---

## Artifact Details

| Metric | Value |
| --- | --- |
| Built date | 2026-08-09 |
| Model type | XGBoost (GPU) |
| Best iteration | 1390 |
| Wall clock time | 254.6s |
| Pickle size | 1.27 MB |

---

## Training Configuration

| Parameter | Value |
| --- | --- |
| Target | `target_ender_60` |
| Era window | 1077 – 1218 |
| Era count | 142 |
| Lookback eras | 142 |
| Trailing eras (feature ranking) | 20 |
| Top-K features selected | 120 |
| Feature pool size | 1506 |
| Fit eras | 132 |
| Early stopping eras | 10 |
| Best iteration | 1390 |
| Benchmark neutralization | 0.1 vs `v53_lgbm_ender20` |

---

_Generated by numerai-weekly MCP on 2026-08-09 12:20._
