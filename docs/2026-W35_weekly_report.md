# Numerai Weekly Report — 2026-W35 | Live Era 1222

**Model:** tailspin | **Built:** 2026-08-30 | **Era window:** 1080 – 1221 | **Live submission era:** 1222

---

## Feature Changes vs Previous Week

> Previous build: **2026-08-24** — era window 1079 – 1220

| | Count |
| --- | --- |
| Total features (current) | 120 |
| Added this week | 17 |
| Removed this week | 17 |
| Retained | 103 |

**Added** (17 features)

| Group | Count | Sample Features |
| --- | --- | --- |
| extra | 1 | `feature_different_wilier_burweed` |
| faith | 4 | `feature_cased_nicene_lymphoma`, `feature_infuscate_taming_totalitarianism`, `feature_monobasic_glairiest_quist`, `feature_peppercorny_archilochian_laggen` |
| quantum | 9 | `feature_attributable_nationalistic_ascertainment`, `feature_chelonian_irrecoverable_zillion`, `feature_comate_impersonal_grysbok`, `feature_imagined_traditionalist_glia`, +5 more |
| strength | 2 | `feature_choreic_sterilized_lagune`, `feature_debonnaire_opulent_stayer` |
| wisdom | 1 | `feature_heliconian_vociferant_cheechako` |

**Removed** (17 features)

| Group | Count | Sample Features |
| --- | --- | --- |
| extra | 1 | `feature_stalworth_rotund_inflammability` |
| faith | 3 | `feature_heathenish_phonotypic_internuncio`, `feature_louche_referenced_esculent`, `feature_stimulative_rueful_exergue` |
| intelligence | 1 | `feature_splanchnic_notional_pint` |
| quantum | 11 | `feature_alaskan_equivocal_althea`, `feature_bare_irrelievable_collimation`, `feature_crisp_concretive_skit`, `feature_diffractive_forehand_archduchy`, +7 more |
| wisdom | 1 | `feature_uncleanly_streamy_gelatinoid` |

**Retained** (103 features)

| Group | Count | Sample Features |
| --- | --- | --- |
| extra | 3 | `feature_imminent_unobserved_lengthening`, `feature_readier_reversed_accusal`, `feature_tonal_illuminating_porgy` |
| faith | 17 | `feature_aerodynamical_exhibitive_keyword`, `feature_attachable_martinique_beg`, `feature_demiurgic_hedgiest_plaque`, `feature_exuvial_curdier_surfperch`, +13 more |
| intelligence | 6 | `feature_capreolate_philharmonic_mazzard`, `feature_esemplastic_droopier_scad`, `feature_flawier_oversized_sophism`, `feature_melismatic_daily_freak`, +2 more |
| quantum | 74 | `feature_acceleratory_purloined_balaklava`, `feature_advisory_environmental_canister`, `feature_agraphic_semifinished_withholder`, `feature_anecdotical_psephological_preventive`, +70 more |
| strength | 1 | `feature_discrete_bicuspidate_bricole` |
| wisdom | 2 | `feature_circulative_devolution_cittern`, `feature_unguessed_abroach_wingman` |


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
| Training: CORR mean | 0.03393 |
| Training: MMC mean | 0.00708 |
| Training Sharpe | 2.858 |
| Validation: CORR mean | 0.03033 |
| Validation: MMC mean | 0.00696 |
| Validation Sharpe | 3.087 |

---

## Live Prediction QA


### Visualization

![Live prediction QA plot](../artifacts/live_prediction_distribution_train_1080_1221.png)

The chart combines the raw histogram, sorted prediction curve, benchmark exposure scatter, and percentile-ranked distribution for the current live batch.


**Distribution Check**

| Metric | Value |
| --- | --- |
| Verdict | PASS |
| Ready for submission | yes |
| Rows scored | 7142 |
| Prediction std | 0.00715 |
| Prediction p99-p01 spread | 0.03246 |
| Duplicate fraction | 0.00000 |
| Benchmark corr | 0.18180 |

| Check | Status | Details |
| --- | --- | --- |
| row_count | PASS | Scored 7,142 live rows. |
| dispersion | PASS | Prediction std is 0.007149. |
| tail_spread | PASS | Prediction p99-p01 spread is 0.032461. |
| duplicates | PASS | Duplicate prediction fraction is 0.000%. |
| benchmark_corr | PASS | abs corr(pred, v53_lgbm_ender20) is 0.182. |

| Artifact | Path |
| --- | --- |
| Distribution plot | `../artifacts/live_prediction_distribution_train_1080_1221.png` |
| Scored CSV | `../artifacts/live_predictions_train_1080_1221.csv` |
| Summary JSON | `../artifacts/live_prediction_distribution_train_1080_1221_summary.json` |

---

## Artifact Details

| Metric | Value |
| --- | --- |
| Built date | 2026-08-30 |
| Model type | XGBoost (GPU) |
| Best iteration | 1884 |
| Wall clock time | 131.8s |
| Pickle size | 1.71 MB |

---

## Training Configuration

| Parameter | Value |
| --- | --- |
| Target | `target_ender_60` |
| Era window | 1080 – 1221 |
| Era count | 142 |
| Lookback eras | 142 |
| Trailing eras (feature ranking) | 20 |
| Top-K features selected | 120 |
| Feature pool size | 1506 |
| Fit eras | 132 |
| Early stopping eras | 10 |
| Best iteration | 1884 |
| Benchmark neutralization | 0.1 vs `v53_lgbm_ender20` |

---

_Generated by numerai-weekly MCP on 2026-08-30 19:59._
