# Feature Analysis Report

This report turns the public feature-analysis artifacts into a browser-readable summary. The raw CSVs remain in `artifacts/feature_analysis/`; this report is the narrative layer that explains what they imply for the live candidate pool.

## Why this exists

This folder contains the public-facing subset of the feature analysis used to
design the walk-forward candidate pool in the autoresearch workflow.

## Included artifacts

- `recommended_features_target_ender_60.csv`
  Public shortlist of strong features for the main live-training target.
- `recommended_unique_target_ender_60.csv`
  Features that were both strong and comparatively underrepresented in the
  standard Numerai medium set.
- `uniqueness_report.csv`
  Overlap analysis showing why `faith` and `rain` were interesting sources of
  novel signal.
- `target_intercorr.csv`
  Spearman inter-correlation of the six investigated targets.
- `group_summary_validation.csv`
  Compact validation summary used during group-level screening.

## Headline Takeaways

- The strongest validation summary in the published slice is **target_ender_60** with Sharpe **0.033** and **51.2%** positive eras.
- The public uniqueness artifact surfaces **963** features outside the standard medium set, while the curated unique shortlist still highlights **6** `rain` candidates.
- The feature shortlist for `target_ender_60` shows why the live workflow leans into a broader pool than the default Numerai feature sets.

## Validation Group Summary

| target | mean | sharpe | positive eras |
| --- | --- | --- | --- |
| target_ender_20 | 0.00026 | 0.024 | 50.9% |
| target_ender_60 | 0.00030 | 0.033 | 51.2% |
| target_teager2b_20 | 0.00024 | 0.023 | 50.9% |
| target_teager2b_60 | 0.00023 | 0.027 | 51.0% |
| target_jasper_20 | 0.00023 | 0.021 | 50.8% |
| target_jasper_60 | 0.00032 | 0.031 | 51.2% |

## Top Recommended Features For `target_ender_60`

| feature | group | val mean | val sharpe | positive eras |
| --- | --- | --- | --- | --- |
| `feature_shouldered_cliffier_chouse` | wisdom | 0.00935 | 1.421 | 90.8% |
| `feature_transisthmian_disbelieving_grillage` | wisdom | 0.01044 | 1.357 | 90.8% |
| `feature_circulative_devolution_cittern` | wisdom | 0.01107 | 1.343 | 90.8% |
| `feature_unguessed_abroach_wingman` | wisdom | 0.00983 | 1.110 | 87.4% |
| `feature_tongued_tricarpellary_inge` | wisdom | 0.00998 | 1.084 | 80.5% |
| `feature_lite_proportionable_mola` | faith | 0.01155 | 1.029 | 87.4% |
| `feature_uncleanly_streamy_gelatinoid` | wisdom | 0.00986 | 1.004 | 80.5% |
| `feature_platy_nonchromosomal_bounty` | wisdom | 0.00702 | 0.985 | 87.4% |
| `feature_disheveled_unmotherly_llandudno` | wisdom | 0.00612 | 0.961 | 81.6% |
| `feature_egotistical_carotid_irrationality` | constitution | 0.00660 | 0.936 | 81.6% |

## Top Unique Features Worth Inspecting

| feature | group | combined score | val sharpe | in medium |
| --- | --- | --- | --- | --- |
| `feature_circulative_devolution_cittern` | wisdom | 0.902 | 1.343 | False |
| `feature_transisthmian_disbelieving_grillage` | wisdom | 0.866 | 1.357 | False |
| `feature_unguessed_abroach_wingman` | wisdom | 0.836 | 1.110 | False |
| `feature_tongued_tricarpellary_inge` | wisdom | 0.722 | 1.084 | False |
| `feature_uncleanly_streamy_gelatinoid` | wisdom | 0.710 | 1.004 | False |
| `feature_brittonic_fortyish_alec` | faith | 0.634 | 0.928 | False |
| `feature_ungodliest_arkansan_gabriel` | wisdom | 0.616 | 0.578 | False |
| `feature_heliconian_vociferant_cheechako` | wisdom | 0.611 | 0.695 | False |
| `feature_touring_silicic_positivism` | quantum | 0.605 | 0.577 | False |
| `feature_donnard_groutier_twinkle` | rain | 0.582 | 0.488 | False |

## Target Inter-Correlation

These correlations help explain why several targets cluster together and why `target_ender_60` can act as a stable live-training anchor while `target_ender_20` remains the fixed CORR evaluation target.

|  | target_ender_20 | target_ender_60 | target_teager2b_20 | target_teager2b_60 | target_jasper_20 | target_jasper_60 |
| --- | --- | --- | --- | --- | --- | --- |
| target_ender_20 | 1.0 | 0.465935733664747 | 0.7937561181637971 | 0.4300340965328804 | 0.7864090296360152 | 0.41676657408070683 |
| target_ender_60 | 0.465935733664747 | 1.0 | 0.42941085244706195 | 0.793289709786667 | 0.41910490161144004 | 0.7913610752379271 |
| target_teager2b_20 | 0.7937561181637971 | 0.42941085244706195 | 1.0 | 0.4464478961584912 | 0.702020963388821 | 0.3853343271474052 |
| target_teager2b_60 | 0.4300340965328804 | 0.793289709786667 | 0.4464478961584912 | 1.0 | 0.3874822966373438 | 0.7070200772923233 |
| target_jasper_20 | 0.7864090296360152 | 0.41910490161144004 | 0.702020963388821 | 0.3874822966373438 | 1.0 | 0.4623422866233813 |
| target_jasper_60 | 0.41676657408070683 | 0.7913610752379271 | 0.3853343271474052 | 0.7070200772923233 | 0.4623422866233813 | 1.0 |

## Artifact Links

- Source folder: `artifacts/feature_analysis/`
- Shortlist CSV: `artifacts/feature_analysis/recommended_features_target_ender_60.csv`
- Unique shortlist CSV: `artifacts/feature_analysis/recommended_unique_target_ender_60.csv`
- Uniqueness scan: `artifacts/feature_analysis/uniqueness_report.csv`
- Correlation matrix: `artifacts/feature_analysis/target_intercorr.csv`
