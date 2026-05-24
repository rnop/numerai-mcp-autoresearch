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

- The strongest validation summary in the published slice is **target_teager2b_60** with Sharpe **0.155** and **55.5%** positive eras.
- The public uniqueness artifact surfaces **297** features outside the standard medium set, while the curated unique shortlist still highlights **51** `rain` candidates.
- The feature shortlist for `target_ender_60` shows why the live workflow leans into a broader pool than the default Numerai feature sets.

## Validation Group Summary

| target | mean | sharpe | positive eras |
| --- | --- | --- | --- |
| target_ender_20 | 0.00129 | 0.107 | 54.1% |
| target_ender_60 | 0.00125 | 0.106 | 53.4% |
| target_teager2b_20 | 0.00147 | 0.126 | 54.9% |
| target_teager2b_60 | 0.00180 | 0.155 | 55.5% |
| target_jasper_20 | 0.00113 | 0.095 | 53.9% |
| target_jasper_60 | 0.00144 | 0.133 | 54.8% |

## Top Recommended Features For `target_ender_60`

| feature | group | val mean | val sharpe | positive eras |
| --- | --- | --- | --- | --- |
| `feature_ishmaelitish_flauntiest_charley` | other | 0.01089 | 1.073 | 86.2% |
| `feature_brittonic_fortyish_alec` | other | 0.01143 | 1.067 | 86.2% |
| `feature_psychiatrical_sphenoid_galaxy` | other | 0.01293 | 0.982 | 86.2% |
| `feature_undivested_vitric_shareholder` | other | 0.01104 | 0.927 | 87.4% |
| `feature_succinct_indusiate_surfacing` | other | 0.01103 | 0.900 | 81.6% |
| `feature_veloce_vulnerary_aluminate` | other | 0.00991 | 0.859 | 79.3% |
| `feature_diplex_parabolic_conk` | other | 0.01063 | 0.771 | 79.3% |
| `feature_triumphal_contortional_brilliance` | other | 0.00848 | 0.754 | 77.0% |
| `feature_unreproached_abrasive_kate` | other | 0.00697 | 0.695 | 77.0% |
| `feature_exhalant_meteorological_excavator` | other | 0.00741 | 0.691 | 72.4% |

## Top Unique Features Worth Inspecting

| feature | group | combined score | val sharpe | in medium |
| --- | --- | --- | --- | --- |
| `feature_circulative_devolution_cittern` | wisdom | 1.091 | 1.172 | False |
| `feature_unguessed_abroach_wingman` | wisdom | 1.089 | 0.990 | False |
| `feature_transisthmian_disbelieving_grillage` | wisdom | 1.037 | 1.242 | False |
| `feature_bridal_fingered_pensioner` | sunshine | 0.958 | 1.102 | False |
| `feature_depressing_punitive_recuperation` | rain | 0.933 | 0.955 | False |
| `feature_stalworth_rotund_inflammability` | rain | 0.925 | 1.279 | False |
| `feature_imminent_unobserved_lengthening` | rain | 0.905 | 1.243 | False |
| `feature_twaddly_eleven_fustet` | sunshine | 0.864 | 0.886 | False |
| `feature_psychiatrical_sphenoid_galaxy` | other | 0.858 | 0.982 | False |
| `feature_gravitational_xeromorphic_myxoma` | rain | 0.857 | 1.176 | False |

## Target Inter-Correlation

These correlations help explain why several targets cluster together and why `target_ender_60` can act as a stable live-training anchor while `target_ender_20` remains the fixed CORR evaluation target.

|  | target_ender_20 | target_ender_60 | target_teager2b_20 | target_teager2b_60 | target_jasper_20 | target_jasper_60 |
| --- | --- | --- | --- | --- | --- | --- |
| target_ender_20 | 1.0 | 0.47045476531398617 | 0.7997595313729902 | 0.4363082949726886 | 0.795817704497426 | 0.4227105405466921 |
| target_ender_60 | 0.47045476531398617 | 1.0 | 0.43369346050214636 | 0.8025776899917662 | 0.42490177856755645 | 0.8002867894146204 |
| target_teager2b_20 | 0.7997595313729902 | 0.43369346050214636 | 1.0 | 0.4489853554396305 | 0.710701970112797 | 0.3904479597818473 |
| target_teager2b_60 | 0.4363082949726886 | 0.8025776899917662 | 0.4489853554396305 | 1.0 | 0.39435154176274767 | 0.716275554708 |
| target_jasper_20 | 0.795817704497426 | 0.42490177856755645 | 0.710701970112797 | 0.39435154176274767 | 1.0 | 0.4656525210301023 |
| target_jasper_60 | 0.4227105405466921 | 0.8002867894146204 | 0.3904479597818473 | 0.716275554708 | 0.4656525210301023 | 1.0 |

## Artifact Links

- Source folder: `artifacts/feature_analysis/`
- Shortlist CSV: `artifacts/feature_analysis/recommended_features_target_ender_60.csv`
- Unique shortlist CSV: `artifacts/feature_analysis/recommended_unique_target_ender_60.csv`
- Uniqueness scan: `artifacts/feature_analysis/uniqueness_report.csv`
- Correlation matrix: `artifacts/feature_analysis/target_intercorr.csv`
