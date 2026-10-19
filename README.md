# SCALE-BUS experimental code

This repository contains the experimental code for **“SCALE-BUS:
Metrologically characterized multiscale lesion measurements across
scanner-heterogeneous breast-ultrasound cohorts.”**

It includes feature extraction, dataset-integrity checks, grouped validation,
source-only transfer, locked external evaluation, ablation and sensitivity
analyses, deep comparators, and deterministic tests.

## Layout

- `src/fractal_extrema/`: regional-extrema, generalized-fractal, wavelet,
  boundary, multizone, and component-tree feature extraction.
- `scripts/`: experiment and analysis entry points.
- `tests/`: deterministic smoke tests using synthetic input.

Names containing `v04`, `v05`, or `v06` identify frozen stages of the
experimental sequence and are retained because the scripts import one another
by those names.

## Core environment

The handcrafted experiments used Python 3.13.5.

Windows PowerShell:

```powershell
py -3.13 -m venv .venv-core
.\.venv-core\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-core.txt
python -m pip install --no-deps -e .
python -m unittest discover -s tests -v
```

Linux or macOS:

```bash
python3 -m venv .venv-core
source .venv-core/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-core.txt
python -m pip install --no-deps -e .
python -m unittest discover -s tests -v
```

## Deep-learning environment

The frozen-embedding and fine-tuned comparators used a separate Python 3.10.19
environment with PyTorch 2.5.1, torchvision 0.20.1, and scikit-learn 1.7.2.
Install `requirements-deep.txt` in a separate environment. Select the PyTorch
wheel appropriate for the local CPU or CUDA installation. Fine-tuning supports
CPU or CUDA; pretrained embedding extraction requires CUDA and downloads the
selected torchvision weights on first use.

## Data access

Original medical images and masks are not redistributed. Obtain them from the
source repositories and follow their licenses and citation requirements:

- BUSI curated derivative: https://doi.org/10.5281/zenodo.21128640
- BUS-UCLM version 3: https://doi.org/10.17632/7fvgj4jsp7.3
- BUS-BRA: https://doi.org/10.5281/zenodo.8231412
- BrEaST / BREAST-LESIONS-USG: https://doi.org/10.7937/9WKK-Q141

Place the downloaded materials below `data/raw/` using the relative paths
defined near the top of the corresponding audit scripts. Run every command
from the repository root. Generated features and results are written below
`results/`, which is intentionally excluded from version control.

`download_busi_zenodo.py` downloads the verified BUSI ZIP. Extract it so that
the image folders are below `data/raw/BUSI_zenodo_21128640/BUSI/` before running
the audit. Extract BUS-BRA so that `bus_data.csv`, `Images/`, and
`Masks/` are in `data/raw/BUS-BRA_zenodo_8231412/BUSBRA/BUSBRA/`.

The initial development-cohort stages are:

```text
python scripts/audit_busuclm.py
python scripts/audit_busi_zenodo.py
python scripts/extract_busuclm_features.py
python scripts/extract_busi_zenodo_features.py
python scripts/run_busuclm_grouped_cv.py --cohort primary --experiment multizone
python scripts/run_busuclm_grouped_cv.py --cohort external_busi --experiment multizone --features results/busi_zenodo_features_v2/features.csv
```

BUS-BRA and BrEaST enter through
`audit_freeze_v04_external_manifests.py`, followed by
`prepare_v04_external_feature_inputs.py` and the `extract_v04_external_*`
scripts. Scripts that accept `--start` and `--limit` may be run in disjoint
chunks; the corresponding `merge_audit_*` script validates row order and joins
the completed chunks.

## Main experiment sequence

The exact entry point depends on whether feature matrices are being extracted
from source images or existing feature matrices are being evaluated. The main
model-selection and evaluation stages are:

```powershell
python scripts/confirm_v04_candidates.py
python scripts/evaluate_v04_locked_external.py
python scripts/run_v04_four_dataset_transfer.py
python scripts/analyze_v04_feature_domain_ablation.py
python scripts/analyze_v05_marginal_block_ablation.py
python scripts/analyze_v05_external_subgroups.py
python scripts/analyze_v05_patient_level_external.py
python scripts/analyze_v05_weighting_sensitivity.py
python scripts/evaluate_v05_external_mask_robustness.py
python scripts/analyze_v06_lodo_paired_ci.py
python scripts/evaluate_v06_random_boundary_robustness.py
python scripts/evaluate_v06_finetuned_deep.py --protocol locked
python scripts/evaluate_v06_finetuned_deep.py --protocol lodo
python scripts/analyze_v06_deep_comparators.py --protocol locked
python scripts/analyze_v06_deep_comparators.py --protocol lodo
```

The extraction, dataset audit, classifier-screen, and comparator scripts in
`scripts/` provide the preceding stages. Outputs are deterministic for the
fixed seeds recorded in the code, subject to the documented software and
hardware environment.

## License and citation

The code is released under the MIT License. Use `CITATION.cff` when citing the
software and cite each original dataset used in an experiment.

## Metrology analyses

The metrology entry points characterize reference artefacts, inter-view
reproducibility, input uncertainty, cross-cohort measurement shift, and
probability recalibration. They reuse the feature definitions and source-only
fitting rules from the preceding stages.

The core and deep environment files include `matplotlib==3.10.0`.
Use the corresponding environment described above.

The reference-geometry calculation uses `matplotlib.path.Path` for polygon
membership. This is a computational dependency even though these entry points
do not generate plots.

Run commands from the repository root. Before running them, complete the
preceding feature-extraction and evaluation stages. Existing inputs include
`results/v04_advanced_features/`, `results/v04_external_features/`,
`results/v04_confirmation/`, `results/v04_locked_external/`, and
`results/v04_four_dataset_transfer/`, together with the images, masks, and
BUS-BRA metadata under `data/raw/`. Image and mask paths recorded in existing
feature tables and accepted manifests must resolve on the machine running the
analyses.

The new entry points use these environment variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `SCALE_BUS_DATA_ROOT` | Repository root | Existing `data/raw/` and `results/` input tree. |
| `SCALE_BUS_OUTPUT_ROOT` | Repository `results/` | Root for newly generated `v07_*` result directories. |
| `SCALE_BUS_WORK_ROOT` | Output root plus `metrology_work/` | Runtime files, joblib temporary storage, and Matplotlib configuration. |
| `SCALE_BUS_MODEL_CACHE` | Output root plus `v07_positive_control/locked_source_models.joblib` | Fitted source-model bundle used by the metrology analyses. |
| `SCALE_BUS_BUSBRA_METADATA` | Data root plus `data/raw/BUS-BRA_zenodo_8231412/BUSBRA/BUSBRA/bus_data.csv` | BUS-BRA lesion metadata for the inter-view analysis. |

Relative environment-variable paths are resolved against the current working
directory. Changing the data root selects the input tree; it does not rewrite
paths stored inside CSV files. The working directory replaces use of the
system temporary directory for these entry points. Numerical-library thread
counts remain fixed at one per process, while the uncertainty-propagation
entry point uses six worker processes by default.

For a separate result tree, set `SCALE_BUS_OUTPUT_ROOT` to a new directory.
Set `SCALE_BUS_DATA_ROOT` only when existing data and preceding results are
stored outside the repository. Set the variables before starting Python so
spawned workers inherit the same configuration. Completed output files are
retained; use a fresh output root for a new run.

| Script | Analysis and dependencies |
| --- | --- |
| `fit_locked_source_models.py` | Refit the locked source models and reproduce their external predictions; creates the model bundle used below. |
| `metrology_reference_calibration.py` | E1: generalized-dimension, wavelet, and digital-boundary reference characterizations; run `gf`, `wavelet`, `shape`, geometry context, then `summary`. |
| `metrology_reference_geometry_context.py` | Relate reference sizes and shapes to each cohort's real lesion geometry after E1 `shape`. |
| `metrology_measurement_invariance.py` | E4: cross-cohort measurand shift and domain classification; run after model fitting and before E2. |
| `metrology_interview_reproducibility.py` | E2: lesion-pair consistency, measurand and score reproducibility, and decision agreement; uses `reproducibility_statistics.py` and the E4 shift table from the output root. |
| `metrology_label_free_calibration.py` | E5: source-OOF probability calibration and label-shift correction; uses existing confirmation OOF and leave-one-dataset-out predictions. |
| `metrology_smooth_contour_dice.py` | Dice agreement of deterministic native-resolution three-pixel erosion/dilation masks with supplied masks. |
| `metrology_uncertainty_propagation.py` | E3: full Monte Carlo propagation over both external cohorts, using six workers by default; approximately two hours on the reference eight-core laptop. |
| `metrology_e3_derived_quantities.py` | Read E3 uncertainty summaries, unperturbed model predictions, and E2 lesion pairs to calculate systematic logit shifts and propagated-to-observed variance ratios. Run after E2 and E3. |

A complete sequence is:

```text
python scripts/fit_locked_source_models.py
python scripts/metrology_reference_calibration.py gf
python scripts/metrology_reference_calibration.py wavelet
python scripts/metrology_reference_calibration.py shape
python scripts/metrology_reference_geometry_context.py
python scripts/metrology_reference_calibration.py summary
python scripts/metrology_measurement_invariance.py
python scripts/metrology_interview_reproducibility.py
python scripts/metrology_label_free_calibration.py
python scripts/metrology_smooth_contour_dice.py
python scripts/metrology_uncertainty_propagation.py --workers 6
```

The required dependencies are model fitting before the other model-based
analyses, E4 before E2, E1 `shape` before geometry context, both E1 phase and
geometry-context outputs before `summary`, and E2 plus E3 before the derived
quantities. E5 and smooth contour Dice do not depend on E2. Run the
derived-quantity entry point with
explicit input paths to the full E3 `per_image_score_uncertainty.csv`, the
unperturbed `busbra_predictions.csv` and `breast_predictions.csv`, and E2
`included_lesion_pairs.csv`; its `--help` lists the input and output options.
This keeps the data used for the variance comparison explicit when multiple
result trees exist.

### Statistical analyses

These entry points compute statistical summaries and uncertainty intervals
from completed metrology analyses.
Keep the generated metrology stages under `SCALE_BUS_OUTPUT_ROOT`; the feature
and locked-external stages remain under `SCALE_BUS_DATA_ROOT/results` using
the directory names produced by the preceding scripts. These entry points
write numerical tables below `metrology_statistics/` in the output root.

| Script | Required inputs | Output directory and contents | Approximate time |
| --- | --- | --- | --- |
| `metrology_mc_convergence.py` | Complete E3 case caches and score/variance tables; E2 included pairs and paired scores. | `mc_convergence/`: batch estimates, standard errors, numerical tolerances, and draw-count comparisons. | A few seconds. |
| `reproducibility_paired_difference.py` | E2 paired scores and the full pair-consistency inventory; locked source-model predictions. | `reproducibility/`: paired ICC, within-lesion spread and reproducibility-coefficient differences, bootstrap intervals, and pair-inclusion sensitivity. | A few seconds. |
| `calibration_intervals.py` | Complete E5 prediction and metric tables for all locked and leave-one-dataset-out settings. | `calibration_intervals/`: probability-quality intervals, binning sensitivity, cross-fitted local recalibration, and labelled-patient sample-size results. | About 4–5 minutes. |
| `auc_aggregation_sensitivity.py` | Locked external predictions from `evaluate_v04_locked_external.py`. | `auc_aggregation/`: group-balanced, image-level and patient-level AUCs and paired bootstrap differences. | About 7–8 minutes. |
| `measurand_block_redundancy.py` | All four advanced-feature tables and the locked model bundle, which supplies the feature-column reference. | `block_redundancy/`: correlations, grouped ridge predictability, effective dimensions, and numerical correlation matrices. | About 10 seconds. |
| `metrology_ratio_interval.py` | Complete E3 caches, E2 pairs and scores, and the completed MC convergence summary. | `ratio_interval/`: full-sample ratios and patient-bootstrap intervals with and without batch-level Monte Carlo variation. | Under a minute. |
| `two_view_selection_audit.py` | BUS-BRA metadata, E2 included pairs, and E1 real-lesion geometry from the shape analysis. | `selection_audit/`: patient-group summaries and standardized mean differences. | A few seconds. |

Run model fitting, E4 then E2, E3 collection and analysis, and the complete E5
analysis before these commands. The selection audit also requires the real
lesion geometry produced by the E1 shape analysis. Run MC convergence
before ratio intervals, which read its completed summary. The other statistical
stages can run independently after their listed inputs are available. The
timings above refer to the same laptop used for the core experiments; sequential
execution totals about 12–13 minutes.

```text
python scripts/metrology_mc_convergence.py
python scripts/reproducibility_paired_difference.py
python scripts/calibration_intervals.py
python scripts/auc_aggregation_sensitivity.py
python scripts/measurand_block_redundancy.py
python scripts/metrology_ratio_interval.py
python scripts/two_view_selection_audit.py
```

## Screening, transfer and audit stages

The following entry points complement the commands listed above, covering
feature preparation, development screening, transfer sensitivity, and result
audits. Each script is in `scripts/`; run it from the repository root and use
`--help` for entry points with command-line options.

| Script | Analysis |
| --- | --- |
| `extract_busuclm_multizone_features.py` | Add lesion, inner-margin, and outer-margin extrema measurements to the existing BUS-UCLM feature table. |
| `extract_v04_advanced_features.py` | Extract and validate the 76 generalized-fractal, wavelet, and boundary measurements for the two development cohorts, with optional row-range extraction. |
| `extract_v04_external_advanced_features.py` | Extract the same 76 measurements from accepted BUS-BRA or BrEaST manifests, with optional row-range extraction. |
| `extract_v04_external_frozen_features.py` | Extract the external reference features and validate the 98 conventional, fractal, and multizone columns in each output part. |
| `merge_audit_v04_advanced_features.py` | Merge BUSI advanced-feature parts and audit both development cohorts for source row order, row counts, 76-column completeness, and finite values. |
| `merge_audit_v04_external_advanced_features.py` | Merge external advanced-feature parts in start-index order and verify manifest row order, row counts, 76-column completeness, and finite values. |
| `merge_audit_v04_external_frozen_features.py` | Merge external reference-feature parts in start-index order and verify manifest row order, row counts, 98-column completeness, and finite values. |
| `extract_v04_deep_embeddings.py` | Extract frozen pretrained CNN embeddings for the selected cohort, encoder, and lesion-support view using CUDA, preserving image order in the saved bundle. |
| `screen_v04_handcrafted.py` | Compare handcrafted feature blocks and combinations through grouped development validation and source-tuned transfer in both directions between BUS-UCLM and BUSI. |
| `screen_v04_embedding.py` | Compare pretrained encoder and lesion-view combinations using fold-fitted standardization and PCA, grouped validation, and bidirectional development-cohort transfer. |
| `screen_v04_classifiers.py` | Screen classifier families on the 76-measurement signature using grouped nested validation and source-only tuning for bidirectional development-cohort transfer. |
| `screen_v04_early_fusion.py` | Evaluate concatenated handcrafted measurements and PCA-reduced inner-margin embeddings with fold-fitted preprocessing in grouped validation and bidirectional transfer. |
| `evaluate_v04_probability_fusion.py` | Compare handcrafted and embedding predictions with probability averaging and logistic stacking in grouped development validation and source-tuned bidirectional transfer. |
| `run_secondary_models.py` | Evaluate RBF-SVM and random-forest comparators across four feature groups using repeated grouped development validation and bootstrap performance comparisons. |
| `run_component_tree_comparator.py` | Extract area-filtered component-tree descriptors and compare their combinations with conventional, fractal, and extrema features in grouped validation and bidirectional transfer. |
| `run_cross_dataset_transfer.py` | Evaluate source-tuned conventional, fractal, and multizone feature combinations in both transfer directions between BUS-UCLM and BUSI, with group-balanced AUCs and bootstrap comparisons. |
| `extract_perturbed_features.py` | Re-extract development-cohort base and multizone features after fixed resolution, speckle, gamma, erosion, and dilation perturbations. |
| `run_cross_dataset_robustness.py` | Apply the clean-source models to aligned clean and perturbed target features in both development-cohort transfer directions and bootstrap the performance comparisons. |
| `extract_v04_external_mask_perturbations.py` | Re-extract the 76 external-cohort measurements after deterministic native-resolution three-pixel mask erosion and dilation. |
| `extract_v06_random_boundary_perturbations.py` | Re-extract the 76 external-cohort measurements after scale-normalized random boundary displacements at two amplitudes and five fixed seeds. |
| `analyze_feature_family_contribution.py` | Quantify feature-family permutation effects, source-bootstrap coefficient stability, and prediction-error complementarity for the development-cohort transfer models. |
| `analyze_clinical_utility.py` | Evaluate group-balanced probability calibration, Brier score, log loss, reliability bins, and decision-curve net benefit from development out-of-fold predictions. |
| `benchmark_feature_runtime.py` | Measure CPU image-loading and feature-extraction times for the 94-variable base and 63-variable multizone audit inventory on fixed samples from both development cohorts. |

The shared `metrology_common.py` and `metrology_paths.py` modules supply
scoring, output, random-number, and path configuration functions to the
metrology entry points; they are imported support modules rather than
standalone analyses.
