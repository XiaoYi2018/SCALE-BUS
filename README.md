# SCALE-BUS experimental code

This repository contains the experimental code for **“SCALE-BUS: An
Interpretable Scale-Coupled Framework for Breast Ultrasound Lesion
Classification under Dataset Shift.”**

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
the audit. The initial development-cohort stages are:

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
