"""Configure input, result, and working paths for metrology analyses.

Import this module before the analysis libraries and supporting analysis modules, including
in process workers. Paths in existing image and mask manifests must remain
readable on the machine running the analyses.
"""

from __future__ import annotations

import importlib
import os
import sys
import tempfile
from pathlib import Path

sys.dont_write_bytecode = True

REPO_ROOT = Path(__file__).resolve().parents[1]


def _configured_path(variable: str, default: Path) -> Path:
    return Path(os.environ.get(variable, str(default))).expanduser().resolve()


DATA_ROOT = _configured_path("SCALE_BUS_DATA_ROOT", REPO_ROOT)
OUTPUT_ROOT = _configured_path("SCALE_BUS_OUTPUT_ROOT", REPO_ROOT / "results")
WORK_ROOT = _configured_path("SCALE_BUS_WORK_ROOT", OUTPUT_ROOT / "metrology_work")
MODEL_CACHE = _configured_path(
    "SCALE_BUS_MODEL_CACHE",
    OUTPUT_ROOT / "v07_positive_control" / "locked_source_models.joblib",
)

# Compatibility names distinguish existing inputs from newly generated outputs.
ROOT = DATA_ROOT
STAGE = WORK_ROOT
CACHE = MODEL_CACHE

for _thread_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_variable] = "1"

WORK_ROOT.mkdir(parents=True, exist_ok=True)
for _variable, _directory in (
    ("MPLCONFIGDIR", WORK_ROOT / "matplotlib"),
    ("JOBLIB_TEMP_FOLDER", WORK_ROOT / "joblib"),
):
    _directory.mkdir(parents=True, exist_ok=True)
    os.environ[_variable] = str(_directory)
tempfile.tempdir = str(WORK_ROOT)

for _directory in (REPO_ROOT / "scripts", REPO_ROOT / "src"):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))


def configure_input_modules() -> None:
    """Redirect the input constants used by the imported analysis helpers.

    Assign after importing each module so import-time defaults cannot replace
    the configured paths. Leave existing output directories and numerical
    settings unchanged. This runs again when a spawned process imports the
    metrology entry point.
    """
    embedding = importlib.import_module("screen_v04_embedding")
    embedding.FEATURE_DIR = DATA_ROOT / "results" / "v04_advanced_features"

    external = importlib.import_module("evaluate_v04_locked_external")
    external.EXTERNAL_FEATURE_DIR = DATA_ROOT / "results" / "v04_external_features"
    external.CONFIRMATION_DIR = DATA_ROOT / "results" / "v04_confirmation"

    for name in (
        "extract_v04_external_mask_perturbations",
        "extract_v06_random_boundary_perturbations",
    ):
        module = importlib.import_module(name)
        module.INPUT_DIR = DATA_ROOT / "results" / "v04_external_features"


configure_input_modules()
