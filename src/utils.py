"""Shared helpers: global seed and project path constants."""
import random
from pathlib import Path

import numpy as np

SEED = 42

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
FEATURE_REPO = PROJECT_ROOT / "feature_repo"
MODELS_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "report"


def set_seed(seed: int = SEED) -> None:
    """Seed Python's `random` and NumPy for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)

# MLflow: sqlite backend at the repo root (absolute, so any cwd works). Artifacts land in ./mlruns.
MLFLOW_DB = PROJECT_ROOT / "mlflow.db"
MLFLOW_TRACKING_URI = f"sqlite:///{MLFLOW_DB.as_posix()}"
MLFLOW_ARTIFACT_ROOT = PROJECT_ROOT / "mlruns"
MLFLOW_EXPERIMENT = "StayGuard"
MLFLOW_REGISTERED_MODEL = "StayGuard"
