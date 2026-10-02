"""Model definitions and the train/test split, free of mlflow/matplotlib imports.

Lives apart from src/train.py so the Kubeflow components (which install only the minimal
runtime deps: pandas, numpy, scikit-learn, joblib, pyyaml) can reuse the exact same model
and split. src/train.py imports from here, so there is one definition.
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from src.utils import SEED

TEST_SIZE = 0.2


def build_models(n_estimators: int = 200, seed: int = SEED) -> dict[str, Pipeline]:
    return {
        "rf": Pipeline([("clf", RandomForestClassifier(
            n_estimators=n_estimators, max_depth=None, min_samples_leaf=1, n_jobs=-1,
            random_state=seed))]),
        "svm": Pipeline([("scaler", StandardScaler()),
                         ("clf", LinearSVC(C=1.0, max_iter=5000, random_state=seed, dual="auto"))]),
        "knn": Pipeline([("scaler", StandardScaler()),
                         ("clf", KNeighborsClassifier(n_neighbors=15, weights="uniform", n_jobs=-1))]),
    }


def split_xy(X: np.ndarray, y: np.ndarray, seed: int = SEED):
    """Stratified 80/20 split used by every training and evaluation path."""
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=seed, stratify=y)
