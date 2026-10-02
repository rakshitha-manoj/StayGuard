"""Train RandomForest, LinearSVC and KNN on both data versions and log every run to MLflow.

CLI: python -m src.train [--data-version v1-raw|v2-clean|all]   (default all -> 6 runs)

Scope choices (documented in report/mlflow_results.md):
- Fixed hyperparameters, no tuning grid. The goal is tracking and comparing, not squeezing scores.
- "SVM" is LinearSVC, not kernel SVC: kernel SVC fits in O(n^2)-O(n^3), impractical at ~68k-95k
  training rows without subsampling; LinearSVC is ~linear-time so it trains on the full split.
  It has no predict_proba, so ROC-AUC uses decision_function.
- KNN uses uniform weights (distance weights make train accuracy trivially 1.0 because each
  training point is its own nearest neighbour at distance 0). Train accuracy for KNN is computed
  on a fixed seeded 10k-row train subsample (prediction on the full train set is slow); the size
  is logged as `train_accuracy_sample_size`.
- Every model is an sklearn Pipeline operating on a plain numeric numpy array (no column names),
  because Vertex's sklearn container feeds a positional numpy array.
- Training stays outside dvc.yaml so `dvc repro` does not create MLflow runs. Lineage is the
  logged data_md5 / dvc_tag / git_commit.
"""
import argparse
import subprocess
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
from mlflow.models import infer_signature
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline

from src.datasets import DATA_VERSIONS, load_dataset
from src.models import TEST_SIZE, build_models, split_xy
from src.utils import (MLFLOW_ARTIFACT_ROOT, MLFLOW_EXPERIMENT, MLFLOW_TRACKING_URI, SEED,
                       set_seed)

KNN_TRAIN_ACC_SAMPLE = 10_000


def setup_mlflow() -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    if mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT) is None:
        mlflow.create_experiment(MLFLOW_EXPERIMENT, artifact_location=MLFLOW_ARTIFACT_ROOT.as_uri())
    mlflow.set_experiment(MLFLOW_EXPERIMENT)


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True).strip()
    except Exception:
        return "unknown"


def flat_params(pipe: Pipeline) -> dict[str, str]:
    """pipeline.get_params(deep=True) minus nested estimator objects/step lists, values stringified."""
    out = {}
    for k, v in pipe.get_params(deep=True).items():
        if hasattr(v, "get_params") or isinstance(v, (list, tuple)):
            continue
        out[k] = str(v)[:500]
    return out


def plot_confusion_matrix(cm: np.ndarray, title: str, path) -> None:
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], ["not canceled", "canceled"])
    ax.set_yticks([0, 1], ["not canceled", "canceled"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def train_one(family: str, pipe: Pipeline, split: dict, meta: dict, tmp_dir) -> str:
    X_tr, X_te, y_tr, y_te = split["X_tr"], split["X_te"], split["y_tr"], split["y_te"]
    version = meta["data_version"]
    run_name = f"{family}__{version}"
    set_seed(SEED)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(flat_params(pipe))
        mlflow.log_params({"data_version": version, "data_md5": meta["data_md5"], "seed": SEED,
                           "test_size": TEST_SIZE, "n_train": len(X_tr), "n_test": len(X_te),
                           "n_features": X_tr.shape[1]})
        mlflow.set_tags({"data_version": version, "model_family": family, "dvc_tag": version,
                         "git_commit": _git("rev-parse", "HEAD"),
                         "git_dirty": str(bool(_git("status", "--porcelain")))})

        t0 = time.perf_counter()
        pipe.fit(X_tr, y_tr)
        fit_seconds = time.perf_counter() - t0

        if family == "knn":
            idx = np.random.RandomState(SEED).choice(len(X_tr), min(KNN_TRAIN_ACC_SAMPLE, len(X_tr)),
                                                     replace=False)
            train_acc = accuracy_score(y_tr[idx], pipe.predict(X_tr[idx]))
            mlflow.log_param("train_accuracy_sample_size", len(idx))
        else:
            train_acc = accuracy_score(y_tr, pipe.predict(X_tr))

        pred = pipe.predict(X_te)
        score = pipe.predict_proba(X_te)[:, 1] if hasattr(pipe, "predict_proba") else pipe.decision_function(X_te)
        metrics = {
            "train_accuracy": train_acc,
            "test_accuracy": accuracy_score(y_te, pred),
            "test_precision": precision_score(y_te, pred, pos_label=1),
            "test_recall": recall_score(y_te, pred, pos_label=1),
            "test_f1": f1_score(y_te, pred, pos_label=1),
            "test_roc_auc": roc_auc_score(y_te, score),
            "fit_seconds": fit_seconds,
        }
        mlflow.log_metrics(metrics)

        cm_path = tmp_dir / f"confusion_matrix_{run_name}.png"
        plot_confusion_matrix(confusion_matrix(y_te, pred, labels=[0, 1]), f"{run_name} (test set)", cm_path)
        mlflow.log_artifact(str(cm_path))

        example = X_tr[:5]
        # MLflow 3.x saves with skops by default; the RF's Tree type must be explicitly trusted
        # (we trained it ourselves, so trusting it is safe).
        mlflow.sklearn.log_model(pipe, name="model", input_example=example,
                                 skops_trusted_types=["sklearn.tree._tree.Tree"],
                                 signature=infer_signature(example, pipe.predict(example)))
        print(f"{run_name}: " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()), flush=True)
        return run.info.run_id


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-version", default="all", choices=[*DATA_VERSIONS, "all"])
    args = ap.parse_args()
    versions = DATA_VERSIONS if args.data_version == "all" else [args.data_version]

    tmp_dir = MLFLOW_ARTIFACT_ROOT / "_tmp_plots"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    setup_mlflow()
    for version in versions:
        X, y, meta = load_dataset(version)
        X_tr, X_te, y_tr, y_te = split_xy(X, y, SEED)
        split = dict(X_tr=X_tr, X_te=X_te, y_tr=y_tr, y_te=y_te)
        print(f"== {version}: {X.shape}, train {len(X_tr)}, test {len(X_te)}, md5 {meta['data_md5']}",
              flush=True)
        for family, pipe in build_models().items():
            train_one(family, pipe, split, meta, tmp_dir)


if __name__ == "__main__":
    main()
