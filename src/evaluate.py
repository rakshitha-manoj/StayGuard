"""Compare the MLflow runs, pick the best v2-clean model and export it.

CLI: python -m src.evaluate

- Reads experiment "StayGuard"; if a (model, data_version) pair has several runs (reruns)
  only the most recent is kept.
- Writes report/mlflow_comparison.{csv,md}, report/mlflow_f1_comparison.png,
  report/confusion_matrix_best.png.
- Best model = highest test F1 among v2-clean runs (deployment uses cleaned data). It is
  exported to models/model.joblib, described in models/best_model.json and registered in
  the MLflow Model Registry as "StayGuard".
"""
import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import sklearn
from mlflow import MlflowClient
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from src.datasets import load_dataset
from src.train import TEST_SIZE, plot_confusion_matrix, setup_mlflow
from src.utils import (MLFLOW_EXPERIMENT, MLFLOW_REGISTERED_MODEL, MODELS_DIR, REPORT_DIR, SEED)

OVERFIT_GAP = 0.05
METRICS = ["train_accuracy", "test_accuracy", "test_precision", "test_recall", "test_f1",
           "test_roc_auc", "fit_seconds"]
MODEL_NAMES = {"rf": "RandomForest", "svm": "LinearSVC", "knn": "KNN"}


def latest_runs() -> pd.DataFrame:
    runs = mlflow.search_runs(experiment_names=[MLFLOW_EXPERIMENT], order_by=["start_time DESC"])
    if runs.empty:
        raise SystemExit("No runs found in experiment StayGuard; run `python -m src.train` first.")
    runs = runs[runs["status"] == "FINISHED"]
    runs = runs.drop_duplicates(subset=["tags.model_family", "tags.data_version"], keep="first")
    df = pd.DataFrame({
        "run_name": runs["tags.mlflow.runName"],
        "run_id": runs["run_id"],
        "model": runs["tags.model_family"],
        "data_version": runs["tags.data_version"],
        "data_md5": runs["params.data_md5"],
    })
    for m in METRICS:
        df[m] = runs[f"metrics.{m}"]
    df["gap"] = df["train_accuracy"] - df["test_accuracy"]
    df["overfit"] = df["gap"] > OVERFIT_GAP
    return df.sort_values(["data_version", "model"], ascending=[False, True]).reset_index(drop=True)


def markdown_table(df: pd.DataFrame) -> str:
    cols = ["run_name", "model", "data_version", "train_accuracy", "test_accuracy", "test_precision",
            "test_recall", "test_f1", "test_roc_auc", "fit_seconds", "gap", "overfit"]
    head = ["Run", "Model", "Data", "Train acc", "Test acc", "Precision", "Recall", "F1", "ROC-AUC",
            "Fit s", "Gap", "Overfit (gap>0.05)"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for _, r in df.iterrows():
        cells = [r.run_name, MODEL_NAMES.get(r.model, r.model), r.data_version]
        cells += [f"{r[c]:.4f}" for c in cols[3:9]] + [f"{r.fit_seconds:.1f}", f"{r.gap:.4f}",
                                                      "yes" if r.overfit else "no"]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def f1_chart(df: pd.DataFrame, path) -> None:
    fams = ["rf", "svm", "knn"]
    versions = ["v1-raw", "v2-clean"]
    colors = {"v1-raw": "#9aa5b1", "v2-clean": "#1f6feb"}
    fig, ax = plt.subplots(figsize=(7, 4.5))
    width = 0.38
    for i, v in enumerate(versions):
        if not (df.data_version == v).any():  # e.g. only `--data-version v2-clean` was trained
            continue
        vals =[df[(df.model == f) & (df.data_version == v)]["test_f1"].iloc[0] for f in fams]
        bars = ax.bar(np.arange(3) + (i - 0.5) * width, vals, width, label=v, color=colors[v])
        ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=9)
    ax.set_xticks(range(3), [MODEL_NAMES[f] for f in fams])
    ax.set_ylabel("Test F1 (positive class = canceled)")
    ax.set_xlabel("Model")
    ax.set_title("Test F1 by model and data version")
    ax.set_ylim(0, 1)
    ax.legend(title="Data version")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    setup_mlflow()
    REPORT_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
    df = latest_runs()
    df.to_csv(REPORT_DIR / "mlflow_comparison.csv", index=False, lineterminator="\n")
    table = markdown_table(df)
    (REPORT_DIR / "mlflow_comparison.md").write_text(
        "# MLflow run comparison (experiment StayGuard)\n\n" + table, encoding="utf-8", newline="\n")
    f1_chart(df, REPORT_DIR / "mlflow_f1_comparison.png")
    print(table)

    best = df[df.data_version == "v2-clean"].sort_values("test_f1", ascending=False).iloc[0]
    print(f"Best (v2-clean, test F1): {best.run_name} ({best.run_id}) F1={best.test_f1:.4f}")
    pipe = mlflow.sklearn.load_model(f"runs:/{best.run_id}/model")
    # compress=3 shrinks the 200-tree RF from ~410 MB to ~65 MB with identical predictions,
    # which keeps the Vertex AI upload small.
    joblib.dump(pipe, MODELS_DIR / "model.joblib", compress=3)

    X, y, meta = load_dataset("v2-clean")
    _, X_te, _, y_te = train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y)
    cm = confusion_matrix(y_te, pipe.predict(X_te), labels=[0, 1])
    plot_confusion_matrix(cm, f"Best model: {best.run_name} (test set)", REPORT_DIR / "confusion_matrix_best.png")

    info = {
        "run_id": best.run_id, "run_name": best.run_name, "model": best.model,
        "data_version": best.data_version, "data_md5": best.data_md5,
        "metrics": {m: float(best[m]) for m in METRICS},
        "n_features": len(meta["feature_names"]), "feature_columns": meta["feature_names"],
        "sklearn_version": sklearn.__version__, "positive_class": 1,
    }
    (MODELS_DIR / "best_model.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8",
                                                newline="\n")

    client = MlflowClient()
    existing = client.search_model_versions(f"name='{MLFLOW_REGISTERED_MODEL}'")
    if any(v.run_id == best.run_id for v in existing):
        print(f"Run already registered as {MLFLOW_REGISTERED_MODEL}; skipping registration.")
    else:
        mv = mlflow.register_model(f"runs:/{best.run_id}/model", MLFLOW_REGISTERED_MODEL)
        print(f"Registered {MLFLOW_REGISTERED_MODEL} version {mv.version}")


if __name__ == "__main__":
    main()
