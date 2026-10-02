"""Kubeflow Pipelines (KFP v2) definition for StayGuard.

Five components: data_collection -> data_validation -> training -> evaluation -> deployment,
with deployment gated by `dsl.If(evaluation F1 > f1_threshold)`.

All logic lives in the `stayguard` package (this repo, `src/`); the component bodies are thin
wrappers that import from `src` at run time. On a real cluster each component installs the repo
from GitHub (`packages_to_install`) at the ref chosen at COMPILE time (default "main"). For
exact reproducibility compile with a commit SHA or tag:

    python -m pipeline.kfp_pipeline compile --ref <commit-sha>

Why an archive URL and not `git+https://`: the base image python:3.11-slim has no git binary,
so pip could not clone. GitHub's /archive/<ref>.zip works for branches, tags and SHAs and needs
only pip + internet access (the repo is public).

Local runs (pipeline/run_local.py) use the same component functions with the repo already
importable from the venv, so they build the components with packages_to_install=[] (nothing is
pip-installed at run time and the venv's pins cannot be touched).

Compile: python -m pipeline.kfp_pipeline compile   -> pipeline/stayguard_pipeline.yaml
"""
import argparse
from pathlib import Path
from typing import NamedTuple

from kfp import compiler, dsl
from kfp.dsl import (Artifact, ClassificationMetrics, Dataset, Input, Metrics, Model, Output)

REPO = "rakshitha-manoj/StayGuard"
DEFAULT_REF = "main"
BASE_IMAGE = "python:3.11-slim"
YAML_PATH = Path(__file__).resolve().parent / "stayguard_pipeline.yaml"
# md5 of data/raw/hotel_bookings.csv, from data/raw/hotel_bookings.csv.dvc (tag v1-raw)
RAW_MD5 = "5bf588c5a949443e021fb7c847d31b27"
VERTEX_SKLEARN_IMAGE = "us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-6:latest"


def package_spec(ref: str = DEFAULT_REF) -> str:
    return f"stayguard @ https://github.com/{REPO}/archive/{ref}.zip"


# --------------------------------------------------------------------------- component bodies
# Each function must be self-contained: KFP serialises its source into the container command,
# so imports happen inside the body and module-level helpers are not available.

def data_collection(source_uri: str, expected_md5: str, raw_data: Output[Dataset]):
    """Fetch the raw CSV (local path or gs://), verify its md5 against the DVC pointer."""
    import hashlib
    import shutil

    if source_uri.startswith("gs://"):
        from google.cloud import storage  # only needed (and installed) for gs:// sources
        bucket_name, _, blob_name = source_uri[len("gs://"):].partition("/")
        storage.Client().bucket(bucket_name).blob(blob_name).download_to_filename(raw_data.path)
    else:
        shutil.copyfile(source_uri, raw_data.path)

    h = hashlib.md5()
    with open(raw_data.path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    actual = h.hexdigest()
    if actual != expected_md5:
        raise RuntimeError(f"raw data md5 mismatch for {source_uri}: expected {expected_md5}, "
                           f"got {actual}. Refusing to continue with unexpected data.")

    from src.ingest import load_raw
    rows = len(load_raw(raw_data.path))
    raw_data.metadata.update({"source_uri": source_uri, "md5": actual, "rows": rows})
    print(f"data_collection ok: {rows} rows, md5 {actual}")


def data_validation(raw_data: Input[Dataset], adr_cap_quantile: float, top_n_countries: int,
                    clean_data: Output[Dataset], validation_metrics: Output[Metrics]):
    """validate_raw -> preprocess -> validate_clean. Raises (fails the run) on bad data.

    Preprocessing lives here because the pipeline has exactly five stages; this step is
    'validate + clean + re-validate' and acts as the data quality gate.
    """
    import hashlib

    from src.ingest import load_raw
    from src.preprocess import preprocess
    from src.validate import validate_clean, validate_raw

    df = load_raw(raw_data.path)
    validate_raw(df)  # raises DataValidationError on hard failures
    log: dict = {}
    clean = preprocess(df, {"adr_cap_quantile": adr_cap_quantile,
                            "top_n_countries": top_n_countries}, log)
    clean_report = validate_clean(clean)  # strict; raises on any violation
    clean.to_csv(clean_data.path, index=False, lineterminator="\n")  # LF => platform-stable md5

    with open(clean_data.path, "rb") as f:
        md5 = hashlib.md5(f.read()).hexdigest()
    clean_data.metadata.update({"md5": md5, "rows": clean_report["rows"],
                                "columns": clean_report["columns"]})
    validation_metrics.log_metric("raw_rows", int(log["rows_in"]))
    validation_metrics.log_metric("clean_rows", clean_report["rows"])
    validation_metrics.log_metric("clean_columns", clean_report["columns"])
    validation_metrics.log_metric("rows_removed", int(log["rows_in"]) - clean_report["rows"])
    validation_metrics.log_metric("clean_cancel_rate", clean_report["cancel_rate"])
    print(f"data_validation ok: {log['rows_in']} -> {clean_report['rows']} rows, md5 {md5}")


def training(clean_data: Input[Dataset], seed: int, n_estimators: int, mlflow_tracking_uri: str,
             model: Output[Model]):
    """Stratified 80/20 split, fit the RandomForest pipeline, save <model.path>/model.joblib.

    A directory containing a file named exactly model.joblib is what Vertex's prebuilt sklearn
    container expects. If mlflow_tracking_uri is set (local runs only) a run tagged
    source=kfp_pipeline is logged to the 'StayGuard' experiment.
    """
    import os
    import time

    import joblib
    import pandas as pd

    from src.models import build_models, split_xy

    df = pd.read_csv(clean_data.path)
    feature_names = [c for c in df.columns if c != "is_canceled"]
    X = df[feature_names].to_numpy(dtype="float64")
    y = df["is_canceled"].to_numpy(dtype=int)
    X_tr, X_te, y_tr, y_te = split_xy(X, y, seed)

    pipe = build_models(n_estimators=n_estimators, seed=seed)["rf"]
    t0 = time.perf_counter()
    pipe.fit(X_tr, y_tr)
    fit_seconds = time.perf_counter() - t0

    os.makedirs(model.path, exist_ok=True)
    joblib.dump(pipe, os.path.join(model.path, "model.joblib"), compress=3)
    model.metadata.update({"framework": "sklearn", "model_family": "rf", "seed": seed,
                           "n_estimators": n_estimators, "n_features": int(X.shape[1]),
                           "data_md5": clean_data.metadata.get("md5", ""),
                           "fit_seconds": round(fit_seconds, 2)})

    if mlflow_tracking_uri:
        import mlflow
        from sklearn.metrics import accuracy_score, f1_score
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment("StayGuard")  # same experiment as src/train.py runs
        with mlflow.start_run(run_name="rf__kfp_pipeline") as run:
            mlflow.set_tags({"source": "kfp_pipeline", "model_family": "rf",
                             "data_md5": clean_data.metadata.get("md5", "")})
            mlflow.log_params({"seed": seed, "n_estimators": n_estimators,
                               "n_train": len(X_tr), "n_test": len(X_te),
                               "n_features": X.shape[1]})
            pred = pipe.predict(X_te)
            mlflow.log_metrics({"fit_seconds": fit_seconds,
                                "train_accuracy": accuracy_score(y_tr, pipe.predict(X_tr)),
                                "test_accuracy": accuracy_score(y_te, pred),
                                "test_f1": f1_score(y_te, pred, pos_label=1)})
            model.metadata["mlflow_run_id"] = run.info.run_id
    print(f"training ok: fit {fit_seconds:.1f}s, saved {model.path}/model.joblib")


def evaluation(clean_data: Input[Dataset], model: Input[Model], seed: int,
               metrics: Output[Metrics], confusion_matrix: Output[ClassificationMetrics]
               ) -> NamedTuple("EvalOutputs", [("f1", float), ("accuracy", float)]):
    """Recompute the same seeded split, score the held-out test set."""
    import os
    from collections import namedtuple

    import joblib
    import pandas as pd
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.metrics import confusion_matrix as sk_cm

    from src.models import split_xy

    df = pd.read_csv(clean_data.path)
    X = df[[c for c in df.columns if c != "is_canceled"]].to_numpy(dtype="float64")
    y = df["is_canceled"].to_numpy(dtype=int)
    _, X_te, _, y_te = split_xy(X, y, seed)

    pipe = joblib.load(os.path.join(model.path, "model.joblib"))
    pred = pipe.predict(X_te)
    f1 = float(f1_score(y_te, pred, pos_label=1))
    acc = float(accuracy_score(y_te, pred))
    metrics.log_metric("f1", f1)
    metrics.log_metric("accuracy", acc)
    metrics.log_metric("n_test", int(len(y_te)))
    confusion_matrix.log_confusion_matrix(["not_canceled", "canceled"],
                                          sk_cm(y_te, pred, labels=[0, 1]).tolist())
    print(f"evaluation ok: f1={f1:.4f} accuracy={acc:.4f}")
    return namedtuple("EvalOutputs", ["f1", "accuracy"])(f1, acc)


def deployment(model: Input[Model], f1: float, deploy_mode: str, target_image: str,
               deployment_record: Output[Artifact]):
    """Gate-passed model -> deployment. dry_run proves the artifact is servable and writes a record.

    'vertex' mode (upload + endpoint) is wired in Phase 6.
    """
    import datetime
    import json
    import os

    import joblib
    import numpy as np
    import sklearn

    if deploy_mode == "vertex":
        raise NotImplementedError("wired in Phase 6")
    if deploy_mode != "dry_run":
        raise ValueError(f"unknown deploy_mode {deploy_mode!r}; expected 'dry_run' or 'vertex'")

    path = os.path.join(model.path, "model.joblib")
    pipe = joblib.load(path)
    # Vertex's sklearn container feeds a plain positional numpy array, so test exactly that.
    sample = np.zeros((2, int(pipe.n_features_in_)))
    pred = pipe.predict(sample)
    if pred.shape != (2,):
        raise RuntimeError(f"unexpected prediction shape {pred.shape}")

    record = {"mode": deploy_mode, "model_uri": model.uri, "model_file": "model.joblib",
              "model_size_mb": round(os.path.getsize(path) / 1e6, 1), "f1": f1,
              "n_features": int(pipe.n_features_in_), "sample_prediction": pred.tolist(),
              "sklearn_version": sklearn.__version__, "target_container_uri": target_image,
              "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    with open(deployment_record.path, "w") as f:
        json.dump(record, f, indent=2)
    deployment_record.metadata.update({"mode": deploy_mode, "f1": f1})
    print("deployment (dry_run) ok:", json.dumps(record))


COMPONENT_FUNCS = {"data_collection": data_collection, "data_validation": data_validation,
                   "training": training, "evaluation": evaluation, "deployment": deployment}


def build_pipeline(packages: list[str] | None):
    """Create the five components + pipeline. `packages` is passed as packages_to_install:
    [package_spec(ref)] for the cluster, [] for local runs where `src` is already importable."""
    # Local builds (packages == []) also skip KFP's own `pip install kfp==...` bootstrap, so no
    # pip command runs at all and the active environment cannot be modified.
    c = {name: dsl.component(func=fn, base_image=BASE_IMAGE, packages_to_install=packages,
                             install_kfp_package=bool(packages))
         for name, fn in COMPONENT_FUNCS.items()}
    if packages:  # gs:// support needs the GCS client in the collection step on a cluster only
        c["data_collection"] = dsl.component(
            func=data_collection, base_image=BASE_IMAGE,
            packages_to_install=packages + ["google-cloud-storage"], install_kfp_package=True)

    @dsl.pipeline(name="stayguard-pipeline",
                  description="StayGuard: collect -> validate/clean -> train -> evaluate -> "
                              "(if F1 > threshold) deploy")
    def stayguard_pipeline(source_uri: str, expected_md5: str = RAW_MD5,
                           f1_threshold: float = 0.60, seed: int = 42, n_estimators: int = 200,
                           adr_cap_quantile: float = 0.995, top_n_countries: int = 10,
                           mlflow_tracking_uri: str = "", deploy_mode: str = "dry_run"):
        collect = c["data_collection"](source_uri=source_uri, expected_md5=expected_md5)
        collect.set_display_name("Data collection")
        collect.set_caching_options(False)  # reads external state; the md5 gate is the real check

        validate = c["data_validation"](raw_data=collect.outputs["raw_data"],
                                        adr_cap_quantile=adr_cap_quantile,
                                        top_n_countries=top_n_countries)
        validate.set_display_name("Data validation and cleaning")

        train = c["training"](clean_data=validate.outputs["clean_data"], seed=seed,
                              n_estimators=n_estimators, mlflow_tracking_uri=mlflow_tracking_uri)
        train.set_display_name("Training (RandomForest)")
        train.set_cpu_request("2").set_cpu_limit("4")  # scalability knob: per-step resources
        train.set_memory_request("4G").set_memory_limit("8G")

        evaluate = c["evaluation"](clean_data=validate.outputs["clean_data"],
                                   model=train.outputs["model"], seed=seed)
        evaluate.set_display_name("Evaluation")

        with dsl.If(evaluate.outputs["f1"] > f1_threshold, name="f1-above-threshold"):
            deploy = c["deployment"](model=train.outputs["model"], f1=evaluate.outputs["f1"],
                                     deploy_mode=deploy_mode,
                                     target_image=VERTEX_SKLEARN_IMAGE)
            deploy.set_display_name("Deployment")
            deploy.set_caching_options(False)  # writes a timestamped record each time

    return stayguard_pipeline, c


def compile_pipeline(ref: str = DEFAULT_REF, out: Path = YAML_PATH) -> Path:
    pipeline, _ = build_pipeline([package_spec(ref)])
    compiler.Compiler().compile(pipeline_func=pipeline, package_path=str(out))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compile", help="compile to pipeline/stayguard_pipeline.yaml")
    c.add_argument("--ref", default=DEFAULT_REF,
                   help="git branch/tag/commit SHA the components install (default: main)")
    args = ap.parse_args()
    print(f"compiled {compile_pipeline(args.ref)} (installs {package_spec(args.ref)})")


if __name__ == "__main__":
    main()
