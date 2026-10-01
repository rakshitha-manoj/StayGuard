# StayGuard

StayGuard predicts whether a hotel booking will be canceled (`is_canceled`) using the Kaggle "Hotel booking demand" dataset. This university MLOps project covers the full lifecycle: versioned data (DVC), experiment tracking (MLflow), a feature store (Feast), a Kubeflow pipeline, and deployment on Vertex AI.

## Repo layout

```
data/raw/            raw CSV (DVC-tracked, contents gitignored)
data/processed/      processed data (DVC-tracked, contents gitignored)
src/                 ingest, validate, preprocess, train, evaluate, monitor, utils
feature_repo/        Feast feature repository
pipeline/            Kubeflow Pipelines definition
deploy/              Vertex AI deploy / sample prediction / teardown scripts
notebooks/           exploration notebooks
report/              project report
.dvc/                DVC config (GCS remote)
requirements.txt     pinned dependencies (Python 3.11)
```

## Setup

With uv (recommended):

```powershell
uv python install 3.11
uv venv --python 3.11 .venv
.venv\Scripts\Activate.ps1
uv pip install -r requirements.txt
```

Plain venv + pip (needs Python 3.11 installed):

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Data

Download `hotel_bookings.csv` from Kaggle (login required) and place it at `data/raw/hotel_bookings.csv`. It is not committed; DVC tracks it (`dvc add data/raw/hotel_bookings.csv`).

## Cloud resources

- GCP project `stayg-510316`, region `us-central1`
- DVC remote bucket: `gs://stayg-510316-stayguard-dvc` (remote path `dvcstore`)

Remove the bucket and its contents with `deploy/teardown_bucket.ps1`, or:
`gcloud storage rm --recursive gs://stayg-510316-stayguard-dvc`

## Data versioning and pipeline (coming in Phase 2)
## Experiment tracking and training (coming in Phase 3)
## Feature store (coming in Phase 4)
## Orchestration and deployment (coming in Phase 5)
## Monitoring (coming in Phase 6)
## Results (coming later)
