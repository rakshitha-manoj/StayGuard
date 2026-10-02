# StayGuard

StayGuard predicts whether a hotel booking will be canceled (`is_canceled`) using the Kaggle "Hotel booking demand" dataset. This university MLOps project covers the full lifecycle: versioned data (DVC), experiment tracking (MLflow), a feature store (Feast), a Kubeflow pipeline, and deployment on Vertex AI.

## Repo layout

```
data/raw/            raw CSV (DVC-tracked, contents gitignored)
data/processed/      processed data (DVC-tracked, contents gitignored)
reports_data/        machine-generated JSON (validation, preprocessing log)
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

## Lifecycle design

Six-stage lifecycle design, stage table and diagram: [report/lifecycle.md](report/lifecycle.md) (diagram source `report/lifecycle.mmd`). Later stages are planned in Phases 2 to 6.

## Data versioning and pipeline

Raw data is tagged `v1-raw`; the cleaned output of the DVC pipeline is tagged `v2-clean`. Pipeline stages (`dvc.yaml`): `validate_raw` -> `preprocess` -> `validate_clean`, parameters in `params.yaml`, machine-generated JSON in `reports_data/`.

`v2-clean` is 85,716 rows x 70 columns (md5 `0bd1d2070641efd38a0d1f526532b8da`), all numeric. Preprocessing drops the leaky columns `reservation_status`, `reservation_status_date` and `assigned_room_type` (set at check-in), and finally drops exact duplicate rows (33,493) so identical rows cannot straddle a train/test split; this trades some legitimate repeated group bookings for honest test metrics. The CSV and JSON outputs are written with LF endings, so hashes match on Windows and Linux.

```powershell
.venv\Scripts\Activate.ps1   # stages call plain `python`
dvc pull                     # fetch data from the GCS remote
dvc repro                    # run the pipeline
dvc diff v1-raw v2-clean     # compare dataset versions
python -m src.data_report    # v1 vs v2 stats table
```

Details, numbers and rollback commands: [report/dvc_comparison.md](report/dvc_comparison.md); `dvc diff` output: [report/dvc_diff.txt](report/dvc_diff.txt).

## Experiment tracking and training (coming in Phase 3)
## Feature store (coming in Phase 4)
## Kubeflow orchestration (coming in Phase 5)
## Vertex AI deployment, cloud comparison and monitoring (coming in Phase 6)
## Demo and results (coming in Phase 7)
