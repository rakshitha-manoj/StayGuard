# StayGuard structure map

Goal: predict `is_canceled` (Kaggle Hotel booking demand). Phase numbers are planned; dvc.yaml arrives in Phase 2.

## Key facts
- Seed: 42 (`src/utils.py: SEED`, `set_seed()`)
- Python 3.11 venv at `.venv` (uv); run tools via `.venv/Scripts/python`
- sklearn pinned 1.6.1 to match Vertex prebuilt `sklearn-cpu.1-6`
- DVC remote `gcs` (default): gs://stayg-510316-stayguard-dvc/dvcstore; ADC not set up yet, no push/pull until it is
- GCP project stayg-510316, region us-central1
- gcloud lives at C:\Users\raksh\AppData\Local\Google\Cloud SDK\google-cloud-sdk\bin (not on PATH)
- Data (hotel_bookings.csv) is placed manually in data/raw/; contents gitignored, .dvc files committed

## Files
| Path | Job | Phase |
|---|---|---|
| src/utils.py | SEED, set_seed, path constants (implemented) | 0 |
| src/ingest.py | load raw CSV | 1 |
| src/validate.py | schema / quality checks | 2 |
| src/preprocess.py | cleaning, features, split | 2 |
| src/train.py | train + MLflow logging | 3 |
| src/evaluate.py | metrics, plots | 3 |
| src/monitor.py | drift / performance monitoring | 6 |
| feature_repo/ | Feast definitions | 4 |
| pipeline/kfp_pipeline.py | KFP pipeline | 5 |
| deploy/deploy_vertex.py | deploy model to Vertex endpoint | 5 |
| deploy/predict_sample.py | sample endpoint request | 5 |
| deploy/teardown.py | delete Vertex resources | 5 |
| deploy/teardown_bucket.ps1 | delete DVC bucket (manual, confirmation prompt) | 0 |
| notebooks/, report/ | EDA, write-up | later |
| requirements.txt | exact pins | 0 |

Update this file when a module moves or changes job.
