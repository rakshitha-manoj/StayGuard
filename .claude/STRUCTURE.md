# StayGuard structure map

Goal: predict `is_canceled` (Kaggle Hotel booking demand). dvc.yaml arrives in Phase 2.

## Phase plan
0 setup | 1 lifecycle design doc (report/lifecycle.md) | 2 DVC versioning + ingest/validate/preprocess + dvc.yaml |
3 MLflow train/evaluate | 4 Feast | 5 Kubeflow pipeline (pipeline/) |
6 Vertex deployment (deploy/*) + cloud comparison + src/monitor.py drift | 7 demo notebook + README + demo script

## Key facts
- Seed: 42 (`src/utils.py: SEED`, `set_seed()`)
- Python 3.11 venv at `.venv` (uv); run tools via `.venv/Scripts/python`
- sklearn pinned 1.6.1 to match Vertex prebuilt `sklearn-cpu.1-6`
- That image (checked 2026-10-01 from its /tmp/pip-requirements.txt layer) runs Python 3.10, numpy==1.26.4, joblib==1.5.3, scikit-learn>=1.6,<1.7; it feeds the model a plain numpy array (no column names)
- DVC remote `gcs` (default): gs://stayg-510316-stayguard-dvc/dvcstore; ADC not set up yet, no push/pull until it is
- GCP project stayg-510316, region us-central1
- gcloud lives at C:\Users\raksh\AppData\Local\Google\Cloud SDK\google-cloud-sdk\bin (not on PATH)
- Data (hotel_bookings.csv) is placed manually in data/raw/; contents gitignored, .dvc files committed

## Files
| Path | Job | Phase |
|---|---|---|
| src/utils.py | SEED, set_seed, path constants (implemented) | 0 |
| src/ingest.py | load raw CSV | 2 |
| src/validate.py | schema / quality checks | 2 |
| src/preprocess.py | cleaning, features, split | 2 |
| src/train.py | train + MLflow logging | 3 |
| src/evaluate.py | metrics, plots | 3 |
| src/monitor.py | drift / performance monitoring | 6 |
| feature_repo/ | Feast definitions | 4 |
| pipeline/kfp_pipeline.py | KFP pipeline | 5 |
| deploy/deploy_vertex.py | deploy model to Vertex endpoint | 6 |
| deploy/predict_sample.py | sample endpoint request | 6 |
| deploy/teardown.py | delete Vertex resources | 6 |
| deploy/teardown_bucket.ps1 | delete DVC bucket (manual, confirmation prompt) | 0 |
| report/lifecycle.md | lifecycle design doc | 1 |
| report/lifecycle.mmd, report/lifecycle.png | diagram source and render (`npx -y @mermaid-js/mermaid-cli -i report/lifecycle.mmd -o report/lifecycle.png -b white -s 2`) | 1 |
| notebooks/ | demo notebook | 7 |
| report/ | cloud comparison (6), final write-up (7) | 1, 6, 7 |
| requirements.txt | exact pins | 0 |

Update this file when a module moves or changes job.
