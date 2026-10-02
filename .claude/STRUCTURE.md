# StayGuard structure map

Goal: predict `is_canceled` (Kaggle Hotel booking demand). dvc.yaml pipeline: validate_raw -> preprocess -> validate_clean (run from activated venv).

## Phase plan
0 setup | 1 lifecycle design doc (report/lifecycle.md) | 2 DVC versioning + ingest/validate/preprocess + dvc.yaml |
3 MLflow train/evaluate | 4 Feast | 5 Kubeflow pipeline (pipeline/) |
6 Vertex deployment (deploy/*) + cloud comparison + src/monitor.py drift | 7 demo notebook + README + demo script

## Key facts
- Seed: 42 (`src/utils.py: SEED`, `set_seed()`)
- Python 3.11 venv at `.venv` (uv); run tools via `.venv/Scripts/python`
- sklearn pinned 1.6.1 to match Vertex prebuilt `sklearn-cpu.1-6`
- That image (checked 2026-10-01 from its /tmp/pip-requirements.txt layer) runs Python 3.10, numpy==1.26.4, joblib==1.5.3, scikit-learn>=1.6,<1.7; it feeds the model a plain numpy array (no column names)
- DVC remote `gcs` (default): gs://stayg-510316-stayguard-dvc/dvcstore; ADC set up; push/pull work. Tags: v1-raw (raw CSV), v2-clean (pipeline output)
- GCP project stayg-510316, region us-central1
- gcloud lives at C:\Users\raksh\AppData\Local\Google\Cloud SDK\google-cloud-sdk\bin (not on PATH)
- Data (hotel_bookings.csv) is placed manually in data/raw/; contents gitignored, .dvc files committed
- .gitignore must ignore data *files* (`data/raw/*.csv`), never `data/raw/*`: DVC's matcher then treats the folder as ignored and silently skips its .dvc files. The v1-raw tag still has the bad pattern, so use `dvc checkout data/raw/hotel_bookings.csv.dvc` there (plain `dvc checkout` works at v2-clean and main)
- .gitattributes forces LF everywhere (dvc.lock hashes src/*.py byte-for-byte); no `-text` exceptions. Processed CSV (to_csv lineterminator LF) and reports_data/*.json (write_text newline LF) are written LF, so md5s match on Windows and Linux
- Processed CSV: 85,716 x 70, md5 0bd1d2070641efd38a0d1f526532b8da. Leaky columns (LEAKY_COLUMNS in validate.py): reservation_status, reservation_status_date, assigned_room_type (check-in; cancel rate 5.4% when differs from reserved vs 41.6% when equal). Exact duplicates (33,493) are dropped as the last preprocess step; validate_clean rejects duplicates and any leaky/assigned_room_type_* column

## Files
| Path | Job | Phase |
|---|---|---|
| src/utils.py | SEED, set_seed, path constants (implemented) | 0 |
| src/ingest.py | EXPECTED_COLUMNS schema, load_raw() | 2 |
| src/validate.py | validate_raw (hard: columns, dtype kinds, target; soft: counted), validate_clean (strict: nulls, leaky cols, duplicates, numeric), DataValidationError, CLI `--stage raw or clean` | 2 |
| src/preprocess.py | cleaning + features + one-hot -> data/processed/hotel_bookings_clean.csv (no split, no scaling), drops exact duplicates last, logs step counts | 2 |
| src/data_report.py | v1 vs v2 stats markdown table from real files | 2 |
| params.yaml | seed, preprocess params (adr cap quantile, top-N countries) | 2 |
| dvc.yaml, dvc.lock | pipeline definition and locked hashes | 2 |
| reports_data/ | machine-generated JSON: validation_raw/clean, preprocess_log | 2 |
| report/dvc_comparison.md, report/dvc_diff.txt | v1 vs v2 write-up, `dvc diff` output | 2 |
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
| report/lifecycle.mmd, report/lifecycle.png | diagram source and render (`npx -y @mermaid-js/mermaid-cli -i report/lifecycle.mmd -o report/lifecycle.png -b white -s 2 --size 2400`) | 1 |
| notebooks/ | demo notebook | 7 |
| report/ | cloud comparison (6), final write-up (7) | 1, 6, 7 |
| requirements.txt | exact pins | 0 |

Update this file when a module moves or changes job.
