# MLflow experiment results (Phase 3)

All numbers below come from the logged MLflow runs (`python -m src.evaluate` -> `report/mlflow_comparison.csv/.md`). Nothing here is hand-edited.

## Setup

- Tracking URI `sqlite:///mlflow.db` (repo root, gitignored; resolved to an absolute path in `src/utils.py`), experiment `StayGuard`, artifacts in `./mlruns` (gitignored).
- Seed 42 (`src/utils.SEED`, `set_seed`). Split: stratified `train_test_split(test_size=0.2, random_state=42)`. Positive class = 1 (canceled); precision/recall/F1 use `pos_label=1`.
- Data versions: `v1-raw` (119,390 rows, md5 `5bf588c5a949443e021fb7c847d31b27`; train 95,512 / test 23,878 / 246 features after minimal prep) and `v2-clean` (85,716 rows, md5 `0bd1d2070641efd38a0d1f526532b8da`; train 68,572 / test 17,144 / 69 features).
- v1 prep (`prepare_v1_minimal` in `src/datasets.py`) is deliberately minimal: drop the 3 leaky columns, naive null fill (numeric 0, categorical "missing"), one-hot every object column as-is (country unbucketed), agent/company as raw numeric codes. No row removal, outlier capping, dedupe or engineered features.
- Models are sklearn Pipelines on a plain numeric numpy array (no column names; the Vertex sklearn container feeds positional arrays), fixed hyperparameters, no tuning grid (a scope choice):
  - `rf`: RandomForest(n_estimators=200, max_depth=None, min_samples_leaf=1, n_jobs=-1)
  - `svm`: StandardScaler + LinearSVC(C=1.0, max_iter=5000, dual="auto"). LinearSVC instead of kernel SVC because kernel SVC fits in O(n^2) to O(n^3) at 68k-95k training rows; LinearSVC is roughly linear-time and trains on the full split without subsampling. It has no probabilities, so ROC-AUC uses `decision_function`.
  - `knn`: StandardScaler + KNeighbors(n_neighbors=15, weights="uniform"). Uniform weights because distance weighting makes train accuracy trivially 1.0. KNN train accuracy is computed on a fixed seeded 10,000-row train subsample (logged as `train_accuracy_sample_size`); all other models use the full train set.
- 12 runs in two rounds of 6, named `<model>__<data_version>` (see Lineage below; reports use the latest round). Each logs all pipeline hyperparameters, `data_version`, `data_md5`, `seed`, `test_size`, `n_train`, `n_test`, `n_features`; tags `data_version`, `model_family`, `git_commit`, `dvc_tag` (plus `git_dirty`); metrics `train_accuracy`, `test_*`, `fit_seconds`; the model (with signature and a 5-row numpy input example) and a test confusion-matrix PNG.
- Training is not a DVC stage (avoids creating MLflow runs on every `dvc repro`). Lineage is the logged `data_md5` / `dvc_tag` / `git_commit`.

## Comparison

| Run | Model | Data | Train acc | Test acc | Precision | Recall | F1 | ROC-AUC | Fit s | Gap | Overfit (gap>0.05) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| knn__v2-clean | KNN | v2-clean | 0.8291 | 0.7925 | 0.6570 | 0.5224 | 0.5820 | 0.8340 | 0.1 | 0.0366 | no |
| rf__v2-clean | RandomForest | v2-clean | 0.9974 | 0.8359 | 0.7433 | 0.6210 | 0.6767 | 0.8933 | 16.2 | 0.1615 | yes |
| svm__v2-clean | LinearSVC | v2-clean | 0.7844 | 0.7817 | 0.6679 | 0.4190 | 0.5150 | 0.8289 | 3.0 | 0.0027 | no |
| knn__v1-raw | KNN | v1-raw | 0.8427 | 0.8188 | 0.7919 | 0.6929 | 0.7391 | 0.8910 | 0.9 | 0.0239 | no |
| rf__v1-raw | RandomForest | v1-raw | 0.9958 | 0.8922 | 0.8913 | 0.8076 | 0.8474 | 0.9578 | 55.3 | 0.1036 | yes |
| svm__v1-raw | LinearSVC | v1-raw | 0.8134 | 0.8116 | 0.8129 | 0.6384 | 0.7152 | 0.8884 | 29.9 | 0.0018 | no |

Gap = train accuracy - test accuracy; overfit flag when gap > 0.05. Chart: `report/mlflow_f1_comparison.png`.

## Best model

`rf__v2-clean` has the highest test F1 among the v2-clean runs (0.6767; accuracy 0.8359, precision 0.7433, recall 0.6210, ROC-AUC 0.8933). Deployment uses cleaned data, so only v2-clean runs compete. Test confusion matrix (`report/confusion_matrix_best.png`): 11,385 TN, 1,017 FP, 1,797 FN, 2,945 TP. It is exported to `models/model.joblib` (about 65 MB with `joblib` `compress=3`, gitignored; sklearn 1.6.1 pipeline), described in `models/best_model.json` (run id, metrics, 69 feature columns in order), and registered in the MLflow Model Registry as `StayGuard` version 2 (version 1 is the same model from round 1).

## Analysis

- **v1 vs v2 scores.** v1 scores higher for every model (F1 0.847 vs 0.677 for RF, 0.715 vs 0.515 for LinearSVC, 0.739 vs 0.582 for KNN), so the metrics alone do not show preprocessing "winning"; the v1 numbers are not comparable. The ablation below (measured with RF) shows why: about 90% of the RF F1 gap comes from v1's duplicate rows. 32.8% of the v1 test rows have an identical feature vector in the train split (0.5% for v2-clean); the RF scores F1 0.983 on those rows and 0.686 on the rest, and deduplicating v1 before the split drops its F1 from 0.847 to 0.692. The higher v1 base rate (37.0% vs 27.7% canceled) is a side effect of the same duplicates, not a separate cause: the duplicated test rows are 58.6% canceled, and deduplicated v1 has 27.5% positives. The remaining ~0.015 F1 (deduplicated v1 0.692 vs v2-clean 0.677) is information that v2 preprocessing throws away: collapsing the `agent`/`company` IDs to `has_agent`/`has_company`. Adding the raw codes back to v2-clean recovers it (F1 0.692, ROC-AUC 0.904 vs 0.893). Bucketing country to top 10 + Other costs nothing (full country one-hot gives F1 0.660). Conclusion: v2-clean gives the honest, leak-free estimate (about 0.68 F1, 0.84 accuracy for the best model); v1 numbers are optimistic mainly because of duplicate leakage, and v2 leaves a small amount of signal on the table in the agent ID.
- **Overfitting.** RF is flagged on both versions (gap 0.162 on v2-clean, 0.104 on v1-raw): fully grown trees fit train accuracy ~0.997. LinearSVC (gap 0.003) and KNN (gap 0.037, measured on the 10k train subsample) are not flagged. LinearSVC's near-zero gap with low train accuracy (0.784) signals underfitting: a linear boundary cannot capture the interactions in this data, hence the lowest F1 (0.515) and recall (0.419). KNN is in between and cheap to fit (0.1 s) but prediction is the costly step and it needs the training set at serving time.
- **Trade-offs.** RF is best on every v2-clean metric, at 16.2 s fit and a 410 MB uncompressed artifact; a smaller `max_depth`/`min_samples_leaf` would shrink the gap and file but was left out of scope (no tuning). Measured in verification (not applied): `min_samples_leaf=5` gives 102 MB at F1 0.653; `joblib.dump(..., compress=3)` stores the same model in 65 MB with identical predictions, and `src/evaluate.py` now exports it that way. LinearSVC is fast, tiny and has no probability output (decision scores only), so thresholding for business use needs calibration. All models have recall below precision on the positive class, i.e. they miss many cancellations; with only 27.7% positives, threshold tuning or class weighting is the obvious next step.
- **Determinism.** Re-running all three v2-clean models gave identical accuracy, precision, recall and F1 to full float precision; the RF ROC-AUC differed in the 7th to 8th decimal (logged 0.89325768; reruns gave 0.89325773 and 0.89325725; parallel summation order in `predict_proba` with `n_jobs=-1`). `evaluate.py` keeps only the most recent run per (model, data version) pair. Those verification reruns were soft-deleted in MLflow.
- **Lineage.** Round 1 (6 runs, `git_commit=a7ac91a`, `git_dirty=True`) was trained before the training code was committed. Round 2 (6 runs) was retrained from the clean commit `7e11459` (`git_dirty=False`) and reproduced round 1's train/test accuracy, precision, recall and F1 exactly (ROC-AUC within 4e-8, the parallel-summation effect above); only fit times differ. Both rounds are kept in the experiment as tracking history; `evaluate.py` reports the latest run per (model, data version), i.e. round 2.
- **Runtime.** Full `python -m src.train` (6 runs including data loading and model logging) took about 6.5 minutes on a 12-core Windows machine; fit times are in the table.

### Ablation (verification)

Measured by the Phase 3 verifier with scratch code (not committed, no MLflow runs). RandomForest with the same hyperparameters as `src/train.py` (200 trees, `min_samples_leaf=1`, seed 42), same stratified 80/20 split with `random_state=42` on each variant. A0 and A3a reproduce the logged `rf__v1-raw` and `rf__v2-clean` runs exactly (F1, accuracy). A3 variants rerun `src.preprocess.preprocess` with one change each, so they include its final deduplication (row counts differ slightly because extra columns make fewer rows identical).

| Variant | Rows | Test rows | Test % canceled | Accuracy | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| A0 v1-raw minimal prep (= logged run) | 119,390 | 23,878 | 37.0 | 0.8922 | 0.8474 | 0.9578 |
| A1 same model, test rows whose feature vector is in train removed | 119,390 | 16,040 | 26.5 | 0.8492 | 0.6860 | 0.9060 |
| A1' same model, only those 7,838 seen-in-train test rows | 119,390 | 7,838 | 58.6 | 0.9804 | 0.9832 | 0.9959 |
| A2 v1-raw minimal prep, exact duplicates dropped before the split | 86,149 | 17,230 | 27.5 | 0.8467 | 0.6920 | 0.9036 |
| A3a v2-clean rebuilt (= logged run) | 85,716 | 17,144 | 27.7 | 0.8359 | 0.6767 | 0.8933 |
| A3b v2-clean + raw `agent`/`company` codes (null = 0) | 85,737 | 17,148 | 27.7 | 0.8426 | 0.6916 | 0.9036 |
| A3c v2-clean + full country one-hot (no top-10 bucketing) | 85,962 | 17,193 | 27.6 | 0.8327 | 0.6604 | 0.8916 |
| A3d v2-clean + both | 85,983 | 17,197 | 27.6 | 0.8442 | 0.6848 | 0.9044 |

Reading: duplicates account for 0.155 of the 0.171 RF F1 gap (A0 to A2), the agent/company ID collapse for about 0.015 (A3a to A3b, which matches A2), country bucketing for none. One split and one seed: differences around 0.01 are not tested for significance.

## Launch the UI

From the repo root (the relative `sqlite:///mlflow.db` resolves against the current directory):

```
.venv\Scripts\Activate.ps1
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Open http://127.0.0.1:5000 (verified: the server returns HTTP 200 and `/api/2.0/mlflow/experiments/search` lists `StayGuard`). Screenshots are not included in the repo; to take them:

1. Open experiment `StayGuard`, tick the 6 latest runs (or all 12 to show both rounds), click **Compare**.
2. In the compare view use the **Parallel coordinates** plot (add `test_f1`, `test_roc_auc`, `fit_seconds`) and the **Scatter plot** with `test_f1` on the Y axis, distinguishing runs by `data_version`; the table view also works.
3. The **Models** tab shows registered model `StayGuard` (versions 1 and 2 = `rf__v2-clean` from rounds 1 and 2).
