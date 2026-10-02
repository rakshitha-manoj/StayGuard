# MLflow run comparison (experiment StayGuard)

| Run | Model | Data | Train acc | Test acc | Precision | Recall | F1 | ROC-AUC | Fit s | Gap | Overfit (gap>0.05) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| knn__v2-clean | KNN | v2-clean | 0.8291 | 0.7925 | 0.6570 | 0.5224 | 0.5820 | 0.8340 | 0.1 | 0.0366 | no |
| rf__v2-clean | RandomForest | v2-clean | 0.9974 | 0.8359 | 0.7433 | 0.6210 | 0.6767 | 0.8933 | 16.2 | 0.1615 | yes |
| svm__v2-clean | LinearSVC | v2-clean | 0.7844 | 0.7817 | 0.6679 | 0.4190 | 0.5150 | 0.8289 | 3.0 | 0.0027 | no |
| knn__v1-raw | KNN | v1-raw | 0.8427 | 0.8188 | 0.7919 | 0.6929 | 0.7391 | 0.8910 | 0.9 | 0.0239 | no |
| rf__v1-raw | RandomForest | v1-raw | 0.9958 | 0.8922 | 0.8913 | 0.8076 | 0.8474 | 0.9578 | 55.3 | 0.1036 | yes |
| svm__v1-raw | LinearSVC | v1-raw | 0.8134 | 0.8116 | 0.8129 | 0.6384 | 0.7152 | 0.8884 | 29.9 | 0.0018 | no |
