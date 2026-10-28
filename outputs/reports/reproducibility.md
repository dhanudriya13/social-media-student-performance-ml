# Reproducibility Record

## Environment

| Component | Version |
|---|---|
| Python | 3.9.6 |
| pandas | 2.3.3 |
| NumPy | 2.0.2 |
| scikit-learn | 1.6.1 |
| Matplotlib | 3.9.4 |
| seaborn | 0.13.2 |

SHAP was not installed or used; permutation importance and partial dependence provide the explainability outputs.

## Data

- Source: `data/raw/Social_media_impact_on_life.csv`
- SHA-256: `893bcfc9165dc2d92caf2511bc10a0753b67df22c2a370a43b513f2417468a3f`
- The raw file is never modified. Derived tables and figures are written to `outputs/`.

## Seeds

- Global split/CV seed: `42` (`sm_project.config.SEED`).
- Train/test split: `train_test_split(..., test_size=0.20, random_state=42)`.
- Cross-validation: `KFold(n_splits=5, shuffle=True, random_state=42)`.
- Random Forest, HistGradientBoosting, permutation importance, and sampling all use seed 42.

## Execution order

1. `notebooks/01_data_audit.ipynb`
2. `notebooks/02_exploratory_analysis.ipynb`
3. `notebooks/03_baseline_models.ipynb`
4. `notebooks/04_model_comparison.ipynb`
5. `notebooks/05_feature_analysis.ipynb`

Run all five plus the test suite with:

```bash
bash scripts/run_pipeline.sh
```

The notebooks resolve the repository root automatically and insert `src/` on the path. In Google Colab, upload the CSV and set either `SM_DATA_PATH` to the file location or place it in `data/raw/`; paths can also be overridden with `SM_DATA_DIR` and `SM_OUTPUT_DIR`.

## Outputs

- `outputs/tables/` — audit, EDA, CV, test, feature-group, and feature-importance tables.
- `outputs/figures/` — PNG figures used in the report.
- `outputs/reports/` — per-stage Markdown summaries plus this record and the final report.

## Evaluation safeguards

- Preprocessing is fitted inside scikit-learn Pipelines, so imputation, scaling, and one-hot encoding are learned only from training folds.
- Hyperparameters were tuned only on the training partition with 3-fold CV.
- Model selection used training CV MAE, never test performance.
- The same eligible observations, split indices, and folds were used for all feature groups and models.
