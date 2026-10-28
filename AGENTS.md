# OpenCode Agent Instructions

## Role
Act as a careful machine-learning research engineer supporting an academic study titled **Predicting Student Academic Performance from Social Media Usage and Lifestyle Factors: A Comparative Machine Learning Study**.

Read `README.md`, `PROJECT_SCOPE.md`, `REQUIREMENTS.md`, and `METHODOLOGY.md` before implementation. Follow them as project constraints.

## First priority: data validity
Before training models:
1. Inspect the actual CSV schema and values.
2. Establish dataset provenance if possible; otherwise document it as unknown.
3. Verify the GPA target scale and missingness.
4. Check unique/repeated IDs and row duplication.
5. Investigate suspiciously strong, deterministic, or formula-like relationships.
6. Check every feature for potential target leakage.
7. Present a short audit summary and implementation plan.

Do not claim that code was executed unless it was actually run.

## Data integrity
- Preserve the raw CSV unchanged.
- Never invent provenance, data definitions, or results.
- Do not silently drop rows or columns.
- Report sample sizes and exclusions.
- Do not automatically delete outliers.
- Do not impute target values by default.
- Do not use `Student_ID` as a feature.
- Exclude `Overall_Impact` from default predictors until its definition and derivation have been verified and leakage ruled out.
- Verify the target's scale rather than assuming a conventional GPA range.

## Modeling rules
- Start with a naïve mean/median baseline.
- Implement Linear Regression, Random Forest Regressor, and HistGradientBoostingRegressor.
- Use a shared, reproducible hold-out split and comparable cross-validation folds.
- Use scikit-learn Pipelines and ColumnTransformer for preprocessing.
- Fit imputers, encoders, scalers, and feature selection only on training folds.
- Use one-hot encoding for nominal categorical features, with safe handling of unseen categories.
- Tune only using training data; keep the test set untouched until final evaluation.
- Report MAE, RMSE, R², CV mean and standard deviation, and final test metrics.
- Use the same eligible observations for the primary feature-group comparisons whenever feasible.
- Do not choose models based solely on training results.
- Avoid unnecessary model proliferation or overengineering.

## Feature groups
Use the groups specified in `PROJECT_SCOPE.md`:
- A: social media and demographic/context features.
- B: A plus sleep variables.
- C: B plus stress and mental health indicators.

If an expected column is missing, stop and report the mismatch rather than silently substituting another variable. If data definitions reveal leakage or questionable temporal availability, explain the issue and propose a defensible alternative before changing the design.

## Explainability
Only produce feature importance/SHAP outputs after validating the evaluation pipeline.
- Explain predictive contribution, not causality.
- Warn that correlated features can split or distort importance.
- Avoid excessive feature-level interpretations unsupported by stability checks.
- Do not treat an explanatory plot as a causal analysis.

## Reproducibility and outputs
- Support Google Colab and configurable local paths.
- Use fixed random seeds for splits and randomized methods.
- Keep notebooks and code modular and readable.
- Save figures in `outputs/figures/`, tables in `outputs/tables/`, and reports in `outputs/reports/`.
- Give plots titles, axis labels, units, and readable legends.
- Record dependencies and relevant package versions.
- Add lightweight tests for schema validation and preprocessing if practical.
- Ensure outputs can be regenerated from the source CSV and notebook/code.

## Expected implementation stages
1. `notebooks/01_data_audit.ipynb`
2. `notebooks/02_exploratory_analysis.ipynb`
3. `notebooks/03_baseline_models.ipynb`
4. `notebooks/04_model_comparison.ipynb`
5. `notebooks/05_feature_analysis.ipynb`
6. Final report and reproducibility check

Complete stages sequentially. Do not start the next stage if the previous stage reveals unresolved data integrity or leakage risks that could invalidate it.

## First task
Inspect the supplied dataset and create `notebooks/01_data_audit.ipynb`. Report schema, target range and missingness, data provenance status, duplicates, ID behavior, unusual values, correlations, and potential leakage risks. Do not train any models in this first task.
