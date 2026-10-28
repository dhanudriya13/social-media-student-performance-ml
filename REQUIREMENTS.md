# Project Requirements

## 1. Functional requirements

### FR-01 — Configurable data loading
Load `Social_media_impact_on_life.csv` from a configurable path for local use or Google Colab.

Acceptance criteria:
- The input path is configured in one place.
- Missing-file errors provide clear instructions.
- The raw file is never overwritten.

### FR-02 — Schema and provenance audit
Inspect actual column names, row/column counts, dtypes, missingness, duplicate rows, duplicate identifiers, and descriptive statistics. Record available provenance and unresolved questions about collection method, sampling, variable definitions, and whether the data are synthetic.

Acceptance criteria:
- The notebook reports the observed schema rather than assuming it.
- Any mismatch against documented expected columns is shown clearly.
- Provenance is not invented.

### FR-03 — Validate the target
Verify `Academic_Performance_GPA` exists, is numeric or can be parsed without ambiguity, and has a plausible scale and distribution. Investigate impossible values and missingness.

Acceptance criteria:
- Report target range, distribution, missing count, and usable sample size.
- Do not assume a 0–4 or 0–5 scale without evidence.
- Target rows with missing values are handled explicitly and the exclusion count is reported.

### FR-04 — Data-quality checks
Check:
- Missing values per column and missingness patterns.
- Exact duplicate rows.
- Duplicate or repeated `Student_ID` values.
- Numeric ranges, impossible values, and suspicious outliers.
- Category levels and inconsistent text labels.
- Strong correlations and suspiciously deterministic relationships.
- Potential synthetic-data fingerprints where feasible.

Do not delete outliers automatically. Flag and investigate them.

### FR-05 — Feature policy and leakage prevention
- Exclude `Student_ID` from all predictors.
- Exclude the target and any direct transformations of the target from features.
- Exclude `Overall_Impact` from default models until its definition and derivation are established and leakage is ruled out.
- Audit every proposed predictor for possible target leakage.
- Treat `Mental_Health_Index` and `Perceived_Stress_Score` as potentially useful but conceptually distinct predictors; report that they may be contemporaneous measures and are not necessarily available in a prospective prediction setting.

### FR-06 — Define feature groups
Implement the feature groups in `PROJECT_SCOPE.md`:
- A: social media and demographic/context features.
- B: A plus sleep features.
- C: B plus perceived stress and mental health indicators.

If a listed feature is absent or invalid, stop with a clear message or use an explicitly documented alternative. Do not silently substitute features.

For fair comparisons, either:
1. compare all feature groups on a common eligible sample, or
2. clearly report that each experiment has a different sample and avoid attributing performance changes solely to feature additions.

Prefer a common eligible sample for the primary comparison, with a secondary sensitivity analysis using all available cases if useful.

### FR-07 — Exploratory data analysis
Produce:
- Target distribution and summary statistics.
- Histograms/boxplots for key numerical variables.
- Missingness summary.
- Correlation matrix for numeric variables, interpreted cautiously.
- GPA summaries by selected categories, where appropriate.
- Scatterplots of GPA against selected numeric features.

Each plot must have a clear title, labels, readable layout, and be saved to disk.

### FR-08 — Preprocessing pipeline
Use scikit-learn pipelines and column transformers where appropriate.

- Numeric features: impute missing predictors using a training-fitted strategy (median is a reasonable default), optionally scale for linear models.
- Categorical features: impute missing categories and use one-hot encoding with safe handling of unseen categories.
- Fit all imputation, encoding, scaling, and feature-selection steps only on training data or within cross-validation folds.
- Do not scale target values unless the need is justified and inverse transformation is correctly handled.
- Record all preprocessing choices.

### FR-09 — Baseline and model training
Train at least:
- Naïve mean or median predictor.
- Linear Regression.
- Random Forest Regressor.
- HistGradientBoostingRegressor, using a compatible preprocessing pipeline.

Ridge Regression or another model may be added if justified. Avoid excessive model proliferation.

### FR-10 — Evaluation protocol
Use a reproducible hold-out test set and cross-validation on training data. Set and record random seeds.

Minimum metrics:
- MAE
- RMSE
- R²

Requirements:
- Keep the test set untouched until model selection and tuning are complete.
- Use the same split and comparable folds across models and feature groups.
- Report cross-validation mean and standard deviation, final test metrics, and sample size.
- Compare every model with the naïve baseline.
- Avoid relying on R² alone.
- Do not report training metrics as evidence of generalization.

If records are repeated per student or there are other grouping/time dependencies, revise the split strategy to prevent leakage across train/test partitions.

### FR-11 — Feature-group comparison
Compare A, B, and C using the same evaluation protocol and preferably the same eligible observations.

Report:
- The model and feature group.
- Cross-validation metrics.
- Test metrics.
- Number of features after preprocessing, if useful.
- Sample size and missing-data policy.

Assess whether performance changes are material relative to cross-validation variability, not merely whether one decimal value is larger.

### FR-12 — Model explanation
After model evaluation:
- Report linear coefficients only with suitable caveats and preprocessing context.
- Use permutation importance on held-out data where practical.
- Optionally use SHAP for a selected tree-based model.
- Use partial dependence or accumulated local effects only where helpful and valid.
- Explain that feature importance is predictive, not causal.
- Check whether correlated features divide or distort importance.

Do not use the test set repeatedly to make modeling decisions. If test-based explanations are produced, label them as final descriptive analyses.

### FR-13 — Outputs
Export:
- Cleaned/analysis-ready data only if appropriate and safe.
- Data-quality report and cleaning log.
- EDA tables and figures.
- Model comparison CSV.
- Predictions for the held-out test set, with a non-identifying row key if needed.
- Feature-importance/explainability tables and figures.
- A concise Markdown or HTML research report.

Save outputs in `outputs/tables/`, `outputs/figures/`, and `outputs/reports/`.

### FR-14 — Reproducibility
Include setup instructions, package versions or a dependency file, random seeds, configurable paths, and notebook execution order. Notebook results must be produced by code, not manually typed.

## 2. Non-functional requirements

- Reproducible on a standard Google Colab CPU.
- Clear separation between raw data, processed data, code, and results.
- Modular and maintainable code without unnecessary abstraction.
- Explicit warnings for schema mismatches and empty subsets.
- No fabricated metrics, results, or data provenance.
- Research limitations documented throughout.
- Charts readable and suitable for inclusion in a paper.
- No unnecessary external data collection.
- No automatic publication or upload of student-level records.

## 3. Technical stack

Core:
- Python 3
- pandas
- NumPy
- scikit-learn
- Matplotlib
- seaborn (optional)

Optional:
- SHAP, for explainability after the modeling pipeline is validated.
- statsmodels, for supplementary statistical diagnostics.
- pytest, for tests of data validation and preprocessing.

Use only necessary dependencies. Document versions used in the final analysis.

## 4. Acceptance checklist

- [ ] Actual schema and target scale verified.
- [ ] Provenance and unresolved data-source questions documented.
- [ ] Raw dataset preserved.
- [ ] Missing values, duplicates, and suspicious records audited.
- [ ] Student ID excluded from predictors.
- [ ] Overall Impact excluded unless leakage is ruled out.
- [ ] Preprocessing fitted inside training folds.
- [ ] Naïve baseline included.
- [ ] At least three regression models evaluated.
- [ ] Hold-out test set not used for tuning.
- [ ] MAE, RMSE, and R² reported.
- [ ] Feature-group comparisons use comparable samples and splits.
- [ ] Feature explanations are non-causal and appropriately caveated.
- [ ] Tables, figures, and reports are reproducible.
