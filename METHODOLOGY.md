# Methodology

## 1. Research design

Quantitative, observational, cross-sectional predictive modeling using a supplied secondary dataset. The study evaluates prediction, not causality.

The primary outcome is `Academic_Performance_GPA`. Its coding, range, provenance, and meaning must be verified before modeling.

## 2. Research questions

- RQ1: How accurately can regression models predict GPA from social media usage and lifestyle variables?
- RQ2: Does adding sleep variables improve predictive performance beyond social media and demographic/context variables?
- RQ3: Does adding stress and mental health indicators provide additional predictive information?
- RQ4: Which predictors contribute most to predictions, and how robust are those contributions?

## 3. Data provenance gate

Before modeling, document:
- Source or provider of the dataset, if known.
- Whether records were collected from real participants, simulated, or synthetically generated.
- Collection date and sampling method, if known.
- Operational definitions and measurement scales.
- Whether GPA is self-reported, institutionally recorded, simulated, or otherwise obtained.
- Whether variables or labels were calculated from one another.

If provenance cannot be established, explicitly label it as unknown and restrict conclusions to the supplied dataset. If synthetic origin is confirmed, frame findings as an analysis of the generated data and do not generalize to real students.

## 4. Unit of analysis and target

- Unit: one verified student record.
- Target: `Academic_Performance_GPA`.
- Exclude `Student_ID` from predictors.
- Verify whether IDs are unique. If repeated records exist, consider group-aware train/test splitting.
- Do not assume a GPA scale before inspecting values and documentation.

## 5. Feature groups

Finalize only after checking definitions and leakage.

### A — Social media and demographic/context features
- `Daily_Usage_Hours`
- `Weekend_Extra_Hours`
- `Late_Night_Usage`
- `Social_Comparison_Frequency`
- `Primary_Platform`
- `Device_Type`
- `Age`
- `Gender`
- `Academic_Level`

### B — Add sleep
Group A plus:
- `Sleep_Duration_Hours`
- `Sleep_Quality_Score`

### C — Add psychological indicators
Group B plus:
- `Perceived_Stress_Score`
- `Mental_Health_Index`

Exclude `Overall_Impact` from default models until its definition and derivation are known and target leakage has been ruled out.

These are contemporaneous features in a cross-sectional dataset. The experiments measure incremental predictive information, not mediation or causal effects.

## 6. Sample definition and missingness

Report missingness for all variables. Because the target is missing for some rows, exclude target-missing records from supervised training and report the resulting sample size.

For the primary feature-group comparison, use a common eligible sample across groups where feasible. This avoids confounding performance differences with changes in the observations used. A secondary experiment can use the maximum available cases per feature group, but must report the differing sample sizes and interpret comparisons cautiously.

Impute predictor missingness within the model pipeline:
- Numeric predictors: median imputation as a default.
- Categorical predictors: most-frequent or explicit missing-category imputation.
Fit imputation only on training folds. Do not impute the target by default.

## 7. Exploratory data analysis

Inspect:
- Target distribution, range, and descriptive statistics.
- Predictor distributions and unusual values.
- Missingness patterns.
- Category frequencies.
- Numeric correlations.
- GPA distributions across selected categories.
- Scatterplots of GPA versus numeric features.

Investigate very high correlations or near-deterministic relationships for possible data-generation artifacts or leakage. Correlation alone does not establish leakage, but unexpectedly strong relationships require scrutiny.

## 8. Train/test and validation strategy

1. Use a reproducible hold-out split, such as 80% training and 20% testing.
2. If no grouping issue is found, use a suitable random split and record the seed.
3. If repeated student IDs or grouped observations exist, use group-aware splitting.
4. Use cross-validation on training data for model comparison and any hyperparameter selection.
5. Keep the hold-out test set untouched until the final selected configurations are fixed.
6. Apply preprocessing through scikit-learn Pipelines and ColumnTransformer so transformations are fitted within each training fold.

For a dataset of this size, 5-fold cross-validation is a reasonable starting point, subject to data quality and runtime.

## 9. Models

Minimum set:
1. DummyRegressor using mean or median prediction.
2. LinearRegression.
3. RandomForestRegressor.
4. HistGradientBoostingRegressor.

Use the same folds and split for comparable experiments. Any hyperparameter tuning must occur inside training data only. Keep model search modest and document the search space.

Linear Regression provides a baseline; tree models test whether nonlinear relationships improve prediction. Model selection must not be based on test-set performance.

## 10. Evaluation metrics

Report:
- **MAE:** mean absolute prediction error in GPA units.
- **RMSE:** root mean squared prediction error in GPA units, more sensitive to large errors.
- **R²:** proportion of target variance accounted for relative to the mean baseline; it may be negative on held-out data.

Report cross-validation mean and standard deviation, final test-set metrics, and sample size. Compare against the naïve baseline. Discuss practical magnitude in the verified GPA scale, not just which model has the best score.

## 11. Feature-group comparison

Evaluate groups A, B, and C with the same split and folds. Prefer the same complete-case target sample for all groups in the primary comparison, while using pipeline imputation for missing predictor values.

For each group, compare the best configuration selected using training cross-validation. Report:
- CV MAE, RMSE, R² (mean and standard deviation).
- Final test MAE, RMSE, R².
- Number of observations.
- Feature list and preprocessing.
- Change relative to the previous group.

Treat changes as evidence of incremental predictive value only within this dataset and validation design. Consider CV variability and confidence intervals or repeated validation if feasible.

## 12. Explainability

After final model selection:
- Use permutation importance on held-out data or cross-validated training data.
- Optionally compute SHAP values for a selected tree model.
- Inspect feature dependence plots for a small number of relevant features.
- Assess stability across folds or model families where feasible.
- Explain that correlated predictors can share or distort importance.
- Do not interpret SHAP values or coefficients as causal effects.

## 13. Robustness and integrity checks

- Compare model results against the naïve baseline.
- Check for target leakage and implausibly high performance.
- Inspect whether target values appear rounded, formula-derived, or otherwise artificial.
- Verify results are not driven by one anomalous feature or record.
- Consider a simple sensitivity analysis with and without questionable variables, but only with documented justification.
- Avoid using the test set repeatedly to guide choices.
- Do not claim generalization to a real student population without credible provenance and sampling information.

## 14. Reporting structure

1. Introduction
2. Related Work
3. Dataset and Data Quality
4. Methods
5. Results
   - Dataset profile
   - Baseline model
   - Model comparison
   - Feature-group comparison
   - Explainability
   - Robustness checks
6. Discussion
7. Limitations
8. Conclusion
9. Reproducibility/data availability statement

Do not fill results sections with hypothetical values. All reported results must be generated by executed analysis code.
