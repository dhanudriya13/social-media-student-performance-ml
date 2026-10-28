# Project Scope

## 1. Project title

**Predicting Student Academic Performance from Social Media Usage and Lifestyle Factors: A Comparative Machine Learning Study**

## 2. Project overview

This project investigates the predictive performance of machine learning regression models for `Academic_Performance_GPA` using social media usage, lifestyle, sleep, psychological, and demographic variables in the supplied dataset `Social_media_impact_on_life.csv`.

The study compares defined feature groups to examine whether sleep and psychological indicators add predictive information beyond social media usage features. It also compares a simple interpretable baseline with nonlinear tree-based models and explores feature contributions after evaluation is complete.

The intended implementation environment is Python in Google Colab, with OpenCode supporting code generation, refactoring, testing, and documentation.

## 3. Research objectives

1. Audit the dataset and establish, as far as possible, its provenance, variable definitions, and suitability for academic-performance prediction.
2. Quantify how accurately selected regression models predict the recorded GPA values.
3. Compare model performance across pre-defined feature groups.
4. Examine whether adding sleep and psychological indicators improves held-out predictive performance beyond social media usage variables.
5. Identify features that contribute to model predictions and assess whether these interpretations are robust.
6. Produce a reproducible workflow, visualizations, performance tables, and a report with clear limitations.

## 4. Research questions

- **RQ1:** How accurately can regression models predict `Academic_Performance_GPA` from the available social media, lifestyle, and demographic variables?
- **RQ2:** Does adding sleep-related variables improve predictive performance compared with using social media usage variables alone?
- **RQ3:** Does adding perceived stress and mental health indicators provide further predictive information?
- **RQ4:** Which predictors contribute most to model predictions, and how consistent are these contributions across model families and feature groups?

## 5. Study design and unit of analysis

- **Design:** Observational, cross-sectional, secondary-data predictive study.
- **Unit of analysis:** One student record, after validating uniqueness and data integrity.
- **Primary target:** `Academic_Performance_GPA`.
- **Environment:** Python, Google Colab, pandas, NumPy, scikit-learn, Matplotlib, and optionally SHAP.
- **Primary task:** Regression, assuming GPA is a numeric continuous outcome and its coding/scale are verified.

The project must not claim that every row is a unique real student until the dataset's provenance and identifier behavior have been checked.

## 6. Scope of features and experiments

Feature groups are designed to test incremental predictive value. The final groups must be confirmed after auditing variable definitions and checking for leakage.

### Experiment A — Social media and demographic baseline
Candidate features:
- `Daily_Usage_Hours`
- `Weekend_Extra_Hours`
- `Late_Night_Usage`
- `Social_Comparison_Frequency`
- `Primary_Platform`
- `Device_Type`
- `Age`
- `Gender`
- `Academic_Level`

Purpose: establish a baseline using social media usage and demographic/context variables.

### Experiment B — Add sleep variables
Experiment A plus:
- `Sleep_Duration_Hours`
- `Sleep_Quality_Score`

Purpose: evaluate whether sleep variables add predictive value.

### Experiment C — Add psychological indicators
Experiment B plus:
- `Perceived_Stress_Score`
- `Mental_Health_Index`

Purpose: evaluate whether stress and mental health indicators add predictive value.

`Overall_Impact` is excluded from the default feature groups because its derivation is unknown and it may encode the target or related variables. It can only be considered if its provenance is documented and leakage is ruled out. `Student_ID` must never be a predictor.

## 7. In scope

- Schema and data-provenance audit.
- Missingness, duplicate, outlier, type, and range checks.
- Exploratory data analysis and descriptive statistics.
- Target-specific missing-value handling and sample-size reporting.
- Leakage checks and investigation of suspiciously strong or deterministic relationships.
- A naïve baseline predictor using the training-set mean or median.
- Linear Regression as an interpretable baseline.
- Random Forest Regressor and HistGradientBoostingRegressor as nonlinear models.
- Optional additional models only when justified, such as Ridge Regression or Gradient Boosting Regressor.
- Repeated or cross-validated evaluation with a held-out test set.
- MAE, RMSE, and R² reporting.
- Comparison of feature groups using the same evaluation protocol.
- Feature importance and model explanation after model evaluation.
- Export of tables, figures, model metadata, and a concise research report.
- Reproducible execution in Google Colab.

## 8. Out of scope

- Causal inference or claims that social media use causes changes in GPA.
- Predicting future GPA over time; no longitudinal structure is assumed.
- Student-level intervention recommendations or high-stakes decisions.
- Deployment of a production application or dashboard.
- Use of external student records or scraping additional personal data.
- Treating feature importance or SHAP values as causal effects.
- Using `Student_ID` as a predictor.
- Using `Academic_Performance_GPA` or any target-derived information to construct predictors.
- Using `Overall_Impact` without confirming its meaning and ruling out target leakage.
- Claims about real student populations unless data provenance, sampling, and collection methods support them.

## 9. Primary deliverables

1. Data audit and provenance/limitations note.
2. Reproducible Colab notebook for data auditing.
3. EDA notebook with descriptive statistics and visualizations.
4. Modeling notebook with baseline models and feature-group experiments.
5. Model comparison table including sample sizes and metrics.
6. Feature analysis notebook with importance/explainability outputs.
7. Exported figures and tables.
8. Concise report with methods, results, limitations, and reproducibility instructions.
9. `requirements.txt` and code/test documentation.

## 10. Success criteria

The project is complete when:
- The source data remain unchanged.
- The schema and target scale have been verified.
- Data provenance is documented as known, unknown, or unresolved rather than guessed.
- Leakage risks are checked and documented.
- Every experiment uses a consistent and defensible evaluation protocol.
- Preprocessing is fitted only on training folds.
- Results include a naïve baseline, held-out test metrics, and cross-validation results.
- Feature-group comparisons report both predictive performance and the exact sample sizes.
- Feature explanations are generated only after the evaluation pipeline is validated.
- All figures and tables can be regenerated.
- Conclusions distinguish predictive association from causality and disclose limitations.
