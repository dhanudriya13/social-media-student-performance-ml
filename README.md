# Student Academic Performance: Comparative Machine Learning Study

## Research title
**Predicting Student Academic Performance from Social Media Usage and Lifestyle Factors: A Comparative Machine Learning Study**

## Purpose
This project evaluates whether student academic performance, represented by `Academic_Performance_GPA`, can be predicted from social media usage and lifestyle variables in `Social_media_impact_on_life.csv`. It compares feature groups and regression algorithms using a reproducible Python workflow intended for Google Colab.

## Read first
1. `PROJECT_SCOPE.md` — study goals, research questions, scope, and deliverables.
2. `REQUIREMENTS.md` — functional, technical, data-quality, and acceptance requirements.
3. `METHODOLOGY.md` — analysis design, feature groups, evaluation, and reporting rules.
4. `AGENTS.md` — instructions for the OpenCode agent.

## Data
Expected source file: `Social_media_impact_on_life.csv`

Expected target: `Academic_Performance_GPA`

Expected fields to verify against the actual CSV:
- `Student_ID`
- `Age`
- `Gender`
- `Academic_Level`
- `Primary_Platform`
- `Daily_Usage_Hours`
- `Weekend_Extra_Hours`
- `Sleep_Duration_Hours`
- `Sleep_Quality_Score`
- `Perceived_Stress_Score`
- `Mental_Health_Index`
- `Academic_Performance_GPA`
- `Social_Comparison_Frequency`
- `Device_Type`
- `Late_Night_Usage`
- `Overall_Impact`

Do not assume the schema or variable definitions without checking the file. The target has missing values in the supplied dataset, so the final target-specific sample size must be reported.

## Main research questions
- RQ1: How well can selected regression models predict GPA in the supplied dataset?
- RQ2: Does adding sleep variables improve predictive performance beyond social media usage variables?
- RQ3: Does adding perceived stress and mental health indicators provide further predictive information?
- RQ4: Which features contribute most to predictions, and are those contributions stable across models or feature groups?

## Important interpretation limits
- Predicting GPA is not the same as explaining or proving causes of academic performance.
- Social media rating/usage variables may be self-reported, synthetic, simulated, or otherwise subject to unknown data-generation processes. Establish provenance before making population-level claims.
- If the dataset is synthetic, findings describe its construction and internal patterns, not real students.
- A model's feature importance or SHAP values are not causal effects.
- This dataset appears cross-sectional; do not claim temporal or causal relationships.
- `Student_ID` is an identifier and must never be used as a predictor.
- `Overall_Impact` may be derived from other columns. Do not use it as a predictor unless its meaning and derivation are established and leakage is ruled out.

## Suggested structure
```text
.
├── AGENTS.md
├── README.md
├── PROJECT_SCOPE.md
├── REQUIREMENTS.md
├── METHODOLOGY.md
├── requirements.txt
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01_data_audit.ipynb
│   ├── 02_exploratory_analysis.ipynb
│   ├── 03_baseline_models.ipynb
│   ├── 04_model_comparison.ipynb
│   └── 05_feature_analysis.ipynb
├── src/
├── outputs/
│   ├── figures/
│   ├── tables/
│   └── reports/
└── tests/
```

## Getting started
Place the CSV in `data/raw/` for local execution or upload it to Google Colab. Keep the source file unchanged. Start by completing the data audit; do not begin model training until schema, provenance questions, target validity, and leakage risks have been reviewed.

## Running the project

Local (Python 3.9+):

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
bash scripts/run_pipeline.sh
```

`scripts/run_pipeline.sh` runs the tests and executes notebooks 01–05 in order. To execute a single notebook, use:

```bash
.venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/01_data_audit.ipynb
```

In Colab, upload the CSV and set `SM_DATA_PATH` to its location, or place it in `data/raw/`. Notebooks insert `src/` on the path automatically.

## Implementation status

All five notebooks are implemented and have been executed:

| Stage | Notebook | Main outputs |
|---|---|---|
| Data audit | `01_data_audit.ipynb` | Schema, missingness, IDs, target, leakage checks |
| EDA | `02_exploratory_analysis.ipynb` | Distributions, correlations, category summaries |
| Baseline models | `03_baseline_models.ipynb` | Dummy, Linear, RF, HGB on group C |
| Comparison | `04_model_comparison.ipynb` | Groups A/B/C × 4 models, same split/folds |
| Feature analysis | `05_feature_analysis.ipynb` | Permutation importance, coefficients, PDP |

Headline result: on the common eligible sample (n = 4,415; 3,532 train / 883 test), Linear Regression with group C achieved held-out MAE 0.2179, RMSE 0.2692, R² 0.5671, versus a naïve mean baseline MAE of 0.3318. Daily usage hours was the dominant predictor. See `outputs/reports/final_report.md`.

**Important:** data provenance is unresolved and the file shows synthetic-data fingerprints. Results describe this dataset only and are not causal or generalizable to real student populations.
