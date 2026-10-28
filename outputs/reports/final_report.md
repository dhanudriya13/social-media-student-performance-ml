# Predicting Student Academic Performance from Social Media Usage and Lifestyle Factors: A Comparative Machine Learning Study

*Report generated from the executed notebooks in this repository. All metrics below are produced by code from the supplied CSV; none are typed by hand.*

## Abstract

This study evaluates whether `Academic_Performance_GPA` can be predicted from social media usage, demographic/context, sleep, and psychological variables in `Social_media_impact_on_life.csv` (4,500 rows × 16 columns). After excluding 85 rows with a missing target, a common eligible sample of 4,415 records was split 80/20 with a fixed seed. Four regression approaches (naïve mean, Linear Regression, Random Forest, HistGradientBoosting) were compared on three nested feature groups (A: social media/demographics; B: A + sleep; C: B + stress/mental health) using 5-fold cross-validation on the training partition only. Linear Regression on group C was the best configuration by training CV MAE (0.2163 ± 0.0035) and achieved held-out MAE 0.2179, RMSE 0.2692, and R² 0.5671, against a naïve baseline held-out MAE of 0.3318 (R² ≈ 0). Adding sleep and then stress/mental-health variables produced small but consistent improvements. Daily usage hours was the dominant predictor in both tree models. Data provenance is unknown and the file shows strong synthetic-data fingerprints, so conclusions apply only to this supplied dataset.

## 1. Introduction

The project asks four questions:

- **RQ1:** How accurately can regression models predict GPA from the available variables?
- **RQ2:** Do sleep variables improve prediction beyond social media and demographic/context features?
- **RQ3:** Do perceived stress and mental-health indicators add further predictive information?
- **RQ4:** Which features contribute most to predictions, and are those contributions consistent across models?

The design is observational, cross-sectional, and predictive. It does not estimate causal effects.

## 2. Related Work

The comparison follows a standard bias–variance logic: a naïve mean baseline establishes the minimum useful performance; Linear Regression captures additive linear structure; Random Forests and gradient boosting capture nonlinearities and interactions. Random Forests are bagged decision-tree ensembles (Breiman, 2001); gradient boosting and regularized linear models are described in Hastie et al. (2009). Implementations follow scikit-learn (Pedregosa et al., 2011). Permutation importance is used as a model-agnostic, predictive attribution and is interpreted cautiously (Fisher et al., 2019). No systematic literature review is claimed.

## 3. Dataset and Data Quality

### 3.1 Provenance

**Status: unknown/unresolved.** The CSV is not accompanied by a codebook, sampling statement, collection date, or provider metadata. It nevertheless exhibits several fingerprints consistent with simulated data:

- numeric predictors lie on regular 0.1 grids;
- GPA is recorded to two decimals with a near-uniform last digit;
- 353 records sit exactly at the maximum of 4.00, indicating a ceiling/cap;
- seven predictor pairs exceed |r| = 0.70, peaking at Daily_Usage_Hours–Mental_Health_Index (r = −0.85);
- most categorical variables have near-zero association with GPA (η² < 0.001); `Late_Night_Usage` is a modest exception (η² ≈ 0.065), while numeric variables carry nearly all of the signal.

These observations do **not** prove synthetic origin, but they justify restricting all conclusions to the supplied file.

### 3.2 Schema and integrity

| Check | Result |
|---|---|
| Rows × columns | 4,500 × 16 |
| Missing expected columns | None |
| Unexpected columns | None |
| Exact duplicate rows | 0 |
| Duplicate `Student_ID` | 0 |
| Unique `Student_ID` | 4,500 |
| Identifier pattern | Contiguous `STU_2024000`–`STU_2028499` |
| Missing `Academic_Performance_GPA` | 85 (1.89%) |
| Missing `Perceived_Stress_Score` | 46 (1.02%) |
| Other missing values | 0 |

The raw file is preserved at `data/raw/Social_media_impact_on_life.csv` (SHA-256 `893bcfc9165dc2d92caf2511bc10a0753b67df22c2a370a43b513f2417468a3f`).

### 3.3 Target

The target is numeric and observed on a **1.90–4.00** scale, not assumed in advance. Mean = 3.429, SD = 0.393, median = 3.460. The upper bound is a hard cap (353 values at 4.00). The 85 target-missing rows were excluded from supervised analysis, leaving **n = 4,415**; target values were not imputed. Missing predictor values (stress only) are median/most-frequent imputed inside each training fold.

### 3.4 Leakage decisions

- `Student_ID` is never a predictor.
- `Overall_Impact` is excluded from all default models. It is strongly associated with the target (η² ≈ 0.44; category means: Beneficial 3.55, Neutral 3.00, Negative 2.54) and its derivation is undocumented.
- No target-derived variable is used as a predictor.

## 4. Methods

### 4.1 Feature groups

| Group | Contents | Input features |
|---|---|---|
| A | Daily/weekend/night usage, comparison frequency, platform, device, age, gender, academic level | 9 |
| B | A + sleep duration + sleep quality | 11 |
| C | B + perceived stress + mental-health index | 13 |

### 4.2 Preprocessing

All preprocessing is inside scikit-learn `Pipeline`/`ColumnTransformer` objects and is therefore fitted only on training folds:

- numeric: median imputation; `StandardScaler` for Linear Regression;
- categorical: most-frequent imputation and one-hot encoding with `handle_unknown="ignore"`;
- no target transformation, no target imputation.

### 4.3 Split and evaluation

- Common eligible sample n = 4,415; one 80/20 split with seed 42 → 3,532 train / 883 test.
- Identical train/test indices and identical 5-fold `KFold(shuffle=True, random_state=42)` folds for every model and feature group.
- Cross-validation is run **on the training partition only**.
- Random Forest and HistGradientBoosting hyperparameters were tuned with a small 3-fold grid search on the training partition (RF: `n_estimators` ∈ {300, 600}, `min_samples_leaf` ∈ {1, 2, 4}; HGB: `learning_rate` ∈ {0.03, 0.05, 0.10}, `max_leaf_nodes` ∈ {15, 31}). Best RF: 600 trees, `min_samples_leaf=4`; best HGB: `learning_rate=0.03`, `max_leaf_nodes=15`.
- Model selection used training CV MAE only. Test metrics are final descriptive evaluations; configurations were never changed in response to test results.

### 4.4 Metrics

MAE (GPA units), RMSE (GPA units), and R², reported as CV mean ± SD across folds and as a single held-out evaluation.

## 5. Results

### 5.1 Dataset profile

Descriptive results are in `outputs/reports/02_exploratory_analysis.md`. The strongest absolute correlations with GPA are Daily_Usage_Hours (−0.71), Mental_Health_Index (0.66), Perceived_Stress_Score (−0.65), Sleep_Duration_Hours (0.60), and Sleep_Quality_Score (0.53). Late-night users have a lower mean GPA (3.346) than non-late-night users (3.548). Most other categorical variables show little separation.

### 5.2 Baseline and model comparison (feature group C)

| Model | CV MAE | CV RMSE | CV R² | Test MAE | Test RMSE | Test R² |
|---|---|---|---|---|---|---|
| Dummy (mean) | 0.3090 ± 0.0037 | 0.3886 ± 0.0067 | −0.0045 ± 0.0063 | 0.3318 | 0.4094 | −0.0012 |
| **Linear Regression** | **0.2163 ± 0.0035** | **0.2665 ± 0.0028** | **0.5269 ± 0.0204** | **0.2179** | **0.2692** | **0.5671** |
| Random Forest | 0.2192 ± 0.0044 | 0.2697 ± 0.0024 | 0.5159 ± 0.0193 | 0.2225 | 0.2746 | 0.5497 |
| HistGradientBoosting | 0.2214 ± 0.0036 | 0.2725 ± 0.0031 | 0.5056 ± 0.0194 | 0.2236 | 0.2756 | 0.5463 |

Linear Regression was selected by training CV MAE. All three learned models clearly beat the naïve baseline; the tree ensembles did not outperform the linear model. This pattern is consistent with an approximately linear/additive data-generating process.

### 5.3 Feature-group comparison

Best model per group (selected by training CV MAE; Linear Regression in all three):

| Group | CV MAE | Test MAE | Test RMSE | Test R² |
|---|---|---|---|---|
| A (9 features) | 0.2244 ± 0.0047 | 0.2273 | 0.2809 | 0.5288 |
| B (11 features) | 0.2204 ± 0.0049 | 0.2234 | 0.2778 | 0.5391 |
| C (13 features) | 0.2163 ± 0.0035 | 0.2179 | 0.2692 | 0.5671 |

Linear Regression incremental changes:

| Comparison | Δ CV MAE | Δ Test MAE | Δ Test R² |
|---|---|---|---|
| B − A | −0.0040 | −0.0039 | +0.0103 |
| C − B | −0.0041 | −0.0055 | +0.0280 |

Relative to fold-to-fold CV variability (SD ≈ 0.004–0.005), both additions yield small improvements of roughly one CV standard deviation. The direction is consistent across models (RF: B−A −0.0078, C−B −0.0066 CV MAE; HGB: B−A −0.0038, C−B −0.0032). Thus sleep and stress/mental-health variables add **modest** predictive information in this dataset, but the effect is small and partly attributable to shared variance among collinear predictors.

### 5.4 Explainability

Permutation importance on the held-out set (`outputs/tables/05_permutation_importance.csv`):

- Random Forest: Daily_Usage_Hours (0.0415 MAE increase), Perceived_Stress_Score (0.0184), Mental_Health_Index (0.0135), Sleep_Duration_Hours (0.0087).
- HistGradientBoosting: Daily_Usage_Hours (0.0800), Perceived_Stress_Score (0.0211), Sleep_Duration_Hours (0.0072), Mental_Health_Index (0.0038).
- RF vs HGB importance ranks: Spearman ρ = 0.687 (p = 0.0095). The dominant feature is stable; secondary rankings differ, as expected under collinearity.
- HGB fold-level importance confirms Daily_Usage_Hours as the dominant and most stable feature (0.0687 ± 0.0079); secondary features are less stable.
- Standardized Linear Regression coefficients: Daily_Usage_Hours −0.170, Perceived_Stress_Score −0.086, Sleep_Duration_Hours +0.049; Mental_Health_Index shrinks to 0.0015 once the other correlated predictors are included.

These are **predictive attributions, not causal effects**. Correlated features can split or distort importance, and permutation can create unrealistic combinations.

### 5.5 Robustness checks

- Every model beats the naïve mean baseline on both CV and held-out data.
- Results are stable across the shared folds; differences among the three learned models are of the same order as CV SD.
- Excluding `Overall_Impact` prevents an obvious leakage-driven inflation of performance.
- The target ceiling (4.00) may compress error for high-achieving records and is not modeled explicitly.
- The incremental feature-group gains are small and should not be over-interpreted.

## 6. Discussion

RQ1: GPA is predictable to a moderate degree from these variables (held-out R² ≈ 0.57, MAE ≈ 0.22 GPA points), far better than the mean baseline.

RQ2: Adding sleep variables improves held-out MAE by about 0.004 and R² by about 0.01. The improvement is consistent in sign across models but small relative to CV variability.

RQ3: Adding stress and mental-health indicators yields a further ~0.005 MAE improvement and ~0.02–0.03 R². Again, the effect is modest. Because stress, mental health, usage, and sleep are strongly inter-correlated, the additional value is partly a redistribution of shared information rather than independent signal.

RQ4: Daily usage hours dominates permutation importance in both tree families, followed by stress and sleep/mental-health measures. Categorical and demographic variables contribute little. Rank agreement between RF and HGB is moderate (ρ ≈ 0.69), with the top feature stable.

The fact that a linear model matches or beats nonlinear ensembles suggests that, in this file, the relationships are largely additive and approximately linear.

## 7. Limitations

1. **Unknown provenance.** The dataset may be synthetic. Findings describe the file's internal patterns, not real students, and cannot support population-level claims.
2. **Cross-sectional design.** No temporal ordering; associations are not causal.
3. **Leakage-prone variables excluded.** `Overall_Impact` was withheld; its true definition is unknown.
4. **Ceiling effect.** 353 GPA values are capped at 4.00, which can bias residuals and metrics.
5. **Collinearity.** Usage, sleep, stress, and mental health overlap heavily, making individual contributions unstable.
6. **Missingness.** 85 target-missing rows were excluded; 46 missing stress values were imputed within pipelines. Predictor missingness is low but not formally MAR/MCAR-tested.
7. **Single split.** One 80/20 split and 5-fold CV; repeated nested validation would give tighter uncertainty estimates.
8. **Test reuse.** The test set was used for final descriptive evaluation in notebooks 03/04 and for permutation importance in 05. No modeling choice was based on test performance, but repeated descriptive use should be acknowledged.
9. **Optional SHAP analysis not run.** Permutation importance and partial dependence were used instead. No causal interpretation is claimed.

## 8. Conclusion

In the supplied dataset, GPA can be predicted moderately well (held-out R² ≈ 0.57) from social media usage, sleep, stress, and mental-health variables. Daily usage is the dominant predictor. Sleep and psychological indicators add small, consistent predictive gains beyond social media/demographic features, but the increments are close to cross-validation noise and reflect shared variance among collinear measures. A linear model performs as well as or better than tree ensembles, indicating predominantly additive structure. Because provenance is unresolved and synthetic fingerprints are present, the results must not be generalized to real student populations.

## 9. Reproducibility and Data Availability

- Raw data: `data/raw/Social_media_impact_on_life.csv` (unchanged; SHA-256 above).
- Execution order: notebooks `01` → `02` → `03` → `04` → `05`.
- Environment and versions: see `outputs/reports/reproducibility.md` and `requirements.txt`.
- Seeds: split and CV seed 42; randomized model components seeded with 42.
- All tables in `outputs/tables/`, figures in `outputs/figures/`, per-stage summaries in `outputs/reports/`.
- Regeneration: `bash scripts/run_pipeline.sh` (or execute notebooks in order). Tests: `python -m pytest`.

## References

- Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5–32.
- Fisher, A., Rudin, C., & Dominici, F. (2019). All Models are Wrong, but Many are Useful: Learning a Variable's Importance by Studying an Entire Class of Prediction Models Simultaneously. *JMLR*, 20(177), 1–81.
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Springer.
- Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *JMLR*, 12, 2825–2830.
