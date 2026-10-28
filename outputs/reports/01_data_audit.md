# Data Audit Summary

- Source file: `Social_media_impact_on_life.csv` (SHA-256 `893bcfc9165dc2d92caf2511bc10a0753b67df22c2a370a43b513f2417468a3f`).
- Shape: 4,500 rows x 16 columns; missing expected columns: none.
- Duplicate rows: 0; duplicated Student_IDs: 0; unique IDs: 4,500.
- Target range: 1.90 to 4.00; mean 3.429; missing 85; usable observations 4,415.
- Observations recorded exactly at the maximum GPA: 353 (possible ceiling/cap).
- Predictor-pair correlations above |0.70|: 7. Strongest: Daily_Usage_Hours-Mental_Health_Index r=-0.85; Sleep_Duration_Hours-Mental_Health_Index r=0.77; Perceived_Stress_Score-Mental_Health_Index r=-0.77; Daily_Usage_Hours-Perceived_Stress_Score r=0.75; Sleep_Duration_Hours-Sleep_Quality_Score r=0.75.
- Overall_Impact category means: Beneficial=3.55; Negative=2.54; Neutral=3.00.

## Provenance status

**Unknown.** No codebook, collection statement, or provider metadata accompanies the CSV. The file nevertheless shows several synthetic-data fingerprints: values on regular decimal grids, a near-uniform last decimal digit of the GPA, a capped maximum, and extremely strong multicollinearity among usage/sleep/stress/mental-health variables. Most categorical variables have near-zero association with the target, with Late_Night_Usage a modest exception (eta-squared about 0.06). These patterns are consistent with simulated data and do not establish real-student provenance.

## Leakage decisions

- `Student_ID` is excluded from all predictors.
- `Overall_Impact` is excluded from default models. It is strongly associated with the target (eta-squared about 0.44) and its derivation is undocumented, so it cannot be treated as a legitimate predictor.
- The target is never used to construct predictors.
- Missing target rows are excluded from supervised analysis; predictor missingness is imputed inside pipelines.

## Modelling gate

Audit checks pass for modelling **only under these exclusions**. Findings describe the supplied file; they must not be generalized to a real student population without documented provenance.