# Exploratory Analysis Summary

- Analysis sample: 4,415 records with an observed target (85 excluded for missing target).
- GPA: mean 3.429, SD 0.393, range 1.90-4.00.
- Strongest absolute correlations with GPA: Daily_Usage_Hours (-0.71), Mental_Health_Index (0.66), Perceived_Stress_Score (-0.65), Sleep_Duration_Hours (0.60), Sleep_Quality_Score (0.53).
- Late_Night_Usage mean GPA: False=3.548; True=3.346.

## Interpretation limits

- Correlations describe this file only and are not causal effects.
- Daily usage, sleep, stress, and mental health are strongly inter-correlated, so univariate correlations overstate unique contributions.
- Most categorical variables (platform, device, gender, academic level, comparison frequency) show little separation in GPA; Late_Night_Usage shows a modest difference, although its unique contribution is small once numeric usage and sleep variables are included in the models.
- The analysis sample contains only records with an observed target; predictor missingness is handled later inside model pipelines.