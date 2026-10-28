# Feature Analysis Summary

- Group C features analysed: 13.
- Top RF permutation-importance features: Daily_Usage_Hours, Perceived_Stress_Score, Mental_Health_Index.
- Top HGB permutation-importance features: Daily_Usage_Hours, Perceived_Stress_Score, Sleep_Duration_Hours.
- RF vs HGB importance rank correlation (Spearman): 0.687 (p=0.0095).

## Caveats

- Permutation importance measures how much predictive performance changes when a feature is shuffled; it is not a causal effect.
- Strongly correlated predictors (usage, sleep, stress, mental health) can share or split importance, and permutation can create unrealistic feature combinations.
- Linear coefficients are conditional on the other included predictors and on scaling; their signs and magnitudes should not be read causally.
- Results describe this supplied dataset, whose provenance is unverified and whose generation process may be synthetic.