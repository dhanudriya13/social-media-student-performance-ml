# Baseline Model Summary

- Feature group C (13 input features); training n=3,532, test n=883.
- Selected by training CV MAE: **Linear Regression** (CV MAE 0.2163).
- Held-out best MAE: 0.2179 (Linear Regression); RMSE 0.2692; R2 0.5671.
- Naive mean baseline held-out MAE: 0.3318; RMSE 0.4094; R2 -0.0012.

The test partition was evaluated once with configurations fixed from training-only CV. Differences should be read against the fold-to-fold CV standard deviations reported above, not as evidence of a uniquely superior model.