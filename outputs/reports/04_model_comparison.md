# Feature-Group Comparison Summary

- Common eligible sample: n=4,415 (train 3,532, test 883); identical split and folds for all groups.
- Group sizes: A=9 input features, B=11, C=13.
- Best configuration by training CV MAE: group C / Linear Regression (CV MAE 0.2163 +/- 0.0035; test MAE 0.2179, RMSE 0.2692, R2 0.5671).
- Mean test MAE by group across models: A=0.2562; B=0.2527; C=0.2490.
- Mean CV MAE by group across models: A=0.2489; B=0.2450; C=0.2415.

## Interpretation

- Adding sleep variables (B vs A) and stress/mental-health indicators (C vs B) is judged against fold-to-fold CV variability, not single decimal differences.
- Because the predictors are strongly collinear, incremental groups may redistribute rather than add information; this is a predictive comparison, not causal mediation.
- The test set was not used to choose any configuration. Test metrics are final descriptive evaluations of fixed pipelines.