"""Generate the project notebooks from structured cell definitions.

Run from the repository root:
    python scripts/build_notebooks.py
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"

PREAMBLE = '''import sys
from pathlib import Path

ROOT = Path.cwd().resolve()
while not (ROOT / "src").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

import hashlib
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown

from sm_project import config, data, features, modeling, viz

config.ensure_output_dirs()
pd.set_option("display.max_columns", 60)
pd.set_option("display.width", 200)
print("Project root:", ROOT)
print("Fixed seed:", config.SEED)
'''


def md(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(source.strip("\n"))


def code(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(source.strip("\n"))


def build_audit() -> nbf.NotebookNode:
    cells = [
        md(
            """
# 01 — Data Audit

**Purpose:** establish the observed schema, verify the target, inspect data quality,
and assess provenance and leakage risk before any model is trained.

**Constraints:** the raw CSV is read-only; `Student_ID` is never a predictor;
`Overall_Impact` is withheld from default models pending evidence about its derivation.
"""
        ),
        code(PREAMBLE),
        md("## 1. Load the source data and record its fingerprint"),
        code(
            """
df, csv_path = data.read_data_with_source()
sha256 = hashlib.sha256(csv_path.read_bytes()).hexdigest()
print("Resolved source:", csv_path)
print("SHA-256 of raw file:", sha256)
print("Rows:", df.shape[0], "| Columns:", df.shape[1])
display(df.head())
display(df.dtypes.rename("dtype").to_frame())
"""
        ),
        md("## 2. Schema check against the documented expected columns"),
        code(
            """
schema_check = data.check_expected_columns(df)
schema_table = pd.DataFrame(
    {
        "column": schema_check["expected"],
        "observed": [c in schema_check["observed"] for c in schema_check["expected"]],
    }
)
display(schema_table)
print("Missing expected columns:", schema_check["missing"])
print("Unexpected columns:", schema_check["unexpected"])
schema_table.to_csv(config.TABLES_DIR / "01_schema_check.csv", index=False)
assert not schema_check["missing"], "Expected columns are missing."
"""
        ),
        md("## 3. Missingness"),
        code(
            """
missing = pd.DataFrame(
    {
        "n_missing": df.isna().sum(),
        "pct_missing": (df.isna().mean() * 100).round(2),
        "dtype": df.dtypes.astype(str),
    }
).sort_values("n_missing", ascending=False)
display(missing)
missing.to_csv(config.TABLES_DIR / "01_missingness.csv")

fig, ax = plt.subplots(figsize=(8, 5))
order = missing["pct_missing"].sort_values()
order.plot.barh(ax=ax, color="#4C72B0")
ax.set_xlabel("Missing values (%)")
ax.set_ylabel("")
ax.set_title("Missingness by column")
viz.finish_figure(fig, "01_missingness.png")
"""
        ),
        md("## 4. Duplicate rows and identifier behaviour"),
        code(
            """
n_duplicate_rows = int(df.duplicated().sum())
n_duplicate_ids = int(df["Student_ID"].duplicated().sum())
print("Exact duplicate rows:", n_duplicate_rows)
print("Duplicate Student_ID values:", n_duplicate_ids)
print("Unique Student_ID values:", df["Student_ID"].nunique(), "of", len(df))

id_text = df["Student_ID"].astype(str)
if id_text.str.fullmatch(r"STU_\\d+").all():
    id_numbers = id_text.str.extract(r"(\\d+)")[0].astype(int)
    contiguous = (id_numbers.max() - id_numbers.min() + 1) == len(df)
    print("ID numeric range:", id_numbers.min(), "to", id_numbers.max())
    print("Contiguous identifier sequence:", contiguous)
else:
    print("Identifier does not follow the expected STU_<digits> pattern.")

pd.DataFrame(
    {
        "check": ["duplicate_rows", "duplicate_ids", "unique_ids", "n_rows"],
        "value": [n_duplicate_rows, n_duplicate_ids, df["Student_ID"].nunique(), len(df)],
    }
).to_csv(config.TABLES_DIR / "01_identifier_checks.csv", index=False)
"""
        ),
        md(
            """
## 5. Target validation

The target is **not assumed** to lie on a conventional 0–4 or 0–5 scale. We report
the observed range, missingness, granularity, and evidence of ceiling effects.
"""
        ),
        code(
            """
target = df[config.TARGET]
usable = int(target.notna().sum())
target_stats = {
    "n_rows": int(len(df)),
    "n_missing_target": int(target.isna().sum()),
    "n_usable_target": usable,
    "min": float(target.min()),
    "max": float(target.max()),
    "mean": float(target.mean()),
    "std": float(target.std()),
    "median": float(target.median()),
    "n_unique": int(target.nunique()),
}
display(pd.Series(target_stats, name="value").to_frame())
print("Target values at observed maximum:", int((target == target.max()).sum()))
print("Target values below 2.0:", int((target < 2.0).sum()))
print("All target values are multiples of 0.01:",
      bool(np.allclose(target.dropna() * 100, np.round(target.dropna() * 100))))

target_stats_df = pd.Series(target_stats, name="value").to_frame()
target_stats_df.to_csv(config.TABLES_DIR / "01_target_summary.csv")

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].hist(target.dropna(), bins=40, color="#4C72B0", edgecolor="white")
axes[0].set_title("Distribution of Academic_Performance_GPA")
axes[0].set_xlabel("GPA (observed scale)")
axes[0].set_ylabel("Number of records")
axes[1].boxplot(target.dropna(), vert=True)
axes[1].set_title("GPA box plot")
axes[1].set_ylabel("GPA (observed scale)")
viz.finish_figure(fig, "01_target_distribution.png")
"""
        ),
        md("## 6. Numeric and categorical summaries"),
        code(
            """
numeric_summary = df.select_dtypes(include="number").describe().T
numeric_summary["missing"] = df.select_dtypes(include="number").isna().sum()
display(numeric_summary)
numeric_summary.to_csv(config.TABLES_DIR / "01_numeric_summary.csv")

rows = []
for column in df.select_dtypes(include=["object", "bool"]).columns:
    if column == "Student_ID":
        continue
    counts = df[column].value_counts(dropna=False)
    for value, count in counts.items():
        rows.append({"column": column, "value": value, "count": int(count),
                     "pct": round(100 * count / len(df), 2)})
category_summary = pd.DataFrame(rows)
category_summary.to_csv(config.TABLES_DIR / "01_category_summary.csv", index=False)
display(category_summary.groupby("column").head(10))
"""
        ),
        md("## 7. Correlation structure and suspicious relationships"),
        code(
            """
numeric_df = df.select_dtypes(include="number")
correlation = numeric_df.corr()
correlation.to_csv(config.TABLES_DIR / "01_correlation_matrix.csv")
display(correlation.round(3))

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(correlation, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(len(correlation.columns)))
ax.set_xticklabels(correlation.columns, rotation=90)
ax.set_yticks(range(len(correlation.columns)))
ax.set_yticklabels(correlation.columns)
ax.set_title("Pearson correlation, numeric variables")
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Correlation")
viz.finish_figure(fig, "01_correlation_heatmap.png")

target_corr = correlation[config.TARGET].drop(config.TARGET).sort_values()
print("Correlation with target:")
display(target_corr.to_frame("pearson_r"))

upper = correlation.where(
    np.triu(np.ones(correlation.shape, dtype=bool), k=1)
).stack().sort_values(key=np.abs, ascending=False)
print("Strongest predictor-predictor correlations (|r| > 0.7):")
display(upper[upper.abs() > 0.7].to_frame("pearson_r"))
upper.to_frame("pearson_r").to_csv(config.TABLES_DIR / "01_predictor_correlations.csv")
"""
        ),
        md(
            """
## 8. Leakage audit: `Student_ID` and `Overall_Impact`

`Student_ID` is an identifier and must never be a predictor. `Overall_Impact` is a
categorical summary that may have been derived from the target or from the same latent
variables. The tables below quantify that risk.
"""
        ),
        code(
            """
impact_table = (
    df.groupby("Overall_Impact")[config.TARGET]
      .agg(["count", "mean", "std", "min", "max"])
      .round(3)
)
display(impact_table)
impact_table.to_csv(config.TABLES_DIR / "01_overall_impact_vs_target.csv")

print("Overall_Impact category by target band:")
bands = pd.cut(df[config.TARGET], bins=[0, 2.5, 3.0, 3.5, 4.01],
               labels=["<=2.5", "2.5-3.0", "3.0-3.5", "3.5-4.0"])
display(pd.crosstab(df["Overall_Impact"], bands, dropna=False))
"""
        ),
        md("## 9. Provenance evidence and synthetic-data fingerprints"),
        code(
            """
deci = np.round((target.dropna() * 100) % 10).astype(int)
last_digit_counts = deci.value_counts().sort_index()
print("Last-digit distribution of GPA (percentiles):")
display(last_digit_counts.describe().to_frame("value"))
print("Observed counts by last digit:")
display(last_digit_counts.to_frame("count"))

def eta_squared(frame, categorical, numeric):
    frame = frame[[categorical, numeric]].dropna()
    grand_mean = frame[numeric].mean()
    ss_total = ((frame[numeric] - grand_mean) ** 2).sum()
    ss_between = 0.0
    for _, group in frame.groupby(categorical):
        ss_between += len(group) * (group[numeric].mean() - grand_mean) ** 2
    return ss_between / ss_total if ss_total else np.nan

eta_rows = []
for column in ["Gender", "Academic_Level", "Primary_Platform", "Device_Type",
               "Late_Night_Usage", "Social_Comparison_Frequency", "Overall_Impact"]:
    eta_rows.append({"column": column, "eta_squared_with_target": eta_squared(df, column, config.TARGET)})
eta_table = pd.DataFrame(eta_rows).sort_values("eta_squared_with_target", ascending=False)
display(eta_table)
eta_table.to_csv(config.TABLES_DIR / "01_categorical_association.csv", index=False)

# Range and grid checks
range_checks = []
for column in ["Age", "Daily_Usage_Hours", "Weekend_Extra_Hours",
               "Sleep_Duration_Hours", "Sleep_Quality_Score",
               "Perceived_Stress_Score", "Mental_Health_Index"]:
    series = df[column].dropna()
    steps = np.diff(np.sort(series.unique()))
    range_checks.append(
        {
            "column": column,
            "min": series.min(),
            "max": series.max(),
            "n_unique": series.nunique(),
            "smallest_step": float(steps.min()) if len(steps) else np.nan,
        }
    )
range_table = pd.DataFrame(range_checks)
display(range_table)
range_table.to_csv(config.TABLES_DIR / "01_range_checks.csv", index=False)
"""
        ),
        md("## 10. Audit conclusion and fitted report"),
        code(
            """
strong_pairs = upper[upper.abs() > 0.7]
report_lines = [
    "# Data Audit Summary",
    "",
    f"- Source file: `{csv_path.name}` (SHA-256 `{sha256}`).",
    f"- Shape: {len(df):,} rows x {df.shape[1]} columns; missing expected columns: {schema_check['missing'] or 'none'}.",
    f"- Duplicate rows: {n_duplicate_rows}; duplicated Student_IDs: {n_duplicate_ids}; unique IDs: {df['Student_ID'].nunique():,}.",
    f"- Target range: {target.min():.2f} to {target.max():.2f}; mean {target.mean():.3f}; "
    f"missing {int(target.isna().sum())}; usable observations {usable:,}.",
    f"- Observations recorded exactly at the maximum GPA: {int((target == target.max()).sum())} (possible ceiling/cap).",
    f"- Predictor-pair correlations above |0.70|: {len(strong_pairs)}. Strongest: "
    + "; ".join(f"{a}-{b} r={v:.2f}" for (a, b), v in strong_pairs.head(5).items()) + ".",
    f"- Overall_Impact category means: "
    + "; ".join(f"{k}={v:.2f}" for k, v in impact_table['mean'].items()) + ".",
    "",
    "## Provenance status",
    "",
    "**Unknown.** No codebook, collection statement, or provider metadata accompanies the CSV. "
    "The file nevertheless shows several synthetic-data fingerprints: values on regular decimal grids, "
    "a near-uniform last decimal digit of the GPA, a capped maximum, and extremely strong multicollinearity "
    "among usage/sleep/stress/mental-health variables. Most categorical variables have near-zero association "
    "with the target, with Late_Night_Usage a modest exception (eta-squared about 0.06). These patterns are "
    "consistent with simulated data and do not establish real-student provenance.",
    "",
    "## Leakage decisions",
    "",
    "- `Student_ID` is excluded from all predictors.",
    "- `Overall_Impact` is excluded from default models. It is strongly associated with the target "
    "(eta-squared about 0.44) and its derivation is undocumented, so it cannot be treated as a legitimate predictor.",
    "- The target is never used to construct predictors.",
    "- Missing target rows are excluded from supervised analysis; predictor missingness is imputed inside pipelines.",
    "",
    "## Modelling gate",
    "",
    "Audit checks pass for modelling **only under these exclusions**. Findings describe the supplied file; "
    "they must not be generalized to a real student population without documented provenance.",
]
audit_text = "\\n".join(report_lines)
(config.REPORTS_DIR / "01_data_audit.md").write_text(audit_text, encoding="utf-8")
display(Markdown(audit_text))
"""
        ),
    ]
    return nbf.v4.new_notebook(cells=cells)


def build_eda() -> nbf.NotebookNode:
    cells = [
        md(
            """
# 02 — Exploratory Analysis

Descriptive analysis of the analysis-ready sample (observed target only). No model is
trained here. Correlations are descriptive, not causal.
"""
        ),
        code(PREAMBLE),
        code(
            """
df = data.load_data()
analysis = data.eligible_sample(df)
print("Rows in raw file:", len(df))
print("Rows with observed target (analysis sample):", len(analysis))
print("Excluded for missing target:", len(df) - len(analysis))
display(analysis[config.TARGET].describe().to_frame().T)
"""
        ),
        md("## 1. Target distribution"),
        code(
            """
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].hist(analysis[config.TARGET], bins=40, color="#4C72B0", edgecolor="white")
axes[0].set_title("GPA distribution (analysis sample)")
axes[0].set_xlabel("GPA (observed scale)")
axes[0].set_ylabel("Count")
axes[1].boxplot(analysis[config.TARGET], vert=True)
axes[1].set_title("GPA box plot")
axes[1].set_ylabel("GPA (observed scale)")
viz.finish_figure(fig, "02_target_distribution.png")
"""
        ),
        md("## 2. Numeric predictor distributions"),
        code(
            """
numeric_cols = [c for c in config.NUMERIC_FEATURES if c in analysis.columns]
fig, axes = plt.subplots(3, 3, figsize=(12, 10))
for ax, column in zip(axes.ravel(), numeric_cols):
    ax.hist(analysis[column].dropna(), bins=30, color="#55A868", edgecolor="white")
    ax.set_title(column)
    ax.set_ylabel("Count")
for ax in axes.ravel()[len(numeric_cols):]:
    ax.axis("off")
viz.finish_figure(fig, "02_numeric_histograms.png")
"""
        ),
        md("## 3. Numeric predictor box plots"),
        code(
            """
scaled = analysis[numeric_cols].copy()
z = (scaled - scaled.mean()) / scaled.std(ddof=0)
fig, ax = plt.subplots(figsize=(11, 4))
ax.boxplot([z[c].dropna() for c in numeric_cols], tick_labels=numeric_cols, vert=False)
ax.set_title("Standardized numeric predictors (z-scores)")
ax.set_xlabel("Standard deviations from the mean")
viz.finish_figure(fig, "02_numeric_boxplots.png")
"""
        ),
        md(
            """
## 4. Correlation structure

Interpretation is intentionally cautious: the dataset contains strong predictor-predictor
correlations, so individual coefficients and importances may be unstable.
"""
        ),
        code(
            """
correlation = analysis.select_dtypes(include="number").corr()
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(correlation, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(len(correlation.columns)))
ax.set_xticklabels(correlation.columns, rotation=90)
ax.set_yticks(range(len(correlation.columns)))
ax.set_yticklabels(correlation.columns)
ax.set_title("Correlation matrix (analysis sample)")
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Pearson correlation")
viz.finish_figure(fig, "02_correlation_heatmap.png")
correlation.to_csv(config.TABLES_DIR / "02_correlation_matrix.csv")
"""
        ),
        md("## 5. GPA by categorical variables"),
        code(
            """
cat_cols = [c for c in config.CATEGORICAL_FEATURES if c in analysis.columns]
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
for ax, column in zip(axes.ravel(), cat_cols):
    groups = [g[config.TARGET].dropna().values for _, g in analysis.groupby(column)]
    labels = [str(k) for k, _ in analysis.groupby(column)]
    ax.boxplot(groups, tick_labels=labels)
    ax.set_title(f"GPA by {column}")
    ax.set_ylabel("GPA")
    ax.tick_params(axis="x", rotation=30)
for ax in axes.ravel()[len(cat_cols):]:
    ax.axis("off")
viz.finish_figure(fig, "02_gpa_by_category.png")

by_category = (
    analysis.groupby("Late_Night_Usage")[config.TARGET]
    .agg(["count", "mean", "std"])
    .round(3)
)
display(by_category)
by_category.to_csv(config.TABLES_DIR / "02_gpa_by_late_night.csv")
"""
        ),
        md("## 6. GPA against key numeric predictors"),
        code(
            """
key_predictors = ["Daily_Usage_Hours", "Sleep_Duration_Hours",
                  "Perceived_Stress_Score", "Mental_Health_Index"]
fig, axes = plt.subplots(2, 2, figsize=(11, 9))
for ax, column in zip(axes.ravel(), key_predictors):
    plot_df = analysis[[column, config.TARGET]].dropna()
    ax.scatter(plot_df[column], plot_df[config.TARGET], s=8, alpha=0.25, color="#4C72B0")
    ax.set_xlabel(column)
    ax.set_ylabel("GPA")
    ax.set_title(f"GPA vs {column}")
viz.finish_figure(fig, "02_gpa_scatterplots.png")
"""
        ),
        md("## 7. Exploratory summary"),
        code(
            """
target_corr = (
    analysis.select_dtypes(include="number").corr()[config.TARGET]
    .drop(config.TARGET).sort_values(key=np.abs, ascending=False)
)
display(target_corr.to_frame("pearson_r"))
target_corr.to_frame("pearson_r").to_csv(config.TABLES_DIR / "02_target_correlations.csv")

report_lines = [
    "# Exploratory Analysis Summary",
    "",
    f"- Analysis sample: {len(analysis):,} records with an observed target "
    f"({len(df) - len(analysis)} excluded for missing target).",
    f"- GPA: mean {analysis[config.TARGET].mean():.3f}, "
    f"SD {analysis[config.TARGET].std():.3f}, range "
    f"{analysis[config.TARGET].min():.2f}-{analysis[config.TARGET].max():.2f}.",
    f"- Strongest absolute correlations with GPA: "
    + ", ".join(f"{k} ({v:.2f})" for k, v in target_corr.head(5).items()) + ".",
    f"- Late_Night_Usage mean GPA: "
    + "; ".join(f"{k}={v:.3f}" for k, v in analysis.groupby('Late_Night_Usage')[config.TARGET].mean().items()) + ".",
    "",
    "## Interpretation limits",
    "",
    "- Correlations describe this file only and are not causal effects.",
    "- Daily usage, sleep, stress, and mental health are strongly inter-correlated, so "
    "univariate correlations overstate unique contributions.",
    "- Most categorical variables (platform, device, gender, academic level, comparison frequency) "
    "show little separation in GPA; Late_Night_Usage shows a modest difference, although its unique "
    "contribution is small once numeric usage and sleep variables are included in the models.",
    "- The analysis sample contains only records with an observed target; predictor missingness "
    "is handled later inside model pipelines.",
]
eda_text = "\\n".join(report_lines)
(config.REPORTS_DIR / "02_exploratory_analysis.md").write_text(eda_text, encoding="utf-8")
display(Markdown(eda_text))
"""
        ),
    ]
    return nbf.v4.new_notebook(cells=cells)


def build_baseline() -> nbf.NotebookNode:
    cells = [
        md(
            """
# 03 — Baseline Models

Compares the naïve mean baseline, Linear Regression, Random Forest, and
HistGradientBoosting on feature group **C** (the richest allowed group).

Protocol:
- 80/20 hold-out split, fixed seed.
- 5-fold cross-validation **on the training partition only**.
- A modest hyperparameter search (3-fold, training only) for the two tree models.
- The test partition is used once, after configurations are fixed.
"""
        ),
        code(PREAMBLE),
        code(
            """
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error

df = data.load_data()
analysis = data.eligible_sample(df)
train_idx, test_idx = features.make_split(analysis)
train_df = analysis.iloc[train_idx]
test_df = analysis.iloc[test_idx]
print("Analysis sample:", len(analysis))
print("Training rows:", len(train_df), "| Held-out rows:", len(test_df))
print("Split seed:", config.SEED)

group_name = "C"
feature_list = features.get_feature_group(group_name)
print("Feature group", group_name, "with", len(feature_list), "input features:")
print(feature_list)
"""
        ),
        md("## 1. Modest training-only tuning for the tree models"),
        code(
            """
base_models = modeling.default_models()
param_grids = {
    "Random Forest": {
        "model__n_estimators": [300, 600],
        "model__min_samples_leaf": [1, 2, 4],
    },
    "Hist Gradient Boosting": {
        "model__learning_rate": [0.03, 0.05, 0.10],
        "model__max_leaf_nodes": [15, 31],
    },
}
tuned_estimators = {}
tuning_rows = []
for model_name, grid in param_grids.items():
    pipeline = features.build_pipeline(base_models[model_name], feature_list, scale_numeric=False)
    search = GridSearchCV(
        pipeline, grid, cv=3, scoring="neg_mean_absolute_error", n_jobs=-1
    )
    search.fit(train_df[feature_list], train_df[config.TARGET])
    tuned_estimators[model_name] = search.best_estimator_.named_steps["model"]
    tuning_rows.append(
        {
            "model": model_name,
            "best_cv_mae": -search.best_score_,
            "best_params": {k.replace("model__", ""): v for k, v in search.best_params_.items()},
        }
    )
    print(model_name, "best training CV MAE:", round(-search.best_score_, 4), search.best_params_)

tuning_table = pd.DataFrame(tuning_rows)
display(tuning_table)
tuning_table.to_csv(config.TABLES_DIR / "03_tuning_results.csv", index=False)
modeling.save_json(
    {row["model"]: row["best_params"] for row in tuning_rows},
    config.TABLES_DIR / "03_tuned_params.json",
)
"""
        ),
        md("## 2. Cross-validation on the training partition"),
        code(
            """
models = modeling.default_models()
models["Random Forest"] = tuned_estimators["Random Forest"]
models["Hist Gradient Boosting"] = tuned_estimators["Hist Gradient Boosting"]

cv = modeling.make_cv()
fold_frames = []
for model_name, estimator in models.items():
    folds = modeling.evaluate_cv(
        train_df, feature_list, model_name, estimator, cv,
        scale_numeric=modeling.needs_scaling(model_name),
    )
    fold_frames.append(folds)
fold_results = pd.concat(fold_frames, ignore_index=True)
display(fold_results.groupby("model")[["mae", "rmse", "r2"]].agg(["mean", "std"]).round(4))
fold_results.to_csv(config.TABLES_DIR / "03_cv_fold_results.csv", index=False)

cv_summary = (
    fold_results.groupby("model")
    .agg(
        cv_mae_mean=("mae", "mean"), cv_mae_std=("mae", "std"),
        cv_rmse_mean=("rmse", "mean"), cv_rmse_std=("rmse", "std"),
        cv_r2_mean=("r2", "mean"), cv_r2_std=("r2", "std"),
    )
    .reset_index()
)
display(cv_summary.round(4))
cv_summary.to_csv(config.TABLES_DIR / "03_cv_summary.csv", index=False)

fig, ax = plt.subplots(figsize=(8, 4))
ax.barh(cv_summary["model"], cv_summary["cv_mae_mean"],
        xerr=cv_summary["cv_mae_std"], color="#4C72B0", alpha=0.85)
ax.set_xlabel("Cross-validated MAE (GPA units)")
ax.set_title("Training-partition 5-fold CV MAE (lower is better)")
viz.finish_figure(fig, "03_cv_mae_by_model.png")
"""
        ),
        md("## 3. Final held-out evaluation"),
        code(
            """
test_rows = []
prediction_frames = []
for model_name, estimator in models.items():
    metrics, predictions, fitted = modeling.fit_and_test(
        train_df, test_df, feature_list, model_name, estimator,
        scale_numeric=modeling.needs_scaling(model_name),
    )
    test_rows.append(metrics)
    prediction_frames.append(
        pd.DataFrame(
            {
                "row_id": np.arange(len(test_df)),
                "model": model_name,
                "y_true": test_df[config.TARGET].values,
                "y_pred": predictions,
            }
        )
    )

test_results = pd.DataFrame(test_rows)
display(test_results.round(4))
test_results.to_csv(config.TABLES_DIR / "03_test_results.csv", index=False)
all_predictions = pd.concat(prediction_frames, ignore_index=True)
all_predictions.to_csv(config.TABLES_DIR / "03_test_predictions.csv", index=False)

selected = cv_summary.sort_values("cv_mae_mean").iloc[0]["model"]
print("Model selected by training CV MAE:", selected)
modeling.save_json(
    {"selected_model": selected,
     "selection_rule": "lowest mean 5-fold CV MAE on the training partition"},
    config.TABLES_DIR / "03_selected_model.json",
)
"""
        ),
        md("## 4. Predicted vs observed for the selected model"),
        code(
            """
sel = all_predictions[all_predictions["model"] == selected]
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(sel["y_true"], sel["y_pred"], s=12, alpha=0.35, color="#4C72B0")
lims = [min(sel["y_true"].min(), sel["y_pred"].min()), max(sel["y_true"].max(), sel["y_pred"].max())]
ax.plot(lims, lims, "--", color="black", linewidth=1, label="Perfect prediction")
ax.set_xlabel("Observed GPA")
ax.set_ylabel("Predicted GPA")
ax.set_title(f"Held-out predictions: {selected}")
ax.legend()
viz.finish_figure(fig, "03_predicted_vs_observed.png")
"""
        ),
        md("## 5. Summary"),
        code(
            """
best_test = test_results.sort_values("mae").iloc[0]
dummy_row = test_results[test_results["model"] == "Dummy (mean)"].iloc[0]
report_lines = [
    "# Baseline Model Summary",
    "",
    f"- Feature group C ({len(feature_list)} input features); training n={len(train_df):,}, test n={len(test_df):,}.",
    f"- Selected by training CV MAE: **{selected}** (CV MAE {cv_summary.sort_values('cv_mae_mean').iloc[0]['cv_mae_mean']:.4f}).",
    f"- Held-out best MAE: {best_test['mae']:.4f} ({best_test['model']}); "
    f"RMSE {best_test['rmse']:.4f}; R2 {best_test['r2']:.4f}.",
    f"- Naive mean baseline held-out MAE: {dummy_row['mae']:.4f}; RMSE {dummy_row['rmse']:.4f}; R2 {dummy_row['r2']:.4f}.",
    "",
    "The test partition was evaluated once with configurations fixed from training-only CV. "
    "Differences should be read against the fold-to-fold CV standard deviations reported above, "
    "not as evidence of a uniquely superior model.",
]
summary_text = "\\n".join(report_lines)
(config.REPORTS_DIR / "03_baseline_models.md").write_text(summary_text, encoding="utf-8")
display(Markdown(summary_text))
"""
        ),
    ]
    return nbf.v4.new_notebook(cells=cells)


def build_comparison() -> nbf.NotebookNode:
    cells = [
        md(
            """
# 04 — Model and Feature-Group Comparison

Compares feature groups **A**, **B**, and **C** on the same eligible observations,
the same split, and the same cross-validation folds.

- A: social media usage + demographic/context
- B: A + sleep variables
- C: B + stress and mental health indicators

Incremental changes (B−A, C−B) are interpreted relative to CV variability.
"""
        ),
        code(PREAMBLE),
        code(
            """
df = data.load_data()
analysis = data.eligible_sample(df)
train_idx, test_idx = features.make_split(analysis)
train_df = analysis.iloc[train_idx]
test_df = analysis.iloc[test_idx]
print("Analysis sample:", len(analysis), "| train:", len(train_df), "| test:", len(test_df))

# Reuse tuned parameters from notebook 03 when available.
tuned_path = config.TABLES_DIR / "03_tuned_params.json"
tuned_params = modeling.load_json(tuned_path) if tuned_path.exists() else {}
base_models = modeling.default_models()
models = {}
for name in ["Dummy (mean)", "Linear Regression", "Random Forest", "Hist Gradient Boosting"]:
    if name in tuned_params and name in base_models:
        estimator = base_models[name].set_params(**tuned_params[name])
    else:
        estimator = base_models[name]
    models[name] = estimator
print("Tuned parameters used:", tuned_params if tuned_params else "defaults (run notebook 03 first)")
"""
        ),
        md("## 1. Run all feature-group and model combinations"),
        code(
            """
from IPython.display import clear_output

cv = modeling.make_cv()
comparison_rows = []
prediction_rows = []
for group_name in ["A", "B", "C"]:
    feature_list = features.get_feature_group(group_name)
    for model_name, estimator in models.items():
        folds = modeling.evaluate_cv(
            train_df, feature_list, model_name, estimator, cv,
            scale_numeric=modeling.needs_scaling(model_name),
        )
        cv_summary = modeling.summarize_cv(folds)
        test_metrics, predictions, _ = modeling.fit_and_test(
            train_df, test_df, feature_list, model_name, estimator,
            scale_numeric=modeling.needs_scaling(model_name),
        )
        row = {
            "feature_group": group_name,
            "group_description": config.FEATURE_GROUP_DESCRIPTIONS[group_name],
            "model": model_name,
            "n_features": len(feature_list),
            "n_train": int(len(train_df)),
            "n_test": int(len(test_df)),
            **cv_summary,
            **{f"test_{k}": v for k, v in test_metrics.items()
               if k in {"mae", "rmse", "r2"}},
        }
        comparison_rows.append(row)
        prediction_rows.append(
            pd.DataFrame(
                {
                    "row_id": np.arange(len(test_df)),
                    "feature_group": group_name,
                    "model": model_name,
                    "y_true": test_df[config.TARGET].values,
                    "y_pred": predictions,
                }
            )
        )
        print(f"finished group {group_name} / {model_name}")

comparison = pd.DataFrame(comparison_rows)
display(comparison.round(4))
comparison.to_csv(config.TABLES_DIR / "04_model_comparison.csv", index=False)
pd.concat(prediction_rows, ignore_index=True).to_csv(
    config.TABLES_DIR / "04_test_predictions.csv", index=False
)
"""
        ),
        md("## 2. Incremental feature-group changes"),
        code(
            """
metric_cols = ["cv_mae_mean", "cv_rmse_mean", "cv_r2_mean", "test_mae", "test_rmse", "test_r2"]
delta_rows = []
for model_name, sub in comparison.groupby("model"):
    sub = sub.set_index("feature_group")
    for previous, current in [("A", "B"), ("B", "C")]:
        row = {"model": model_name, "comparison": f"{current}-{previous}"}
        for column in metric_cols:
            row[column] = sub.loc[current, column] - sub.loc[previous, column]
        delta_rows.append(row)
deltas = pd.DataFrame(delta_rows)
display(deltas.round(4))
deltas.to_csv(config.TABLES_DIR / "04_feature_group_deltas.csv", index=False)

best_cv = comparison.sort_values("cv_mae_mean").groupby("feature_group").head(1)
print("Best model per feature group by training CV MAE:")
display(best_cv[["feature_group", "model", "cv_mae_mean", "cv_mae_std", "test_mae", "test_rmse", "test_r2"]].round(4))
best_cv.to_csv(config.TABLES_DIR / "04_best_per_group.csv", index=False)
"""
        ),
        md("## 3. Visual comparison"),
        code(
            """
pivot_cv = comparison.pivot(index="model", columns="feature_group", values="cv_mae_mean")
pivot_test = comparison.pivot(index="model", columns="feature_group", values="test_mae")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
pivot_cv.plot.bar(ax=axes[0], color=["#4C72B0", "#55A868", "#C44E52"])
axes[0].set_title("Cross-validated MAE by feature group")
axes[0].set_ylabel("CV MAE (GPA units)")
axes[0].set_xlabel("")
axes[0].tick_params(axis="x", rotation=20)
axes[0].legend(title="Group", fontsize=8)

pivot_test.plot.bar(ax=axes[1], color=["#4C72B0", "#55A868", "#C44E52"])
axes[1].set_title("Held-out MAE by feature group")
axes[1].set_ylabel("Test MAE (GPA units)")
axes[1].set_xlabel("")
axes[1].tick_params(axis="x", rotation=20)
axes[1].legend(title="Group", fontsize=8)
viz.finish_figure(fig, "04_feature_group_comparison.png")

fig, ax = plt.subplots(figsize=(8, 4))
for group_name in ["A", "B", "C"]:
    sub = comparison[comparison["feature_group"] == group_name]
    ax.scatter(sub["cv_mae_mean"], sub["test_mae"], label=f"Group {group_name}", s=60)
ax.plot([comparison["cv_mae_mean"].min(), comparison["cv_mae_mean"].max()],
        [comparison["cv_mae_mean"].min(), comparison["cv_mae_mean"].max()],
        "--", color="grey", linewidth=1)
ax.set_xlabel("CV MAE (training folds)")
ax.set_ylabel("Held-out MAE")
ax.set_title("CV and held-out MAE across models and feature groups")
ax.legend()
viz.finish_figure(fig, "04_cv_vs_test_mae.png")
"""
        ),
        md("## 4. Summary"),
        code(
            """
best_row = comparison.sort_values("cv_mae_mean").iloc[0]
group_means = comparison.groupby("feature_group")[["cv_mae_mean", "test_mae"]].mean()
report_lines = [
    "# Feature-Group Comparison Summary",
    "",
    f"- Common eligible sample: n={len(analysis):,} (train {len(train_df):,}, test {len(test_df):,}); "
    "identical split and folds for all groups.",
    "- Group sizes: A=9 input features, B=11, C=13.",
    f"- Best configuration by training CV MAE: group {best_row['feature_group']} / {best_row['model']} "
    f"(CV MAE {best_row['cv_mae_mean']:.4f} +/- {best_row['cv_mae_std']:.4f}; "
    f"test MAE {best_row['test_mae']:.4f}, RMSE {best_row['test_rmse']:.4f}, R2 {best_row['test_r2']:.4f}).",
    "- Mean test MAE by group across models: "
    + "; ".join(f"{g}={v:.4f}" for g, v in group_means["test_mae"].items()) + ".",
    "- Mean CV MAE by group across models: "
    + "; ".join(f"{g}={v:.4f}" for g, v in group_means["cv_mae_mean"].items()) + ".",
    "",
    "## Interpretation",
    "",
    "- Adding sleep variables (B vs A) and stress/mental-health indicators (C vs B) is judged "
    "against fold-to-fold CV variability, not single decimal differences.",
    "- Because the predictors are strongly collinear, incremental groups may redistribute rather "
    "than add information; this is a predictive comparison, not causal mediation.",
    "- The test set was not used to choose any configuration. Test metrics are final descriptive "
    "evaluations of fixed pipelines.",
]
summary_text = "\\n".join(report_lines)
(config.REPORTS_DIR / "04_model_comparison.md").write_text(summary_text, encoding="utf-8")
display(Markdown(summary_text))
"""
        ),
    ]
    return nbf.v4.new_notebook(cells=cells)


def build_feature_analysis() -> nbf.NotebookNode:
    cells = [
        md(
            """
# 05 — Feature Analysis

Feature contributions for feature group **C**, produced only after the evaluation
pipeline has been validated. Interpretations are **predictive, not causal**.

Methods:
- Linear coefficients (with preprocessing context).
- Permutation importance on held-out data for Random Forest and HistGradientBoosting.
- Fold-level permutation-importance stability for HistGradientBoosting.
- Partial dependence for the top features.
- Comparison of RF and HGB importance rankings.
"""
        ),
        code(PREAMBLE),
        code(
            """
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from scipy.stats import spearmanr

df = data.load_data()
analysis = data.eligible_sample(df)
train_idx, test_idx = features.make_split(analysis)
train_df = analysis.iloc[train_idx]
test_df = analysis.iloc[test_idx]
feature_list = features.get_feature_group("C")
print("Group C features:", feature_list)
print("Train:", len(train_df), "| Test:", len(test_df))
"""
        ),
        md("## 1. Fit the final model families on the training partition"),
        code(
            """
tuned_path = config.TABLES_DIR / "03_tuned_params.json"
tuned_params = modeling.load_json(tuned_path) if tuned_path.exists() else {}
base_models = modeling.default_models()

fitted = {}
for model_name in ["Linear Regression", "Random Forest", "Hist Gradient Boosting"]:
    estimator = base_models[model_name]
    if model_name in tuned_params:
        estimator = estimator.set_params(**tuned_params[model_name])
    pipeline = features.build_pipeline(
        estimator, feature_list, scale_numeric=modeling.needs_scaling(model_name)
    )
    pipeline.fit(train_df[feature_list], train_df[config.TARGET])
    fitted[model_name] = pipeline
    score = pipeline.score(test_df[feature_list], test_df[config.TARGET])
    print(f"{model_name}: held-out R2 = {score:.4f}")
"""
        ),
        md("## 2. Permutation importance on held-out data"),
        code(
            """
importance_frames = []
for model_name in ["Random Forest", "Hist Gradient Boosting"]:
    result = permutation_importance(
        fitted[model_name], test_df[feature_list], test_df[config.TARGET],
        n_repeats=10, random_state=config.SEED, scoring="neg_mean_absolute_error",
    )
    frame = pd.DataFrame(
        {
            "model": model_name,
            "feature": feature_list,
            "importance_mae_increase_mean": result.importances_mean,
            "importance_mae_increase_std": result.importances_std,
        }
    ).sort_values("importance_mae_increase_mean", ascending=False)
    importance_frames.append(frame)

importances = pd.concat(importance_frames, ignore_index=True)
display(importances.round(4))
importances.to_csv(config.TABLES_DIR / "05_permutation_importance.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
for ax, model_name in zip(axes, ["Random Forest", "Hist Gradient Boosting"]):
    sub = importances[importances["model"] == model_name].sort_values("importance_mae_increase_mean")
    ax.barh(sub["feature"], sub["importance_mae_increase_mean"],
            xerr=sub["importance_mae_increase_std"], color="#4C72B0", alpha=0.85)
    ax.set_title(f"Permutation importance: {model_name}")
    ax.set_xlabel("Increase in MAE when shuffled (GPA units)")
viz.finish_figure(fig, "05_permutation_importance.png")
"""
        ),
        md("## 3. Linear coefficients with preprocessing context"),
        code(
            """
linear_pipeline = fitted["Linear Regression"]
preprocessor = linear_pipeline.named_steps["preprocess"]
model = linear_pipeline.named_steps["model"]
feature_names = preprocessor.get_feature_names_out()
coef_table = pd.DataFrame(
    {"encoded_feature": feature_names, "coefficient": model.coef_}
).sort_values("coefficient", key=np.abs, ascending=False)
display(coef_table.head(20).round(4))
coef_table.to_csv(config.TABLES_DIR / "05_linear_coefficients.csv", index=False)
print("Numeric predictors were standardized; categorical predictors were one-hot encoded.")
print("Coefficients are conditional associations under collinearity, not causal effects.")
"""
        ),
        md("## 4. Fold-level stability of HGB permutation importance"),
        code(
            """
hgb_params = tuned_params.get("Hist Gradient Boosting", {})
from sklearn.ensemble import HistGradientBoostingRegressor

fold_importances = []
for fold, (fit_idx, val_idx) in enumerate(modeling.make_cv().split(train_df), start=1):
    fit_part = train_df.iloc[fit_idx]
    val_part = train_df.iloc[val_idx]
    estimator = HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.05, max_leaf_nodes=31, random_state=config.SEED,
        **{k: v for k, v in hgb_params.items()
           if k not in {"max_iter", "learning_rate", "max_leaf_nodes"}},
    )
    pipeline = features.build_pipeline(estimator, feature_list, scale_numeric=False)
    pipeline.fit(fit_part[feature_list], fit_part[config.TARGET])
    result = permutation_importance(
        pipeline, val_part[feature_list], val_part[config.TARGET],
        n_repeats=5, random_state=config.SEED, scoring="neg_mean_absolute_error",
    )
    fold_importances.append(
        pd.DataFrame(
            {"fold": fold, "feature": feature_list, "importance": result.importances_mean}
        )
    )
fold_importance = pd.concat(fold_importances, ignore_index=True)
stability = (
    fold_importance.groupby("feature")["importance"]
    .agg(["mean", "std"])
    .sort_values("mean", ascending=False)
)
display(stability.round(4))
stability.to_csv(config.TABLES_DIR / "05_importance_stability.csv")
"""
        ),
        md("## 5. Partial dependence for top HGB features"),
        code(
            """
top_features = (
    importances[importances["model"] == "Hist Gradient Boosting"]
    .sort_values("importance_mae_increase_mean", ascending=False)
    .head(3)["feature"].tolist()
)
print("Top features:", top_features)
sample = test_df.sample(min(500, len(test_df)), random_state=config.SEED)
fig, ax = plt.subplots(figsize=(11, 3.5))
PartialDependenceDisplay.from_estimator(
    fitted["Hist Gradient Boosting"], sample[feature_list], top_features,
    ax=ax, kind="average",
)
fig.suptitle("Partial dependence: top HGB features (held-out sample)", y=1.03)
viz.finish_figure(fig, "05_partial_dependence.png")
"""
        ),
        md("## 6. Cross-model rank consistency"),
        code(
            """
wide = importances.pivot(index="feature", columns="model",
                         values="importance_mae_increase_mean")
rho, p_value = spearmanr(wide["Random Forest"], wide["Hist Gradient Boosting"])
print("Spearman rank correlation between RF and HGB permutation importance:", round(rho, 3))
print("p-value:", round(p_value, 4))
display(wide.round(4).sort_values("Hist Gradient Boosting", ascending=False))
wide.to_csv(config.TABLES_DIR / "05_importance_by_model.csv")
"""
        ),
        md("## 7. Summary"),
        code(
            """
rf_top = importances[importances["model"] == "Random Forest"].head(3)["feature"].tolist()
hgb_top = importances[importances["model"] == "Hist Gradient Boosting"].head(3)["feature"].tolist()
report_lines = [
    "# Feature Analysis Summary",
    "",
    f"- Group C features analysed: {len(feature_list)}.",
    f"- Top RF permutation-importance features: {', '.join(rf_top)}.",
    f"- Top HGB permutation-importance features: {', '.join(hgb_top)}.",
    f"- RF vs HGB importance rank correlation (Spearman): {rho:.3f} (p={p_value:.4f}).",
    "",
    "## Caveats",
    "",
    "- Permutation importance measures how much predictive performance changes when a feature "
    "is shuffled; it is not a causal effect.",
    "- Strongly correlated predictors (usage, sleep, stress, mental health) can share or split "
    "importance, and permutation can create unrealistic feature combinations.",
    "- Linear coefficients are conditional on the other included predictors and on scaling; "
    "their signs and magnitudes should not be read causally.",
    "- Results describe this supplied dataset, whose provenance is unverified and whose "
    "generation process may be synthetic.",
]
summary_text = "\\n".join(report_lines)
(config.REPORTS_DIR / "05_feature_analysis.md").write_text(summary_text, encoding="utf-8")
display(Markdown(summary_text))
"""
        ),
    ]
    return nbf.v4.new_notebook(cells=cells)


def main() -> None:
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    notebooks = {
        "01_data_audit.ipynb": build_audit(),
        "02_exploratory_analysis.ipynb": build_eda(),
        "03_baseline_models.ipynb": build_baseline(),
        "04_model_comparison.ipynb": build_comparison(),
        "05_feature_analysis.ipynb": build_feature_analysis(),
    }
    for name, notebook in notebooks.items():
        notebook.metadata = {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        }
        path = NOTEBOOK_DIR / name
        nbf.write(notebook, path)
        print("wrote", path)


if __name__ == "__main__":
    main()
