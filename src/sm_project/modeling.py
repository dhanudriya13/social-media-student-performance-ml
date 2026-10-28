"""Model factories, cross-validation, and evaluation helpers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate

from .config import N_SPLITS, SEED
from .features import build_pipeline


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """Compute MAE, RMSE, and R² in target units."""
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def make_cv(n_splits: int = N_SPLITS, seed: int = SEED) -> KFold:
    """Shared shuffled K-fold splitter used across models and feature groups."""
    return KFold(n_splits=n_splits, shuffle=True, random_state=seed)


def default_models(seed: int = SEED) -> dict[str, object]:
    """Return the required model set with documented, fixed hyperparameters."""
    return {
        "Dummy (mean)": DummyRegressor(strategy="mean"),
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=400,
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=seed,
            n_jobs=-1,
        ),
        "Hist Gradient Boosting": HistGradientBoostingRegressor(
            max_iter=300,
            learning_rate=0.05,
            max_leaf_nodes=31,
            random_state=seed,
        ),
    }


def needs_scaling(model_name: str) -> bool:
    """Linear models benefit from scaling; trees do not require it."""
    return model_name == "Linear Regression"


def evaluate_cv(
    df: pd.DataFrame,
    features: list[str],
    model_name: str,
    estimator,
    cv: KFold,
    scale_numeric: bool,
) -> pd.DataFrame:
    """Run cross-validation and return per-fold MAE, RMSE, and R²."""
    from .features import prepare_xy

    X, y = prepare_xy(df, features)
    pipeline = build_pipeline(estimator, features, scale_numeric=scale_numeric)
    scores = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring={
            "mae": "neg_mean_absolute_error",
            "rmse": "neg_root_mean_squared_error",
            "r2": "r2",
        },
        n_jobs=1,
        return_train_score=False,
    )
    folds = pd.DataFrame(
        {
            "model": model_name,
            "fold": np.arange(1, len(scores["test_mae"]) + 1),
            "mae": -scores["test_mae"],
            "rmse": -scores["test_rmse"],
            "r2": scores["test_r2"],
        }
    )
    return folds


def summarize_cv(folds: pd.DataFrame) -> dict[str, float]:
    """Aggregate fold metrics into mean and standard deviation."""
    out: dict[str, float] = {}
    for metric in ("mae", "rmse", "r2"):
        out[f"cv_{metric}_mean"] = float(folds[metric].mean())
        out[f"cv_{metric}_std"] = float(folds[metric].std(ddof=1))
    out["cv_n_folds"] = int(len(folds))
    return out


def fit_and_test(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: list[str],
    model_name: str,
    estimator,
    scale_numeric: bool,
):
    """Fit a pipeline on training rows and evaluate once on held-out rows."""
    from .features import prepare_xy

    X_train, y_train = prepare_xy(train_df, features)
    X_test, y_test = prepare_xy(test_df, features)
    pipeline = build_pipeline(estimator, features, scale_numeric=scale_numeric)
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    metrics = regression_metrics(y_test, predictions)
    metrics["n_train"] = int(len(train_df))
    metrics["n_test"] = int(len(test_df))
    metrics["n_features_input"] = int(len(features))
    metrics["model"] = model_name
    return metrics, predictions, pipeline


def save_json(obj, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, default=str)


def load_json(path: str | Path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)
