"""Feature groups, data splitting, and preprocessing pipelines."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import (
    CATEGORICAL_FEATURES,
    FEATURE_GROUPS,
    NUMERIC_FEATURES,
    SEED,
    TARGET,
    TEST_SIZE,
)


def get_feature_group(name: str) -> list[str]:
    """Return the documented feature list for group A, B, or C.

    A missing group, or a group containing an unexpected column, raises an
    explicit error so features are never silently substituted.
    """
    key = str(name).upper()
    if key not in FEATURE_GROUPS:
        raise KeyError(f"Unknown feature group {name!r}; expected one of {list(FEATURE_GROUPS)}.")
    return list(FEATURE_GROUPS[key])


def split_feature_types(features: list[str]) -> tuple[list[str], list[str]]:
    """Split a feature list into numeric and categorical columns."""
    numeric = [f for f in features if f in NUMERIC_FEATURES]
    categorical = [f for f in features if f in CATEGORICAL_FEATURES]
    unknown = [f for f in features if f not in numeric and f not in categorical]
    if unknown:
        raise ValueError(f"Features lack a declared type: {unknown}")
    return numeric, categorical


def build_preprocessor(
    numeric: list[str], categorical: list[str], scale_numeric: bool = False
) -> ColumnTransformer:
    """Build a preprocessing transformer.

    Imputation and encoding are fitted inside each training fold because the
    transformer is placed in a scikit-learn Pipeline.
    """
    numeric_steps: list[tuple[str, object]] = [
        ("imputer", SimpleImputer(strategy="median")),
    ]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_pipe = Pipeline(numeric_steps)

    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    return ColumnTransformer(
        [
            ("num", numeric_pipe, numeric),
            ("cat", categorical_pipe, categorical),
        ],
        remainder="drop",
    )


def build_pipeline(model, features: list[str], scale_numeric: bool = False) -> Pipeline:
    """Create a preprocessing + model Pipeline for a feature group."""
    numeric, categorical = split_feature_types(features)
    preprocessor = build_preprocessor(numeric, categorical, scale_numeric=scale_numeric)
    return Pipeline([("preprocess", preprocessor), ("model", model)])


def make_split(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Return reproducible positional train/test indices.

    The same indices are reused for every model and feature group so that
    comparisons are made on identical observations.
    """
    positions = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        positions, test_size=test_size, random_state=seed, shuffle=True
    )
    return np.sort(train_idx), np.sort(test_idx)


def prepare_xy(
    df: pd.DataFrame, features: list[str]
) -> tuple[pd.DataFrame, pd.Series]:
    """Return the predictor frame and target vector for a feature group."""
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f"Feature group references absent columns: {missing}")
    return df[features].copy(), df[TARGET].copy()
