"""Data loading and validation helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import EXPECTED_COLUMNS, TARGET, resolve_data_path


def load_data(path: str | Path | None = None) -> pd.DataFrame:
    """Load the raw CSV without modifying it.

    Parameters
    ----------
    path:
        Optional explicit CSV path. When omitted the path is resolved from
        the environment, data/raw/, or common Colab locations.
    """
    csv_path = resolve_data_path(path)
    df = pd.read_csv(csv_path)
    return df


def read_data_with_source(path: str | Path | None = None) -> tuple[pd.DataFrame, Path]:
    """Load the CSV and also return the resolved path (useful for logging)."""
    csv_path = resolve_data_path(path)
    return pd.read_csv(csv_path), csv_path


def check_expected_columns(df: pd.DataFrame) -> dict:
    """Compare the observed schema against the documented expected columns.

    Returns a dictionary with present/expected/missing/unexpected lists so that
    notebooks can display any mismatch explicitly instead of assuming a schema.
    """
    expected = list(EXPECTED_COLUMNS)
    observed = list(df.columns)
    missing = [c for c in expected if c not in observed]
    unexpected = [c for c in observed if c not in expected]
    return {
        "expected": expected,
        "observed": observed,
        "missing": missing,
        "unexpected": unexpected,
        "matches": not missing and not unexpected,
    }


def validate_schema(df: pd.DataFrame) -> None:
    """Raise a descriptive error if required columns are absent."""
    check = check_expected_columns(df)
    if check["missing"]:
        raise ValueError(
            "Dataset is missing expected columns: " + ", ".join(check["missing"])
        )
    if TARGET not in df.columns:
        raise ValueError(f"Target column {TARGET!r} is absent from the dataset.")


def target_missing_mask(df: pd.DataFrame) -> pd.Series:
    """Boolean mask of rows with a missing target."""
    return df[TARGET].isna()


def eligible_sample(df: pd.DataFrame) -> pd.DataFrame:
    """Rows with an observed target, preserving the original index and file order."""
    return df.loc[df[TARGET].notna()].copy()
