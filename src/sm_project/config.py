"""Project configuration: paths, columns, feature groups, and seeds.

Paths are resolved relative to the repository root by default, but every
path can be overridden with an environment variable so that the same code
runs locally and in Google Colab.
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = Path(os.environ.get("SM_DATA_DIR", PROJECT_ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

OUTPUT_DIR = Path(os.environ.get("SM_OUTPUT_DIR", PROJECT_ROOT / "outputs"))
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"
REPORTS_DIR = OUTPUT_DIR / "reports"

DATA_FILENAME = "Social_media_impact_on_life.csv"

TARGET = "Academic_Performance_GPA"
ID_COLUMN = "Student_ID"

# Never used as predictors. Overall_Impact is excluded because its derivation
# is undocumented and it strongly tracks the target (potential leakage).
EXCLUDED_FROM_MODELS = [ID_COLUMN, "Overall_Impact", TARGET]

EXPECTED_COLUMNS = [
    "Student_ID",
    "Age",
    "Gender",
    "Academic_Level",
    "Primary_Platform",
    "Daily_Usage_Hours",
    "Weekend_Extra_Hours",
    "Sleep_Duration_Hours",
    "Sleep_Quality_Score",
    "Perceived_Stress_Score",
    "Mental_Health_Index",
    "Academic_Performance_GPA",
    "Social_Comparison_Frequency",
    "Device_Type",
    "Late_Night_Usage",
    "Overall_Impact",
]

NUMERIC_FEATURES = [
    "Age",
    "Daily_Usage_Hours",
    "Weekend_Extra_Hours",
    "Sleep_Duration_Hours",
    "Sleep_Quality_Score",
    "Perceived_Stress_Score",
    "Mental_Health_Index",
]

CATEGORICAL_FEATURES = [
    "Gender",
    "Academic_Level",
    "Primary_Platform",
    "Device_Type",
    "Late_Night_Usage",
    "Social_Comparison_Frequency",
]

FEATURE_GROUPS = {
    "A": [
        "Daily_Usage_Hours",
        "Weekend_Extra_Hours",
        "Late_Night_Usage",
        "Social_Comparison_Frequency",
        "Primary_Platform",
        "Device_Type",
        "Age",
        "Gender",
        "Academic_Level",
    ],
    "B": [
        "Daily_Usage_Hours",
        "Weekend_Extra_Hours",
        "Late_Night_Usage",
        "Social_Comparison_Frequency",
        "Primary_Platform",
        "Device_Type",
        "Age",
        "Gender",
        "Academic_Level",
        "Sleep_Duration_Hours",
        "Sleep_Quality_Score",
    ],
    "C": [
        "Daily_Usage_Hours",
        "Weekend_Extra_Hours",
        "Late_Night_Usage",
        "Social_Comparison_Frequency",
        "Primary_Platform",
        "Device_Type",
        "Age",
        "Gender",
        "Academic_Level",
        "Sleep_Duration_Hours",
        "Sleep_Quality_Score",
        "Perceived_Stress_Score",
        "Mental_Health_Index",
    ],
}

FEATURE_GROUP_DESCRIPTIONS = {
    "A": "Social media usage + demographic/context",
    "B": "A + sleep variables",
    "C": "B + stress and mental health indicators",
}

SEED = 42
TEST_SIZE = 0.20
N_SPLITS = 5


def resolve_data_path(path: str | Path | None = None) -> Path:
    """Return the CSV path, checking an explicit path, env var, and common locations."""
    if path is not None:
        candidate = Path(path)
        if not candidate.exists():
            raise FileNotFoundError(
                f"Data file not found at {candidate}. Place {DATA_FILENAME} in "
                f"data/raw/ or set SM_DATA_PATH."
            )
        return candidate

    env_path = os.environ.get("SM_DATA_PATH")
    candidates = [
        Path(env_path) if env_path else None,
        RAW_DIR / DATA_FILENAME,
        PROJECT_ROOT / DATA_FILENAME,
        Path("/content") / DATA_FILENAME,
        Path.cwd() / DATA_FILENAME,
        Path.cwd() / "data" / "raw" / DATA_FILENAME,
    ]
    for candidate in candidates:
        if candidate is not None and candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Could not locate {DATA_FILENAME}. Place it in data/raw/, pass an "
        "explicit path, or set the SM_DATA_PATH environment variable."
    )


def ensure_output_dirs() -> None:
    for directory in (FIGURES_DIR, TABLES_DIR, REPORTS_DIR, PROCESSED_DIR):
        directory.mkdir(parents=True, exist_ok=True)
