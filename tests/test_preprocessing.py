import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from sm_project import features


def make_frame():
    return pd.DataFrame(
        {
            "Age": [18.0, 19.0, np.nan, 21.0],
            "Daily_Usage_Hours": [2.0, np.nan, 6.0, 4.0],
            "Gender": ["Female", "Male", None, "Male"],
            "Late_Night_Usage": [True, False, True, False],
            "Academic_Performance_GPA": [3.8, 3.2, 3.5, 2.9],
        }
    )


def test_pipeline_handles_missing_and_unseen_categories():
    df = make_frame()
    features_list = ["Age", "Daily_Usage_Hours", "Gender", "Late_Night_Usage"]
    pipeline = features.build_pipeline(LinearRegression(), features_list, scale_numeric=True)
    pipeline.fit(df[features_list], df["Academic_Performance_GPA"])

    unseen = pd.DataFrame(
        {
            "Age": [np.nan, 22.0],
            "Daily_Usage_Hours": [5.0, np.nan],
            "Gender": ["Non-Binary", None],
            "Late_Night_Usage": [False, True],
        }
    )
    predictions = pipeline.predict(unseen)
    assert predictions.shape == (2,)
    assert np.isfinite(predictions).all()


def test_missing_predictors_are_imputed_not_rows_dropped():
    df = make_frame()
    features_list = ["Age", "Daily_Usage_Hours", "Gender", "Late_Night_Usage"]
    pipeline = features.build_pipeline(LinearRegression(), features_list)
    pipeline.fit(df[features_list], df["Academic_Performance_GPA"])
    transformed = pipeline.named_steps["preprocess"].transform(df[features_list])
    assert np.isfinite(transformed).all()
    assert transformed.shape[0] == len(df)
