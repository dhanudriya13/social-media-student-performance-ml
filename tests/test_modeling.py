import numpy as np

from sm_project import modeling


def test_regression_metrics_are_correct_for_perfect_prediction():
    y = np.array([1.0, 2.0, 3.0])
    metrics = modeling.regression_metrics(y, y)
    assert metrics["mae"] == 0.0
    assert metrics["rmse"] == 0.0
    assert metrics["r2"] == 1.0


def test_regression_metrics_rmse_is_root_mse():
    y_true = np.array([0.0, 0.0])
    y_pred = np.array([1.0, -1.0])
    metrics = modeling.regression_metrics(y_true, y_pred)
    assert metrics["mae"] == 1.0
    assert metrics["rmse"] == 1.0


def test_required_models_are_present():
    models = modeling.default_models()
    assert "Dummy (mean)" in models
    assert "Linear Regression" in models
    assert "Random Forest" in models
    assert "Hist Gradient Boosting" in models
