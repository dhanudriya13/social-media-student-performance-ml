import numpy as np
import pandas as pd
import pytest


def test_groups_are_nested_and_use_declared_types():
    from sm_project import config, features

    a = features.get_feature_group("A")
    b = features.get_feature_group("B")
    c = features.get_feature_group("C")
    assert set(a).issubset(b)
    assert set(b).issubset(c)
    numeric, categorical = features.split_feature_types(c)
    assert set(numeric).issubset(config.NUMERIC_FEATURES)
    assert set(categorical).issubset(config.CATEGORICAL_FEATURES)
    assert len(numeric) + len(categorical) == len(c)


def test_unknown_group_raises():
    from sm_project import features

    with pytest.raises(KeyError):
        features.get_feature_group("Z")


def test_missing_feature_column_raises():
    from sm_project import features

    df = pd.DataFrame({"Age": [20], "Academic_Performance_GPA": [3.0]})
    with pytest.raises(ValueError):
        features.prepare_xy(df, features.get_feature_group("A"))


def test_split_is_reproducible_and_disjoint():
    from sm_project import config, features

    df = pd.DataFrame({"x": np.arange(100)})
    train1, test1 = features.make_split(df, seed=config.SEED)
    train2, test2 = features.make_split(df, seed=config.SEED)
    np.testing.assert_array_equal(train1, train2)
    np.testing.assert_array_equal(test1, test2)
    assert set(train1).isdisjoint(set(test1))
    assert len(train1) + len(test1) == len(df)
