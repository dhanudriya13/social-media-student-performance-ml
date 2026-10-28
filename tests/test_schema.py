import pandas as pd

from sm_project import config, data


def test_expected_columns_present():
    df = data.load_data()
    check = data.check_expected_columns(df)
    assert check["missing"] == []
    data.validate_schema(df)


def test_identifier_is_unique_and_not_a_feature():
    df = data.load_data()
    assert df[config.ID_COLUMN].is_unique
    assert df[config.ID_COLUMN].notna().all()
    for features in config.FEATURE_GROUPS.values():
        assert config.ID_COLUMN not in features


def test_no_duplicate_rows_and_target_numeric():
    df = data.load_data()
    assert df.duplicated().sum() == 0
    assert pd.api.types.is_numeric_dtype(df[config.TARGET])
    assert df[config.TARGET].notna().sum() > 0


def test_overall_impact_excluded_from_all_groups():
    for features in config.FEATURE_GROUPS.values():
        assert "Overall_Impact" not in features
        assert config.TARGET not in features
