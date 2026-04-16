
import pandas as pd
import pytest
from src.data.loader import load_daily_activity


def test_load_returns_dataframe():
    df = load_daily_activity()
    assert isinstance(df, pd.DataFrame)


def test_required_columns_present():
    required_columns = {
        "user_id", "date", "steps",
        "calories", "active_minutes", "sedentary_minutes",
    }
    df = load_daily_activity()
    missing = required_columns - set(df.columns)
    assert not missing, f"Отсутствуют колонки: {missing}"


def test_date_column_is_datetime():
    df = load_daily_activity()
    assert pd.api.types.is_datetime64_any_dtype(df["date"])


def test_no_negative_values():
    df = load_daily_activity()
    for col in ["steps", "calories", "active_minutes", "sedentary_minutes"]:
        assert (df[col] >= 0).all(), f"Отрицательные значения в '{col}'"


def test_multiple_users():
    df = load_daily_activity()
    assert df["user_id"].nunique() > 1


def test_dataset_not_empty():
    df = load_daily_activity()
    assert len(df) > 0


def test_no_duplicate_rows():
    df = load_daily_activity()
    duplicates = df.duplicated(subset=["user_id", "date"]).sum()
    assert duplicates == 0, f"Найдено {duplicates} дублей"