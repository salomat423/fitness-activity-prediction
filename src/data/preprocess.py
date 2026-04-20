import pandas as pd
import numpy as np
from pathlib import Path

RAW_PATH = Path("data/raw/dailyActivity_merged.csv")
PROCESSED_PATH = Path("data/processed")


def load_data() -> pd.DataFrame:
    df = pd.read_csv(RAW_PATH)
    df.columns = [
        "user_id", "date", "steps", "total_distance", "tracker_distance",
        "logged_distance", "very_active_dist", "moderate_active_dist",
        "light_active_dist", "sedentary_dist", "very_active_min",
        "fairly_active_min", "lightly_active_min", "sedentary_min", "calories"
    ]
    df["date"] = pd.to_datetime(df["date"])
    df["active_minutes"] = (
        df["very_active_min"] + df["fairly_active_min"] + df["lightly_active_min"]
    )
    return df


def remove_invalid_days(df: pd.DataFrame) -> pd.DataFrame:
    mask = (df["steps"] > 0) & (df["active_minutes"] >= 100)
    cleaned = df[mask].copy()
    removed = len(df) - len(cleaned)
    print(f"remove_invalid_days: удалено {removed} строк, осталось {len(cleaned)}")
    return cleaned


def fill_missing_dates(df: pd.DataFrame) -> pd.DataFrame:
    all_users = df["user_id"].unique()
    date_min = df["date"].min()
    date_max = df["date"].max()
    full_index = pd.date_range(date_min, date_max, freq="D")

    parts = []
    for uid in all_users:
        user_df = df[df["user_id"] == uid].set_index("date")
        user_df = user_df.reindex(full_index)
        user_df["user_id"] = uid
        user_df.index.name = "date"
        parts.append(user_df.reset_index())

    filled = pd.concat(parts, ignore_index=True)
    added = len(filled) - len(df)
    print(f"fill_missing_dates: добавлено {added} строк с пропущенными датами")
    return filled


def split_by_time(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.sort_values("date").reset_index(drop=True)
    dates = df["date"].sort_values().unique()

    n = len(dates)
    train_end = dates[int(n * train_ratio) - 1]
    val_end = dates[int(n * (train_ratio + val_ratio)) - 1]

    train = df[df["date"] <= train_end].copy()
    val = df[(df["date"] > train_end) & (df["date"] <= val_end)].copy()
    test = df[df["date"] > val_end].copy()

    print(f"split_by_time:")
    print(f"  train: {len(train)} строк | {train['date'].min().date()} — {train['date'].max().date()}")
    print(f"  val:   {len(val)} строк | {val['date'].min().date()} — {val['date'].max().date()}")
    print(f"  test:  {len(test)} строк | {test['date'].min().date()} — {test['date'].max().date()}")

    return train, val, test


def main():
    print("=== Preprocessing pipeline ===\n")

    PROCESSED_PATH.mkdir(parents=True, exist_ok=True)

    df = load_data()
    print(f"Загружено: {len(df)} строк, {df['user_id'].nunique()} пользователей\n")

    df = remove_invalid_days(df)
    df = fill_missing_dates(df)
    print()

    train, val, test = split_by_time(df)

    train.to_parquet(PROCESSED_PATH / "train.parquet", index=False)
    val.to_parquet(PROCESSED_PATH / "val.parquet", index=False)
    test.to_parquet(PROCESSED_PATH / "test.parquet", index=False)

    print(f"\nФайлы сохранены в {PROCESSED_PATH}/")
    print("  train.parquet, val.parquet, test.parquet")


if __name__ == "__main__":
    main()
