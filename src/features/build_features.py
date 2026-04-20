import pandas as pd
import numpy as np
from pathlib import Path

PROCESSED_PATH = Path("data/processed")


def add_lag_features(df: pd.DataFrame, lags: list[int] = [1, 3, 7]) -> pd.DataFrame:
    df = df.sort_values(["user_id", "date"]).copy()
    for lag in lags:
        df[f"steps_lag_{lag}"] = (
            df.groupby("user_id")["steps"].shift(lag)
        )
        df[f"calories_lag_{lag}"] = (
            df.groupby("user_id")["calories"].shift(lag)
        )
    return df


def add_rolling_features(df: pd.DataFrame, windows: list[int] = [3, 7, 14]) -> pd.DataFrame:
    df = df.sort_values(["user_id", "date"]).copy()
    for window in windows:
        df[f"steps_roll_mean_{window}"] = (
            df.groupby("user_id")["steps"]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        )
        df[f"steps_roll_std_{window}"] = (
            df.groupby("user_id")["steps"]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=1).std())
        )
    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["day_of_week"] = df["date"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["day_of_month"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    return df


def add_activity_streak(df: pd.DataFrame, threshold: int = 5000) -> pd.DataFrame:
    df = df.sort_values(["user_id", "date"]).copy()

    def streak(series):
        result = []
        count = 0
        for val in series:
            if pd.isna(val):
                result.append(0)
                count = 0
            elif val > threshold:
                count += 1
                result.append(count)
            else:
                count = 0
                result.append(0)
        return result

    df["activity_streak"] = (
        df.groupby("user_id")["steps"]
        .transform(lambda x: pd.Series(streak(x.shift(1).values), index=x.index))
    )
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_calendar_features(df)
    df = add_activity_streak(df)
    return df


def main():
    print("=== Feature Engineering ===\n")

    Path(PROCESSED_PATH).mkdir(parents=True, exist_ok=True)

    for split in ["train", "val", "test"]:
        path = PROCESSED_PATH / f"{split}.parquet"
        df = pd.read_parquet(path)
        df = build_features(df)
        out_path = PROCESSED_PATH / f"{split}_features.parquet"
        df.to_parquet(out_path, index=False)

        new_cols = [c for c in df.columns if c not in pd.read_parquet(path).columns]
        print(f"{split}: {len(df)} строк, {len(new_cols)} новых признаков")

    print("\nНовые признаки:")
    df_sample = pd.read_parquet(PROCESSED_PATH / "train_features.parquet")
    original = pd.read_parquet(PROCESSED_PATH / "train.parquet").columns.tolist()
    new_features = [c for c in df_sample.columns if c not in original]
    for f in new_features:
        print(f"  {f}")

    print(f"\nВсего новых признаков: {len(new_features)}")
    print("Файлы сохранены: train_features.parquet, val_features.parquet, test_features.parquet")


if __name__ == "__main__":
    main()
