
import pandas as pd
from pathlib import Path

RAW_DATA_PATH = Path(__file__).parents[2] / "data" / "raw" / "dailyActivity_merged.csv"


def load_daily_activity(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
   

    df = pd.read_csv(path, parse_dates=["ActivityDate"])


    df = df.rename(columns={
        "Id":                    "user_id",
        "ActivityDate":          "date",
        "TotalSteps":            "steps",
        "Calories":              "calories",
        "VeryActiveMinutes":     "very_active_minutes",
        "FairlyActiveMinutes":   "fairly_active_minutes",
        "LightlyActiveMinutes":  "lightly_active_minutes",
        "SedentaryMinutes":      "sedentary_minutes",
    })

    df["active_minutes"] = (
        df["very_active_minutes"]
        + df["fairly_active_minutes"]
        + df["lightly_active_minutes"]
    )

    columns = [
        "user_id",
        "date",
        "steps",
        "calories",
        "active_minutes",
        "sedentary_minutes",
    ]

    return df[columns].copy()

if __name__ == "__main__":
    df = load_daily_activity()
    print(f"Размер датасета: {df.shape}")
    print(f"Пользователей:   {df['user_id'].nunique()}")
    print(f"Период:          {df['date'].min().date()} — {df['date'].max().date()}")
    print()
    print(df.head())
    print()
    print(df.dtypes)