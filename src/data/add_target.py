import pandas as pd
from pathlib import Path

PROCESSED_PATH = Path("data/processed")


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["user_id", "date"]).copy()
    df["target"] = df.groupby("user_id")["steps"].shift(-1)
    before = len(df)
    df = df.dropna(subset=["target"])
    print(f"add_target: удалено {before - len(df)} строк с NaN target, осталось {len(df)}")
    return df


def main():
    print("=== Adding target column ===\n")
    for split in ["train", "val", "test"]:
        path = PROCESSED_PATH / f"{split}_features.parquet"
        df = pd.read_parquet(path)
        df = add_target(df)
        df.to_parquet(path, index=False)
        print(f"{split}: {len(df)} строк, target диапазон: {df['target'].min():.0f} — {df['target'].max():.0f}\n")
    print("Готово!")


if __name__ == "__main__":
    main()
