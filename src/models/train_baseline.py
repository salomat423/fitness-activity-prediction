import pandas as pd
import numpy as np
import pickle
from pathlib import Path

import lightgbm as lgb
import mlflow
import mlflow.lightgbm

from src.eval.metrics import evaluate

PROCESSED_PATH = Path("data/processed")
MODELS_PATH = Path("models")
FEATURE_COLS = [
    "steps_lag_1", "steps_lag_3", "steps_lag_7",
    "calories_lag_1", "calories_lag_3", "calories_lag_7",
    "steps_roll_mean_3", "steps_roll_mean_7", "steps_roll_mean_14",
    "steps_roll_std_3", "steps_roll_std_7", "steps_roll_std_14",
    "day_of_week", "is_weekend", "day_of_month", "month",
    "activity_streak",
]
TARGET_COL = "target"


def load_split(split: str) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_parquet(PROCESSED_PATH / f"{split}_features.parquet")
    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL])
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]
    return X, y


def main():
    MODELS_PATH.mkdir(parents=True, exist_ok=True)

    X_train, y_train = load_split("train")
    X_val, y_val = load_split("val")

    print(f"Train: {X_train.shape}, Val: {X_val.shape}")

    if X_val.shape[0] == 0:
        print("Val пустой — используем последние 20% train для валидации")
        split_idx = int(len(X_train) * 0.8)
        X_val = X_train.iloc[split_idx:]
        y_val = y_train.iloc[split_idx:]
        X_train = X_train.iloc[:split_idx]
        y_train = y_train.iloc[:split_idx]
        print(f"Train: {X_train.shape}, Val: {X_val.shape}")

    params = {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_child_samples": 10,
        "random_state": 42,
        "verbose": -1,
    }

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("fitness-activity-prediction")

    with mlflow.start_run(run_name="baseline_lgbm"):
        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(False)],
        )

        y_pred = model.predict(X_val)
        metrics = evaluate(y_val.values, y_pred)

        print("\n=== Val Metrics ===")
        for k, v in metrics.items():
            print(f"  {k.upper()}: {v:.2f}")

        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.lightgbm.log_model(model, "model")

        model_path = MODELS_PATH / "baseline_lgbm.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        print(f"\nМодель сохранена: {model_path}")
        print("Эксперимент залогирован в MLflow → http://localhost:5000")


if __name__ == "__main__":
    main()
