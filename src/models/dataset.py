import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from pathlib import Path

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


class FitnessSequenceDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        sequence_length: int = 7,
        scaler: StandardScaler = None,
        fit_scaler: bool = False,
    ):
        df = df.dropna(subset=FEATURE_COLS).copy()
        df = df.sort_values(["user_id", "date"]).reset_index(drop=True)

        X_raw = df[FEATURE_COLS].values.astype(np.float32)
        y_raw = df[TARGET_COL].values.astype(np.float32)

        if fit_scaler:
            self.scaler = StandardScaler()
            X_raw = self.scaler.fit_transform(X_raw)
            MODELS_PATH.mkdir(parents=True, exist_ok=True)
            with open(MODELS_PATH / "scaler.pkl", "wb") as f:
                pickle.dump(self.scaler, f)
            print(f"Scaler сохранён: {MODELS_PATH / 'scaler.pkl'}")
        elif scaler is not None:
            self.scaler = scaler
            X_raw = self.scaler.transform(X_raw)
        else:
            self.scaler = None

        self.sequences = []
        self.targets = []

        user_ids = df["user_id"].values
        for i in range(sequence_length, len(df)):
            if user_ids[i] == user_ids[i - sequence_length]:
                seq = X_raw[i - sequence_length:i]
                target = y_raw[i]
                if not np.isnan(seq).any() and not np.isnan(target) and not np.isinf(seq).any():
                    self.sequences.append(seq)
                    self.targets.append(target)

        self.sequences = np.array(self.sequences, dtype=np.float32)
        self.targets = np.array(self.targets, dtype=np.float32)
        print(f"Dataset: {len(self.sequences)} sequences, shape={self.sequences.shape}")

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        X = torch.tensor(self.sequences[idx], dtype=torch.float32)
        y = torch.tensor(self.targets[idx], dtype=torch.float32).unsqueeze(0)
        return X, y
