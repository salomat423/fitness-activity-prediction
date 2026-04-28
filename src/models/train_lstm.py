import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path

import mlflow
import mlflow.pytorch

from src.models.dataset import FitnessSequenceDataset
from src.models.lstm import LSTMRegressor
from src.eval.metrics import evaluate

PROCESSED_PATH = Path("data/processed")
MODELS_PATH = Path("models")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        pred = model(X)
        loss = criterion(pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss = 0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            pred = model(X)
            total_loss += criterion(pred, y).item()
            all_preds.extend(pred.cpu().numpy().flatten())
            all_targets.extend(y.cpu().numpy().flatten())
    return total_loss / len(loader), np.array(all_preds), np.array(all_targets)


def main():
    MODELS_PATH.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(PROCESSED_PATH / "train_features.parquet")
    n = len(df)
    split_idx = int(n * 0.8)
    train_df = df.iloc[:split_idx].copy()
    val_df = df.iloc[split_idx:].copy()

    with open(MODELS_PATH / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    train_dataset = FitnessSequenceDataset(train_df, sequence_length=7, fit_scaler=True)
    val_dataset = FitnessSequenceDataset(val_df, sequence_length=7, scaler=train_dataset.scaler)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Device: {DEVICE}")

    params = {
        "input_size": 17,
        "hidden_size": 64,
        "num_layers": 2,
        "dropout": 0.2,
        "lr": 0.001,
        "epochs": 50,
        "batch_size": 32,
    }

    model = LSTMRegressor(
        input_size=params["input_size"],
        hidden_size=params["hidden_size"],
        num_layers=params["num_layers"],
        dropout=params["dropout"],
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("fitness-activity-prediction")

    best_val_loss = float("inf")
    best_model_state = None

    with mlflow.start_run(run_name="lstm_v1"):
        mlflow.log_params(params)

        for epoch in range(1, params["epochs"] + 1):
            train_loss = train_epoch(model, train_loader, optimizer, criterion)
            val_loss, val_preds, val_targets = eval_epoch(model, val_loader, criterion)
            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = {k: v.clone() for k, v in model.state_dict().items()}

            if epoch % 10 == 0:
                print(f"Epoch {epoch:3d} | train_loss={train_loss:.2f} | val_loss={val_loss:.2f}")
                mlflow.log_metrics({"train_loss": train_loss, "val_loss": val_loss}, step=epoch)

        model.load_state_dict(best_model_state)
        _, val_preds, val_targets = eval_epoch(model, val_loader, criterion)
        metrics = evaluate(val_targets, val_preds)

        print("\n=== Val Metrics (LSTM) ===")
        for k, v in metrics.items():
            print(f"  {k.upper()}: {v:.2f}")

        mlflow.log_metrics(metrics)

        model_path = MODELS_PATH / "lstm_v1.pt"
        torch.save(model.state_dict(), model_path)
        print(f"\nМодель сохранена: {model_path}")


if __name__ == "__main__":
    main()
