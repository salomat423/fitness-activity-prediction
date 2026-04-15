# Fitness Activity Prediction

## Goal
ѕрогнозирование дневной активности пользовател€ (steps на следующий день) на основе исторических данных фитнес-трекера.

## Stack
Python 3.11, PyTorch, LightGBM, FastAPI, MLflow, Docker, Pandas, scikit-learn.

## Quick Start
1. ”становить Python 3.11 и Poetry.
2. ”становить зависимости: `poetry install`.
3. ѕоложить датасет в `data/raw/`.

## Data
»сточник данных: Fitbit Fitness Tracker Data (Kaggle, Mobius).

## Task
–егресси€: предсказать `steps` на следующий день дл€ каждого пользовател€ по окну исторических признаков.

## Results
| Model | MAE | RMSE | MAPE |
|------|-----|------|------|
| baseline_lgbm | TBD | TBD | TBD |
| lstm_v1 | TBD | TBD | TBD |
