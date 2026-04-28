# Fitness Activity Prediction

## Goal

Прогнозирование дневной активности пользователя (steps на следующий день) на основе исторических данных фитнес-трекера.

## Stack

Python 3.11, PyTorch, LightGBM, FastAPI, MLflow, Docker, Pandas, scikit-learn.

## Quick Start

1. Установить Python 3.11 и Poetry.
2. Установить зависимости: poetry install.
3. Положить датасет в data/raw/.

## Data

Источник данных: Fitbit Fitness Tracker Data (Kaggle, Mobius).

## Task

Регрессия: предсказать steps на следующий день для каждого пользователя по окну исторических признаков.

## Results

|Model|MAE|RMSE|MAPE|
|-|-|-|-|
|baseline\_lgbm|TBD|TBD|TBD|
|lstm\_v1|TBD|TBD|TBD|



\## API



Запустить сервер:



&#x20;   poetry run uvicorn src.api.main:app --reload



Проверить здоровье сервиса:



&#x20;   curl http://127.0.0.1:8000/health



Пример запроса на предсказание:



&#x20;   curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "{\\"steps\_lag\_1\\":8000,\\"steps\_lag\_3\\":7500,\\"steps\_lag\_7\\":7000,\\"calories\_lag\_1\\":2000,\\"calories\_lag\_3\\":1900,\\"calories\_lag\_7\\":1800,\\"steps\_roll\_mean\_3\\":7800,\\"steps\_roll\_mean\_7\\":7600,\\"steps\_roll\_mean\_14\\":7400,\\"steps\_roll\_std\_3\\":500,\\"steps\_roll\_std\_7\\":600,\\"steps\_roll\_std\_14\\":700,\\"day\_of\_week\\":1,\\"is\_weekend\\":0,\\"day\_of\_month\\":15,\\"month\\":4,\\"activity\_streak\\":3}"



Пример ответа:



&#x20;   {"predicted\_steps": 6386.6, "model\_version": "1.0.0-lgbm-baseline"}



Документация доступна по адресу: http://127.0.0.1:8000/docsss

