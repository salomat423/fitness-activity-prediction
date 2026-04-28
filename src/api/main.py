import pickle
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = Path("models/baseline_lgbm.pkl")
MODEL_VERSION = "1.0.0-lgbm-baseline"

app = FastAPI(title="Fitness Activity Prediction API", version=MODEL_VERSION)

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)


class PredictionRequest(BaseModel):
    steps_lag_1: float = Field(..., ge=0, le=100000, description="Шаги 1 день назад")
    steps_lag_3: float = Field(..., ge=0, le=100000, description="Шаги 3 дня назад")
    steps_lag_7: float = Field(..., ge=0, le=100000, description="Шаги 7 дней назад")
    calories_lag_1: float = Field(..., ge=0, le=10000, description="Калории 1 день назад")
    calories_lag_3: float = Field(..., ge=0, le=10000, description="Калории 3 дня назад")
    calories_lag_7: float = Field(..., ge=0, le=10000, description="Калории 7 дней назад")
    steps_roll_mean_3: float = Field(..., ge=0, le=100000)
    steps_roll_mean_7: float = Field(..., ge=0, le=100000)
    steps_roll_mean_14: float = Field(..., ge=0, le=100000)
    steps_roll_std_3: float = Field(..., ge=0, le=50000)
    steps_roll_std_7: float = Field(..., ge=0, le=50000)
    steps_roll_std_14: float = Field(..., ge=0, le=50000)
    day_of_week: int = Field(..., ge=0, le=6, description="0=Пн, 6=Вс")
    is_weekend: int = Field(..., ge=0, le=1)
    day_of_month: int = Field(..., ge=1, le=31)
    month: int = Field(..., ge=1, le=12)
    activity_streak: int = Field(..., ge=0, le=365)


class PredictionResponse(BaseModel):
    predicted_steps: float
    model_version: str


@app.get("/health")
def health():
    return {"status": "ok", "model_version": MODEL_VERSION}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    features = np.array([[
        request.steps_lag_1, request.steps_lag_3, request.steps_lag_7,
        request.calories_lag_1, request.calories_lag_3, request.calories_lag_7,
        request.steps_roll_mean_3, request.steps_roll_mean_7, request.steps_roll_mean_14,
        request.steps_roll_std_3, request.steps_roll_std_7, request.steps_roll_std_14,
        request.day_of_week, request.is_weekend, request.day_of_month,
        request.month, request.activity_streak,
    ]])

    prediction = model.predict(features)[0]
    prediction = max(0.0, float(prediction))

    return PredictionResponse(
        predicted_steps=round(prediction, 2),
        model_version=MODEL_VERSION,
    )
