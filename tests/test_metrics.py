import numpy as np
import pytest
from src.eval.metrics import mae, rmse, mape, evaluate


def test_mae_perfect():
    y = np.array([100.0, 200.0, 300.0])
    assert mae(y, y) == 0.0


def test_mae_basic():
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([110.0, 190.0, 310.0])
    assert mae(y_true, y_pred) == pytest.approx(10.0)


def test_rmse_perfect():
    y = np.array([100.0, 200.0, 300.0])
    assert rmse(y, y) == 0.0


def test_rmse_basic():
    y_true = np.array([0.0, 0.0])
    y_pred = np.array([3.0, 4.0])
    assert rmse(y_true, y_pred) == pytest.approx(np.sqrt(12.5))


def test_mape_basic():
    y_true = np.array([100.0, 200.0])
    y_pred = np.array([110.0, 200.0])
    assert mape(y_true, y_pred) == pytest.approx(5.0)


def test_mape_zero_division():
    y_true = np.array([0.0, 0.0])
    y_pred = np.array([1.0, 2.0])
    result = mape(y_true, y_pred)
    assert np.isnan(result)


def test_evaluate_returns_dict():
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([110.0, 190.0, 290.0])
    result = evaluate(y_true, y_pred)
    assert set(result.keys()) == {"mae", "rmse", "mape"}
    assert all(isinstance(v, float) for v in result.values())


def test_evaluate_perfect():
    y = np.array([500.0, 1000.0, 1500.0])
    result = evaluate(y, y)
    assert result["mae"] == 0.0
    assert result["rmse"] == 0.0
    assert result["mape"] == 0.0
