import numpy as np
import pytest

from sense_energy.evaluation import metrics


@pytest.fixture
def y():
    return np.array([100.0, 110.0, 90.0, 105.0])


def test_perfect_forecast_scores_zero(y):
    assert metrics.mae(y, y) == 0
    assert metrics.rmse(y, y) == 0
    assert metrics.mape(y, y) == pytest.approx(0)
    assert metrics.bias(y, y) == 0


def test_mae_and_rmse(y):
    pred = y + 10
    assert metrics.mae(y, pred) == pytest.approx(10)
    assert metrics.rmse(y, pred) == pytest.approx(10)


def test_bias_sign_indicates_over_forecast(y):
    assert metrics.bias(y, y + 5) > 0
    assert metrics.bias(y, y - 5) < 0


def test_mase_beats_one_for_a_good_forecast():
    rng = np.random.default_rng(0)
    train = 100 + 10 * np.sin(np.arange(1000) * 2 * np.pi / 336) + rng.normal(0, 1, 1000)
    truth = np.full(50, 100.0)
    assert metrics.mase(truth, truth + 0.1, train, season_length=336) < 1.0


def test_all_metrics_returns_expected_keys(y):
    assert set(metrics.all_metrics(y, y + 1)) == {"mae", "rmse", "mape", "smape", "bias"}
