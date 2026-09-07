import numpy as np

from sense_energy.evaluation.backtest import rolling_origin_splits
from sense_energy.features import calendar
from sense_energy.models.baseline import ProfileMean, SeasonalNaive


def test_seasonal_naive_repeats_last_season(synthetic_demand):
    site = synthetic_demand[synthetic_demand["site_id"] == "SITE_A"].sort_values("timestamp")
    model = SeasonalNaive(season_length=336).fit(site, site["value"])
    preds = model.predict(site.head(336))
    assert np.allclose(preds, site["value"].tail(336).to_numpy())


def test_profile_mean_predicts_per_cell_average(synthetic_demand):
    df = calendar.add_calendar_features(synthetic_demand)
    model = ProfileMean().fit(df, df["value"])
    preds = model.predict(df)
    assert len(preds) == len(df)
    assert np.isfinite(preds).all()


def test_rolling_origin_splits_never_look_ahead(synthetic_demand):
    site = synthetic_demand[synthetic_demand["site_id"] == "SITE_A"]
    folds = list(rolling_origin_splits(site, n_splits=3, test_size=48, gap=48))
    assert len(folds) == 3
    for train_df, test_df in folds:
        assert train_df["timestamp"].max() < test_df["timestamp"].min()
