import numpy as np

from sense_energy.evaluation.backtest import rolling_origin_splits
from sense_energy.features import calendar
from sense_energy.models.baseline import ProfileMean, SeasonalNaive


def test_seasonal_naive_repeats_last_season(synthetic_demand):
    site = synthetic_demand[synthetic_demand["mpxn"] == "1000000000001"].sort_values("datetime")
    model = SeasonalNaive(season_length=336).fit(site, site["consumption_kwh"])
    preds = model.predict(site.head(336))
    assert np.allclose(preds, site["consumption_kwh"].tail(336).to_numpy())


def test_profile_mean_predicts_per_cell_average(synthetic_demand):
    df = calendar.add_calendar_features(synthetic_demand)
    model = ProfileMean().fit(df, df["consumption_kwh"])
    preds = model.predict(df)
    assert len(preds) == len(df)
    assert np.isfinite(preds).all()


def test_rolling_origin_splits_never_look_ahead(synthetic_demand):
    site = synthetic_demand[synthetic_demand["mpxn"] == "1000000000001"]
    folds = list(rolling_origin_splits(site, n_splits=3, test_size=48, gap=48))
    assert len(folds) == 3
    for train_df, test_df in folds:
        assert train_df["datetime"].max() < test_df["datetime"].min()
