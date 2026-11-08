"""Year-on-year changes in the estates returns, and the steps in demand that go with them.

ERIC 2023/24 against 2024/25 for the panel sites: solar generation appearing or growing, heat
pumps installed, LED coverage rising, CHP units changing, floor area and annual electricity
moving. Each change is an installation event with a year but no date. The site's own demand
supplies candidate dates: a PELT changepoint search (``ruptures``, L2 cost) on the site's daily
mean demand, log-transformed and net of its day-of-week pattern, gives the steps; each ERIC
change is matched to the nearest step that falls in or near the 2024/25 reporting year. The
result is a list of dated treatments for later event studies, and a check of which returned
changes actually show in the meter.

Outputs (``code/reports/estates/``): ``eric_changes.csv`` (one row per panel site in both
returns), ``demand_steps.csv`` (one row per detected step), ``summary.json`` is extended by the
caller's log only.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..config import INTERIM_DIR, PROJECT_ROOT
from ..logging_utils import get_logger
from .cooling import daily_demand

logger = get_logger(__name__)

OUT_DIR = PROJECT_ROOT / "code" / "reports" / "estates"
FY = (pd.Timestamp("2024-04-01"), pd.Timestamp("2025-04-01"))
MATCH_MARGIN = pd.Timedelta(days=92)  # a step up to a quarter either side of the reporting year
MIN_SEGMENT_DAYS = 28
MIN_STEP = 0.15  # relative change in level that counts as a step
#: PELT penalty = PEN_FACTOR x sigma^2 x log(n), sigma the robust SD of day-to-day changes; the
#: CHP sites switch plant and are flagged rather than trusted for installation dates.
PEN_FACTOR = 6.0
FIELDS = [
    "solar_generated_kwh",
    "solar_consumed_kwh",
    "heat_pumps",
    "led_pct",
    "chp_units",
    "gia_m2",
    "elec_kwh",
    "gas_kwh",
    "ev_charge_points",
    "overheating_occurrences",
    "max_demand_kw",
    "capacity_kva",
]


def panel_sites() -> list[str]:
    from ..config import PROCESSED_DIR

    return list(pd.read_parquet(PROCESSED_DIR / "elec" / "site" / "wide_raw.parquet").columns)


def eric_changes() -> pd.DataFrame:
    """Side-by-side fields of the two returns with deltas and change flags."""
    a = pd.read_parquet(INTERIM_DIR / "eric_2023_24_site.parquet").drop_duplicates("site_code")
    b = pd.read_parquet(INTERIM_DIR / "eric_2024_25_site.parquet").drop_duplicates("site_code")
    sites = panel_sites()
    a = a[a["site_code"].isin(sites)].set_index("site_code")
    b = b[b["site_code"].isin(sites)].set_index("site_code")
    both = a.index.intersection(b.index)
    a, b = a.loc[both], b.loc[both]
    out = pd.DataFrame(index=both)
    out["site_name"] = a["site_name"]
    sites_tbl = pd.read_parquet(INTERIM_DIR / "sites.parquet").drop_duplicates("site_code")
    chp = sites_tbl.set_index("site_code")["fossil_fuel_led_chp_units_operated_on_site"]
    out["chp_site"] = chp.reindex(both).fillna(0).gt(0).to_numpy()
    for f in FIELDS:
        out[f"{f}_2324"] = a[f]
        out[f"{f}_2425"] = b[f]
        out[f"{f}_delta"] = b[f] - a[f]
    pos = lambda s: s.fillna(0) > 0  # noqa: E731
    out["solar_new"] = ~pos(a["solar_generated_kwh"]) & pos(b["solar_generated_kwh"])
    out["solar_grew"] = pos(a["solar_generated_kwh"]) & (
        b["solar_generated_kwh"] > 1.5 * a["solar_generated_kwh"]
    )
    out["heat_pumps_new"] = ~pos(a["heat_pumps"]) & pos(b["heat_pumps"])
    out["heat_pumps_more"] = pos(a["heat_pumps"]) & (b["heat_pumps"] > a["heat_pumps"])
    out["led_up_10pts"] = (b["led_pct"] - a["led_pct"]) >= 10
    out["chp_changed"] = b["chp_units"].fillna(0) != a["chp_units"].fillna(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        out["gia_change_pct"] = 100 * (b["gia_m2"] / a["gia_m2"] - 1)
        out["elec_change_pct"] = 100 * (b["elec_kwh"] / a["elec_kwh"] - 1)
        out["gas_change_pct"] = 100 * (b["gas_kwh"] / a["gas_kwh"] - 1)
    out["any_change"] = out[
        [
            "solar_new",
            "solar_grew",
            "heat_pumps_new",
            "heat_pumps_more",
            "led_up_10pts",
            "chp_changed",
        ]
    ].any(axis=1)
    return out.rename_axis("site_code")


def demand_steps(min_step: float = MIN_STEP) -> pd.DataFrame:
    """PELT changepoints in each site's log daily demand net of its day-of-week pattern; the
    level before and after each step is the mean of the adjoining segments."""

    kw = daily_demand()  # date x site, mean kW per local day
    rows = []
    for s in kw.columns:
        for step in steps_in_series(kw[s], min_step):
            rows.append({"site_code": s, **step})
    return pd.DataFrame(rows)


def steps_in_series(kw: pd.Series, min_step: float = MIN_STEP) -> list[dict]:
    """PELT steps in one site's daily mean demand (log, net of day-of-week means)."""
    import ruptures as rpt

    y = np.log(kw.where(kw > 0)).dropna()
    if len(y) < 4 * MIN_SEGMENT_DAYS:
        return []
    y = y - y.groupby(y.index.dayofweek).transform("mean")
    x = y.to_numpy()
    d = np.diff(x)
    sigma = 1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2)
    pen = PEN_FACTOR * max(sigma, 1e-3) ** 2 * np.log(len(x))
    bkps = rpt.Pelt(model="l2", min_size=MIN_SEGMENT_DAYS, jump=1).fit(x).predict(pen=pen)
    edges = [0, *bkps]
    out = []
    for i in range(1, len(edges) - 1):
        before = x[edges[i - 1] : edges[i]].mean()
        after = x[edges[i] : edges[i + 1]].mean()
        change = float(np.expm1(after - before))
        if abs(change) >= min_step:
            out.append(
                {
                    "date": y.index[edges[i]],
                    "change": change,
                    "level_before_kw": float(np.exp(before + np.log(kw.mean()))),
                    "segment_before_days": edges[i] - edges[i - 1],
                    "segment_after_days": edges[i + 1] - edges[i],
                }
            )
    return out


def match_steps(changes: pd.DataFrame, steps: pd.DataFrame) -> pd.DataFrame:
    """For every site with a returned change, the nearest demand step in or near 2024/25."""
    lo, hi = FY[0] - MATCH_MARGIN, FY[1] + MATCH_MARGIN
    window = steps[(steps["date"] >= lo) & (steps["date"] < hi)]
    out = changes.copy()
    out["steps_in_window"] = (
        window.groupby("site_code").size().reindex(out.index).fillna(0).astype(int)
    )
    biggest = window.loc[window.groupby("site_code")["change"].apply(lambda s: s.abs().idxmax())]
    biggest = biggest.set_index("site_code")
    out["step_date"] = biggest["date"].reindex(out.index)
    out["step_change"] = biggest["change"].reindex(out.index)
    return out


def run() -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    changes = eric_changes()
    steps = demand_steps()
    steps.to_csv(OUT_DIR / "demand_steps.csv", index=False)
    matched = match_steps(changes, steps)
    matched.to_csv(OUT_DIR / "eric_changes.csv")
    flags = [
        "solar_new",
        "solar_grew",
        "heat_pumps_new",
        "heat_pumps_more",
        "led_up_10pts",
        "chp_changed",
    ]
    summary = {
        "sites_in_both_returns": int(len(matched)),
        "changes": {f: int(matched[f].sum()) for f in flags},
        "sites_with_any_change": int(matched["any_change"].sum()),
        "changed_sites_with_a_step_in_window": int(
            (matched["any_change"] & (matched["steps_in_window"] > 0)).sum()
        ),
        "changed_non_chp_sites_with_a_step_in_window": int(
            (matched["any_change"] & ~matched["chp_site"] & (matched["steps_in_window"] > 0)).sum()
        ),
        "steps_total": int(len(steps)),
        "steps_in_window": int(
            ((steps["date"] >= FY[0] - MATCH_MARGIN) & (steps["date"] < FY[1] + MATCH_MARGIN)).sum()
        ),
        "median_abs_elec_change_pct": round(float(matched["elec_change_pct"].abs().median()), 1),
    }
    (OUT_DIR / "eric_changes_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    logger.info("ERIC changes: %s", summary)
    return {"changes": matched, "steps": steps, "summary": summary}
