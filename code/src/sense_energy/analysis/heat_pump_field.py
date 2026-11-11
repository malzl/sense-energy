"""Heat pumps in the field: what the estates returns say they did to gas and electricity.

The returns give a count of heat pumps per site and the site's annual gas and electricity. Between
two returns a site either reports its first heat pump, more of them, or no change. If the heat
pumps replaced gas heating, gas falls and electricity rises by the heat displaced over the
coefficient of performance (COP): ``electricity added = 0.85 x gas removed / COP`` with the boiler
efficiency of the headroom work. Three steps on every English site in both returns, CHP sites
left out (their gas also makes electricity):

1. ``did``: the change in gas and in electricity at sites with new heat pumps against sites that
   never reported one (differences in median log changes, bootstrap intervals), and the share of
   sites whose gas fell by a fifth or more.
2. ``switchers``: sites whose gas fell by ``SWITCH_FALL`` or more on an unchanged floor area while
   electricity rose: the implied COP site by site, for sites with and without reported heat pumps.
3. ``slope``: Theil-Sen slope of the electricity change on the gas removed (kWh per m2) among the
   sites with new heat pumps, which is ``0.85 / COP`` if the pumps replace gas, with the same
   slope among sites without heat pumps as the placebo.

The returns are annual and self-reported: reporting scopes change, units slip, and the heat-pump
field counts every unit whatever its size. Sites whose floor area or electricity moved
implausibly are flagged and left out of the estimates; a fall in gas is never a reason to leave
a site out.

Year pairs come from ``data.eric.YEARS``: every consecutive pair whose site tables are on disk.
Outputs in ``code/reports/heat_pump_field/``.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import theilslopes

from ..config import INTERIM_DIR, PROJECT_ROOT
from ..logging_utils import get_logger

logger = get_logger(__name__)

OUT_DIR = PROJECT_ROOT / "code" / "reports" / "heat_pump_field"
BOILER_EFFICIENCY = 0.85
SWITCH_FALL = 0.25  # gas down by a quarter or more
MAX_AREA_CHANGE = 0.05
MAX_LOG_CHANGE = np.log(3.0)  # a threefold change in a year is a reporting change
N_BOOT = 2000
STATUS_ORDER = ("never", "had already", "new", "more")


def year_pairs() -> list[tuple[str, str]]:
    from ..data.eric import YEARS

    have = [y for y in YEARS if (INTERIM_DIR / f"eric_{y}_site.parquet").exists()]
    return list(zip(have[:-1], have[1:], strict=True))


def changes(before: str, after: str) -> pd.DataFrame:
    """One row per site in both returns: heat-pump status and the changes in gas, electricity
    and floor area, with the plausibility flag."""
    a = pd.read_parquet(INTERIM_DIR / f"eric_{before}_site.parquet")
    b = pd.read_parquet(INTERIM_DIR / f"eric_{after}_site.parquet")
    cols = ["site_code", "elec_kwh", "gas_kwh", "gia_m2", "heat_pumps", "chp_units"]
    m = a[cols].merge(
        b[[*cols, "site_name", "site_type", "trust_name"]], on="site_code", suffixes=("_a", "_b")
    )
    for c in ("heat_pumps", "chp_units", "gas_kwh"):
        for s in ("_a", "_b"):
            m[c + s] = m[c + s].fillna(0.0)
    hp_a, hp_b = m["heat_pumps_a"], m["heat_pumps_b"]
    m["status"] = np.select(
        [(hp_a == 0) & (hp_b > 0), (hp_a > 0) & (hp_b > hp_a), (hp_a > 0) & (hp_b == hp_a)],
        ["new", "more", "had already"],
        default="never",
    )
    m.loc[hp_b < hp_a, "status"] = "fewer"
    m["pair"] = f"{before.replace('_', '/')} to {after.replace('_', '/')}"
    with np.errstate(divide="ignore", invalid="ignore"):
        m["dlog_elec"] = np.log(m["elec_kwh_b"] / m["elec_kwh_a"])
        m["dlog_gas"] = np.log(m["gas_kwh_b"] / m["gas_kwh_a"])
        m["dlog_area"] = np.log(m["gia_m2_b"] / m["gia_m2_a"])
    m["d_elec_kwh"] = m["elec_kwh_b"] - m["elec_kwh_a"]
    m["d_gas_kwh"] = m["gas_kwh_b"] - m["gas_kwh_a"]
    m["no_chp"] = (m["chp_units_a"] == 0) & (m["chp_units_b"] == 0)
    m["plausible"] = (
        (m["elec_kwh_a"] > 0)
        & (m["elec_kwh_b"] > 0)
        & (m["gia_m2_a"] > 0)
        & (m["dlog_area"].abs() <= MAX_AREA_CHANGE)
        & (m["dlog_elec"].abs() <= MAX_LOG_CHANGE)
    )
    m["has_gas"] = m["gas_kwh_a"] > 0
    # gas may fall to nothing: that is the switch being looked for, so it is not trimmed (the
    # estimates are medians and rank-based slopes)
    m["usable"] = m["no_chp"] & m["plausible"] & m["has_gas"]
    return m


def _boot_diff(x: np.ndarray, y: np.ndarray, stat=np.median, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    d = [stat(rng.choice(x, len(x))) - stat(rng.choice(y, len(y))) for _ in range(N_BOOT)]
    return float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))


def did(m: pd.DataFrame) -> pd.DataFrame:
    """Median log change in gas and electricity by heat-pump status, and the difference from the
    sites that never reported a heat pump, in per cent."""
    u = m[m["usable"]]
    ref = u[u["status"] == "never"]
    rows = []
    for st in STATUS_ORDER:
        g = u[u["status"] == st]
        if len(g) < 3:
            continue
        row = {"pair": m["pair"].iloc[0], "status": st, "sites": len(g)}
        for v, name in (("dlog_gas", "gas"), ("dlog_elec", "elec")):
            row[f"{name}_median_pct"] = 100 * (np.exp(g[v].median()) - 1)
            if st != "never":
                lo, hi = _boot_diff(g[v].to_numpy(), ref[v].to_numpy())
                row[f"{name}_did_pct"] = 100 * (g[v].median() - ref[v].median())
                row[f"{name}_did_lo"], row[f"{name}_did_hi"] = 100 * lo, 100 * hi
        row["gas_fell_fifth_share"] = float((g["dlog_gas"] <= np.log(0.8)).mean())
        row["elec_rose_tenth_share"] = float((g["dlog_elec"] >= np.log(1.1)).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def switchers(m: pd.DataFrame) -> pd.DataFrame:
    """Sites whose gas fell by ``SWITCH_FALL`` or more on an unchanged floor area: the heat
    removed, the electricity added net of the trend of the never-sites, and the implied COP."""
    u = m[m["no_chp"] & m["plausible"] & m["has_gas"]].copy()
    trend = float(u.loc[u["status"] == "never", "dlog_elec"].median())
    s = u[u["gas_kwh_b"] <= (1 - SWITCH_FALL) * u["gas_kwh_a"]].copy()
    s["heat_removed_kwh"] = -s["d_gas_kwh"] * BOILER_EFFICIENCY
    s["elec_added_kwh"] = s["elec_kwh_b"] - s["elec_kwh_a"] * np.exp(trend)
    s["elec_change_pct"] = 100 * (np.exp(s["dlog_elec"]) - 1)
    s["gas_change_pct"] = 100 * (s["gas_kwh_b"] / s["gas_kwh_a"] - 1)
    s["implied_cop"] = np.where(
        s["elec_added_kwh"] > 0, s["heat_removed_kwh"] / s["elec_added_kwh"], np.nan
    )
    s["reports_heat_pump"] = s["heat_pumps_b"] > 0
    cols = [
        "pair",
        "site_code",
        "site_name",
        "site_type",
        "status",
        "heat_pumps_a",
        "heat_pumps_b",
        "reports_heat_pump",
        "gas_change_pct",
        "elec_change_pct",
        "heat_removed_kwh",
        "elec_added_kwh",
        "implied_cop",
        "gia_m2_a",
    ]
    return s[cols].sort_values("gas_change_pct")


def slope(m: pd.DataFrame) -> pd.DataFrame:
    """Theil-Sen slope of the electricity change on the gas removed, both per m2 of floor area,
    by heat-pump status; ``BOILER_EFFICIENCY / slope`` is the COP if the pumps replace gas."""
    u = m[m["usable"]].copy()
    trend = float(u.loc[u["status"] == "never", "dlog_elec"].median())
    u["y"] = (u["elec_kwh_b"] - u["elec_kwh_a"] * np.exp(trend)) / u["gia_m2_a"]
    u["x"] = -u["d_gas_kwh"] / u["gia_m2_a"]
    rows = []
    for st, g in (
        ("new or more", u[u["status"].isin(["new", "more"])]),
        ("never", u[u["status"] == "never"]),
    ):
        if len(g) < 8:
            continue
        b, _, lo, hi = theilslopes(g["y"].to_numpy(), g["x"].to_numpy(), 0.95)
        rows.append(
            {
                "pair": m["pair"].iloc[0],
                "status": st,
                "sites": len(g),
                "slope": b,
                "slope_lo": lo,
                "slope_hi": hi,
                "implied_cop": BOILER_EFFICIENCY / b if b > 0 else np.nan,
                "implied_cop_lo": BOILER_EFFICIENCY / hi if hi > 0 else np.nan,
                "implied_cop_hi": BOILER_EFFICIENCY / lo if lo > 0 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def panel_sites(m: pd.DataFrame) -> pd.DataFrame:
    """The panel sites that report new or more heat pumps, with the metered weather-normalised
    change next to the returned gas and electricity."""
    path = PROJECT_ROOT / "code" / "reports" / "installation_savings" / "site_changes.csv"
    if not path.exists():
        return pd.DataFrame()
    s = pd.read_csv(path)
    s = s[s["heat_pumps_new"] | s["heat_pumps_more"]][["site_code", "nac_change_pct", "chp_site"]]
    out = s.merge(m, on="site_code", how="left")
    out["gas_change_pct"] = 100 * (out["gas_kwh_b"] / out["gas_kwh_a"] - 1)
    out["elec_change_pct"] = 100 * (out["elec_kwh_b"] / out["elec_kwh_a"] - 1)
    return out[
        [
            "site_code",
            "site_name",
            "chp_site",
            "heat_pumps_a",
            "heat_pumps_b",
            "nac_change_pct",
            "gas_kwh_a",
            "gas_kwh_b",
            "gas_change_pct",
            "elec_change_pct",
            "plausible",
        ]
    ]


def run() -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pairs = year_pairs()
    all_m = [changes(a, b) for a, b in pairs]
    m = pd.concat(all_m, ignore_index=True)
    m.drop(columns=["trust_name"]).to_csv(OUT_DIR / "site_changes.csv", index=False)
    d = pd.concat([did(x) for x in all_m], ignore_index=True)
    d.to_csv(OUT_DIR / "did.csv", index=False)
    sw = pd.concat([switchers(x) for x in all_m], ignore_index=True)
    sw.to_csv(OUT_DIR / "switchers.csv", index=False)
    sl = pd.concat([slope(x) for x in all_m], ignore_index=True)
    sl.to_csv(OUT_DIR / "slope.csv", index=False)
    ps = panel_sites(all_m[0])
    ps.to_csv(OUT_DIR / "panel_sites.csv", index=False)

    first = all_m[0]
    hp_sw = sw[sw["reports_heat_pump"] & sw["implied_cop"].between(1.0, 8.0)]
    new = d[d["status"] == "new"].iloc[0] if (d["status"] == "new").any() else None
    never = d[d["status"] == "never"].iloc[0]
    summary = {
        "pairs": [x["pair"].iloc[0] for x in all_m],
        "sites_in_both": int(len(first)),
        "status_counts": first["status"].value_counts().to_dict(),
        "usable_counts": first[first["usable"]]["status"].value_counts().to_dict(),
        "new_heat_pump_sites": None
        if new is None
        else {
            "sites": int(new["sites"]),
            "gas_did_pct": round(float(new["gas_did_pct"]), 1),
            "gas_did_ci": [round(float(new["gas_did_lo"]), 1), round(float(new["gas_did_hi"]), 1)],
            "elec_did_pct": round(float(new["elec_did_pct"]), 1),
            "elec_did_ci": [
                round(float(new["elec_did_lo"]), 1),
                round(float(new["elec_did_hi"]), 1),
            ],
            "gas_fell_fifth_share": round(float(new["gas_fell_fifth_share"]), 3),
            "gas_fell_fifth_share_never": round(float(never["gas_fell_fifth_share"]), 3),
        },
        "switchers": {
            "sites": int(len(sw)),
            "reporting_heat_pumps": int(sw["reports_heat_pump"].sum()),
            "heat_pump_sites_with_plausible_cop": int(len(hp_sw)),
            "median_implied_cop": round(float(hp_sw["implied_cop"].median()), 2)
            if len(hp_sw)
            else None,
            "implied_cops": hp_sw["implied_cop"].round(2).tolist(),
        },
        "slope": sl.round(3).to_dict(orient="records"),
        "panel_sites": ps.round(1).to_dict(orient="records"),
    }
    text = json.dumps(summary, indent=2, default=float).replace("NaN", "null")
    (OUT_DIR / "summary.json").write_text(text + "\n")
    logger.info("heat pumps in the field: %s", json.dumps(summary["switchers"]))
    return {"summary": summary}
