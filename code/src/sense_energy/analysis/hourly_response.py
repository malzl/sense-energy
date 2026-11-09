"""Where in the day the weather response lives, site by site.

For every local hour of the day, the hour's mean demand of each site-day is regressed on site x
(cold dose, heat dose, sunshine) interactions with specification B's fixed effects (site x
year-month, site x weekday, date), on the clean sample of ``heat_drivers`` (no CHP, no meter
gaps). Twenty-four regressions give every site a 24-hour profile of its heat, cold and sunshine
response; per hour and term a random-effects decomposition across sites gives the pooled
response and empirical-Bayes posteriors for the site profiles.

Derived per site: the hour at which the heat response peaks, the share of the positive heat
response that falls in the 16:00-19:00 distribution red band (the time-of-use window most DNOs
charge most for), and whether a hot day moves the site's demand peak (hour of the highest
weekday demand on summer days at or above 25 degC maximum against other summer weekdays).

The engine for the counterfactual threads (``estate_counterfactuals``): the demand of a site in
hour h under a different temperature is the observed demand times exp(beta_heat[s, h] x change
in the heat dose + beta_cold[s, h] x change in the cold dose).

A check of the JEPA counterfactual machine (``jepa_check``): the weather-conditioned JEPA's
predicted change of a summer day's profile when the day is made 5 degC warmer, decoded to half
hours, against the same change implied by the regression posteriors.

Outputs (``code/reports/hourly_response/``): ``hourly_terms.csv`` (site x hour x term: slope,
SE, posterior), ``pooled.csv`` (hour x term), ``site_profiles.csv`` (one row per site with the
derived measures), ``jepa_check.csv``, ``summary.json``.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..config import PROCESSED_DIR, PROJECT_ROOT
from ..logging_utils import get_logger
from . import heat_drivers as hd
from . import weather_causal as wc
from . import weather_heterogeneity as wh
from . import weather_response as wr

logger = get_logger(__name__)

OUT_DIR = PROJECT_ROOT / "code" / "reports" / "hourly_response"
TERMS = hd.TERMS
HOURS = range(24)
RED_BAND = (16, 17, 18)  # 16:00-19:00 local
HOT_TMAX = 25.0
MIN_HOUR_PERIODS = 2


def hour_means(w: pd.DataFrame) -> pd.DataFrame:
    """Mean kW of each site in each local hour of each local day (both half-hours present)."""
    local = pd.to_datetime(w.index, utc=True).tz_convert(wr.TZ)
    day = local.normalize().tz_localize(None).rename("date")
    hour = pd.Index(local.hour, name="hour")
    n = w.notna().groupby([day, hour]).sum()
    mean = w.groupby([day, hour]).mean().where(n >= MIN_HOUR_PERIODS)
    out = mean.stack().rename("hour_kw").rename_axis(["date", "hour", "site_code"]).reset_index()
    return out[out["hour_kw"] > 0]


def hourly_terms(df: pd.DataFrame) -> pd.DataFrame:
    sub = hd.subset(df)
    sites = sorted(sub["site_code"].unique())
    w = pd.read_parquet(PROCESSED_DIR / "elec" / "site" / "wide_raw.parquet")[sites] * 2.0
    hm = hour_means(w)
    rows = []
    for h in HOURS:
        x = sub.merge(
            hm.loc[hm["hour"] == h, ["date", "site_code", "hour_kw"]],
            on=["date", "site_code"],
            how="inner",
        )
        fit, names, fsites = hd.fit_sites(x, np.log(x["hour_kw"].to_numpy()), TERMS)
        rows.append(hd.extract(fit, names, fsites, TERMS).assign(hour=h))
        logger.info("hour %02d done (%d site-days)", h, len(x))
    return pd.concat(rows, ignore_index=True)


def shrink_hourly(t: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per hour and term: random-effects pooled slope and tau across sites; posterior per site."""
    out, pooled = [], []
    for (h, term), d in t.groupby(["hour", "term"]):
        d = d[d["se"].notna() & (d["se"] > 0)].copy()
        re = wh.random_effects(d["slope"], d["se"])
        tau2 = re["tau"] ** 2
        b = tau2 / (tau2 + d["se"] ** 2)
        d["posterior"] = b * d["slope"] + (1 - b) * re["pooled"]
        out.append(d)
        w = 1 / (d["se"] ** 2 + tau2)
        se_pooled = float(np.sqrt(1 / w.sum()))
        pooled.append(
            {
                "hour": h,
                "term": term,
                "pooled": re["pooled"],
                "pooled_se": se_pooled,
                "tau": re["tau"],
                "i2": re["i2"],
                "sites": len(d),
            }
        )
    return pd.concat(out, ignore_index=True), pd.DataFrame(pooled)


def hot_day_peaks(sites: list[str], df: pd.DataFrame) -> pd.DataFrame:
    """Per site: hour of the highest mean weekday demand on summer days with T_max >= 25 degC and
    on other summer weekdays, and the ratio of the two peak levels."""
    w = pd.read_parquet(PROCESSED_DIR / "elec" / "site" / "wide_raw.parquet")[sites] * 2.0
    hm = hour_means(w)
    d = df[["site_code", "date", "t_max", "holiday"]]
    hm = hm.merge(d, on=["site_code", "date"], how="inner")
    hm = hm[
        hm["date"].dt.month.isin([6, 7, 8, 9])
        & (hm["date"].dt.dayofweek < 5)
        & (hm["holiday"] == 0)
    ]
    hm["hot"] = hm["t_max"] >= HOT_TMAX
    prof = hm.groupby(["site_code", "hot", "hour"])["hour_kw"].mean().unstack("hour")
    rows = []
    for s in sites:
        if (s, True) not in prof.index or (s, False) not in prof.index:
            continue
        hot, other = prof.loc[(s, True)], prof.loc[(s, False)]
        rows.append(
            {
                "site_code": s,
                "peak_hour_hot": int(hot.idxmax()),
                "peak_hour_other": int(other.idxmax()),
                "peak_ratio_hot": float(hot.max() / other.max()),
                "hot_days": int(hm[(hm["site_code"] == s) & hm["hot"]]["date"].nunique()),
            }
        )
    return pd.DataFrame(rows).set_index("site_code")


def site_profiles(post: pd.DataFrame, peaks: pd.DataFrame) -> pd.DataFrame:
    heat = post[post["term"] == "cdd20"].pivot(
        index="site_code", columns="hour", values="posterior"
    )
    out = pd.DataFrame(index=heat.index)
    out["heat_peak_hour"] = heat.idxmax(axis=1)
    pos = heat.clip(lower=0)
    out["heat_redband_share"] = pos[list(RED_BAND)].sum(axis=1) / pos.sum(axis=1).replace(0, np.nan)
    out["heat_daily_mean_pct"] = 100 * heat.mean(axis=1)
    out["heat_redband_pct"] = 100 * heat[list(RED_BAND)].mean(axis=1)
    out["heat_night_pct"] = 100 * heat[list(range(0, 6))].mean(axis=1)
    out["heat_evening_excess_pct"] = out["heat_redband_pct"] - out["heat_night_pct"]
    cold = post[post["term"] == "hdd14"].pivot(
        index="site_code", columns="hour", values="posterior"
    )
    out["cold_daily_mean_pct"] = 100 * cold.mean(axis=1)
    out["cold_peak_hour"] = cold.idxmax(axis=1)
    return out.join(peaks, how="left")


# --------------------------------------------------------------------------- #
# The JEPA counterfactual machine, checked against the regression
# --------------------------------------------------------------------------- #


def jepa_check(
    post: pd.DataFrame, delta_c: float = 5.0, device: str | None = None, steps: int = 3000
) -> pd.DataFrame:
    """For every summer window of every site in ``post``: the weather-conditioned JEPA predicts
    the last seven days' embeddings under the actual weather and with their temperatures raised
    by ``delta_c``; a decoder from the (standardised) target space to the day's scaled profile
    turns both into half-hourly values. The relative change per hour is set against the
    regression's exp(beta_heat[s, h] x change in the heat dose) - 1 on the same days."""
    import torch

    from ..experiments import jepa

    device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    model, cfg = jepa.load(jepa.OUT_DIR_WEATHER, device)
    tokens = jepa.day_tokens(weather=True)
    store = jepa.Store(tokens, device)
    names = list(jepa.WEATHER_VARS)
    std = tokens["wx_std"]
    shift = torch.zeros(store.k, device=device)
    for v in jepa.TEMPERATURE_VARS:
        shift[names.index(v)] = delta_c / float(std[names.index(v)])
    # decoder from standardised target embeddings to scaled profiles, on all tiled days
    tiles = jepa._tiling(tokens["valid"])
    Z, X = [], []
    for i in range(0, len(tiles), 512):
        chunk = tiles[i : i + 512]
        s = torch.tensor([a for a, _ in chunk], device=device)
        d0 = torch.tensor([b for _, b in chunk], device=device)
        x, side, pos = store.gather(s, d0)
        with torch.no_grad():
            full = model.target(x, side, pos).float()
            Z.append(
                torch.nn.functional.layer_norm(full, full.shape[-1:]).reshape(-1, full.shape[-1])
            )
            X.append(x.reshape(-1, x.shape[-1]))
    Z, X = torch.cat(Z), torch.cat(X)
    dec = torch.nn.Sequential(
        torch.nn.Linear(Z.shape[1], 256),
        torch.nn.GELU(),
        torch.nn.Linear(256, 256),
        torch.nn.GELU(),
        torch.nn.Linear(256, jepa.SLOTS),
    ).to(device)
    opt = torch.optim.AdamW(dec.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(steps):
        b = torch.randint(0, len(Z), (2048,), device=device)
        loss = torch.nn.functional.mse_loss(dec(Z[b]), X[b])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    logger.info("JEPA check: decoder MSE %.4f on %d days", loss.item(), len(Z))
    # counterfactual on summer windows (last seven days in June-September)
    heat = post[post["term"] == "cdd20"].pivot(
        index="site_code", columns="hour", values="posterior"
    )
    keep = [
        (s, d)
        for s, d in tiles
        if tokens["sites"][s] in heat.index
        and tokens["dates"][d + jepa.WINDOW - 1].month in (6, 7, 8, 9)
    ]
    mask_row = torch.zeros(jepa.WINDOW, dtype=torch.bool, device=device)
    mask_row[jepa.WINDOW - jepa.N_TARGET :] = True
    tmean_i = names.index("t_mean")
    rows = []
    for i in range(0, len(keep), 256):
        chunk = keep[i : i + 256]
        B = len(chunk)
        s = torch.tensor([a for a, _ in chunk], device=device)
        d0 = torch.tensor([b for _, b in chunk], device=device)
        x, side, pos = store.gather(s, d0)
        xc, sc, pc, st, pt, _ = jepa.split(x, side, pos, mask_row[None].expand(B, -1))
        with torch.no_grad():
            ctx = model.encoder(xc, sc, pc)
            base = dec(model.predictor(ctx, pc, st, pt).float())
            st2 = st.clone()
            st2[:, :, 9:] += shift
            warm = dec(model.predictor(ctx, pc, st2, pt).float())
        rel = (warm.clamp(min=1e-3) / base.clamp(min=1e-3) - 1).cpu().numpy()  # B x 7 x 48
        t_std = st[:, :, 9 + tmean_i].cpu().numpy() * float(std[tmean_i]) + float(
            tokens["wx_mean"][tmean_i]
        )
        for b, (si, _di) in enumerate(chunk):
            site = tokens["sites"][si]
            beta = heat.loc[site].to_numpy()
            for k in range(jepa.N_TARGET):
                t = t_std[b, k]
                dcdd = max(t + delta_c - 20.0, 0.0) - max(t - 20.0, 0.0)
                jepa_h = rel[b, k].reshape(24, 2).mean(axis=1)
                reg_h = np.expm1(beta * dcdd)
                rows.append(
                    {
                        "site_code": site,
                        "t_mean": t,
                        "dcdd": dcdd,
                        **{f"jepa_h{h}": jepa_h[h] for h in HOURS},
                        **{f"reg_h{h}": reg_h[h] for h in HOURS},
                    }
                )
    out = pd.DataFrame(rows)
    logger.info("JEPA check: %d summer site-days", len(out))
    return out


def summarise_check(chk: pd.DataFrame) -> pd.DataFrame:
    """Per site: the mean daily relative change from the JEPA and from the regression, and the
    correlation of their hour profiles, on days where the +5 degC shift changes the heat dose."""
    c = chk[chk["dcdd"] > 0]
    jh = [f"jepa_h{h}" for h in HOURS]
    rh = [f"reg_h{h}" for h in HOURS]
    g = c.groupby("site_code")
    out = pd.DataFrame(
        {
            "days": g.size(),
            "jepa_mean_pct": 100 * g[jh].mean().mean(axis=1),
            "reg_mean_pct": 100 * g[rh].mean().mean(axis=1),
        }
    )
    jm, rm = g[jh].mean(), g[rh].mean()
    out["shape_corr"] = [
        np.corrcoef(jm.loc[s].to_numpy(), rm.loc[s].to_numpy())[0, 1]
        if jm.loc[s].std() > 0 and rm.loc[s].std() > 0
        else np.nan
        for s in out.index
    ]
    return out


def run(with_jepa: bool = True, device: str | None = None) -> dict:
    from threadpoolctl import threadpool_limits

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = wc.panel()
    with threadpool_limits(4):
        t = hourly_terms(df)
    post, pooled = shrink_hourly(t)
    post.to_csv(OUT_DIR / "hourly_terms.csv", index=False)
    pooled.to_csv(OUT_DIR / "pooled.csv", index=False)
    sites = sorted(post["site_code"].unique())
    peaks = hot_day_peaks(sites, df)
    prof = site_profiles(post, peaks)
    prof.to_csv(OUT_DIR / "site_profiles.csv")
    summary = {
        "sites": len(sites),
        "pooled_heat_pct_by_hour": {
            int(r.hour): round(100 * r.pooled, 2)
            for r in pooled[pooled["term"] == "cdd20"].itertuples()
        },
        "pooled_cold_pct_by_hour": {
            int(r.hour): round(100 * r.pooled, 2)
            for r in pooled[pooled["term"] == "hdd14"].itertuples()
        },
        "heat_peak_hour_counts": prof["heat_peak_hour"].value_counts().sort_index().to_dict(),
        "median_redband_share": round(float(prof["heat_redband_share"].median()), 3),
        "sites_peak_moves_later_on_hot_days": int(
            (prof["peak_hour_hot"] > prof["peak_hour_other"]).sum()
        ),
        "sites_peak_moves_earlier_on_hot_days": int(
            (prof["peak_hour_hot"] < prof["peak_hour_other"]).sum()
        ),
        "sites_with_hot_day_peaks": int(prof["peak_hour_hot"].notna().sum()),
    }
    if with_jepa:
        try:
            chk = jepa_check(post, device=device)
            chk.to_csv(OUT_DIR / "jepa_check_days.csv", index=False)
            sc = summarise_check(chk)
            sc.to_csv(OUT_DIR / "jepa_check.csv")
            summary["jepa_check"] = {
                "sites": int(len(sc)),
                "corr_daily_mean_across_sites": round(
                    float(sc[["jepa_mean_pct", "reg_mean_pct"]].corr().iloc[0, 1]), 3
                ),
                "median_shape_corr": round(float(sc["shape_corr"].median()), 3),
                "median_jepa_pct": round(float(sc["jepa_mean_pct"].median()), 2),
                "median_reg_pct": round(float(sc["reg_mean_pct"].median()), 2),
            }
        except Exception as exc:  # the check is optional
            logger.warning("JEPA check skipped: %s", exc)
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    logger.info("hourly response: %s", json.dumps(summary)[:600])
    return {"terms": post, "pooled": pooled, "profiles": prof, "summary": summary}
