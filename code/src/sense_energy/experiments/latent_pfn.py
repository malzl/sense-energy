"""The LaT-PFN route: a prior-fitted network that forecasts in the JEPA's latent space.

Two small models on top of the weather-conditioned JEPA (``experiments/jepa.py``):

* **Decoder**: embedding of a day -> its 48 half-hour values (scaled by the site's trailing
  month), an MLP fitted on every embedded day. It is the way back from latent space to kWh.
* **Latent PFN**: an in-context regressor. A task is one site: a context set of days, each a
  token of (day embedding, that day's weather, calendar), and query days given only weather and
  calendar; a transformer attends across the set and predicts the query days' embeddings. It is
  trained across all sites and periods before the test window, so at forecast time it adapts to
  a site from its own recent days with no fitting, the way TabPFN adapts to a table. This is
  LaT-PFN's idea (in-context forecasting in a learned latent space) with the panel as the prior.

**Day-ahead evaluation** on the proof-of-concept's site-days (test window June 2025 to May 2026):
context = the site's last 42 embedded days before the target day; query weather = realised ERA5
(an upper bound on weather) or the 00z IFS ENS run of the day before, aggregated to the same
daily variables; profile = decoder(predicted embedding) x the trailing scale. Scored as in
``analysis/forecast_heat_days`` (nMAE per site by day class), next to the existing models and
to an *oracle* that decodes the day's true embedding (what the latent space can express at all).

Outputs (``code/reports/latent_pfn/``): ``decoder.pt``, ``pfn.pt``, ``train_log.csv``,
``forecasts.parquet`` (site, date, slot, variant, kWh), ``scores_by_day_class.csv``,
``summary.json``.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import INTERIM_DIR, LOCAL_TZ, PROJECT_ROOT
from ..logging_utils import get_logger
from . import jepa

logger = get_logger(__name__)

OUT_DIR = PROJECT_ROOT / "code" / "reports" / "latent_pfn"
POC = PROJECT_ROOT / "code" / "reports" / "poc"
TRAIN_END = pd.Timestamp("2025-05-31")
TEST_START, TEST_END = pd.Timestamp("2025-06-01"), pd.Timestamp("2026-05-08")
CONTEXT_DAYS = 42
LOOKBACK_DAYS = 120  # context days are sampled from this many days before the queries
N_QUERY = 8
DEFAULT = {
    "d_model": 128,
    "n_heads": 4,
    "layers": 4,
    "ff_mult": 2,
    "dropout": 0.1,
    "batch_tasks": 96,
    "steps": 3000,
    "lr": 5e-4,
    "weight_decay": 0.05,
    "warmup": 150,
    "decoder_steps": 3000,
    "decoder_hidden": 256,
    "seed": 0,
}


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #


def load_latent(out_dir: Path = jepa.OUT_DIR_WEATHER) -> dict:
    """Day embeddings aligned to the token arrays: ``e`` (S x D x d) with ``has`` (S x D)."""
    tokens = jepa.day_tokens(weather=True)
    days = pd.read_parquet(out_dir / "day_embeddings.parquet")
    cols = [c for c in days.columns if c.startswith("e")]
    S, D = tokens["valid"].shape
    e = np.zeros((S, D, len(cols)), dtype=np.float32)
    has = np.zeros((S, D), dtype=bool)
    si = pd.Index(tokens["sites"]).get_indexer(days["site_code"])
    di = pd.Index(tokens["dates"]).get_indexer(pd.to_datetime(days["date"]))
    ok = (si >= 0) & (di >= 0)
    e[si[ok], di[ok]] = days.loc[ok, cols].to_numpy(dtype=np.float32)
    has[si[ok], di[ok]] = True
    tokens["e"], tokens["has"] = e, has & tokens["valid"]
    tokens["cal"] = jepa.calendar_features(tokens["dow"], tokens["doy"])
    return tokens


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #


def build_decoder(d_in: int, hidden: int):
    from torch import nn

    return nn.Sequential(
        nn.Linear(d_in, hidden),
        nn.GELU(),
        nn.Linear(hidden, hidden),
        nn.GELU(),
        nn.Linear(hidden, jepa.SLOTS),
    )


def build_pfn(cfg: dict, d_latent: int, n_side: int):
    import torch
    from torch import nn

    d = cfg["d_model"]

    class LatentPFN(nn.Module):
        def __init__(self):
            super().__init__()
            self.ctx_in = nn.Linear(d_latent + n_side, d)
            self.qry_in = nn.Linear(
                n_side + d_latent, d
            )  # the query carries the last context day's embedding
            self.kind = nn.Parameter(torch.zeros(2, d))  # context / query flags
            layer = nn.TransformerEncoderLayer(
                d,
                cfg["n_heads"],
                dim_feedforward=cfg["ff_mult"] * d,
                dropout=cfg["dropout"],
                batch_first=True,
                norm_first=True,
                activation="gelu",
            )
            self.body = nn.TransformerEncoder(layer, cfg["layers"])
            self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d_latent))

        def forward(self, ctx_e, ctx_side, qry_side):
            c = self.ctx_in(torch.cat([ctx_e, ctx_side], dim=-1)) + self.kind[0]
            last = ctx_e[:, -1:, :].expand(
                -1, qry_side.shape[1], -1
            )  # context days are in time order
            q = self.qry_in(torch.cat([qry_side, last], dim=-1)) + self.kind[1]
            z = self.body(torch.cat([c, q], dim=1))
            return self.out(z[:, c.shape[1] :])

    return LatentPFN()


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #


def _side(tokens, s, days, ref_day: int | None = None):
    """(weather, calendar, relative time) side features for site s and day indices; the relative
    time is the number of days before ``ref_day`` (the first query day) over 100, so the model can
    weight recent context more than old."""
    rel = np.zeros((len(days), 1), dtype=np.float32)
    if ref_day is not None:
        rel[:, 0] = (ref_day - np.asarray(days)) / 100.0
    parts = [tokens["wx"][s, days], tokens["cal"][days], rel]
    return np.concatenate(parts, axis=-1).astype(np.float32)


def sample_task(tokens, rng, s: int, train_end_idx: int):
    """Context days from a lookback window and query days just after it, all with embeddings."""
    has = tokens["has"][s, : train_end_idx + 1]
    idx = np.where(has)[0]
    if len(idx) < CONTEXT_DAYS + N_QUERY + 10:
        return None
    q_end = rng.integers(CONTEXT_DAYS + N_QUERY + 10, len(idx))
    q = idx[q_end - N_QUERY : q_end]
    pool = idx[max(0, q_end - N_QUERY - LOOKBACK_DAYS) : q_end - N_QUERY]
    if len(pool) < CONTEXT_DAYS:
        return None
    c = np.sort(rng.choice(pool, CONTEXT_DAYS, replace=False))
    return c, q


def train(
    cfg: dict | None = None,
    device: str | None = None,
    tokens: dict | None = None,
    out_dir: Path = OUT_DIR,
) -> dict:
    import torch

    cfg = {**DEFAULT, **(cfg or {})}
    device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    tokens = tokens if tokens is not None else load_latent()
    S, D, d_latent = tokens["e"].shape
    n_side = tokens["wx"].shape[-1] + 9 + 1
    train_end_idx = int(np.searchsorted(tokens["dates"], TRAIN_END, side="right") - 1)
    use_amp = device.startswith("cuda")
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- decoder: embedding -> scaled profile, on training-period days ---------------------
    has_tr = tokens["has"].copy()
    has_tr[:, train_end_idx + 1 :] = False
    si, di = np.where(has_tr)
    E = torch.tensor(tokens["e"][si, di], device=device)
    X = torch.tensor(tokens["x"][si, di], device=device)
    dec = build_decoder(d_latent, cfg["decoder_hidden"]).to(device)
    opt = torch.optim.AdamW(dec.parameters(), lr=1e-3, weight_decay=1e-4)
    log = []
    t0 = time.time()
    for step in range(int(cfg["decoder_steps"])):
        b = torch.randint(0, len(E), (1024,), device=device)
        loss = torch.nn.functional.mse_loss(dec(E[b]), X[b])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 500 == 0 or step == cfg["decoder_steps"] - 1:
            log.append(
                {"model": "decoder", "step": step, "loss": loss.item(), "seconds": time.time() - t0}
            )
    logger.info("decoder: %d training days, final MSE %.4f (scaled units)", len(E), log[-1]["loss"])

    # --- latent PFN ---------------------------------------------------------------------
    pfn = build_pfn(cfg, d_latent, n_side).to(device)
    opt = torch.optim.AdamW(pfn.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    steps = int(cfg["steps"])
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: (
            min(1.0, (s + 1) / cfg["warmup"])
            * 0.5
            * (1 + math.cos(math.pi * min(s, steps) / steps))
        ),
    )
    eligible = [
        s
        for s in range(S)
        if tokens["has"][s, : train_end_idx + 1].sum() >= CONTEXT_DAYS + N_QUERY + 10
    ]
    t0 = time.time()
    for step in range(steps):
        tasks = []
        while len(tasks) < cfg["batch_tasks"]:
            s = int(rng.choice(eligible))
            t = sample_task(tokens, rng, s, train_end_idx)
            if t is not None:
                tasks.append((s, *t))
        ce = torch.tensor(np.stack([tokens["e"][s, c] for s, c, _ in tasks]), device=device)
        cs = torch.tensor(
            np.stack([_side(tokens, s, c, int(q[0])) for s, c, q in tasks]), device=device
        )
        qs = torch.tensor(
            np.stack([_side(tokens, s, q, int(q[0])) for s, _, q in tasks]), device=device
        )
        qe = torch.tensor(np.stack([tokens["e"][s, q] for s, _, q in tasks]), device=device)
        pfn.train()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            pred = pfn(ce, cs, qs)
            loss = torch.nn.functional.smooth_l1_loss(pred.float(), qe)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(pfn.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 100 == 0 or step == steps - 1:
            log.append(
                {"model": "pfn", "step": step, "loss": loss.item(), "seconds": time.time() - t0}
            )
            if step % 500 == 0 or step == steps - 1:
                logger.info("latent PFN step %d loss %.4f", step, loss.item())
    torch.save(
        {"cfg": cfg, "state": dec.state_dict(), "d_latent": d_latent}, out_dir / "decoder.pt"
    )
    torch.save(
        {"cfg": cfg, "state": pfn.state_dict(), "d_latent": d_latent, "n_side": n_side},
        out_dir / "pfn.pt",
    )
    pd.DataFrame(log).to_csv(out_dir / "train_log.csv", index=False)
    return {
        "decoder": dec,
        "pfn": pfn,
        "cfg": cfg,
        "tokens": tokens,
        "train_end_idx": train_end_idx,
    }


# --------------------------------------------------------------------------- #
# Day-ahead forecasts and scoring
# --------------------------------------------------------------------------- #


def ifs_daily(sites: list[str], dates: pd.DatetimeIndex) -> np.ndarray:
    """The 00z IFS ENS run of the day before, aggregated to the daily weather variables of the
    JEPA tokens (member mean), for each site and target day; NaN where no run is held."""
    files = sorted((INTERIM_DIR / "forecast_ifs_ens").glob("*.parquet"))
    k = len(jepa.WEATHER_VARS)
    import pyarrow.parquet as pq

    wanted = ["site_code", "run_time", "valid_time", "member", "t2m", "ssrd", "u10", "v10", "tp"]
    frames = []
    for p in files:
        present = set(pq.read_schema(p).names)
        f = pd.read_parquet(p, columns=[c for c in wanted if c in present])
        for c in wanted:  # the 0.4-degree era lacks radiation
            if c not in f.columns:
                f[c] = np.nan
        frames.append(f[f["site_code"].isin(sites)])
    f = pd.concat(frames, ignore_index=True)
    f["t2m"] -= 273.15
    f["wind"] = np.hypot(f["u10"], f["v10"])
    g = (
        f.groupby(["site_code", "run_time", "valid_time"], observed=True)[
            ["t2m", "ssrd", "wind", "tp"]
        ]
        .mean()
        .reset_index()
    )
    g["local"] = g["valid_time"].dt.tz_convert(LOCAL_TZ)
    g["date"] = g["local"].dt.normalize().dt.tz_localize(None)
    g["run_date"] = g["run_time"].dt.tz_convert("UTC").dt.normalize().dt.tz_localize(None)
    g = g[(g["date"] - g["run_date"]).dt.days == 1]  # the day after the 00z run
    g = g[g["run_time"].dt.hour == 0]
    agg = g.groupby(["site_code", "date"]).agg(
        t_mean=("t2m", "mean"),
        t_max=("t2m", "max"),
        t_min=("t2m", "min"),
        ssrd_max=("ssrd", "max"),
        ssrd_min=("ssrd", "min"),
        wind_mean=("wind", "mean"),
        tp_max=("tp", "max"),
        tp_min=("tp", "min"),
        n=("t2m", "size"),
    )
    agg = agg[agg["n"] >= 4]  # the ENS steps are six-hourly: four to five per local day
    agg["solar_kwh_m2"] = (agg["ssrd_max"] - agg["ssrd_min"]) / 3.6e6
    agg["precip_mm"] = (agg["tp_max"] - agg["tp_min"]) * 1000.0
    wx = np.full((len(sites), len(dates), k), np.nan, dtype=np.float32)
    si = pd.Index(sites).get_indexer(agg.index.get_level_values(0))
    di = pd.Index(dates).get_indexer(agg.index.get_level_values(1))
    ok = (si >= 0) & (di >= 0)
    wx[si[ok], di[ok]] = agg.loc[ok, list(jepa.WEATHER_VARS)].to_numpy(dtype=np.float32)
    return wx


def poc_site_days() -> pd.DataFrame:
    f = pd.read_parquet(POC / "forecasts_lightgbm_ifs.parquet", columns=["site_code", "target"])
    f["date"] = (
        pd.to_datetime(f["target"], utc=True)
        .dt.tz_convert(LOCAL_TZ)
        .dt.normalize()
        .dt.tz_localize(None)
    )
    return f[["site_code", "date"]].drop_duplicates()


def forecast(
    models: dict, tokens: dict, device: str, variants=("era5", "ifs"), batch: int = 256
) -> pd.DataFrame:
    """Day-ahead profiles on the PoC site-days: latent PFN with realised (ERA5) or forecast (IFS)
    weather for the target day, and the oracle decode of the true embedding."""
    import torch

    dec, pfn = models["decoder"].eval(), models["pfn"].eval()
    sites, dates = tokens["sites"], tokens["dates"]
    targets = poc_site_days()
    targets = targets[(targets["date"] >= TEST_START) & (targets["date"] <= TEST_END)]
    si = pd.Index(sites).get_indexer(targets["site_code"])
    di = pd.Index(dates).get_indexer(targets["date"])
    keep = (si >= 0) & (di >= 0)
    targets, si, di = targets[keep].reset_index(drop=True), si[keep], di[keep]
    wx_ifs = ifs_daily(sites, dates) if "ifs" in variants else None
    if wx_ifs is not None:
        wx_ifs = (wx_ifs - tokens["wx_mean"]) / tokens["wx_std"]
    rows = []
    tasks = []
    for s, d in zip(si, di, strict=True):
        hist = np.where(tokens["has"][s, :d])[0]
        if (
            len(hist) < CONTEXT_DAYS
            or not np.isfinite(tokens["scale"][s, d])
            or not tokens["valid"][s, d]
        ):
            continue
        tasks.append((int(s), int(d), hist[-CONTEXT_DAYS:]))
    logger.info("latent PFN: %d site-days to forecast", len(tasks))
    for i in range(0, len(tasks), batch):
        chunk = tasks[i : i + batch]
        ce = torch.tensor(np.stack([tokens["e"][s, c] for s, _, c in chunk]), device=device)
        cs = torch.tensor(np.stack([_side(tokens, s, c, d) for s, d, c in chunk]), device=device)
        q_era5 = torch.tensor(
            np.stack([_side(tokens, s, np.array([d]), d) for s, d, _ in chunk]), device=device
        )
        with torch.no_grad():
            preds = {"latent PFN + ERA5": dec(pfn(ce, cs, q_era5)[:, 0]).cpu().numpy()}
            if wx_ifs is not None:
                q_ifs = np.stack(
                    [
                        np.concatenate([wx_ifs[s, d], tokens["cal"][d], [0.0]])[None]
                        for s, d, _ in chunk
                    ]
                ).astype(np.float32)
                ok_ifs = np.isfinite(q_ifs).all(axis=(1, 2))
                q_ifs = torch.tensor(np.where(np.isfinite(q_ifs), q_ifs, 0.0), device=device)
                p_ifs = dec(pfn(ce, cs, q_ifs)[:, 0]).cpu().numpy()
                p_ifs[~ok_ifs] = np.nan
                preds["latent PFN + IFS ENS"] = p_ifs
            preds["oracle: decoded true embedding"] = (
                dec(torch.tensor(np.stack([tokens["e"][s, d] for s, d, _ in chunk]), device=device))
                .cpu()
                .numpy()
            )
        for b, (s, d, _) in enumerate(chunk):
            scale = tokens["scale"][s, d]
            for variant, arr in preds.items():
                if not np.isfinite(arr[b]).all():
                    continue
                prof = np.clip(arr[b], 0, None) * scale
                rows.append((sites[s], dates[d], variant, *prof))
    cols = ["site_code", "date", "variant", *[f"slot{j}" for j in range(jepa.SLOTS)]]
    return pd.DataFrame(rows, columns=cols)


def score(fc: pd.DataFrame, tokens: dict) -> pd.DataFrame:
    """nMAE per site and day class for the latent forecasts and, on the same site-days, the
    existing models' forecasts (``analysis/forecast_heat_days``)."""
    from ..analysis import forecast_heat_days as fh

    w = pd.read_parquet(
        PROJECT_ROOT / "code" / "data" / "processed" / "elec" / "site" / "wide_raw.parquet"
    )
    local = pd.to_datetime(w.index, utc=True).tz_convert(LOCAL_TZ)
    actual = w.copy()
    actual["date"] = local.normalize().tz_localize(None)
    actual["slot"] = (local.hour * 2 + local.minute // 30).to_numpy()
    actual = actual.melt(id_vars=["date", "slot"], var_name="site_code", value_name="y").dropna(
        subset=["y"]
    )
    actual = actual.drop_duplicates(
        ["site_code", "date", "slot"]
    )  # the 25-hour day repeats two slots
    long = fc.melt(id_vars=["site_code", "date", "variant"], var_name="slot", value_name="point")
    long["slot"] = long["slot"].str[4:].astype(int)
    m = long.merge(
        actual[["site_code", "date", "slot", "y"]], on=["site_code", "date", "slot"], how="inner"
    )
    m = m.merge(fh.day_classes(), on=["site_code", "date"], how="left")
    m["day_class"] = m["day_class"].fillna(fh.CLASS_ORDER[4])
    m["ae"] = (m["y"] - m["point"]).abs()
    site_mean = actual.groupby("site_code")["y"].mean()
    per = (
        m.groupby(["variant", "day_class", "site_code"])
        .agg(n=("y", "size"), mae=("ae", "mean"))
        .reset_index()
    )
    per["nmae"] = per["mae"] / per["site_code"].map(site_mean)
    pooled = (
        per[per["n"] >= 48]
        .groupby(["variant", "day_class"])
        .agg(
            sites=("site_code", "nunique"),
            site_days=("n", lambda s: float(s.sum() / 48)),
            nmae=("nmae", "mean"),
        )
        .reset_index()
    )
    # the existing models on the same site-days
    keys = fc[["site_code", "date"]].drop_duplicates()
    ref = fh.load_forecasts(
        ("seasonal_naive", "lightgbm_ifs", "lightgbm_era5", "chronos2_ifs", "timesfm3_ifs")
    ).merge(fh.actuals(), on=["target", "site_code"], how="inner")
    ref["date"] = ref["target"].dt.tz_convert(LOCAL_TZ).dt.normalize().dt.tz_localize(None)
    ref = ref.merge(keys, on=["site_code", "date"], how="inner").merge(
        fh.day_classes(), on=["site_code", "date"], how="left"
    )
    ref["day_class"] = ref["day_class"].fillna(fh.CLASS_ORDER[4])
    ref["ae"] = (ref["y"] - ref["point"]).abs()
    rper = (
        ref.groupby(["model", "day_class", "site_code"])
        .agg(n=("y", "size"), mae=("ae", "mean"))
        .reset_index()
    )
    rper["nmae"] = rper["mae"] / rper["site_code"].map(site_mean)
    rpooled = (
        rper[rper["n"] >= 48]
        .groupby(["model", "day_class"])
        .agg(
            sites=("site_code", "nunique"),
            site_days=("n", lambda s: float(s.sum() / 48)),
            nmae=("nmae", "mean"),
        )
        .reset_index()
        .rename(columns={"model": "variant"})
    )
    return pd.concat([pooled, rpooled], ignore_index=True)


def load_models(device: str, out_dir: Path = OUT_DIR) -> dict:
    """The trained decoder and latent PFN with the latent tokens they were trained on."""
    import torch

    tokens = load_latent()
    ck_d = torch.load(out_dir / "decoder.pt", map_location=device, weights_only=False)
    ck_p = torch.load(out_dir / "pfn.pt", map_location=device, weights_only=False)
    dec = build_decoder(ck_d["d_latent"], ck_d["cfg"]["decoder_hidden"]).to(device)
    dec.load_state_dict(ck_d["state"])
    pfn = build_pfn(ck_p["cfg"], ck_p["d_latent"], ck_p["n_side"]).to(device)
    pfn.load_state_dict(ck_p["state"])
    return {"decoder": dec, "pfn": pfn, "cfg": ck_p["cfg"], "tokens": tokens}


def run(cfg: dict | None = None, device: str | None = None, retrain: bool = True) -> dict:
    import torch

    device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    if not retrain and (OUT_DIR / "pfn.pt").exists():
        res = load_models(device)
    else:
        res = train(cfg, device)
    fc = forecast(res, res["tokens"], device)
    fc.to_parquet(OUT_DIR / "forecasts.parquet", index=False)
    sc = score(fc, res["tokens"])
    sc.to_csv(OUT_DIR / "scores_by_day_class.csv", index=False)
    overall = fc.groupby("variant").size().to_dict()
    hot = sc[sc["day_class"] == "28 °C and above"].set_index("variant")["nmae"].round(4).to_dict()
    allc = (
        sc.groupby("variant")
        .apply(lambda d: float(np.average(d["nmae"], weights=d["site_days"])), include_groups=False)
        .round(4)
        .to_dict()
    )
    summary = {"forecast_rows": overall, "nmae_weighted_all_classes": allc, "nmae_28C_plus": hot}
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    logger.info("latent PFN: %s", summary)
    return {"scores": sc, "forecasts": fc, "summary": summary}
