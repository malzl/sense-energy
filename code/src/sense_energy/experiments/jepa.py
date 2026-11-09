"""A joint-embedding predictive architecture (JEPA) on the half-hourly demand panel.

Self-supervised site representations. A window of ``W`` consecutive local days of one site's
demand is tokenised day by day (48 half-hours scaled by the site's trailing 28-day mean, plus
weekday and season, and, in the weather-conditioned version, the day's weather). A *context
encoder* sees the window with ``M`` target days removed; a *target encoder*, an exponential
moving average of the context encoder, sees the whole window; a *predictor* is asked for the
target days' embeddings from the context embeddings and the target days' calendar (and
weather). The loss is the distance between predicted and target embeddings, so nothing is
reconstructed: the model is pushed to represent what a day is rather than its noise. A variance
term keeps the embedding from collapsing, which matters with 134 sites.

**Weather conditioning** (``cfg["weather"] = True``): the day tokens and the predictor's target
tokens carry the day's temperature (mean, maximum, minimum), solar radiation, wind and rain. The
model then has to represent how a site's profile *responds* to weather, not only what it looks
like, and the predictor becomes a counterfactual machine: the same context, a target day 5 degC
warmer, a different predicted embedding. ``response_embeddings`` averages that difference per
site: a learned response signature to probe against the regression slopes.

Outputs (``code/reports/jepa/`` or ``jepa_weather/``): ``site_embeddings.parquet`` (mean day
embedding over the history and over June-September; with weather, the heat and sunshine
response vectors ``h*`` and ``s*`` and their norms), ``day_embeddings.parquet``,
``surprise.parquet`` (latent prediction error of each day held out of its window: a learned
anomaly score), ``jepa.pt``, ``train_log.csv``, ``summary.json``.
"""

from __future__ import annotations

import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import INTERIM_DIR, LOCAL_TZ, PROCESSED_DIR, PROJECT_ROOT
from ..logging_utils import get_logger

logger = get_logger(__name__)

OUT_DIR = PROJECT_ROOT / "code" / "reports" / "jepa"
OUT_DIR_WEATHER = PROJECT_ROOT / "code" / "reports" / "jepa_weather"
SLOTS = 48
WINDOW = 28  # days per window
N_TARGET = 7  # target days per window: one block of four and three singles
SCALE_DAYS = 28  # trailing mean that scales a site's day tokens
MIN_SLOTS = 44  # a day needs this many of its 48 half-hours
WEATHER_VARS = ("t_mean", "t_max", "t_min", "solar_kwh_m2", "wind_mean", "precip_mm")
TEMPERATURE_VARS = ("t_mean", "t_max", "t_min")
#: Counterfactuals for the response embeddings: degrees added to the day's temperatures, and
#: kWh/m2 added to its solar radiation.
HEAT_DELTA_C = 5.0
SUN_DELTA_KWH = 2.0
DEFAULT = {
    "d_model": 96,
    "n_heads": 4,
    "enc_layers": 4,
    "pred_layers": 2,
    "ff_mult": 2,
    "dropout": 0.1,
    "batch_size": 256,
    "steps": 4000,
    "lr": 5e-4,
    "weight_decay": 0.05,
    "warmup": 200,
    "ema_start": 0.996,
    "ema_end": 1.0,
    "var_weight": 1.0,
    "weather": False,
    "seed": 0,
}


# --------------------------------------------------------------------------- #
# Data: day tokens per site
# --------------------------------------------------------------------------- #


def day_tokens(w: pd.DataFrame | None = None, weather: bool = False) -> dict:
    """Day matrices from the half-hourly table: ``x`` (sites x days x 48) scaled by the site's
    trailing 28-day mean, ``valid`` (sites x days), ``dates``, ``sites``, ``dow``, ``doy``, and
    with ``weather`` the standardised daily weather ``wx`` (sites x days x k) with its scale."""
    if w is None:
        w = pd.read_parquet(PROCESSED_DIR / "elec" / "site" / "wide_raw.parquet")
    local = pd.to_datetime(w.index, utc=True).tz_convert(LOCAL_TZ)
    day = local.normalize().tz_localize(None)
    slot = local.hour * 2 + local.minute // 30
    dates = pd.DatetimeIndex(sorted(day.unique()))
    sites = list(w.columns)
    S, D = len(sites), len(dates)
    x = np.full((S, D, SLOTS), np.nan, dtype=np.float32)
    di = pd.Index(dates).get_indexer(day)
    arr = w.to_numpy(dtype=np.float32)
    seen = np.zeros(
        (D, SLOTS), dtype=bool
    )  # first value per (day, slot): the 25-hour day repeats two
    for t in range(len(w)):
        d, s = di[t], slot[t]
        if not seen[d, s]:
            x[:, d, s] = arr[t]
            seen[d, s] = True
    present = np.isfinite(x).sum(axis=2)
    valid = present >= MIN_SLOTS
    for s in range(S):
        for d in np.where(valid[s] & (present[s] < SLOTS))[0]:
            v = x[s, d]
            ok = np.isfinite(v)
            x[s, d] = np.interp(np.arange(SLOTS), np.where(ok)[0], v[ok])
    with np.errstate(all="ignore"):
        daily_mean = np.where(
            valid, np.nanmean(np.where(valid[:, :, None], x, np.nan), axis=2), np.nan
        )
    scale = (
        pd.DataFrame(daily_mean.T).rolling(SCALE_DAYS, min_periods=7).mean().shift(1).to_numpy().T
    )
    scale = np.where(np.isfinite(scale) & (scale > 0), scale, np.nan)
    valid &= np.isfinite(scale)
    x = np.clip(x / scale[:, :, None], 0.0, 6.0).astype(np.float32)
    x = np.where(valid[:, :, None], x, 0.0)
    out = {
        "x": x,
        "valid": valid,
        "dates": dates,
        "sites": sites,
        "dow": dates.dayofweek.to_numpy(),
        "doy": dates.dayofyear.to_numpy(),
        "daily_mean": daily_mean,
        "scale": scale,
        "wx": np.zeros((S, D, 0), dtype=np.float32),
        "wx_mean": np.zeros(0, dtype=np.float32),
        "wx_std": np.ones(0, dtype=np.float32),
    }
    if weather:
        out.update(weather_tokens(sites, dates))
        out["valid"] &= out["wx_valid"]
    return out


def weather_tokens(
    sites: list[str], dates: pd.DatetimeIndex, table: pd.DataFrame | None = None
) -> dict:
    """Daily weather per site and day, standardised over the panel; ``wx_valid`` marks days
    with every variable present."""
    if table is None:
        table = pd.read_parquet(
            INTERIM_DIR / "weather_daily.parquet", columns=["site_code", "date", *WEATHER_VARS]
        )
    k = len(WEATHER_VARS)
    wx = np.full((len(sites), len(dates), k), np.nan, dtype=np.float32)
    t = table[table["site_code"].isin(sites)]
    si = pd.Index(sites).get_indexer(t["site_code"])
    di = pd.Index(dates).get_indexer(t["date"])
    ok = (si >= 0) & (di >= 0)
    wx[si[ok], di[ok]] = t.loc[ok, list(WEATHER_VARS)].to_numpy(dtype=np.float32)
    mean = np.nanmean(wx.reshape(-1, k), axis=0)
    std = np.nanstd(wx.reshape(-1, k), axis=0)
    std = np.where(std > 0, std, 1.0)
    z = (wx - mean) / std
    valid = np.isfinite(z).all(axis=2)
    return {
        "wx": np.where(valid[:, :, None], z, 0.0).astype(np.float32),
        "wx_valid": valid,
        "wx_mean": mean.astype(np.float32),
        "wx_std": std.astype(np.float32),
    }


def calendar_features(dow: np.ndarray, doy: np.ndarray) -> np.ndarray:
    """Weekday one-hot and season (sin, cos): 9 features."""
    f = np.zeros((len(dow), 9), dtype=np.float32)
    f[np.arange(len(dow)), dow] = 1.0
    f[:, 7] = np.sin(2 * np.pi * doy / 365.25)
    f[:, 8] = np.cos(2 * np.pi * doy / 365.25)
    return f


def sample_mask(
    rng: np.random.Generator, window: int = WINDOW, n_target: int = N_TARGET
) -> np.ndarray:
    """Boolean mask of target days: one block of four consecutive days plus singles."""
    m = np.zeros(window, dtype=bool)
    start = rng.integers(0, window - 3)
    m[start : start + 4] = True
    while m.sum() < n_target:
        m[rng.integers(0, window)] = True
    return m


def windows(tokens: dict, window: int = WINDOW) -> list[tuple[int, int]]:
    """Every (site, start day) whose ``window`` days are all valid."""
    valid = tokens["valid"]
    S, D = valid.shape
    out = []
    k = np.ones(window, dtype=int)
    for s in range(S):
        full = np.convolve(valid[s].astype(int), k, mode="valid") == window
        out.extend((s, int(d)) for d in np.where(full)[0])
    return out


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #


def build(cfg: dict, n_weather: int = 0):
    import torch
    from torch import nn

    d, h = cfg["d_model"], cfg["n_heads"]

    def block(layers: int):
        layer = nn.TransformerEncoderLayer(
            d,
            h,
            dim_feedforward=cfg["ff_mult"] * d,
            dropout=cfg["dropout"],
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        return nn.TransformerEncoder(layer, layers)

    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Linear(SLOTS + 9 + n_weather, d)
            self.pos = nn.Parameter(torch.zeros(WINDOW, d))
            self.body = block(cfg["enc_layers"])
            self.norm = nn.LayerNorm(d)

        def forward(self, x, side, pos_idx):
            # x: B x L x 48, side: B x L x (9 + k), pos_idx: B x L
            z = self.embed(torch.cat([x, side], dim=-1)) + self.pos[pos_idx]
            return self.norm(self.body(z))

    class Predictor(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Linear(d, d)
            self.mask_token = nn.Parameter(torch.zeros(d))
            self.side = nn.Linear(9 + n_weather, d)
            self.pos = nn.Parameter(torch.zeros(WINDOW, d))
            self.body = block(cfg["pred_layers"])
            self.out = nn.Linear(d, d)

        def forward(self, ctx, ctx_pos, tgt_side, tgt_pos):
            c = self.inp(ctx) + self.pos[ctx_pos]
            t = self.mask_token + self.side(tgt_side) + self.pos[tgt_pos]
            z = self.body(torch.cat([c, t], dim=1))
            return self.out(z[:, c.shape[1] :])

    class JEPA(nn.Module):
        def __init__(self):
            super().__init__()
            self.n_weather = n_weather
            self.encoder = Encoder()
            self.predictor = Predictor()
            self.target = copy.deepcopy(self.encoder)
            for p in self.target.parameters():
                p.requires_grad_(False)

        @torch.no_grad()
        def ema(self, m: float):
            for p, q in zip(self.encoder.parameters(), self.target.parameters(), strict=True):
                q.mul_(m).add_(p.detach(), alpha=1 - m)

    return JEPA()


def variance_term(z, gamma: float = 1.0, eps: float = 1e-4):
    """VICReg's hinge on the per-dimension standard deviation of the batch embeddings."""
    import torch

    std = torch.sqrt(z.reshape(-1, z.shape[-1]).var(dim=0) + eps)
    return torch.relu(gamma - std).mean()


# --------------------------------------------------------------------------- #
# Device-resident data and batches
# --------------------------------------------------------------------------- #


class Store:
    """The token arrays on the device, and vectorised window gathering."""

    def __init__(self, tokens: dict, device: str):
        import torch

        self.device = device
        self.x = torch.tensor(tokens["x"], device=device)  # S x D x 48
        self.cal = torch.tensor(calendar_features(tokens["dow"], tokens["doy"]), device=device)
        self.wx = torch.tensor(tokens["wx"], device=device)  # S x D x k
        self.k = int(self.wx.shape[-1])
        pairs = windows(tokens)
        self.ps = torch.tensor([s for s, _ in pairs], device=device)
        self.pd0 = torch.tensor([d for _, d in pairs], device=device)
        self.offsets = torch.arange(WINDOW, device=device)
        self.n = len(pairs)

    def gather(self, s, d0):
        """x (B x W x 48), side (B x W x (9+k)), pos (B x W) for sites ``s`` and starts ``d0``."""
        import torch

        days = d0[:, None] + self.offsets[None]
        x = self.x[s[:, None], days]
        side = torch.cat([self.cal[days], self.wx[s[:, None], days]], dim=-1)
        pos = self.offsets[None].expand(len(s), -1)
        return x, side, pos

    def masks(self, B: int, gen) -> torch.Tensor:  # noqa: F821
        """One block of four consecutive days plus singles, vectorised."""
        import torch

        start = torch.randint(0, WINDOW - 3, (B,), device=self.device, generator=gen)
        block = (self.offsets[None] >= start[:, None]) & (self.offsets[None] < start[:, None] + 4)
        key = torch.rand(B, WINDOW, device=self.device, generator=gen)
        key[block] = 2.0
        singles = torch.argsort(key, dim=1)[:, : N_TARGET - 4]
        mask = block.clone()
        mask.scatter_(1, singles, True)
        return mask


def split(x, side, pos, mask):
    """Context and target parts of a window (fixed sizes), by a stable sort on the mask."""
    import torch

    order = torch.argsort(mask.to(torch.int8), dim=1, stable=True)
    n_ctx = WINDOW - int(mask[0].sum())
    ctx_idx, tgt_idx = order[:, :n_ctx], order[:, n_ctx:]
    g = lambda a, i: torch.gather(a, 1, i.unsqueeze(-1).expand(-1, -1, a.shape[-1]))  # noqa: E731
    return (
        g(x, ctx_idx),
        g(side, ctx_idx),
        torch.gather(pos, 1, ctx_idx),
        g(side, tgt_idx),
        torch.gather(pos, 1, tgt_idx),
        tgt_idx,
    )


def _targets(model, x, side, pos, tgt_idx):
    import torch

    full = model.target(x, side, pos)
    t = torch.gather(full, 1, tgt_idx.unsqueeze(-1).expand(-1, -1, full.shape[-1]))
    return torch.nn.functional.layer_norm(t.float(), t.shape[-1:])


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #


def train(
    cfg: dict | None = None,
    device: str | None = None,
    tokens: dict | None = None,
    out_dir: Path | None = None,
) -> dict:
    import torch

    cfg = {**DEFAULT, **(cfg or {})}
    device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    out_dir = out_dir or (OUT_DIR_WEATHER if cfg["weather"] else OUT_DIR)
    torch.manual_seed(cfg["seed"])
    gen = torch.Generator(device=device)
    gen.manual_seed(cfg["seed"])
    tokens = tokens if tokens is not None else day_tokens(weather=cfg["weather"])
    store = Store(tokens, device)
    logger.info(
        "JEPA%s: %d sites, %d days, %d training windows of %d days; device %s",
        " (weather)" if cfg["weather"] else "",
        len(tokens["sites"]),
        len(tokens["dates"]),
        store.n,
        WINDOW,
        device,
    )
    model = build(cfg, store.k).to(device)
    n_par = sum(p.numel() for p in model.encoder.parameters()) + sum(
        p.numel() for p in model.predictor.parameters()
    )
    opt = torch.optim.AdamW(
        list(model.encoder.parameters()) + list(model.predictor.parameters()),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )
    steps = int(cfg["steps"])
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: (
            min(1.0, (s + 1) / cfg["warmup"])
            * 0.5
            * (1 + math.cos(math.pi * min(s, steps) / steps))
        ),
    )
    log, t0 = [], time.time()
    use_amp = device.startswith("cuda")
    B = int(cfg["batch_size"])
    for step in range(steps):
        idx = torch.randint(0, store.n, (B,), device=device, generator=gen)
        x, side, pos = store.gather(store.ps[idx], store.pd0[idx])
        mask = store.masks(B, gen)
        xc, sc, pc, st, pt, tgt_idx = split(x, side, pos, mask)
        model.train()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            with torch.no_grad():
                target = _targets(model, x, side, pos, tgt_idx)
            ctx = model.encoder(xc, sc, pc)
            pred = model.predictor(ctx, pc, st, pt)
            loss_pred = torch.nn.functional.smooth_l1_loss(pred.float(), target)
            loss_var = variance_term(ctx.float())
            loss = loss_pred + cfg["var_weight"] * loss_var
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        m = cfg["ema_start"] + (cfg["ema_end"] - cfg["ema_start"]) * step / max(1, steps - 1)
        model.ema(m)
        if step % 100 == 0 or step == steps - 1:
            with torch.no_grad():
                std = ctx.float().reshape(-1, ctx.shape[-1]).std(dim=0).mean().item()
            log.append(
                {
                    "step": step,
                    "loss": loss.item(),
                    "loss_pred": loss_pred.item(),
                    "loss_var": loss_var.item(),
                    "embed_std": std,
                    "lr": sched.get_last_lr()[0],
                    "seconds": time.time() - t0,
                }
            )
            if step % 500 == 0 or step == steps - 1:
                logger.info(
                    "step %d loss %.4f (pred %.4f, var %.4f) embed std %.3f",
                    step,
                    loss.item(),
                    loss_pred.item(),
                    loss_var.item(),
                    std,
                )
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "cfg": cfg,
            "state": model.state_dict(),
            "n_weather": store.k,
            "wx_mean": tokens["wx_mean"],
            "wx_std": tokens["wx_std"],
        },
        out_dir / "jepa.pt",
    )
    pd.DataFrame(log).to_csv(out_dir / "train_log.csv", index=False)
    logger.info(
        "JEPA trained: %.2fM parameters, %d steps in %.0f s", n_par / 1e6, steps, time.time() - t0
    )
    return {
        "model": model,
        "cfg": cfg,
        "tokens": tokens,
        "log": pd.DataFrame(log),
        "store": store,
        "out_dir": out_dir,
    }


def load(out_dir: Path = OUT_DIR, device: str | None = None):
    import torch

    device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    ck = torch.load(out_dir / "jepa.pt", map_location=device, weights_only=False)
    model = build(ck["cfg"], ck.get("n_weather", 0)).to(device)
    model.load_state_dict(ck["state"])
    model.eval()
    return model, ck["cfg"]


# --------------------------------------------------------------------------- #
# Embeddings, surprise and counterfactual responses
# --------------------------------------------------------------------------- #


def _tiling(valid: np.ndarray) -> list[tuple[int, int]]:
    """Non-overlapping full windows per site, plus a trailing window to cover the tail."""
    S, D = valid.shape
    starts = []
    for s in range(S):
        d = 0
        while d + WINDOW <= D:
            if valid[s, d : d + WINDOW].all():
                starts.append((s, d))
                d += WINDOW
            else:
                d += 1
        if D - WINDOW >= 0 and valid[s, D - WINDOW :].all():
            starts.append((s, D - WINDOW))
    return starts


def _chunks(starts, batch):
    for i in range(0, len(starts), batch):
        yield starts[i : i + batch]


def embed_days(
    model, tokens: dict, device: str, batch: int = 512, store: Store | None = None
) -> pd.DataFrame:
    """The target encoder's embedding of every valid day."""
    import torch

    store = store or Store(tokens, device)
    model.eval()
    rows = []
    for chunk in _chunks(_tiling(tokens["valid"]), batch):
        s = torch.tensor([a for a, _ in chunk], device=device)
        d0 = torch.tensor([b for _, b in chunk], device=device)
        x, side, pos = store.gather(s, d0)
        with (
            torch.no_grad(),
            torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")),
        ):
            z = model.target(x, side, pos).float().cpu().numpy()
        for b, (si, di) in enumerate(chunk):
            for k in range(WINDOW):
                rows.append((tokens["sites"][si], tokens["dates"][di + k], *z[b, k]))
    cols = ["site_code", "date", *[f"e{j}" for j in range(z.shape[-1])]]
    return pd.DataFrame(rows, columns=cols).groupby(["site_code", "date"], as_index=False).mean()


def surprise(
    model, tokens: dict, device: str, batch: int = 512, store: Store | None = None
) -> pd.DataFrame:
    """Latent prediction error of each day when it alone is held out of its window."""
    import torch

    store = store or Store(tokens, device)
    model.eval()
    tasks = [(s, d, k) for s, d in _tiling(tokens["valid"]) for k in range(WINDOW)]
    rows = []
    for chunk in _chunks(tasks, batch):
        B = len(chunk)
        s = torch.tensor([a for a, _, _ in chunk], device=device)
        d0 = torch.tensor([b for _, b, _ in chunk], device=device)
        x, side, pos = store.gather(s, d0)
        mask = torch.zeros(B, WINDOW, dtype=torch.bool, device=device)
        mask[
            torch.arange(B, device=device), torch.tensor([k for _, _, k in chunk], device=device)
        ] = True
        xc, sc, pc, st, pt, tgt_idx = split(x, side, pos, mask)
        with (
            torch.no_grad(),
            torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")),
        ):
            target = _targets(model, x, side, pos, tgt_idx)
            pred = model.predictor(model.encoder(xc, sc, pc), pc, st, pt).float()
            err = ((pred - target) ** 2).mean(dim=(1, 2)).cpu().numpy()
        for b, (si, di, k) in enumerate(chunk):
            rows.append((tokens["sites"][si], tokens["dates"][di + k], float(err[b])))
    return pd.DataFrame(rows, columns=["site_code", "date", "surprise"])


def response_embeddings(
    model, tokens: dict, device: str, batch: int = 512, store: Store | None = None
) -> pd.DataFrame:
    """Weather-conditioned models only. For every full window the last seven days are the
    targets; their embeddings are predicted under the actual weather and under counterfactual
    weather (temperatures + 5 degC; solar radiation + 2 kWh/m2). The mean difference per site is
    the site's learned heat (``h*``) and sunshine (``s*``) response, with their norms."""
    import torch

    store = store or Store(tokens, device)
    if store.k == 0:
        raise ValueError("response embeddings need a weather-conditioned model")
    model.eval()
    std = tokens["wx_std"]
    names = list(WEATHER_VARS)
    d_heat = torch.zeros(store.k, device=device)
    for v in TEMPERATURE_VARS:
        d_heat[names.index(v)] = HEAT_DELTA_C / float(std[names.index(v)])
    d_sun = torch.zeros(store.k, device=device)
    d_sun[names.index("solar_kwh_m2")] = SUN_DELTA_KWH / float(std[names.index("solar_kwh_m2")])
    mask_row = torch.zeros(WINDOW, dtype=torch.bool, device=device)
    mask_row[WINDOW - N_TARGET :] = True
    acc_h, acc_s, counts = {}, {}, {}
    for chunk in _chunks(_tiling(tokens["valid"]), batch):
        B = len(chunk)
        s = torch.tensor([a for a, _ in chunk], device=device)
        d0 = torch.tensor([b for _, b in chunk], device=device)
        x, side, pos = store.gather(s, d0)
        mask = mask_row[None].expand(B, -1)
        xc, sc, pc, st, pt, _ = split(x, side, pos, mask)
        with (
            torch.no_grad(),
            torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")),
        ):
            ctx = model.encoder(xc, sc, pc)
            base = model.predictor(ctx, pc, st, pt).float()
            st_h = st.clone()
            st_h[:, :, 9:] += d_heat
            st_s = st.clone()
            st_s[:, :, 9:] += d_sun
            dh = (model.predictor(ctx, pc, st_h, pt).float() - base).mean(dim=1).cpu().numpy()
            ds = (model.predictor(ctx, pc, st_s, pt).float() - base).mean(dim=1).cpu().numpy()
        for b, (si, _) in enumerate(chunk):
            site = tokens["sites"][si]
            acc_h[site] = acc_h.get(site, 0) + dh[b]
            acc_s[site] = acc_s.get(site, 0) + ds[b]
            counts[site] = counts.get(site, 0) + 1
    rows = []
    for site in acc_h:
        h, sv = acc_h[site] / counts[site], acc_s[site] / counts[site]
        rows.append(
            {
                "site_code": site,
                **{f"h{j}": float(v) for j, v in enumerate(h)},
                **{f"s{j}": float(v) for j, v in enumerate(sv)},
                "h_norm": float(np.linalg.norm(h)),
                "s_norm": float(np.linalg.norm(sv)),
                "windows": counts[site],
            }
        )
    return pd.DataFrame(rows).set_index("site_code")


def site_embeddings(days: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in days.columns if c.startswith("e")]
    allm = days.groupby("site_code")[cols].mean()
    summer = days[days["date"].dt.month.isin([6, 7, 8, 9])].groupby("site_code")[cols].mean()
    summer.columns = [f"s_{c}" for c in cols]
    out = allm.join(summer)
    out["days"] = days.groupby("site_code").size()
    return out


def run(cfg: dict | None = None, device: str | None = None) -> dict:
    import torch

    device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    res = train(cfg, device)
    model, tokens, store, out_dir = res["model"], res["tokens"], res["store"], res["out_dir"]
    days = embed_days(model, tokens, device, store=store)
    days.to_parquet(out_dir / "day_embeddings.parquet", index=False)
    sites = site_embeddings(days)
    if store.k:
        sites = sites.join(response_embeddings(model, tokens, device, store=store))
    sites.to_parquet(out_dir / "site_embeddings.parquet")
    sp = surprise(model, tokens, device, store=store)
    sp.to_parquet(out_dir / "surprise.parquet", index=False)
    summary = {
        "sites": int(len(sites)),
        "days_embedded": int(len(days)),
        "final_loss": float(res["log"]["loss"].iloc[-1]),
        "final_embed_std": float(res["log"]["embed_std"].iloc[-1]),
        "weather": bool(store.k),
        "cfg": res["cfg"],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    logger.info("JEPA outputs: %s", summary)
    return {"sites": sites, "days": days, "surprise": sp, "summary": summary}
