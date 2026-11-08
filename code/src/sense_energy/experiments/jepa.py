"""A joint-embedding predictive architecture (JEPA) on the half-hourly demand panel.

Self-supervised site representations. A window of ``W`` consecutive local days of one site's
demand is tokenised day by day (48 half-hours scaled by the site's trailing 28-day mean, plus
weekday and season). A *context encoder* sees the window with ``M`` target days removed; a
*target encoder*, an exponential moving average of the context encoder, sees the whole window;
a *predictor* is asked for the target days' embeddings from the context embeddings and the
target days' calendar. The loss is the distance between predicted and target embeddings, so
nothing is reconstructed: the model is pushed to represent what a day is (shape, level against
the site's own recent past, weekday, season, state of the plant) rather than its noise. A
variance term keeps the embedding from collapsing, which matters with 134 sites.

Outputs (``code/reports/jepa/``): ``site_embeddings.parquet`` (one row per site, the mean day
embedding over the whole history and over June-September), ``day_embeddings.parquet`` (site,
date, embedding), ``surprise.parquet`` (site, date: latent prediction error of the day when it
is the one held out of its window, a learned anomaly score), ``jepa.pt``, ``train_log.csv``.
"""

from __future__ import annotations

import copy
import json
import math
import time

import numpy as np
import pandas as pd

from ..config import LOCAL_TZ, PROCESSED_DIR, PROJECT_ROOT
from ..logging_utils import get_logger

logger = get_logger(__name__)

OUT_DIR = PROJECT_ROOT / "code" / "reports" / "jepa"
SLOTS = 48
WINDOW = 28  # days per window
N_TARGET = 7  # target days per window: one block of four and three singles
SCALE_DAYS = 28  # trailing mean that scales a site's day tokens
MIN_SLOTS = 44  # a day needs this many of its 48 half-hours
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
    "seed": 0,
}


# --------------------------------------------------------------------------- #
# Data: day tokens per site
# --------------------------------------------------------------------------- #


def day_tokens(w: pd.DataFrame | None = None) -> dict:
    """Day matrices from the half-hourly table: ``x`` (sites x days x 48) scaled by the site's
    trailing 28-day mean, ``valid`` (sites x days), ``dates``, ``sites``, ``dow``, ``doy``."""
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
    # first value per (day, slot): the 25-hour day carries slots 2-3 twice
    seen = np.zeros((D, SLOTS), dtype=bool)
    for t in range(len(w)):
        d, s = di[t], slot[t]
        if not seen[d, s]:
            x[:, d, s] = arr[t]
            seen[d, s] = True
    present = np.isfinite(x).sum(axis=2)  # S x D
    valid = present >= MIN_SLOTS
    # fill the few missing slots of a valid day by linear interpolation along the day
    for s in range(S):
        for d in np.where(valid[s] & (present[s] < SLOTS))[0]:
            v = x[s, d]
            ok = np.isfinite(v)
            x[s, d] = np.interp(np.arange(SLOTS), np.where(ok)[0], v[ok])
    daily_mean = np.where(valid, np.nanmean(np.where(valid[:, :, None], x, np.nan), axis=2), np.nan)
    scale = (
        pd.DataFrame(daily_mean.T).rolling(SCALE_DAYS, min_periods=7).mean().shift(1).to_numpy().T
    )
    scale = np.where(np.isfinite(scale) & (scale > 0), scale, np.nan)
    valid &= np.isfinite(scale)
    x = np.clip(x / scale[:, :, None], 0.0, 6.0).astype(np.float32)
    x = np.where(valid[:, :, None], x, 0.0)
    return {
        "x": x,
        "valid": valid,
        "dates": dates,
        "sites": sites,
        "dow": dates.dayofweek.to_numpy(),
        "doy": dates.dayofyear.to_numpy(),
        "daily_mean": daily_mean,
        "scale": scale,
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


def build(cfg: dict):
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
            self.embed = nn.Linear(SLOTS + 9, d)
            self.pos = nn.Parameter(torch.zeros(WINDOW, d))
            self.body = block(cfg["enc_layers"])
            self.norm = nn.LayerNorm(d)

        def forward(self, x, cal, pos_idx):
            # x: B x L x 48, cal: B x L x 9, pos_idx: B x L (position in the window)
            z = self.embed(torch.cat([x, cal], dim=-1)) + self.pos[pos_idx]
            return self.norm(self.body(z))

    class Predictor(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Linear(d, d)
            self.mask_token = nn.Parameter(torch.zeros(d))
            self.cal = nn.Linear(9, d)
            self.pos = nn.Parameter(torch.zeros(WINDOW, d))
            self.body = block(cfg["pred_layers"])
            self.out = nn.Linear(d, d)

        def forward(self, ctx, ctx_pos, tgt_cal, tgt_pos):
            # ctx: B x Lc x d, tgt_cal: B x Lt x 9; returns B x Lt x d
            c = self.inp(ctx) + self.pos[ctx_pos]
            t = self.mask_token + self.cal(tgt_cal) + self.pos[tgt_pos]
            z = self.body(torch.cat([c, t], dim=1))
            return self.out(z[:, c.shape[1] :])

    class JEPA(nn.Module):
        def __init__(self):
            super().__init__()
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
# Training
# --------------------------------------------------------------------------- #


def _batch(tokens, cal, pairs, idx, rng, device):
    import torch

    B = len(idx)
    x = np.zeros((B, WINDOW, SLOTS), dtype=np.float32)
    c = np.zeros((B, WINDOW, 9), dtype=np.float32)
    masks = np.zeros((B, WINDOW), dtype=bool)
    for b, k in enumerate(idx):
        s, d0 = pairs[k]
        x[b] = tokens["x"][s, d0 : d0 + WINDOW]
        c[b] = cal[d0 : d0 + WINDOW]
        masks[b] = sample_mask(rng)
    pos = np.broadcast_to(np.arange(WINDOW), (B, WINDOW))
    t = lambda a, dt=torch.float32: torch.tensor(a, dtype=dt, device=device)  # noqa: E731
    return t(x), t(c), torch.tensor(masks, device=device), t(np.ascontiguousarray(pos), torch.long)


def _split(x, c, pos, mask):
    """Context (unmasked) and target (masked) parts, each as padded B x L' tensors; the number
    of target days is fixed, the number of context days too (WINDOW - N_TARGET)."""
    import torch

    B = x.shape[0]
    ctx_idx = torch.stack([torch.where(~mask[b])[0] for b in range(B)])
    tgt_idx = torch.stack([torch.where(mask[b])[0] for b in range(B)])
    g = lambda a, i: torch.gather(a, 1, i.unsqueeze(-1).expand(-1, -1, a.shape[-1]))  # noqa: E731
    return (
        g(x, ctx_idx),
        g(c, ctx_idx),
        torch.gather(pos, 1, ctx_idx),
        g(c, tgt_idx),
        torch.gather(pos, 1, tgt_idx),
        tgt_idx,
    )


def train(cfg: dict | None = None, device: str | None = None, tokens: dict | None = None) -> dict:
    import torch

    cfg = {**DEFAULT, **(cfg or {})}
    device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    tokens = tokens or day_tokens()
    cal = calendar_features(tokens["dow"], tokens["doy"])
    pairs = windows(tokens)
    logger.info(
        "JEPA: %d sites, %d days, %d training windows of %d days; device %s",
        len(tokens["sites"]),
        len(tokens["dates"]),
        len(pairs),
        WINDOW,
        device,
    )
    model = build(cfg).to(device)
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
    for step in range(steps):
        idx = rng.integers(0, len(pairs), cfg["batch_size"])
        x, c, mask, pos = _batch(tokens, cal, pairs, idx, rng, device)
        xc, cc, pc, ct, pt, tgt_idx = _split(x, c, pos, mask)
        model.train()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            with torch.no_grad():
                full = model.target(x, c, pos)  # B x W x d
                target = torch.gather(full, 1, tgt_idx.unsqueeze(-1).expand(-1, -1, full.shape[-1]))
                target = torch.nn.functional.layer_norm(target.float(), target.shape[-1:])
            ctx = model.encoder(xc, cc, pc)
            pred = model.predictor(ctx, pc, ct, pt)
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
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"cfg": cfg, "state": model.state_dict()}, OUT_DIR / "jepa.pt")
    pd.DataFrame(log).to_csv(OUT_DIR / "train_log.csv", index=False)
    logger.info(
        "JEPA trained: %.2fM parameters, %d steps in %.0f s", n_par / 1e6, steps, time.time() - t0
    )
    return {"model": model, "cfg": cfg, "tokens": tokens, "log": pd.DataFrame(log)}


def load(device: str | None = None):
    import torch

    device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    ck = torch.load(OUT_DIR / "jepa.pt", map_location=device, weights_only=False)
    model = build(ck["cfg"]).to(device)
    model.load_state_dict(ck["state"])
    model.eval()
    return model, ck["cfg"]


# --------------------------------------------------------------------------- #
# Embeddings and surprise
# --------------------------------------------------------------------------- #


def embed_days(model, tokens: dict, device: str, batch: int = 512) -> pd.DataFrame:
    """The target encoder's embedding of every valid day, from non-overlapping full windows
    (a trailing window where the last stretch is shorter than ``WINDOW``)."""
    import torch

    cal = calendar_features(tokens["dow"], tokens["doy"])
    valid, x_all = tokens["valid"], tokens["x"]
    S, D = valid.shape
    rows, starts = [], []
    for s in range(S):
        d = 0
        while d + WINDOW <= D:
            if valid[s, d : d + WINDOW].all():
                starts.append((s, d))
                d += WINDOW
            else:
                d += 1
        if D - WINDOW >= 0 and valid[s, D - WINDOW :].all():
            starts.append((s, D - WINDOW))  # covers the tail; duplicates averaged below
    model.eval()
    pos = torch.arange(WINDOW, device=device)[None].expand(batch, -1)
    for i in range(0, len(starts), batch):
        chunk = starts[i : i + batch]
        x = torch.tensor(np.stack([x_all[s, d : d + WINDOW] for s, d in chunk]), device=device)
        c = torch.tensor(np.stack([cal[d : d + WINDOW] for s, d in chunk]), device=device)
        with (
            torch.no_grad(),
            torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")),
        ):
            z = model.target(x, c, pos[: len(chunk)]).float().cpu().numpy()
        for b, (s, d) in enumerate(chunk):
            for k in range(WINDOW):
                rows.append((tokens["sites"][s], tokens["dates"][d + k], *z[b, k]))
    cols = ["site_code", "date", *[f"e{j}" for j in range(z.shape[-1])]]
    df = pd.DataFrame(rows, columns=cols)
    return df.groupby(["site_code", "date"], as_index=False).mean()


def surprise(model, tokens: dict, device: str, batch: int = 512) -> pd.DataFrame:
    """Latent prediction error of each day when it alone is held out of its window."""
    import torch

    cal = calendar_features(tokens["dow"], tokens["doy"])
    valid, x_all = tokens["valid"], tokens["x"]
    S, D = valid.shape
    tasks = []
    for s in range(S):
        d = 0
        while d + WINDOW <= D:
            if valid[s, d : d + WINDOW].all():
                tasks.extend((s, d, k) for k in range(WINDOW))
                d += WINDOW
            else:
                d += 1
    model.eval()
    rows = []
    for i in range(0, len(tasks), batch):
        chunk = tasks[i : i + batch]
        B = len(chunk)
        x = torch.tensor(np.stack([x_all[s, d : d + WINDOW] for s, d, _ in chunk]), device=device)
        c = torch.tensor(np.stack([cal[d : d + WINDOW] for s, d, _ in chunk]), device=device)
        pos = torch.arange(WINDOW, device=device)[None].expand(B, -1)
        mask = torch.zeros(B, WINDOW, dtype=torch.bool, device=device)
        mask[torch.arange(B), torch.tensor([k for _, _, k in chunk], device=device)] = True
        xc, cc, pc, ct, pt, tgt_idx = _split(x, c, pos, mask)
        with (
            torch.no_grad(),
            torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")),
        ):
            full = model.target(x, c, pos)
            target = torch.gather(full, 1, tgt_idx.unsqueeze(-1).expand(-1, -1, full.shape[-1]))
            target = torch.nn.functional.layer_norm(target.float(), target.shape[-1:])
            pred = model.predictor(model.encoder(xc, cc, pc), pc, ct, pt).float()
            err = ((pred - target) ** 2).mean(dim=(1, 2)).cpu().numpy()
        for b, (s, d, k) in enumerate(chunk):
            rows.append((tokens["sites"][s], tokens["dates"][d + k], float(err[b])))
    return pd.DataFrame(rows, columns=["site_code", "date", "surprise"])


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
    model, tokens = res["model"], res["tokens"]
    days = embed_days(model, tokens, device)
    days.to_parquet(OUT_DIR / "day_embeddings.parquet", index=False)
    sites = site_embeddings(days)
    sites.to_parquet(OUT_DIR / "site_embeddings.parquet")
    sp = surprise(model, tokens, device)
    sp.to_parquet(OUT_DIR / "surprise.parquet", index=False)
    summary = {
        "sites": int(len(sites)),
        "days_embedded": int(len(days)),
        "final_loss": float(res["log"]["loss"].iloc[-1]),
        "final_embed_std": float(res["log"]["embed_std"].iloc[-1]),
        "cfg": res["cfg"],
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    logger.info("JEPA outputs: %s", summary)
    return {"sites": sites, "days": days, "surprise": sp, "summary": summary}
