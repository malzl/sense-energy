"""Transformer neural processes (TNPs) as day-ahead forecasters, meta-trained on the panel.

A TNP maps a *context set* of observed points (x, y) and a set of target inputs to a
predictive distribution at the targets (Nguyen & Grover 2022; Ashman et al. 2025). Here
one task is one site at one forecast origin:

* context: the last ``context_periods`` half-hours up to the origin, y = demand divided
  by the site's mean over the last 28 days (short gaps imputed, the rest filled);
* targets: the next ``H`` half-hours, covering the target day;
* x: time to the origin in weeks, plus either the local calendar (plain models) or the
  full known-ahead covariate set of the foundation-model runs (``+ ERA5`` / ``+ IFS ENS``).

The model is meta-trained on tasks drawn at random origins in the training window
(every site, every half-hour) with a heteroscedastic Gaussian likelihood, early-stopped
on the negative log-likelihood of 16:30 origins in the last ``valid_last_days`` of that
window, then applied at the test origins. Covariate models are trained with reanalysis
weather and evaluated twice: with reanalysis (``+ ERA5``) and with the IFS ENS forecast
(``+ IFS ENS``) over the horizon.

Three architectures from the vendored gridded-tnp code (``code/third_party/gridded_tnp``):

* ``tnp``   - TNP: context self-attention, targets cross-attend to the context;
* ``pttnp`` - pseudo-token TNP: an induced-set transformer with learned latents;
* ``gtnp``  - gridded TNP along the time axis: context points are encoded onto a grid of
  pseudo-tokens (one per ``grid_spacing_periods``), processed with shifted-window (SWIN)
  attention and decoded at each target from its nearest grid tokens. The grid only uses
  the time coordinate; calendar and covariates enter through the token embeddings.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from typing import Any

import numpy as np
import pandas as pd

from ..config import PROJECT_ROOT
from ..logging_utils import get_logger
from .features import _holiday_set
from .poc import Origin, Panel, forecast_frame, make_origins, outputs_dir

logger = get_logger(__name__)

ARCHS = ("tnp", "pttnp", "gtnp")
WEEK = 336


def parse_name(name: str) -> tuple[str, str]:
    """'pttnp_ifs' -> ('pttnp', 'ifs'); 'gtnp' -> ('gtnp', 'none')."""
    for arch in sorted(ARCHS, key=len, reverse=True):
        if name == arch:
            return arch, "none"
        if name.startswith(arch + "_"):
            weather = name[len(arch) + 1 :]
            if weather not in ("era5", "ifs"):
                raise KeyError(name)
            return arch, weather
    raise KeyError(name)


def calendar_features(panel: Panel) -> np.ndarray:
    """(T, 6): half-hour of day and day of week as sine/cosine, weekend, bank holiday."""
    tod, dow = panel.tod.astype(float), panel.dow.astype(float)
    hol = _holiday_set(range(2020, 2028))
    return np.column_stack(
        [
            np.sin(2 * np.pi * tod / 48),
            np.cos(2 * np.pi * tod / 48),
            np.sin(2 * np.pi * dow / 7),
            np.cos(2 * np.pi * dow / 7),
            (dow >= 5).astype(float),
            np.isin(panel.local_date, list(hol)).astype(float),
        ]
    ).astype("float32")


def _filled(y: pd.DataFrame) -> np.ndarray:
    f = y.ffill().bfill()
    return f.fillna(f.median()).fillna(0.0).to_numpy(dtype="float32")


# --------------------------------------------------------------------------- #
# Model construction
# --------------------------------------------------------------------------- #


def _time_gridded_encoder_cls():
    import torch
    from tnp.models.gridded_tnp import GriddedTNPEncoder
    from tnp.utils.helpers import preprocess_observations

    class TimeGriddedTNPEncoder(GriddedTNPEncoder):
        """Gridded TNP encoder whose grid spans the time coordinate (x[..., 0]) only; the
        remaining inputs (calendar, covariates) are part of the token embeddings."""

        def forward(self, xc, yc, xt, time_grid=None):
            yc, yt = preprocess_observations(xt, yc)
            zc = self.xy_encoder(torch.cat((self.x_encoder(xc), yc), dim=-1))
            zt = self.xy_encoder(torch.cat((self.x_encoder(xt), yt), dim=-1))
            # the pseudo-token scatter (index_put) needs one dtype: run it in float32
            with torch.autocast(device_type=zc.device.type, enabled=False):
                xc_grid, zc_grid = self.grid_encoder(xc[..., :1].float(), zc.float())
            return self.transformer_encoder(xc_grid, zc_grid, xt[..., :1], zt)

    return TimeGriddedTNPEncoder


def build_model(arch: str, dx: int, spec: dict[str, Any], context: int, horizon: int):
    from tnp.likelihoods.gaussian import HeteroscedasticNormalLikelihood
    from tnp.models.tnp import TNP, TNPDecoder, TNPEncoder
    from tnp.networks.attention_layers import (
        MultiHeadCrossAttentionLayer,
        MultiHeadSelfAttentionLayer,
    )
    from tnp.networks.mlp import MLP
    from tnp.networks.transformer import TNPTransformerEncoder

    e = int(spec["embed_dim"])
    layers = int(spec["num_layers"])
    att = {
        "embed_dim": e,
        "num_heads": int(spec["num_heads"]),
        "head_dim": int(spec["head_dim"]),
        "feedforward_dim": e,
        "norm_first": True,
    }
    xy_encoder = MLP(in_dim=dx + 2, out_dim=e, num_layers=2, width=e)  # x, y, target flag
    decoder = TNPDecoder(z_decoder=MLP(in_dim=e, out_dim=2, num_layers=2, width=e))
    likelihood = HeteroscedasticNormalLikelihood(min_noise=float(spec.get("min_noise", 1e-3)))
    if arch == "tnp":
        enc = TNPEncoder(
            transformer_encoder=TNPTransformerEncoder(
                num_layers=layers,
                mhca_layer=MultiHeadCrossAttentionLayer(**att),
                mhsa_layer=MultiHeadSelfAttentionLayer(**att),
            ),
            xy_encoder=xy_encoder,
        )
        return TNP(enc, decoder, likelihood)
    if arch == "pttnp":
        from tnp.models.pttnp import PTTNP, PTTNPEncoder
        from tnp.networks.transformer import ISTEncoder

        ist = ISTEncoder(
            num_latents=int(spec["num_latents"]),
            mhca_ctoq_layer=MultiHeadCrossAttentionLayer(**att),
            mhca_qtoc_layer=MultiHeadCrossAttentionLayer(**att),
            mhca_qtot_layer=MultiHeadCrossAttentionLayer(**att),
            num_layers=layers,
        )
        return PTTNP(
            PTTNPEncoder(transformer_encoder=ist, xy_encoder=xy_encoder), decoder, likelihood
        )
    if arch == "gtnp":
        from tnp.models.gridded_tnp import GriddedTNP
        from tnp.networks.grid_decoders import MHCAGridDecoder
        from tnp.networks.grid_encoders import PseudoTokenGridEncoder
        from tnp.networks.swin_attention import SWINAttentionLayer
        from tnp.networks.transformer import GriddedTransformerEncoder

        spacing = int(spec["grid_spacing_periods"])
        window = int(spec["window_tokens"])
        n_grid = int(math.ceil((context - 1 + horizon) / spacing)) + 1
        # whole number of windows: the SWIN padding path of the upstream code fails otherwise,
        # so the grid is extended past the horizon at the same spacing instead
        n_grid = int(math.ceil(n_grid / window)) * window
        lo = -(context - 1) / WEEK
        hi = lo + (n_grid - 1) * spacing / WEEK
        grid_encoder = PseudoTokenGridEncoder(
            embed_dim=e,
            mhca_layer=MultiHeadCrossAttentionLayer(**att),
            grid_range=((lo, hi),),
            points_per_dim=(n_grid,),
        )
        swin = SWINAttentionLayer(
            mhsa_layer=MultiHeadSelfAttentionLayer(**att),
            window_sizes=(window,),
            shift_sizes=(window // 2,),
        )
        transformer = GriddedTransformerEncoder(
            num_layers=layers,
            grid_decoder=MHCAGridDecoder(
                mhca_layer=MultiHeadCrossAttentionLayer(**att), top_k_ctot=int(spec["top_k_ctot"])
            ),
            mhsa_layer=swin,
        )
        enc = _time_gridded_encoder_cls()(
            transformer_encoder=transformer, grid_encoder=grid_encoder, xy_encoder=xy_encoder
        )
        return GriddedTNP(enc, decoder, likelihood)
    raise KeyError(arch)


# --------------------------------------------------------------------------- #
# Data on the GPU
# --------------------------------------------------------------------------- #


class TaskData:
    """Panel arrays on the device and a task assembler (context, targets, scale)."""

    def __init__(self, panel: Panel, config: dict[str, Any], covariates: bool, device: str):
        import torch

        spec = config["tnp"]
        self.panel = panel
        self.device = device
        self.L = int(spec["context_periods"])
        self.H = int(
            max(
                o.target_pos[-1] - o.origin_pos
                for o in make_origins(panel, config, config["test_start"], config["test_start"], 1)
            )
        )
        sites = panel.sites
        self.y_ctx = torch.tensor(_filled(panel.y_imp[sites]), device=device)  # T x S
        self.y_obs = torch.tensor(panel.y_raw[sites].to_numpy(dtype="float32"), device=device)
        self.covariates = covariates
        if covariates:
            from .fm_covariates import CovariateWindows

            cw = CovariateWindows(config, panel, "era5")
            self.cov_names = list(cw.names)
            T, S = len(panel.index), len(sites)
            arrs = []
            for n in self.cov_names:
                a = cw.shared[n][:, None].repeat(S, axis=1) if n in cw.shared else cw.per_site[n]
                arrs.append(np.asarray(a, dtype="float32").reshape(T, S))
            cov = np.stack(arrs, axis=-1)  # T x S x F
            tr = self._train_mask(config)
            self.mu = np.nanmean(cov[tr], axis=(0, 1))
            self.sd = np.nanstd(cov[tr], axis=(0, 1))
            self.sd = np.where(self.sd > 1e-6, self.sd, 1.0)
            cov = np.nan_to_num((cov - self.mu) / self.sd).astype("float32")
            self.x_static = torch.tensor(cov, device=device)  # T x S x F
        else:
            self.cov_names = [
                "hod_sin",
                "hod_cos",
                "dow_sin",
                "dow_cos",
                "is_weekend",
                "is_holiday",
            ]
            cal = calendar_features(panel)
            self.x_static = torch.tensor(cal, device=device)[:, None, :].expand(-1, len(sites), -1)
        self.dx = 1 + self.x_static.shape[-1]
        offs = np.arange(-self.L + 1, self.H + 1)
        self.offsets = torch.tensor(offs, device=device)
        self.tau = torch.tensor(offs / WEEK, dtype=torch.float32, device=device)

    def _train_mask(self, config) -> np.ndarray:
        idx = self.panel.index
        return (idx >= pd.Timestamp(config["train_start"], tz="UTC")) & (
            idx <= pd.Timestamp(config["train_end"], tz="UTC")
        )

    def assemble(self, pos, site, x_future=None):
        """pos, site: (B,) long tensors (origin position, site column). Returns xc, yc, xt,
        yt (observed, NaN where missing) and the per-task scale."""
        import torch

        idx = pos[:, None] + self.offsets[None, :]  # B x (L+H)
        y = self.y_ctx[idx, site[:, None]]  # B x (L+H)
        ctx_recent = y[:, self.L - 48 * 28 : self.L]
        scale = ctx_recent.mean(dim=1).clamp_min(1e-3)
        x = self.x_static[idx, site[:, None]]  # B x (L+H) x F
        if x_future is not None:  # replace the horizon's covariates (forecast weather)
            x = x.clone()
            x[:, self.L :, :] = x_future
        tau = self.tau[None, :, None].expand(len(pos), -1, -1)
        x = torch.cat([tau, x], dim=-1)
        xc, xt = x[:, : self.L], x[:, self.L :]
        yc = (y[:, : self.L] / scale[:, None])[..., None]
        yt = self.y_obs[idx[:, self.L :], site[:, None]] / scale[:, None]
        return xc, yc, xt, yt, scale


def _gaussian_nll(loc, scale, y):
    import torch

    ok = torch.isfinite(y)
    z = (torch.nan_to_num(y) - loc) / scale
    nll = torch.log(scale) + 0.5 * z**2 + 0.5 * math.log(2 * math.pi)
    return (nll * ok).sum() / ok.sum().clamp_min(1)


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #


def _fingerprint(arch: str, covariates: bool, config: dict[str, Any]) -> str:
    keys = {
        k: config[k] for k in ("energy", "level", "train_start", "train_end", "issue_time_local")
    }
    blob = json.dumps(
        {"arch": arch, "cov": covariates, "tnp": config["tnp"], **keys}, sort_keys=True, default=str
    )
    return hashlib.sha1(blob.encode()).hexdigest()[:10]


def train(panel: Panel, config: dict[str, Any], arch: str, covariates: bool, device: str):
    import torch
    from check_shapes import ShapeCheckingState, set_enable_check_shapes

    set_enable_check_shapes(ShapeCheckingState.DISABLED)
    spec = config["tnp"]
    torch.manual_seed(int(spec.get("seed", 0)))
    rng = np.random.default_rng(int(spec.get("seed", 0)))
    data = TaskData(panel, config, covariates, device)
    model = build_model(arch, data.dx, spec, data.L, data.H).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    idx = panel.index
    t0_pos = int(idx.searchsorted(pd.Timestamp(config["train_start"], tz="UTC"))) + data.L
    valid_start = pd.Timestamp(config["train_end"], tz="UTC") - pd.Timedelta(
        days=int(spec["valid_last_days"])
    )
    t1_pos = int(idx.searchsorted(valid_start)) - data.H - 1
    val_origins = make_origins(panel, config, f"{valid_start:%Y-%m-%d}", config["train_end"], 1)
    S = len(panel.sites)
    val_pos = torch.tensor(np.repeat([o.origin_pos for o in val_origins], S), device=device)
    val_site = torch.tensor(np.tile(np.arange(S), len(val_origins)), device=device)
    B = int(spec["batch_size"])
    steps, warm = int(spec["max_steps"]), int(spec["warmup_steps"])
    opt = torch.optim.AdamW(
        model.parameters(), lr=float(spec["lr"]), weight_decay=float(spec.get("weight_decay", 1e-4))
    )
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)),
    )
    logger.info(
        "%s (%s): %.2fM parameters, dx %d, context %d, horizon %d, %d train positions x %d sites, %d validation tasks, device %s",
        arch,
        "covariates" if covariates else "calendar",
        n_params / 1e6,
        data.dx,
        data.L,
        data.H,
        t1_pos - t0_pos,
        S,
        len(val_pos),
        device,
    )

    # the gridded encoder runs one attention per (task, pseudo-token): keep task x tokens
    # below CUDA's 65,535 grid limit
    eval_bs = int(spec.get("eval_batch_size", 32))

    def validate():
        model.eval()
        tot, n = 0.0, 0
        with (
            torch.no_grad(),
            torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")),
        ):
            for s in range(0, len(val_pos), eval_bs):
                xc, yc, xt, yt, _ = data.assemble(
                    val_pos[s : s + eval_bs], val_site[s : s + eval_bs]
                )
                dist = model(xc, yc, xt)
                nll = _gaussian_nll(dist.loc.float()[..., 0], dist.scale.float()[..., 0], yt)
                k = int(torch.isfinite(yt).sum())
                tot += float(nll) * k
                n += k
        model.train()
        return tot / max(n, 1)

    best, best_state, bad, history = float("inf"), None, 0, []
    t0 = time.time()
    model.train()
    for step in range(1, steps + 1):
        pos = torch.tensor(rng.integers(t0_pos, t1_pos, B), device=device)
        site = torch.tensor(rng.integers(0, S, B), device=device)
        xc, yc, xt, yt, _ = data.assemble(pos, site)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
            dist = model(xc, yc, xt)
        loss = _gaussian_nll(dist.loc.float()[..., 0], dist.scale.float()[..., 0], yt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(spec.get("grad_clip", 0.5)))
        opt.step()
        sched.step()
        if step % int(spec["eval_every"]) == 0 or step == steps:
            v = validate()
            history.append(
                {
                    "step": step,
                    "train_nll": float(loss),
                    "valid_nll": v,
                    "seconds": time.time() - t0,
                }
            )
            logger.info(
                "%s step %d: train NLL %.3f, validation NLL %.4f (%.0f s)",
                arch,
                step,
                float(loss),
                v,
                time.time() - t0,
            )
            if v < best - 1e-4:
                best, bad = v, 0
                best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
            else:
                bad += 1
                if bad >= int(spec["patience"]):
                    logger.info(
                        "%s: early stop at step %d (best validation NLL %.4f)", arch, step, best
                    )
                    break
    model.load_state_dict(best_state)
    model.eval()
    return model, data, pd.DataFrame(history)


def trained(panel: Panel, config: dict[str, Any], arch: str, covariates: bool, device: str):
    """Train, or load the checkpoint written by an earlier run with the same settings."""
    import torch

    ckdir = outputs_dir(config) / "tnp_models"
    ckdir.mkdir(parents=True, exist_ok=True)
    fp = _fingerprint(arch, covariates, config)
    path = ckdir / f"{arch}_{'cov' if covariates else 'cal'}_{fp}.pt"
    if path.exists():
        data = TaskData(panel, config, covariates, device)
        model = build_model(arch, data.dx, config["tnp"], data.L, data.H).to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        logger.info("%s: loaded %s", arch, path.name)
        return model, data
    model, data, hist = train(panel, config, arch, covariates, device)
    torch.save(model.state_dict(), path)
    hist.to_csv(path.with_suffix(".history.csv"), index=False)
    return model, data


# --------------------------------------------------------------------------- #
# Forecasting the test origins
# --------------------------------------------------------------------------- #


def run_tnp(panel: Panel, origins: list[Origin], config: dict[str, Any], name: str) -> pd.DataFrame:
    import torch
    from check_shapes import ShapeCheckingState, set_enable_check_shapes
    from scipy.stats import norm

    from ..compute import select_device

    set_enable_check_shapes(ShapeCheckingState.DISABLED)
    arch, weather = parse_name(name)
    device = select_device()
    model, data = trained(panel, config, arch, weather != "none", device)
    quantiles = [float(q) for q in config["quantiles"]]
    z = torch.tensor(norm.ppf(quantiles), dtype=torch.float32, device=device)
    S = len(panel.sites)
    cw = None
    if weather == "ifs":
        from .fm_covariates import CovariateWindows

        cw = CovariateWindows(config, panel, "ifs")
        mu = torch.tensor(data.mu, device=device)
        sd = torch.tensor(data.sd, device=device)
    eval_bs = int(config["tnp"].get("eval_batch_size", 32))
    frames = []
    t0 = time.time()
    for o in origins:
        if o.target_pos[-1] - o.origin_pos > data.H:
            raise ValueError("test horizon longer than the trained horizon")
        x_future_all = None
        if cw is not None:
            fut = []
            for j, s in enumerate(panel.sites):
                _, f = cw.window(o, j, s, data.L, data.H)
                fut.append(np.stack([f[n] for n in data.cov_names], axis=-1))
            x_future_all = (
                torch.tensor(np.stack(fut), dtype=torch.float32, device=device) - mu
            ) / sd
            x_future_all = torch.nan_to_num(x_future_all)
        locs, scs = [], []
        for c0 in range(0, S, eval_bs):
            site = torch.arange(c0, min(S, c0 + eval_bs), device=device)
            pos = torch.full((len(site),), o.origin_pos, device=device)
            xf = None if x_future_all is None else x_future_all[c0 : c0 + len(site)]
            with (
                torch.no_grad(),
                torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")),
            ):
                xc, yc, xt, _, scale = data.assemble(pos, site, xf)
                dist = model(xc, yc, xt)
            locs.append(dist.loc.float()[..., 0] * scale[:, None])
            scs.append(dist.scale.float()[..., 0] * scale[:, None])
        loc, sc = torch.cat(locs), torch.cat(scs)
        q = (loc[..., None] + sc[..., None] * z).clamp_min(0).cpu().numpy()  # S x H x Q
        steps = o.target_pos - o.origin_pos - 1
        for j, s in enumerate(panel.sites):
            frames.append(
                forecast_frame(
                    name,
                    s,
                    o,
                    panel,
                    quantiles,
                    q[j, steps],
                    loc[j, steps].clamp_min(0).cpu().numpy(),
                )
            )
    out = pd.concat(frames, ignore_index=True)
    logger.info("%s: %d origins forecast in %.0f s", name, len(origins), time.time() - t0)
    return out


__all__ = ["ARCHS", "build_model", "parse_name", "run_tnp", "train", "PROJECT_ROOT"]
