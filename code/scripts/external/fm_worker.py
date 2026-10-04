"""Run TiRex-2 or T0-beta on a batch of contexts. Plain numpy + torch + the model package,
so it runs inside the model's own environment (Python 3.11 venvs outside this project).

Input npz: contexts (N, L) float32, horizon, quantiles (Q,), optional covariates (N, F, L + H).
Output npz: quantiles (N, H, K) and levels (K,).
"""

from __future__ import annotations

import argparse
import time

import numpy as np


def run_tirex2(ctx: np.ndarray, horizon: int, cov, batch_size: int, bf16: bool = False):
    import torch
    from tirex2 import TimeseriesType, load_model

    model = load_model("NX-AI/TiRex-2", device="cuda" if torch.cuda.is_available() else "cpu")
    series = [
        TimeseriesType(
            target=torch.tensor(ctx[i], dtype=torch.float32).unsqueeze(0),
            past_covariates=None,
            future_covariates=None if cov is None else torch.tensor(cov[i], dtype=torch.float32),
        )
        for i in range(len(ctx))
    ]
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16 and torch.cuda.is_available()):
        raw = model.forecast(
            series, prediction_length=horizon, output_type="numpy", batch_size=batch_size
        )
    levels = None
    for holder in (model, getattr(model, "model", None)):
        for name in ("quantiles", "quantile_levels"):
            v = getattr(holder, name, None) if holder is not None else None
            if v is not None:
                levels = np.asarray(
                    [
                        float(x)
                        for x in np.asarray(v.detach().cpu() if hasattr(v, "detach") else v).ravel()
                    ]
                )
                break
        if levels is not None:
            break
    out = []
    for arr in raw:
        a = np.asarray(arr, dtype="float32").squeeze()
        if a.ndim != 2:
            raise ValueError(f"unexpected TiRex-2 output shape {a.shape}")
        if a.shape[0] != horizon and a.shape[1] == horizon:
            a = a.T
        out.append(a)
    Q = np.stack(out)
    if levels is None or len(levels) != Q.shape[2]:
        levels = (
            np.linspace(0.1, 0.9, Q.shape[2])
            if Q.shape[2] == 9
            else np.linspace(0.05, 0.95, Q.shape[2])
        )
    return Q, levels


def run_t0beta(
    ctx: np.ndarray, horizon: int, cov, batch_size: int, levels: np.ndarray, bf16: bool = False
):
    import torch
    from t0 import T0Forecaster

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = T0Forecaster.from_pretrained("theforecastingcompany/t0-beta").eval().to(device)
    out = []
    with (
        torch.no_grad(),
        torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16 and torch.cuda.is_available()),
    ):
        for s in range(0, len(ctx), batch_size):
            x = torch.tensor(ctx[s : s + batch_size], dtype=torch.float32, device=device)
            kw = {}
            if cov is not None:
                kw["future_covariates"] = torch.tensor(
                    cov[s : s + batch_size], dtype=torch.float32, device=device
                )
            pred = model.predict(
                x, horizon=horizon, quantile_levels=[float(v) for v in levels], **kw
            )
            out.append(pred.quantiles.float().cpu().numpy())
    return np.concatenate(out), np.asarray(levels, dtype=float)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["tirex2", "t0beta"])
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--bf16", action="store_true")
    a = ap.parse_args()
    data = np.load(a.input)
    ctx = data["contexts"]
    horizon = int(data["horizon"])
    cov = data["covariates"] if "covariates" in data.files else None
    t0 = time.time()
    if a.model == "tirex2":
        Q, levels = run_tirex2(ctx, horizon, cov, a.batch_size, a.bf16)
    else:
        Q, levels = run_t0beta(ctx, horizon, cov, a.batch_size, data["quantiles"], a.bf16)
    np.savez(a.output, quantiles=Q.astype("float32"), levels=levels)
    print(
        f"{a.model}: {len(ctx)} contexts, horizon {horizon}, {Q.shape[2]} levels, "
        f"covariates {'yes' if cov is not None else 'no'}, {time.time() - t0:.0f} s"
    )


if __name__ == "__main__":
    main()
