"""Zero-shot models that live in their own Python environments (TiRex-2, T0-beta).

Both packages need Python 3.11 and their own torch builds, so they are not
installed in this project's environment. The runner writes the contexts (and
covariates) of every (site, origin) to an ``npz`` file, calls the model's
interpreter on ``code/scripts/external/fm_worker.py`` with the GPU chosen
through ``CUDA_VISIBLE_DEVICES``, and reads the quantiles back. The worker is
plain numpy + torch + the model package, so it runs in those environments.

The returned quantile levels are interpolated onto the project's reporting
grid when they differ (T0-beta is asked for the grid directly; TiRex-2 returns
its own nine levels).
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..compute import select_device
from ..config import PROJECT_ROOT
from ..logging_utils import get_logger
from .fm_covariates import covariate_inputs
from .poc import Origin, Panel
from .zero_shot import _assemble, contexts_for, model_name

logger = get_logger(__name__)

WORKER = PROJECT_ROOT / "code" / "scripts" / "external" / "fm_worker.py"


def to_grid(q: np.ndarray, levels: np.ndarray, grid: list[float]) -> np.ndarray:
    """(N, H, K) quantiles at ``levels`` -> (N, H, Q) at ``grid`` by linear interpolation in tau."""
    levels = np.asarray(levels, dtype=float)
    if len(levels) == len(grid) and np.allclose(levels, grid):
        return q
    q = np.sort(q, axis=2)
    flat = q.reshape(-1, q.shape[2])
    out = np.stack([np.interp(grid, levels, row) for row in flat])
    return out.reshape(q.shape[0], q.shape[1], len(grid))


def run_external(
    panel: Panel, origins: list[Origin], config: dict[str, Any], base: str, weather: str = "none"
) -> pd.DataFrame:
    spec = config["external_models"][base]
    name = model_name(base, weather)
    quantiles = [float(q) for q in config["quantiles"]]
    keys, X, horizon = contexts_for(panel, origins, config)
    payload = {
        "contexts": X.astype("float32"),
        "horizon": np.int64(horizon),
        "quantiles": np.asarray(quantiles),
    }
    if weather != "none":
        windows, names = covariate_inputs(
            panel, keys, config, weather, X.shape[1], horizon, standardised=True
        )
        payload["covariates"] = np.stack(
            [np.stack([np.concatenate([p[n], f[n]]) for n in names]) for p, f in windows]
        ).astype("float32")  # (N, F, L + H)
    device = select_device()
    env = dict(os.environ)
    if device.startswith("cuda:"):  # map the visible index back to the physical GPU id
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        idx = int(device.split(":")[1])
        env["CUDA_VISIBLE_DEVICES"] = visible.split(",")[idx] if visible else str(idx)
    env.setdefault("HF_HUB_OFFLINE", "1")  # the checkpoints are in the cache; never re-download
    # the interpreter's own bin dir (ninja for JIT kernels) and any configured extras on PATH
    prepend = [str(Path(spec["python"]).parent), *spec.get("path_prepend", [])]
    env["PATH"] = os.pathsep.join([*prepend, env.get("PATH", "")])
    env.update({k: str(v) for k, v in spec.get("env", {}).items()})
    with tempfile.TemporaryDirectory(prefix=f"{base}_") as tmp:
        inp, outp = Path(tmp) / "input.npz", Path(tmp) / "output.npz"
        np.savez(inp, **payload)
        cmd = [
            str(spec["python"]),
            str(WORKER),
            "--model",
            base,
            "--input",
            str(inp),
            "--output",
            str(outp),
            "--batch-size",
            str(int(spec.get("batch_size", 64))),
            *(["--bf16"] if spec.get("bf16", False) else []),
        ]
        logger.info("%s: %d contexts, horizon %d, worker on %s", name, len(keys), horizon, device)
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
        tail = (proc.stderr or "")[-3000:]
        if proc.returncode != 0:
            raise RuntimeError(f"{base} worker failed (exit {proc.returncode}):\n{tail}")
        for line in (proc.stdout or "").strip().splitlines()[-5:]:
            logger.info("%s worker: %s", base, line)
        res = np.load(outp)
        Q, levels = res["quantiles"], res["levels"]
    Q = to_grid(Q.astype("float64"), levels, quantiles).astype("float32")
    P = Q[:, :, quantiles.index(0.5)]
    assert Q.shape[:2] == (len(keys), horizon), Q.shape
    return _assemble(name, keys, Q, P, panel, quantiles)


if __name__ == "__main__":  # pragma: no cover
    print(sys.argv)
