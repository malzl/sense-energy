"""Does a learned site embedding carry the heat response? Probes of the JEPA embeddings.

The question the heat work left: the site's diurnal signature explains 46 % of the true
between-site variance of the heat slope, the estates and certificate descriptors nothing, the
causal forest 24 %. Here every candidate representation of a site is put through the same
harness: predict the site's fixed-effect heat slope (net of sunshine) out of fold, over five site
folds repeated with different seeds, with a ridge probe and with TabPFN (in-context regression,
suited to eighty training sites), and report the out-of-fold R2 and the share of the slopes' true
variance recovered (DerSimonian-Laird, as in ``heat_causal_ml``).

Representations: the JEPA site embedding (``experiments/jepa.py``, all days and June-September
only), the Chronos-2 encoder embedding of the site's demand (mean-pooled patch states over
windows spread through the history: a foundation-model teacher for comparison), the observable
descriptors of ``heat_causal_ml``, the diurnal signature, and combinations. Secondary targets:
the cold slope, the sunshine slope, and classification of building use and the DEC environment.
Clusters: k-means on the JEPA embedding against the k = 4 demand-shape clusters, by adjusted Rand
index and by how much of the heat-slope variance cluster membership explains.

Outputs (``code/reports/jepa/``): ``probe.csv``, ``probe_classification.csv``, ``clusters.csv``,
``chronos_embeddings.parquet``, ``probe_summary.json``.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..config import PROCESSED_DIR
from ..experiments.jepa import OUT_DIR
from ..logging_utils import get_logger
from . import heat_causal_ml as hcm
from . import heat_drivers as hd

logger = get_logger(__name__)

N_FOLDS, N_REPEATS, SEED = 5, 10, 20261108
PCA_DIMS = 16
CHRONOS_WINDOW_HOURS = 14 * 24
CHRONOS_WINDOWS = 12
TARGETS = {"heat": "heat slope net of sunshine", "cold": "cold slope", "sun": "sunshine slope"}
CLASS_TARGETS = {"use": "building use", "dec_main_environment": "DEC environment"}


# --------------------------------------------------------------------------- #
# Representations
# --------------------------------------------------------------------------- #


def jepa_embeddings() -> tuple[pd.DataFrame, pd.DataFrame]:
    e = pd.read_parquet(OUT_DIR / "site_embeddings.parquet")
    allm = e[[c for c in e.columns if c.startswith("e")]]
    summer = e[[c for c in e.columns if c.startswith("s_e")]]
    summer.columns = [c[2:] for c in summer.columns]
    return allm, summer


def chronos_embeddings(device: str | None = None, refresh: bool = False) -> pd.DataFrame:
    """Mean-pooled Chronos-2 encoder states of each site's demand over windows spread through
    the history (cached)."""
    path = OUT_DIR / "chronos_embeddings.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    import torch

    from ..models.foundation import load_chronos2

    device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    pipe = load_chronos2(device)
    w = pd.read_parquet(PROCESSED_DIR / "elec" / "site" / "wide_raw.parquet")
    hourly = w.resample("1h").sum(min_count=2)
    rows = {}
    n = CHRONOS_WINDOW_HOURS
    for site in hourly.columns:
        y = hourly[site].to_numpy(dtype=np.float32)
        ok = np.isfinite(y)
        starts = np.linspace(0, len(y) - n, CHRONOS_WINDOWS).astype(int)
        contexts = [y[s : s + n] for s in starts if ok[s : s + n].mean() > 0.95]
        if not contexts:
            continue
        contexts = [np.where(np.isfinite(c), c, np.nanmean(c)) for c in contexts]
        contexts = [c / max(float(np.mean(c)), 1e-6) for c in contexts]  # level-free, as the JEPA
        emb, _ = pipe.embed([torch.tensor(c) for c in contexts])
        pooled = torch.stack([e[0, 1:-1].float().mean(dim=0) for e in emb]).mean(dim=0)
        rows[site] = pooled.cpu().numpy()
    out = pd.DataFrame.from_dict(rows, orient="index")
    out.columns = [f"c{j}" for j in range(out.shape[1])]
    out.index.name = "site_code"
    out.to_parquet(path)
    logger.info("Chronos-2 embeddings: %d sites x %d", *out.shape)
    return out


def representations(
    desc: pd.DataFrame, features: pd.DataFrame, device: str | None = None
) -> dict[str, pd.DataFrame]:
    jepa_all, jepa_summer = jepa_embeddings()
    obs = hcm.site_covariates(desc, features, signature=False)
    sig = features[hcm.SIGNATURE].copy()
    reps = {
        "observables": obs,
        "diurnal signature": sig,
        "observables + signature": obs.join(sig, how="left"),
        "JEPA": jepa_all,
        "JEPA (June–September)": jepa_summer,
        "JEPA + observables": jepa_all.join(obs, how="inner"),
        "JEPA + signature": jepa_all.join(sig, how="inner"),
    }
    try:
        reps["Chronos-2"] = chronos_embeddings(device)
    except Exception as exc:  # the teacher is optional
        logger.warning("Chronos-2 embeddings unavailable: %s", exc)
    return reps


# --------------------------------------------------------------------------- #
# Probes
# --------------------------------------------------------------------------- #


def _pca(x_train: np.ndarray, x_test: np.ndarray, dims: int):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler().fit(x_train)
    a, b = sc.transform(x_train), sc.transform(x_test)
    k = min(dims, a.shape[1], a.shape[0] - 1)
    if a.shape[1] > k:
        p = PCA(n_components=k, random_state=0).fit(a)
        a, b = p.transform(a), p.transform(b)
    return a, b


def oof_predictions(x: pd.DataFrame, y: pd.Series, model: str, seed: int) -> pd.Series:
    """Out-of-fold predictions over site folds; ridge or TabPFN on PCA-reduced inputs."""
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import KFold

    x = x.reindex(y.index).fillna(0.0)
    pred = pd.Series(np.nan, index=y.index)
    kf = KFold(N_FOLDS, shuffle=True, random_state=seed)
    for tr, te in kf.split(x):
        a, b = _pca(x.iloc[tr].to_numpy(dtype=float), x.iloc[te].to_numpy(dtype=float), PCA_DIMS)
        if model == "ridge":
            m = RidgeCV(alphas=np.logspace(-2, 4, 25)).fit(a, y.iloc[tr])
            pred.iloc[te] = m.predict(b)
        elif model == "tabpfn":
            from tabpfn import TabPFNRegressor

            m = TabPFNRegressor(n_estimators=4, random_state=seed)
            m.fit(a, y.iloc[tr].to_numpy())
            pred.iloc[te] = m.predict(b)
        else:
            raise ValueError(model)
    return pred


def probe(
    reps: dict[str, pd.DataFrame], features: pd.DataFrame, models=("ridge", "tabpfn")
) -> pd.DataFrame:
    rows = []
    for target, label in TARGETS.items():
        y_all = features[target]
        se_all = features[f"{target}_se"]
        for name, x in reps.items():
            idx = x.index.intersection(y_all.dropna().index)
            idx = idx[(se_all.reindex(idx) > 0).to_numpy()]
            y, se = y_all.loc[idx], se_all.loc[idx]
            for model in models:
                preds = []
                for r in range(N_REPEATS if model == "ridge" else max(1, N_REPEATS // 5)):
                    preds.append(oof_predictions(x.loc[idx], y, model, SEED + r))
                p = pd.concat(preds, axis=1).mean(axis=1)
                r2 = 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()
                ag = hcm.agreement(p, y, se)
                rows.append(
                    {
                        "target": target,
                        "target_name": label,
                        "representation": name,
                        "model": model,
                        "sites": int(len(idx)),
                        "dims": int(x.shape[1]),
                        "r2_oof": round(float(r2), 3),
                        "corr": ag["corr"],
                        "share_true_variance": ag["share_true_variance_recovered"],
                    }
                )
                logger.info(
                    "%s | %s | %s: R2 %.3f, share %.3f",
                    label,
                    name,
                    model,
                    r2,
                    ag["share_true_variance_recovered"],
                )
    return pd.DataFrame(rows)


def probe_classification(reps: dict[str, pd.DataFrame], desc: pd.DataFrame) -> pd.DataFrame:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import KFold

    rows = []
    for target, label in CLASS_TARGETS.items():
        y_all = desc[target].dropna()
        y_all = y_all[y_all.map(y_all.value_counts()) >= 5]
        for name, x in reps.items():
            idx = x.index.intersection(y_all.index)
            y = y_all.loc[idx]
            correct = []
            for r in range(N_REPEATS):
                kf = KFold(N_FOLDS, shuffle=True, random_state=SEED + r)
                xx = x.loc[idx].fillna(0.0)
                hit = pd.Series(False, index=idx)
                for tr, te in kf.split(xx):
                    a, b = _pca(
                        xx.iloc[tr].to_numpy(dtype=float),
                        xx.iloc[te].to_numpy(dtype=float),
                        PCA_DIMS,
                    )
                    m = LogisticRegression(C=0.5, max_iter=2000).fit(a, y.iloc[tr])
                    hit.iloc[te] = m.predict(b) == y.iloc[te].to_numpy()
                correct.append(hit.mean())
            rows.append(
                {
                    "target": target,
                    "target_name": label,
                    "representation": name,
                    "sites": int(len(idx)),
                    "classes": int(y.nunique()),
                    "accuracy": round(float(np.mean(correct)), 3),
                    "majority": round(float(y.value_counts(normalize=True).iloc[0]), 3),
                }
            )
    return pd.DataFrame(rows)


def clusters(
    reps: dict[str, pd.DataFrame], features: pd.DataFrame, desc: pd.DataFrame, k: int = 4
) -> pd.DataFrame:
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score
    from sklearn.preprocessing import StandardScaler

    base = desc["cluster"].dropna()
    rows = []
    heat = features["heat"]
    for name in ("JEPA", "JEPA (June–September)", "Chronos-2", "observables"):
        if name not in reps:
            continue
        x = reps[name].dropna()
        z = StandardScaler().fit_transform(x.to_numpy(dtype=float))
        lab = pd.Series(KMeans(k, n_init=20, random_state=SEED).fit_predict(z), index=x.index)
        common = lab.index.intersection(base.index)
        ari = adjusted_rand_score(base.loc[common], lab.loc[common])
        h = heat.reindex(lab.index).dropna()
        g = h.groupby(lab.loc[h.index])
        r2 = 1 - (g.transform("mean") - h).pow(2).sum() / (h - h.mean()).pow(2).sum()
        rows.append(
            {
                "representation": name,
                "k": k,
                "sites": int(len(lab)),
                "ari_vs_shape_clusters": round(float(ari), 3),
                "heat_slope_r2_by_cluster": round(float(r2), 3),
                "cluster_sizes": lab.value_counts().sort_index().tolist(),
            }
        )
    h = heat.reindex(base.index).dropna()
    g = h.groupby(base.loc[h.index])
    rows.append(
        {
            "representation": "demand-shape clusters (k = 4)",
            "k": 4,
            "sites": int(len(base)),
            "ari_vs_shape_clusters": 1.0,
            "heat_slope_r2_by_cluster": round(
                float(1 - (g.transform("mean") - h).pow(2).sum() / (h - h.mean()).pow(2).sum()), 3
            ),
            "cluster_sizes": base.value_counts().sort_index().tolist(),
        }
    )
    return pd.DataFrame(rows)


def run(device: str | None = None) -> dict:
    from threadpoolctl import threadpool_limits

    features = pd.read_csv(hd.OUT_DIR / "site_features.csv").set_index("site_code")
    desc = pd.read_csv(hd.OUT_DIR / "site_descriptors.csv").set_index("site_code")
    reps = representations(desc, features, device)
    with threadpool_limits(8):
        p = probe(reps, features)
        pc = probe_classification(reps, desc)
        cl = clusters(reps, features, desc)
    p.to_csv(OUT_DIR / "probe.csv", index=False)
    pc.to_csv(OUT_DIR / "probe_classification.csv", index=False)
    cl.to_csv(OUT_DIR / "clusters.csv", index=False)
    heat = p[p["target"] == "heat"]
    summary = {
        "heat_share_true_variance": {
            f"{r.representation} | {r.model}": r.share_true_variance for r in heat.itertuples()
        },
        "heat_r2_oof": {f"{r.representation} | {r.model}": r.r2_oof for r in heat.itertuples()},
        "classification_accuracy": {
            f"{r.target} | {r.representation}": (r.accuracy, r.majority) for r in pc.itertuples()
        },
        "clusters": cl.set_index("representation")[
            ["ari_vs_shape_clusters", "heat_slope_r2_by_cluster"]
        ].to_dict("index"),
    }
    (OUT_DIR / "probe_summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    logger.info("probe: %s", json.dumps(summary["heat_share_true_variance"]))
    return {"probe": p, "classification": pc, "clusters": cl, "summary": summary}
