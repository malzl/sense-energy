"""LaTeX tables for the PI report, generated from the frozen score files (no number is retyped)."""

from pathlib import Path

import pandas as pd

from sense_energy.config import INTERIM_DIR, PROJECT_ROOT
from sense_energy.visualization import style

POC = PROJECT_ROOT / "code" / "reports" / "poc"
HIER = PROJECT_ROOT / "code" / "reports" / "poc_hierarchy"
CLUS = PROJECT_ROOT / "code" / "reports" / "clustering"
OUT = Path(__file__).parent / "tables"
LEVELS = ["meter", "site", "trust", "region", "total"]


def tex(s: str) -> str:
    return str(s).replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")


def write(
    name: str, header: list[str], rows: list[list[str]], align: str, rules: set[int] = frozenset()
):
    lines = [
        rf"\begin{{tabular}}{{{align}}}",
        r"\toprule",
        " & ".join(header) + r" \\",
        r"\midrule",
    ]
    for i, r in enumerate(rows):
        if i in rules:
            lines.append(r"\addlinespace[3pt]")
        lines.append(" & ".join(r) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / f"{name}.tex").write_text("\n".join(lines) + "\n")


# --- 1. site-level results -------------------------------------------------
pooled = pd.read_csv(POC / "scores_pooled.csv").set_index("model")
per = pd.read_csv(POC / "scores_per_site.csv")
beat = per[per["mae_skill_vs_naive"] > 0].groupby("model").size() / per.groupby("model").size()
main = [
    "seasonal_naive", "profile_quantiles", "sarima",
    "lightgbm", "lightgbm_era5", "lightgbm_ifs",
    "chronos2", "chronos2_era5", "chronos2_ifs",
    "timesfm3", "timesfm3_era5", "timesfm3_ifs",
    "tabpfn_ts", "tabpfn_ts_era5", "tabpfn_ts_ifs",
    "tirex2", "tirex2_era5", "tirex2_ifs",
    "t0beta", "t0beta_era5", "t0beta_ifs",
]  # fmt: skip
best_nmae = pooled.loc[main, "nmae"].min()
best_crps = pooled.loc[main, "crps_q"].min()
rows = []
for m in main:
    r = pooled.loc[m]
    naive = m == "seasonal_naive"
    nm, cr = f"{100 * r['nmae']:.1f}", f"{r['crps_q']:.2f}"
    if abs(r["nmae"] - best_nmae) < 5e-4:
        nm = rf"\textbf{{{nm}}}"
    if abs(r["crps_q"] - best_crps) < 5e-3:
        cr = rf"\textbf{{{cr}}}"
    rows.append(
        [
            tex(style.METHOD_NAMES[m]),
            nm,
            cr,
            "--" if naive else f"{100 * r['coverage_80']:.0f}",
            "--" if naive else f"{100 * r['crps_skill_vs_naive_median']:.0f}",
            "--" if naive else f"{100 * beat.get(m, 0.0):.0f}",
        ]
    )
write(
    "site_results",
    [
        "Method",
        r"nMAE (\%)",
        "CRPS",
        r"Coverage (\%)",
        r"CRPS skill (\%)",
        r"Sites better than naive (\%)",
    ],
    rows,
    "lrrrrr",
    rules={3, 6, 9, 12, 15, 18},
)

# --- 2. weather ensemble ---------------------------------------------------
rows = []
for m, label in (
    ("chronos2", "No covariates"),
    ("chronos2_ifs", "Ensemble mean"),
    ("chronos2_ifs_ctrl", "Control member"),
    ("chronos2_ifs_members", "51 members, mixed"),
    ("chronos2_era5", "Reanalysis (perfect weather)"),
):
    r = pooled.loc[m]
    rows.append(
        [label, f"{100 * r['nmae']:.2f}", f"{r['crps_q']:.3f}", f"{100 * r['coverage_80']:.0f}"]
    )
write(
    "ensemble",
    ["Weather given to Chronos-2", r"nMAE (\%)", "CRPS", r"Coverage (\%)"],
    rows,
    "lrrr",
    rules={1, 4},
)

# --- 3. hierarchy: direct forecasts ----------------------------------------
s = pd.read_csv(HIER / "summary.csv")
d = (
    s[s["kind"] == "direct"].pivot_table(index="model", columns="level", values="nmae")[LEVELS]
    * 100
)
n = s[s["kind"] == "direct"].groupby("level")["sites"].max().reindex(LEVELS).astype(int)
rows = [[r"\textit{Series}"] + [f"\\textit{{{v}}}" for v in n]]
for m in ("seasonal_naive", "lightgbm_ifs", "timesfm3_ifs", "chronos2_ifs"):
    rows.append([tex(style.METHOD_NAMES[m])] + [f"{v:.1f}" for v in d.loc[m]])
write(
    "hierarchy_direct",
    ["Direct forecast", "Meter", "Site", "Trust", "Region", "National"],
    rows,
    "lrrrrr",
    rules={1},
)

# --- 4. reconciliation -----------------------------------------------------
r = pd.read_csv(HIER / "reconciliation.csv")
rc = (
    r[r["model"] == "chronos2_ifs"].pivot_table(index="method", columns="level", values="nmae")[
        LEVELS
    ]
    * 100
)
order = [
    "base",
    "bottom_up",
    "wls_var",
    "mint_shrink",
    "wls_struct",
    "middle_out",
    "top_down",
    "ols",
]
best = rc.loc[[m for m in order if m != "base"]].min()
rows = []
for m in order:
    cells = []
    for lv in LEVELS:
        v = f"{rc.loc[m, lv]:.1f}"
        cells.append(
            rf"\textbf{{{v}}}" if m != "base" and abs(rc.loc[m, lv] - best[lv]) < 0.05 else v
        )
    rows.append([tex(style.RECONCILIATION_NAMES[m])] + cells)
write(
    "reconciliation",
    ["Approach", "Meter", "Site", "Trust", "Region", "National"],
    rows,
    "lrrrrr",
    rules={1},
)

# --- 5. clusters (k = 4, electricity sites) --------------------------------
t = pd.read_csv(CLUS / "elec_site_clusters.csv").set_index("series_id")
lab = pd.read_csv(CLUS / "elec_site_labels_all_k.csv").set_index("series_id")["k4"]
size_order = t["mean_kw"].groupby(lab).mean().sort_values(ascending=False).index
lab = lab.map({old: new for new, old in enumerate(size_order)})
t["k4"] = lab
names = {}
g = t.groupby("k4")
med = g[
    [
        "mean_kw",
        "night_day_ratio",
        "weekend_ratio",
        "winter_summer_ratio",
        "temp_slope_per_c",
        "daily_cv",
    ]
].median()
for k in med.index:
    nd = med.loc[k, "night_day_ratio"]
    names[k] = (
        "Night-heavy"
        if nd > 1.5
        else "Round-the-clock"
        if nd > 0.7
        else "Daytime"
        if nd > 0.45
        else "Office-like"
    )
rows = []
for k in sorted(med.index, key=lambda k: -med.loc[k, "night_day_ratio"]):
    m = med.loc[k]
    rows.append(
        [
            f"{names[k]} (cluster {k})",
            f"{int((t['k4'] == k).sum())}",
            f"{m['mean_kw']:.0f}",
            f"{m['night_day_ratio']:.2f}",
            f"{m['weekend_ratio']:.2f}",
            f"{m['winter_summer_ratio']:.2f}",
            f"{100 * m['temp_slope_per_c']:.1f}",
            f"{m['daily_cv']:.2f}",
        ]
    )
write(
    "clusters",
    [
        "Shape",
        "Sites",
        "Median kW",
        "Night/day",
        "Weekend/weekday",
        "Winter/summer",
        r"\% per $^\circ$C",
        "Daily CV",
    ],
    rows,
    "lrrrrrrr",
)

# --- 6. public activity data ----------------------------------------------
tr = pd.read_parquet(INTERIM_DIR / "nhs_activity_trusts.parquet")
ae = pd.read_parquet(INTERIM_DIR / "nhs_activity_ae_monthly.parquet")
amb = pd.read_parquet(INTERIM_DIR / "nhs_activity_ambsys_monthly.parquet")
kh = pd.read_parquet(INTERIM_DIR / "nhs_activity_kh03_quarterly.parquet")
n_amb = int(tr["organisation_name"].str.contains("Ambulance").sum())
rows = [
    [
        "A\\&E attendances and emergency admissions",
        "monthly",
        f"{ae['period'].min():%b %Y} -- {ae['period'].max():%b %Y}",
        f"{int(tr['in_ae'].sum())} of {len(tr)}",
    ],
    [
        "Ambulance systems indicators",
        "monthly",
        f"{amb['period'].min():%b %Y} -- {amb['period'].max():%b %Y}",
        f"{int(tr['in_ambsys'].sum())} of {n_amb} ambulance trusts",
    ],
    [
        "Overnight beds available and occupied",
        "quarterly",
        f"{kh['snapshot'].min():%Y} -- {kh['snapshot'].max():%b %Y}",
        f"{int(tr['in_kh03'].sum())} of {len(tr)}",
    ],
]
write("activity", ["Public series", "Resolution", "Period", "Our trusts covered"], rows, "llll")
print("tables written:", sorted(p.name for p in OUT.glob("*.tex")))
