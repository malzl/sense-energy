"""LaTeX tables for the PI report, generated from the frozen score files (no number is retyped)."""

import json
from pathlib import Path

import pandas as pd

from sense_energy.config import INTERIM_DIR, PROCESSED_DIR, PROJECT_ROOT
from sense_energy.visualization import style

POC = PROJECT_ROOT / "code" / "reports" / "poc"
HIER = PROJECT_ROOT / "code" / "reports" / "poc_hierarchy"
CLUS = PROJECT_ROOT / "code" / "reports" / "clustering"
REPORTS = PROJECT_ROOT / "code" / "reports"
OUT = Path(__file__).parent / "tables"
LEVELS = ["meter", "site", "trust", "region", "total"]


def signed(v: float) -> str:
    """One decimal with an explicit sign; a typographic minus in LaTeX."""
    return f"{v:+.1f}".replace("-", "$-$")


def tex(s: str) -> str:
    return str(s).replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")


def write(
    name: str, header: list[str], rows: list[list[str]], align: str, rules: set[int] = frozenset()
):
    lines = [rf"\begin{{tabular}}{{{align}}}", r"\toprule"]
    if any(header):
        lines += [" & ".join(header) + r" \\", r"\midrule"]
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
    "lightgbm_ifs_noid", "lightgbm_ifs_eric",
    "tnp", "tnp_ifs", "pttnp", "pttnp_ifs", "gtnp", "gtnp_ifs",
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
    rules={3, 6, 9, 12, 15, 18, 21, 23},
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

# --- 7. the data at a glance -----------------------------------------------
sites = pd.read_parquet(INTERIM_DIR / "sites.parquet")
hh = pd.read_parquet(
    INTERIM_DIR / "consumption_halfhourly.parquet",
    columns=["energy_type", "datetime", "consumption_kwh"],
)
hh = hh[hh["consumption_kwh"].notna()]
span = hh.groupby("energy_type")["datetime"].agg(["min", "max"])
si_e = pd.read_parquet(PROCESSED_DIR / "elec" / "site" / "series_index.parquet")
si_m = pd.read_parquet(PROCESSED_DIR / "elec" / "meter" / "series_index.parquet")
si_g = pd.read_parquet(PROCESSED_DIR / "gas" / "site" / "series_index.parquet")
w = pd.read_parquet(PROCESSED_DIR / "elec" / "site" / "wide_raw.parquet") * 2
mean_kw = w.mean().sort_values(ascending=False)
chp = pd.read_csv(REPORTS / "chp" / "group_summary.csv").set_index("group")
flags = pd.read_csv(REPORTS / "estates" / "coverage_flags.csv")
wx = json.loads((REPORTS / "weather_events" / "summary.json").read_text())


def period(e):
    return f"{span.loc[e, 'min']:%-d %b %Y} -- {span.loc[e, 'max']:%-d %b %Y}"


rows = [
    [
        "Sites in the extract",
        f"{len(sites)} sites, {sites['organisation_name'].nunique()} trusts, 7 NHS regions",
    ],
    [
        "Half-hourly electricity",
        f"{len(si_e)} sites ({si_e['trust_id'].nunique()} trusts), {len(si_m)} meters, {period('elec')}",
    ],
    ["Half-hourly gas", f"{len(si_g)} sites, {period('gas')}"],
    [
        "Mean electricity demand",
        f"{mean_kw.sum() / 1000:.1f} MW in total; median site {mean_kw.median():.0f} kW, largest {mean_kw.iloc[0] / 1000:.1f} MW",
    ],
    [
        "Concentration",
        f"10 largest sites carry {100 * mean_kw.head(10).sum() / mean_kw.sum():.0f}\\% of demand",
    ],
    [
        "Sites with on-site CHP",
        f"{int(chp.loc['on-site CHP', 'sites'])}, carrying {100 * chp.loc['on-site CHP', 'demand_share']:.0f}\\% of demand",
    ],
    [
        "With ERIC 2023/24 electricity",
        f"{int((flags['coverage_flag'] != 'no_eric').sum())} sites; annual total confirmed for {int((flags['coverage_flag'] == 'ok').sum())}",
    ],
    [
        "Weather",
        f"ERA5 hourly at {wx['sites']} located sites, {pd.Timestamp(wx['weather_window'][0]):%-d %b %Y} -- {pd.Timestamp(wx['weather_window'][1]):%-d %b %Y}",
    ],
]
write("data", ["", ""], rows, "ll")

# --- 8. sites with on-site CHP ----------------------------------------------
rows = []
for g in ("on-site CHP", "acute, no CHP", "other sites"):
    r = chp.loc[g]
    rows.append(
        [
            tex(g),
            f"{int(r['sites'])}",
            f"{100 * r['demand_share']:.0f}",
            f"{r['mean_kw']:.0f}",
            f"{r['load_factor']:.2f}",
            f"{100 * r['daily_cv']:.0f}",
            f"{100 * r['jump_day_share']:.1f}",
            f"{100 * r['nmae']:.1f}",
            f"{100 * r['crps_skill']:.0f}",
            f"{r['cooling_pct_per_cdd']:.1f}",
        ]
    )
write(
    "chp",
    [
        "Group",
        "Sites",
        r"Demand (\%)",
        "Median kW",
        "Load factor",
        r"Daily CV (\%)",
        r"Jump days (\%)",
        r"nMAE (\%)",
        r"CRPS skill (\%)",
        r"\% per CDD",
    ],
    rows,
    "lrrrrrrrrr",
)

# --- 9. ERIC coverage flags -------------------------------------------------
desc = {
    "ok": "annual meter total within 0.65--1.5 of ERIC",
    "meter_gaps": "fewer than 80\\% of the year's half-hours observed",
    "no_eric": "no ERIC electricity figure",
    "chp_mismatch": "outside the range at a site with on-site CHP",
    "aggregate_supply": "one meter feeds more than the ERIC site",
    "partial_supply": "meters cover part of the site",
}
fc = flags["coverage_flag"].value_counts()
ok_ratio = flags.loc[flags["coverage_flag"] == "ok", "energy_ratio"]
rows = [
    [tex(style.COVERAGE_FLAG_NAMES.get(k, "no ERIC figure")), f"{int(fc.get(k, 0))}", desc[k]]
    for k in desc
]
write("eric_flags", ["Meter check", "Sites", "Rule"], rows, "lrl")

# --- 10. nonlinear temperature response ------------------------------------
cv = pd.read_csv(REPORTS / "weather_response" / "curve.csv")
cv = cv[cv["outcome"] == "daily mean"]
groups = list(style.SITE_GROUP_ORDER)
n_sites = cv.groupby("group")["sites"].first()
show = ["<-2", "0", "8", "14", "20", "22", ">=24"]
names = {"<-2": r"below $-2$", ">=24": r"24 and above"}
rows = []
for b in show:
    cells = []
    for g in groups:
        r = cv[(cv["group"] == g) & (cv["bin"] == b)].iloc[0]
        if b == "14":
            cells.append("reference")
        else:
            cells.append(f"{signed(r['coef_pct'])} [{signed(r['lo_pct'])}, {signed(r['hi_pct'])}]")
    label = names[b] if b in names else f"{float(b):g} -- {float(b) + 2:g}"
    rows.append([label] + cells)
write(
    "temperature_response",
    [r"Daily mean ($^\circ$C)"] + [f"{tex(g)} ({int(n_sites[g])})" for g in groups],
    rows,
    "lrrr",
    rules={3, 4},
)

# --- 11. extreme-weather events ----------------------------------------------
nat = pd.read_csv(REPORTS / "weather_events" / "national_events.csv")
defs = {
    "heatwave": r"$\geq$3 days with $T_{max}$ at or above the county threshold (Met Office)",
    "cold spell": r"daily mean $\leq$2\,$^\circ$C on $\geq$2 days (UKHSA cold alert)",
    "snow": r"$\geq$5\,mm water equivalent in a day",
    "heavy rain": r"$\geq$25\,mm in a day",
    "high wind": r"gusts $\geq$22\,m/s (about 50\,mph)",
}
rows = []
for k in style.WEATHER_EVENT_ORDER:
    e = nat[nat["type"] == k]
    rec = e[e["in_electricity_record"]]
    scored = rec[rec["demand_site_days"] >= 20]["demand_ratio_median"]
    rows.append(
        [
            k,
            defs[k],
            f"{len(e)}",
            f"{len(rec)}",
            f"{int(wx['site_events'][k]):,}",
            signed(100 * (scored.median() - 1)) if len(scored) else "--",
        ]
    )
write(
    "events",
    ["Event", "Definition", "Events", "In record", "Site events", r"Demand (\%)"],
    rows,
    r"lp{6.2cm}rrrr",
)

# --- 12. quasi-experimental temperature effects ------------------------------
cz = pd.read_csv(REPORTS / "weather_causal" / "twfe_specs.csv")
cz = cz[cz["group"] == "no CHP"]
iv = pd.read_csv(REPORTS / "weather_causal" / "identifying_variation.csv")
iv = iv[iv["group"] == "no CHP"].set_index(["spec", "dose"])["within_sd"]
rows = []
for spec in style.SPEC_ORDER:
    cells = [tex(style.SPEC_NAMES[spec])]
    for dose in ("hdd14", "cdd20"):
        r = cz[(cz["spec"] == spec) & (cz["term"] == dose)].iloc[0]
        cells.append(f"{signed(r['coef_pct'])} [{signed(r['lo_pct'])}, {signed(r['hi_pct'])}]")
        cells.append(f"{iv[(spec, dose)]:.2f}")
    rows.append(cells)
write(
    "causal_specs",
    [
        "Fixed effects",
        r"Cold (\% per $^\circ$C below 14)",
        r"SD ($^\circ$C)",
        r"Heat (\% per $^\circ$C above 20)",
        r"SD ($^\circ$C)",
    ],
    rows,
    "lrrrr",
)

ev = pd.read_csv(REPORTS / "weather_causal" / "event_per_degree.csv")
es = pd.read_csv(REPORTS / "weather_causal" / "event_study.csv")
samples = {
    (d["type"], d["group"]): d
    for d in json.loads((REPORTS / "weather_causal" / "summary.json").read_text())["event_samples"]
}
rows = []
for kind in ("heatwave", "cold spell"):
    r = ev[(ev["type"] == kind) & (ev["group"] == "no CHP")].iloc[0]
    p_pre = es[(es["type"] == kind) & (es["group"] == "no CHP") & (es["outcome"] == "demand")][
        "p_leads"
    ].iloc[0]
    n = samples[(kind, "no CHP")]
    exposure = abs(r["temperature_att_c"])  # degrees hotter (heatwave) or colder (cold spell)
    rows.append(
        [
            kind,
            f"{n['stacks']}",
            f"{n['treated_sites']} / {n['control_sites']}",
            f"{signed(r['temperature_att_c'])}",
            f"{signed(r['demand_att_pct'])} [{signed(r['demand_att_lo'])}, {signed(r['demand_att_hi'])}]",
            f"{signed(r['demand_att_pct'] / exposure)} ({r['pct_per_c_se']:.1f})",
            f"{p_pre:.2f}",
        ]
    )
write(
    "causal_events",
    [
        "Event",
        "Stacks",
        "Treated / control sites",
        r"Exposure ($^\circ$C)",
        r"Demand (\%)",
        r"\% per $^\circ$C of exposure (SE)",
        "Pre-trend $p$",
    ],
    rows,
    "lrrrrrr",
)

# --- 13. extreme days, neighbour placebo, inference ---------------------------
ex = pd.read_csv(REPORTS / "weather_causal" / "twfe_extreme.csv")
ex = ex[ex["group"] == "no CHP"]
rows = []
for spec in style.SPEC_ORDER:
    cells = [tex(style.SPEC_NAMES[spec])]
    for term in ("hot28", "hot30"):
        r = ex[(ex["spec"] == spec) & (ex["term"] == term)].iloc[0]
        cells.append(f"{signed(r['coef_pct'])} [{signed(r['lo_pct'])}, {signed(r['hi_pct'])}]")
    rows.append(cells)
n30 = int(ex["site_days_30plus"].iloc[0])
write(
    "causal_extreme",
    [
        "Fixed effects",
        r"28--30\,$^\circ$C day (\%)",
        rf"30\,$^\circ$C+ day (\%), {n30} site-days",
    ],
    rows,
    "lrr",
)

lagt = pd.read_csv(REPORTS / "weather_causal" / "spatial_lag.csv")
rows = []
for variant in ("own and neighbours", "neighbours only"):
    for spec in ("A", "B"):
        d = lagt[(lagt["variant"] == variant) & (lagt["spec"] == spec)].set_index("term")
        cells = [tex(f"{style.SPEC_NAMES[spec]}, {variant}")]
        for term in ("hdd14", "cdd20", "hdd14_nb", "cdd20_nb"):
            if term in d.index:
                r = d.loc[term]
                cells.append(
                    f"{signed(r['coef_pct'])} [{signed(r['lo_pct'])}, {signed(r['hi_pct'])}]"
                )
            else:
                cells.append("--")
        rows.append(cells)
write(
    "causal_spatial_lag",
    ["Specification", "Own cold", "Own heat", "Neighbours' cold", "Neighbours' heat"],
    rows,
    "lrrrr",
    rules={2},
)

inf = pd.read_csv(REPORTS / "weather_causal" / "inference.csv")
rows = []
for cl in ("site and date", "region x week", "ICB"):
    d = inf[inf["cluster"] == cl].set_index("term")
    rows.append(
        [
            tex(cl),
            tex(str(d["n_clusters"].iloc[0])),
            f"{d.loc['hdd14', 'se_pct']:.2f}",
            f"{d.loc['cdd20', 'se_pct']:.2f}",
        ]
    )
write(
    "causal_inference",
    ["Clusters", "Number", r"SE cold (\% per $^\circ$C)", r"SE heat (\% per $^\circ$C)"],
    rows,
    "llrr",
)
print("tables written:", sorted(p.name for p in OUT.glob("*.tex")))


# --- what moves heat ---------------------------------------------------------
hd = pd.read_csv(REPORTS / "heat_drivers" / "group_decomposition.csv")
hd = hd[hd["model"] == "night and day"].set_index(["sign", "term"])
hs = json.loads((REPORTS / "heat_drivers" / "summary.json").read_text())
feat = pd.read_csv(REPORTS / "heat_drivers" / "site_features.csv")
share = feat.groupby("sign")["mean_kw"].sum() / feat["mean_kw"].sum()
rows = []
for sign in ("rises with heat", "falls with heat", "neither"):
    cells = [
        tex(sign),
        f"{int(hd.loc[(sign, 'hot_afternoon'), 'sites'])}",
        f"{100 * share.get(sign, 0):.0f}",
    ]
    for term in ("hot_afternoon", "warm_night", "solar_kwh_m2"):
        r = hd.loc[(sign, term)]
        cells.append(f"{signed(r['coef_pct'])} [{signed(r['lo_pct'])}, {signed(r['hi_pct'])}]")
    rows.append(cells)
write(
    "heat_decomposition",
    [
        "Daily heat slope",
        "Sites",
        r"Demand (\%)",
        r"Hot afternoon (\% per $^\circ$C)",
        r"Warm night (\% per $^\circ$C)",
        r"Sunshine (\% per kWh/m$^2$)",
    ],
    rows,
    "lrrrrr",
)
