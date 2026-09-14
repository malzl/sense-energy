"""Print the PoC results table (markdown) from the frozen score files, in the fixed method order."""

import pandas as pd
from _common import NAIVE, RESULTS, display, ordered_models, per_site

pooled = pd.read_csv(RESULTS / "scores_pooled.csv").set_index("model")
per = per_site()
beat = per[per["mae_skill_vs_naive"] > 0].groupby("model").size() / per.groupby("model").size()
rows = []
for m in ordered_models(pooled.index):
    r = pooled.loc[m]
    rows.append(
        "| {name} | {nmae:.1f} | {mae:.2f} | {crps:.2f} | {cov} | {ms:+.1f} | {cs:+.1f} | {beat:.0f} |".format(
            name=display(m),
            nmae=100 * r["nmae"],
            mae=r["mae"],
            crps=r["crps_q"],
            cov="–" if m == NAIVE else f"{100 * r['coverage_80']:.0f}",
            ms=100 * r["mae_skill_vs_naive_median"],
            cs=100 * r["crps_skill_vs_naive_median"],
            beat=100 * beat.get(m, 0.0),
        )
    )
header = (
    "| Candidate method | nMAE (%) | MAE (kWh/½h) | CRPS | Coverage (%, target 80) | MAE skill (%) | CRPS skill (%) | Sites beating naive (%) |\n"
    "|---|---|---|---|---|---|---|---|"
)
print(header)
print("\n".join(rows))
