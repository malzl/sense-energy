#!/bin/bash
# Build the PI report: tables from the score files, figures copied from the figure
# folders (vector PDFs), two pdflatex passes. Run from anywhere.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../.." && pwd)"
"$root/.venv/bin/python" "$here/make_tables.py"
mkdir -p "$here/figures"
for f in crps_skill_by_method covariate_gain calibration ensemble_spread_skill nmae_by_hour \
         map_site_skill nmae_by_level reconciliation example_day; do
  cp "$root/code/reports/figures/poc/$f.pdf" "$here/figures/"
done
for f in cluster_profiles_elec_site_k4 cluster_map_elec_site_k4; do
  cp "$root/code/reports/figures/clustering/$f.pdf" "$here/figures/"
done
cd "$here"
pdflatex -interaction=nonstopmode -halt-on-error pi_report.tex > build.log
pdflatex -interaction=nonstopmode -halt-on-error pi_report.tex >> build.log
rm -f pi_report.aux pi_report.out pi_report.log build.log
echo "wrote $here/pi_report.pdf"
