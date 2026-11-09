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
for f in chp_variability chp_generation dec_recommendations eric_changes installation_savings chp_dispatch_red meter_vs_eric headroom_by_type; do
  cp "$root/code/reports/figures/estates/$f.pdf" "$here/figures/"
done
for f in temperature_response_curve temperature_response_hourly weather_event_timeline \
         weather_event_demand weather_persistence causal_specs causal_event_study \
         causal_lags causal_randomisation causal_extreme causal_spatial \
         causal_response_mean causal_response_tmax temperature_response_tmax \
         thermometer_agreement thermometer_effects thermometer_response \
         map_thermometer_network map_thermometer_error_era5 map_hot_day_bias \
         map_heatwave_exposure site_slopes_hdd14 site_slopes_cdd20 moderators \
         hourly_by_use_cold map_site_slopes_hdd14 map_site_slopes_cdd20 \
         heat_sunshine_control heat_night_day heat_blocks_heat heat_group_decomposition \
         heat_moderators_continuous heat_variance_explained heat_moderators_categorical \
         map_heat_types heat_dec_aircon heat_dec_heating cate_vs_slope_observables cate_importance \
         forecast_heat_days jepa_probe jepa_map jepa_surprise jepa_response latent_pfn \
         hourly_pooled warming_capacity heatwave_attribution solar_watch; do
  cp "$root/code/reports/figures/weather/$f.pdf" "$here/figures/"
done
cd "$here"
pdflatex -interaction=nonstopmode -halt-on-error pi_report.tex > build.log
pdflatex -interaction=nonstopmode -halt-on-error pi_report.tex >> build.log
rm -f pi_report.aux pi_report.out pi_report.log build.log
echo "wrote $here/pi_report.pdf"
