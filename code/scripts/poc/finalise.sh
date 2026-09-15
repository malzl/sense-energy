#!/bin/bash
# Wait for every running PoC job, then re-score, rebuild the hierarchy and
# reconciliation tables, render all figures and write the results table.
# Runs unattended under nohup; progress in code/reports/poc/finalise.log.
cd /home/malzn/sense-energy || exit 1
SE=.venv/bin/sense-energy
PY=.venv/bin/python
log() { echo "$(date '+%F %T') | $*"; }
running() { ps -eo args | grep -qE "^/home/malzn/sense-energy/.venv/bin/python3 .venv/bin/sense-energy --log-level INFO $1"; }
wait_for() { while running "$1"; do sleep 120; done; log "finished: $1"; }
# anchored on the interpreter so a shell whose command line merely mentions the script never matches
wait_script() { while ps -eo args | grep -qE "^/bin/bash .*$1"; do sleep 120; done; log "finished: $1"; }
step() { log "== $1"; shift; "$@" && log "ok" || log "FAILED (exit $?)"; }

log "finaliser started; waiting for running jobs"
wait_for "run-poc --model sarima"
wait_for "run-poc --model chronos2_ifs_members"
wait_script "run_levels_lgbm.sh"
wait_for "run-poc --model tabpfn_ts_ifs"
log "all jobs done; post-processing"

step "score site level" $SE --log-level WARNING run-poc --score-only
for L in meter trust region total; do
  step "score $L level" $SE --log-level WARNING run-poc --score-only --level $L
done
step "hierarchy bottom-up + summary" $SE --log-level WARNING poc-hierarchy
step "reconciliation LightGBM" $SE --log-level WARNING poc-reconcile --model lightgbm --model lightgbm_ifs
step "reconciliation foundation models" $SE --log-level WARNING poc-reconcile --model chronos2 --model chronos2_ifs --model timesfm3 --model timesfm3_ifs
step "figures" bash -c "cd code/scripts/poc && ../../../$PY run_all.py"
step "results table" bash -c "cd code/scripts/poc && ../../../$PY results_table.py > ../../reports/poc/results_table.md"
log "finaliser done"
