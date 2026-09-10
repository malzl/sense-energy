#!/usr/bin/env bash
# Takes over from the running TIGGE client without killing its in-flight ECDS jobs.
cd /home/malzn/sense-energy
while [ -n "2906993" ] && kill -0 2906993 2>/dev/null; do sleep 120; done
exec /home/malzn/sense-energy/.venv/bin/sense-energy --log-level INFO fetch-tigge --config code/configs/tigge.yaml --until-complete >> code/data/external/weather/fetch_tigge.log 2>&1
