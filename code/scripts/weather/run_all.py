"""Run every weather figure script, each in its own process, from the repository root."""

import subprocess
import sys
from pathlib import Path

here = Path(__file__).parent
scripts = sorted(p for p in here.glob("*.py") if p.name.startswith(("fig_", "map_")))
failed = []
# Scripts that take an argument run once per value.
ARGS = {
    "map_thermometer_error.py": ["ERA5", "ERA5-Land"],
    "fig_site_slopes.py": ["hdd14", "cdd20"],
    "map_site_slopes.py": ["hdd14", "cdd20"],
    "fig_hourly_by_use.py": ["cold", "hot"],
    "fig_heat_blocks.py": ["cdd20", "solar_kwh_m2"],
    "fig_heat_moderators.py": ["categorical", "continuous"],
    "fig_cate_vs_slope.py": ["observables", "signature"],
}
runs = [(s, a) for s in scripts for a in ARGS.get(s.name, [None])]
for script, arg in runs:
    print(f"=== {script.name}" + (f" {arg}" if arg else ""))
    cmd = [sys.executable, str(script)] + ([arg] if arg else [])
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout.strip())
    if r.returncode != 0:
        failed.append(script.name)
        print(r.stderr.strip()[-1500:])
print(
    f"\n{len(runs) - len(failed)}/{len(runs)} figures written"
    + (f"; failed: {failed}" if failed else "")
)
sys.exit(1 if failed else 0)
