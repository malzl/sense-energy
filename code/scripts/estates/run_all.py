"""Render every estates figure from the repository root."""

import subprocess
import sys
from pathlib import Path

here = Path(__file__).parent
failed = []
#: scripts that render one figure per argument
ARGS = {"fig_chp_dispatch.py": ["cold", "price", "red"]}
scripts = sorted(here.glob("fig_*.py"))
for script in scripts:
    for arg in ARGS.get(script.name, [None]):
        cmd = [sys.executable, str(script)] + ([arg] if arg else [])
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=here)
        print(f"=== {script.name} {arg or ''}\n{r.stdout.strip()}")
        if r.returncode:
            failed.append(f"{script.name} {arg or ''}".strip())
            print(r.stderr.strip()[-1500:])
print(f"{len(scripts) - len(failed)} figures written" + (f"; failed {failed}" if failed else ""))
sys.exit(1 if failed else 0)
