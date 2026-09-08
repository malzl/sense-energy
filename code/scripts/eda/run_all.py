"""Run every EDA figure script, each in its own process, from the repository root."""

import subprocess
import sys
from pathlib import Path

here = Path(__file__).parent
scripts = sorted(p for p in here.glob("*.py") if p.name.startswith(("fig_", "map_")))
failed = []
for script in scripts:
    print(f"=== {script.name}")
    r = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
    print(r.stdout.strip())
    if r.returncode != 0:
        failed.append(script.name)
        print(r.stderr.strip()[-1500:])
print(
    f"\n{len(scripts) - len(failed)}/{len(scripts)} figures written"
    + (f"; failed: {failed}" if failed else "")
)
sys.exit(1 if failed else 0)
