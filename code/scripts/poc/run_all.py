"""Run every PoC figure script from the repository root."""

import subprocess
import sys
from pathlib import Path

here = Path(__file__).parent
failed = []
for script in sorted(here.glob("fig_*.py")):
    r = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, cwd=here)
    print(f"=== {script.name}\n{r.stdout.strip()}")
    if r.returncode:
        failed.append(script.name)
        print(r.stderr.strip()[-1200:])
print(
    f"{len(list(here.glob('fig_*.py'))) - len(failed)} figures written"
    + (f"; failed {failed}" if failed else "")
)
sys.exit(1 if failed else 0)
