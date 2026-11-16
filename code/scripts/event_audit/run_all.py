"""Run every figure script of the event audit (the specification curve once per number)."""

import subprocess
import sys
from pathlib import Path

from _common import METRICS

here = Path(__file__).parent
for script in sorted(here.glob("fig_*.py")):
    for arg in list(METRICS) if script.name == "fig_specification_curve.py" else [None]:
        print(">>", script.name, arg or "")
        subprocess.run([sys.executable, str(script), *([arg] if arg else [])], check=True, cwd=here)
