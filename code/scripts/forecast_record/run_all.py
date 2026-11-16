"""Run every whole-record forecast figure script."""

import subprocess
import sys
from pathlib import Path

here = Path(__file__).parent
for script in sorted(here.glob("fig_*.py")):
    print(">>", script.name)
    subprocess.run([sys.executable, str(script)], check=True, cwd=here)
