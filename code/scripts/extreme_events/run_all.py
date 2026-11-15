"""Run every extreme-event figure script (those with an event type once per type)."""

import subprocess
import sys
from pathlib import Path

here = Path(__file__).parent
BY_TYPE = {"fig_hour_profile.py", "map_reaction.py"}
for script in sorted([*here.glob("fig_*.py"), *here.glob("map_*.py")]):
    for arg in ("heatwave", "cold") if script.name in BY_TYPE else (None,):
        cmd = [sys.executable, str(script), *([arg] if arg else [])]
        print(">>", script.name, arg or "")
        subprocess.run(cmd, check=True, cwd=here)
