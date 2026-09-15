"""Render every clustering figure from the repository root."""

import subprocess
import sys
from pathlib import Path

here = Path(__file__).parent
jobs = [
    ("fig_cluster_profiles.py", ["elec", "site"]),
    ("fig_cluster_profiles.py", ["elec", "site", "4"]),
    ("fig_cluster_profiles.py", ["elec", "meter"]),
    ("fig_cluster_profiles.py", ["elec", "trust"]),
    ("fig_cluster_profiles.py", ["gas", "meter"]),
    ("fig_cluster_profiles.py", ["gas", "site"]),
    ("fig_cluster_silhouette.py", []),
    ("fig_cluster_pca.py", []),
    ("fig_cluster_composition.py", []),
    ("fig_cluster_map.py", []),
    ("fig_cluster_skill.py", []),
    ("fig_cluster_dendrogram.py", []),
    ("fig_cluster_descriptors.py", []),
]
failed = []
for script, argv in jobs:
    r = subprocess.run(
        [sys.executable, str(here / script), *argv], capture_output=True, text=True, cwd=here
    )
    print(f"=== {script} {' '.join(argv)}\n{r.stdout.strip()}")
    if r.returncode:
        failed.append(f"{script} {' '.join(argv)}")
        print(r.stderr.strip()[-1200:])
print(f"{len(jobs) - len(failed)} figures written" + (f"; failed {failed}" if failed else ""))
sys.exit(1 if failed else 0)
