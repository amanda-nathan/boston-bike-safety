import subprocess
import sys
from pathlib import Path

SCRIPTS = [
    "download_data.py",
    "build_graph.py",
    "train_gnn.py",
    "build_site.py",
    "build_readme.py",
]

SRC = Path(__file__).resolve().parent

for script in SCRIPTS:
    result = subprocess.run(
        [sys.executable, str(SRC / script)],
        cwd=str(SRC),
    )
    if result.returncode != 0:
        sys.exit(result.returncode)
