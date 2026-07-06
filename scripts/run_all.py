"""Regenerate all Paper II results, figures and tables for a given scale.

Usage:
    python scripts/run_all.py [preview|standard|publication]

Resumable stages (phase) are looped until they report completion, so this single
command works both locally (one pass) and under a wall-clock budget (set the
ECSL_BUDGET environment variable, in seconds, to force chunked runs). Running
from a clean checkout regenerates every result, figure and table from fixed
seeds; nothing is produced or edited by hand.
"""
import os
import subprocess
import sys

import numpy as np

import common

PY = sys.executable
HERE = os.path.dirname(__file__)


def run(script, scale):
    subprocess.run([PY, os.path.join(HERE, script), scale], check=True)


def loop_until_done(script, result_file, scale, key="done"):
    """Re-invoke a resumable/budgeted stage until its result marks all rows done."""
    path = f"{common.RESULTS}/{result_file}"
    for _ in range(200):
        run(script, scale)
        if os.path.exists(path):
            d = np.load(path)
            if key in d and bool(np.all(d[key])):
                return
    raise RuntimeError(f"{script} did not complete")


def main():
    scale = common.get_scale()
    print(f"== regenerating all Paper II results at scale '{scale}' ==")
    run("run_experiments.py", scale)                              # experiments_<scale>.{npz,json}
    run("run_ablations.py", scale)                                # ablations_<scale>.{npz,json}
    loop_until_done("run_phase.py", f"phase_main_{scale}.npz", scale)  # phase_main/secondary_<scale>.npz
    run("make_figures.py", scale)                                 # Monte-Carlo figures (2-7)
    run("make_gt_figures.py", scale)                              # game-theory figures (phase portrait, hysteresis)
    run("make_tables.py", scale)                                  # results / ablation / param tables
    print("== done: figures in paper/figures, tables in paper/tables ==")


if __name__ == "__main__":
    main()
