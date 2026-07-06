"""Shared paths for Paper II scripts."""
import os
import sys
import json

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))   # code/
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

RESULTS = os.path.join(ROOT, "results")
# Locate the manuscript folder robustly: flat layout (paper/ under the repo root)
# or a nested code/ + manuscript/ layout.
if os.path.isdir(os.path.join(ROOT, "paper")):
    _MAN = os.path.join(ROOT, "paper")
else:
    _MAN = os.path.join(ROOT, "..", "manuscript")
FIGDIR = os.path.abspath(os.path.join(_MAN, "figures"))
TABDIR = os.path.abspath(os.path.join(_MAN, "tables"))
for d in (RESULTS, FIGDIR, TABDIR):
    os.makedirs(d, exist_ok=True)

from legitimacy.config import SCALE                                     # noqa: E402


def get_scale(default="standard"):
    for a in sys.argv[1:]:
        if a in SCALE:
            return a
    return os.environ.get("ECSL_SCALE", default)


def dump_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=float)
