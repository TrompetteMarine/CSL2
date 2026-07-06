# Reproducibility — Paper II (CSL2)

This package is engineered so that a clean checkout regenerates **every** result,
figure and table from fixed seeds, in a **fully version-locked** environment, on
CPython 3.10 / 3.11 / 3.12. Nothing is produced or edited by hand.

## 1. Locked environment
`requirements.txt` is an exact-pin lock (not a floor list): the versions the
manuscript was generated against, chosen so every wheel ships for the whole
3.10–3.12 matrix, i.e. **all interpreters resolve an identical environment**.
`requirements-dev.txt` adds the pinned test toolchain (`pytest` and its closure),
with environment markers for the `< 3.11` back-compat shims.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt        # exact, closed dependency set
```

Abstract ranges for `pip install -e .` live in `pyproject.toml`; the lock is the
source of truth for reproduction. To bump deliberately, upgrade, re-run
`make verify`, and copy the new `==` pins back into `requirements.txt`.

## 2. Determinism guarantees
- **Seeded randomness.** All stochasticity uses `numpy.random.default_rng`
  (PCG64), whose streams are stable across NumPy versions; identical seeds give
  identical outputs (`tests/test_legitimacy.py::…seed_reproducibility`).
- **`PYTHONHASHSEED=0`** and **`MPLBACKEND=Agg`** are exported in CI so hashing
  and rendering are deterministic and headless.
- **No `text.usetex`.** Figures render with mathtext, so no LaTeX is needed to
  reproduce the PDFs of the plots (only to build the manuscript).
- **Warnings are errors** in the test step (`-W error::DeprecationWarning
  -W error::FutureWarning`), so any future library drift fails loudly instead of
  silently changing numbers.

## 3. One-command reproduction
```bash
PYTHONPATH=. python scripts/run_all.py standard   # or: make reproduce
```
`run_all.py` runs, in dependency order: `run_experiments` → `run_ablations` →
`run_phase` (resumable) → `make_figures` → `make_gt_figures` → `make_tables`.
Resumable sweeps are looped until complete, so the single command works both in
one pass and under a wall-clock budget (`ECSL_BUDGET`, seconds).

Verify first:
```bash
PYTHONPATH=. python -m pytest tests/ -q      # 14 tests
```

Build the manuscript (needs a LaTeX toolchain with TikZ/pgfplots):
```bash
make paper
```

## 4. Profiles
`preview` (fast smoke) / `standard` (reported figures) / `publication` (large
sweeps). Profiles change **only** replicate counts and grid resolution; the
mechanism parameters are identical across profiles.

## 5. Continuous integration
`.github/workflows/ci.yml` has two jobs: **verify** (pytest on 3.10/3.11/3.12
from the lock, warnings-as-errors) and **reproduce-smoke** (the full
`run_all.py preview` pipeline on 3.11, asserting figures and tables are
regenerated, uploaded as artifacts).

## 6. Notes on what makes this *exactly* reproducible
Earlier the deps were floor-pinned (`>=`), so CI resolved whatever NumPy was
newest on the run date and different matrix legs could resolve different NumPy
majors; a size-1-array→scalar cast (`test_legitimacy.py`) sat on NumPy's
deprecation-to-error path; and `run_all.py` omitted stages that the figure/table
makers consume. All three are fixed here: exact pins, `.item()` on the cast, and
a complete dependency-ordered driver.
