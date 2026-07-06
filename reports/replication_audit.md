# Paper II — replication audit

## Environment
Python 3.10; NumPy, SciPy, Matplotlib, pytest. OS tested: Linux. LaTeX+TikZ for
the manuscript. All randomness seeded; ablations use common random numbers.
Docker/Julia not executed in the build sandbox (see `submission_readiness.md`).

## Commands (from `code/`)
```bash
python -m pytest tests/ -q            # 11/11 pass
python scripts/run_experiments.py     # II.1–II.6  -> results/experiments_standard.{npz,json}
python scripts/run_phase.py           # main + secondary phase grids (resumable)
python scripts/run_ablations.py       # A1–A7 under CRN -> ablations_standard.{npz,json}
python scripts/make_figures.py        # Figures 1–7 -> manuscript/figures
python scripts/make_tables.py         # Tables -> manuscript/tables
```

## Status
- Tests: **11/11 pass**.
- Experiments II.1–II.6, two phase diagrams (11×11), and ablations A1–A7 (CRN,
  Wilson + paired-bootstrap intervals) all run; results saved.
- Figures 1–7 and tables (results, ablation, parameters, comparison, variables)
  regenerate from results only.
- Manuscript compiles to **13 pages**, **0 undefined references/citations**.
- Runtime: experiments ≈35 s; phase grids ≈3 min; ablations ≈17 s (standard).

## Separation from Paper I
No code, configs, results, or figures shared with Paper I; Paper II
re-implements the credibility backbone and cites Paper I only as background.
