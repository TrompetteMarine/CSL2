# CSL2 — Legitimacy-Gated Social Learning

Reference implementation and replication package for
**"Endogenous Legitimacy in Social Learning: Fairness, Motivated Cognition, and
Behavioural Equilibrium"** (Cognitive Social Learning, Paper II; sibling of
[CSL](../CSL)).

Reward feedback disciplines beliefs only to the extent that the reward rule is
treated as **legitimate**, and legitimacy is itself learned from socially
transmitted **fairness benchmarks**. The effective prediction error is
`δ_eff = L · δ`: a legitimacy gate on evidence. The model produces four regimes —
legitimate, correct-but-illegitimate, defensive-wrong, and fairness-polarised —
and a behavioural-equilibrium analysis (QRE / M-equilibrium, a mean-field phase
portrait, and a motivated-cognition **cusp with legitimacy hysteresis**).

Every figure and table is regenerated from fixed seeds by the commands below.

---

## 1. Install

Python 3.10+.

```bash
git clone https://github.com/<owner>/CSL2.git
cd CSL2
python -m venv .venv && source .venv/bin/activate      # optional
pip install -r requirements.txt        # numpy, scipy, matplotlib, pyyaml, pytest
```

The package imports as `legitimacy` and runs with the repo root on the path
(`PYTHONPATH=.`). An editable install (`pip install -e .`) also works.

## 2. Verify

```bash
PYTHONPATH=. python -m pytest tests/ -q       # 14 tests
```

The suite checks legitimacy and value bounds, the effective-prediction-error
identity `δ_eff=Lδ`, the discomfort asymmetry `φ⁻>φ⁺`, the acceptability
threshold, modal-benchmark misperception, the regime classifier, seed
reproducibility, and the mean-field reduction (collapse threshold, basins, and
the calibrated **cusp bistability**).

## 3. Reproduce

```bash
PYTHONPATH=. python scripts/run_all.py standard      # experiments, phase, ablations
PYTHONPATH=. python scripts/make_figures.py          # Figs 2–7 (Monte-Carlo)
PYTHONPATH=. python scripts/make_gt_figures.py       # phase portrait + hysteresis
PYTHONPATH=. python scripts/make_tables.py           # results / ablation / param tables
```

or `make verify` / `make reproduce` / `make paper` (see the Makefile).

Profiles (`preview` / `standard` / `publication`) change replicate counts and
grid resolution only; the mechanism parameters are shared.

## 4. Layout

```
legitimacy/           core model (importable package)
  model.py            agent loop: gated value update + fairness/legitimacy
  fairness.py         discomfort, acceptability, legitimacy update, gate
  modal.py            transmissible modal-agent benchmark
  meanfield.py        deterministic (Δ,L) reduction: QRE map, cusp, hysteresis,
                      calibrate_cusp (parameters fit from the model rates)
  network.py, metrics.py, uncertainty.py, confidence.py, scenarios.py, config.py
scripts/              run_all + experiments/phase/ablations, figure/table makers,
                      make_gt_figures (game-theory figures), plotstyle (shared grammar)
tests/                pytest verification suite
paper/                manuscript (.tex/.bib/.pdf) + figures/ + tables/
reports/              AB-comment audit, literature positioning, replication/robustness
```

## 5. The behavioural equilibrium (one paragraph)

Choices are a quantal (logistic) response to value; values learn from a
legitimacy-gated prediction error; legitimacy is a **self-reinforcing attitude**
(motivated cognition) on a tilted double well whose tilt is perceived unfairness.
Rest points are QRE / M-equilibria; under heterogeneous, weakly-transmitted
benchmarks the equilibrium is set-valued (fairness polarisation). The legitimacy
coordinate is bistable, so sweeping the benchmark traces a **fold with
hysteresis** (a cusp catastrophe) — `legitimacy/meanfield.py`, proved in the
manuscript appendix.

## 6. License & citation
MIT (see `LICENSE`). Please cite the article and this repository; release tag /
DOI to be added on publication.
