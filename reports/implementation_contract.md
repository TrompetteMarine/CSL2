# Paper II — Implementation contract

## Build order (do not skip)
1. Foundations: `config.py`, `network.py` (balanced block + quotient), `confidence.py`.
2. Fairness/legitimacy: `fairness.py` (ΔF, asymmetric DF, acceptability, legitimacy update), `modal.py` (modal-agent social learning).
3. Simulator: `model.py` (vectorised over replications×agents, synchronous simultaneous update, legitimacy gate).
4. Observables + regimes: `metrics.py`.
5. Verification: 8 tests in `tests/test_legitimacy.py`, all green, **before** any experiment.
6. Experiments II.1–II.6: `scripts/run_experiments.py` (elementary II.1–II.4 first).
7. Figures/tables from saved results only (`make_figures.py`, `make_tables.py`).

## Acceptance criteria
- Legitimacy `L ∈ [0,1]` always; `δ^eff = L·δ` exactly; `DF` asymmetry `φ⁻>φ⁺ ⇒ |ΔL|_{disadv} > |ΔL|_{adv}`.
- Acceptability threshold partitions outcomes; modal-agent misperception yields `Lbar_1 ≠ Lbar_2`.
- Network update preserves the balanced-block quotient for `Mhat`.
- Regime classifier returns the four named regimes on hand-built inputs.
- Seed reproducibility: identical seed ⇒ identical outputs.
- Legitimacy-gated wrong consensus is **more persistent** than the ungated baseline (II.4).

## Outputs
- `results/*.npz|json`, `manuscript/figures/fig*.pdf`, `manuscript/tables/*.tex`,
  reports (`claim_audit`, `replication_audit`, `robustness_summary`).

## Reproduce
```bash
cd code
python -m pytest tests/ -q
python scripts/run_experiments.py
python scripts/make_figures.py
python scripts/make_tables.py
```
