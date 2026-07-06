# ODD-to-code map — Paper II

| Model element | Equation (manuscript) | Code | Test |
|---|---|---|---|
| Anticipatory credibility field | `S_i,t(a)=Σ_d B_cd φ_d,t-1(a)` | `legitimacy/network.py: field_block`; `model.py` | `test_network_quotient_field_exact` |
| Augmented value & choice | Eq. (choice) | `legitimacy/model.py: simulate` | `test_seed_reproducibility` |
| Confidence (balance of evidence) | `C=|2P(A=1)-1|` | `legitimacy/confidence.py` | (bounds) `test_legitimacy_and_value_bounds` |
| Fairness discrepancy | Eq. (1) `ΔF=R-M̂` | `legitimacy/fairness.py: fairness_discrepancy` | `test_acceptability_threshold` |
| Asymmetric discomfort | Eq. (2) `DF` | `legitimacy/fairness.py: discomfort` | `test_fairness_discomfort_asymmetry` |
| Legitimacy update | Eq. (3) | `legitimacy/fairness.py: legitimacy_update` | `test_acceptability_threshold` |
| Gated prediction error | Eq. (4) `δ^eff=Lδ` | `legitimacy/fairness.py: gated_prediction_error` | `test_effective_prediction_error` |
| Value update | Eq. (5) | `legitimacy/model.py: simulate` | `test_legitimacy_and_value_bounds` |
| Modal-agent transmission | Eq. (6) | `legitimacy/modal.py: modal_update` | `test_modal_agent_misperception` |
| Regime classifier | §4 | `legitimacy/metrics.py: classify_regime` | `test_regime_classifier` |
| Legitimacy drift (Prop. 1) | §3 | `fairness.legitimacy_update` (expectation) | matches II.2 sweep |

Scenarios (per-experiment parameters): `legitimacy/scenarios.py`.
Experiment drivers: `scripts/run_experiments.py` (II.1–II.6).
Figures/tables: `scripts/make_figures.py`, `scripts/make_tables.py`.
