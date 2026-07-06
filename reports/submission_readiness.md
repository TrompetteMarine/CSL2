# Paper II — submission readiness

Acceptance gate (prompt §4.14). Status verified in a Python sandbox without
Docker/Julia/GitHub execution (see "Out of sandbox scope").

```
[x] Modal-agent benchmark theoretically defined (M true, Mhat perceived, r^F=f(a,Mhat); effort form behind a flag).
[x] Environment clearly interpreted (two-action reward rule; effort-reward reading available).
[x] Legitimacy gate central and tested (delta_eff = L*delta; test_effective_prediction_error, test_no_gate_ablation).
[x] Fairness discomfort asymmetry tested (test_fairness_discomfort_asymmetry).
[x] Acceptability threshold tested (test_acceptability_threshold; experiment II.2).
[x] Modal-agent misrepresentation tested (test_modal_agent_misperception; experiment II.3).
[x] Network geometry has a phase diagram (secondary phase diagram, Gamma x psi).
[x] Ablations isolate the mechanism (A1–A7, CRN, paired counterfactual intervals).
[x] ODD+D appendix complete (A.1–A.10).
[x] Scenario/parameter tables exist (param, scenario via scenarios.py, variables, comparison, ablation).
[x] Uncertainty intervals reported (Wilson + bootstrap + paired bootstrap).
[ ] Docker replication passes — NOT RUN in this sandbox (no Docker).
[x] Paper is self-contained.
[x] Paper cites Paper I only as background.
```

## Out of sandbox scope (must be completed on a full machine before submission)
- Build and run the Docker image; record image hash; run `make paper-ii` inside it.
- Optional Julia kernels for publication-scale phase sweeps (Python is sufficient
  at the reported scale).
- Insert repository URL, release tag/commit hash, and archive DOI.
- Figure 8 (robustness/sensitivity sweep): scaffolded in `robustness_summary.md`;
  reuses `simulate` + the regime classifier.

## Risks
- Ablation operating point is one contested point (defensive wrong); the phase
  diagrams provide the global map for the other regimes.
- Fairness-in-choice term (`zeta_F`) is implemented but off by default; the v1
  mechanism operates through legitimacy gating of the prediction error.
