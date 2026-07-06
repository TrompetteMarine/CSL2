# Paper II — robustness summary (v1)

This is a v1 (first working-paper) project. The qualitative regimes are
established and analytically anchored; a full multi-axis robustness sweep is the
immediate next step and is scaffolded here.

## Established (v1)
- **Analytical collapse boundary (Proposition 1):** legitimacy drifts down iff
  `ν_F·E[(DF−τ^F)_+] > ν_R·P(DF≤τ^F)`. The II.2 threshold sweep is the empirical
  counterpart (mean L: 0.00→1.00 across τ^F), so the legitimate/illegitimate
  boundary is not a single hand-picked point but follows the closed-form drift.
- **Discomfort asymmetry (II.1):** monotone in `|ΔF|` and one-sided
  (disadvantageous only), by construction `φ⁻>φ⁺`.
- **Network geometry (II.5):** the fairness-polarisation → mixing transition is
  smooth in permeability `Γ` (G^L: 0.63→0.00; mean L: 0.68→0.98), not a knife edge.

## Planned sweep (next step, mirrors Paper I Appendix D)
| Axis | Range | Question |
|------|-------|----------|
| terminal threshold θ | 0.80/0.90/0.95 | regime-area invariance |
| horizon T | 200–1000 | is legitimacy collapse a finite-horizon artefact? |
| benchmark strength M̂₀ | 0.4–0.9 | how unfair must the rule feel to delegitimise? |
| loss/recovery ν_F/ν_R | grid | collapse-boundary map vs Proposition 1 |
| permeability Γ | 0–0.5 | polarisation vs global collapse |
| random graphs | balanced/SBM/noisy-W | regimes off exact symmetry |

These reuse `legitimacy.model.simulate` and the regime classifier; only a sweep
driver (`scripts/run_robustness.py`) need be added. No result above depends on
the sweep being completed; the headline regimes (II.4, II.6) are already
Monte Carlo frequencies, not single runs.
