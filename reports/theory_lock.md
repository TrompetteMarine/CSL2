# Paper II — Theory lock (v1 frozen design)

*Legitimacy Gating in Social Learning.* This document freezes the v1 model.
Changes require an explicit edit here first.

## Research question
**Primary:** When do agents stop treating reward feedback as valid evidence
because the reward rule is perceived as unfair?
**Secondary:** How does network geometry modulate the diffusion of fairness
benchmarks, legitimacy collapse, and defensive wrong consensus?

## Relation to Paper I
Builds on the credibility-transmission mechanism of Bontemps & Banerjee (2026),
but studies a different behavioural layer: whether reward feedback is accepted as
legitimate evidence. Paper II is self-contained and re-implements every
component; it shares **no code, configs, or figures** with Paper I.

## Agent states (per agent i, trial t)
`Q_i,t(a)` value; `A_i,t` action; `R_i,t` reward; `C_i,t` confidence;
`S_i,t(a)` credibility-weighted social signal; `L_i,t` legitimacy of the reward
rule; `Mhat_i,t` perceived modal-agent fairness benchmark; `ΔF_i,t` fairness
discrepancy; `DF_i,t` fairness discomfort; `δ_i,t` raw prediction error;
`δ_i,t^eff` legitimacy-gated prediction error.

## Frozen equations
1. Anticipatory signal: `S_i,t(a) = Σ_j W_ij 1{A_{j,t-1}=a} C_{j,t-1}` (self-included; exact quotient under balanced blocks).
2. Choice: `v = β(V_i,t(1)-V_i,t(2))`, `V_i,t(a)=Q_i,t(a)+λ S_i,t(a) [− ζ_F·DF-term, behind a flag]`; `P(A=1)=logistic(2 v a/σ²)`.
3. Confidence (balance of evidence): `C_i,t = |2 P(A=1) − 1|`, bounded in (0,1).
4. Fairness benchmark: `rhat^F_i,t = Mhat_i,t`; discrepancy `ΔF_i,t = R_i,t − Mhat_i,t`.
5. Asymmetric discomfort: `DF_i,t = φ⁻[−ΔF_i,t]_+ + φ⁺[ΔF_i,t]_+`, with `φ⁻ > φ⁺ ≥ 0`.
6. Acceptability threshold `τ^F`: outcome acceptable iff `DF ≤ τ^F`.
7. Legitimacy update: `L_{i,t+1} = Π_{[0,1]}( L_i,t − ν_F(DF − τ^F)_+ + ν_R·1{DF ≤ τ^F} )`.
8. Gated prediction error: `δ_i,t = R_i,t − Q_i,t(A_i,t)`, `δ^eff_i,t = L_i,t · δ_i,t`.
9. Value update: `Q_{i,t+1}(a)=Π_{[0,1]}( Q_i,t(a) + 1{A_i,t=a} α δ^eff_i,t + η g(Cbar_i,t(a)) δ^soc_i,t(a) )`.
10. Modal-agent social learning: `Mhat_{i,t+1}=Π_{[0,1]}( (1−ψ)Mhat_i,t + ψ Σ_d B_{g(i),d} Mbar_{d,t} + ξ_i,t )`.

Timing: synchronous; all prediction errors and the legitimacy gate use
**pre-update** `L_i,t`, `Q_i,t`; updates applied simultaneously.

## Regimes (terminal)
- **Legitimate consensus:** every community `m_c(a*) ≥ θ` and mean legitimacy high (`≥ L_hi`).
- **Correct but illegitimate consensus:** `m_c(a*) ≥ θ` but mean legitimacy low (`≤ L_lo`).
- **Defensive wrong consensus:** every community `m_c(a≠a*) ≥ θ` and legitimacy low.
- **Fairness polarisation:** communities differ in legitimacy (`|Lbar_1−Lbar_2| ≥ G`) or in action.
- **Unresolved:** otherwise. (Modal-agent polarisation `||Mhat_1−Mhat_2||` is reported as a diagnostic.)

## Experiments (run order: elementary first)
II.1 advantageous vs disadvantageous unfairness (asymmetric `ΔL`);
II.2 acceptability-threshold sweep (`τ^F`);
II.3 modal-agent misrepresentation → fairness polarisation;
II.4 legitimacy-gated wrong consensus (gate off / on / on+misrep);
II.5 network geometry (cohesion/permeability);
II.6 correct-but-illegitimate consensus.

## Excluded from v1
strategic lying; endogenous rewiring; multi-arm; non-stationary rewards;
institutional rule change; empirical calibration; categorical identity groups;
competing fairness philosophies; per-period equilibrium solving; language agents.
One fairness benchmark only (modal-agent reward benchmark with asymmetric
discomfort). Fairness-in-choice term `ζ_F` is implemented behind a config flag,
**off** by default; the v1 mechanism operates through legitimacy gating of the
prediction error.
