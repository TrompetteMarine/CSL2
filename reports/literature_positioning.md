# Paper II — literature positioning (from the shared corpus + SOTA)

Mined the uploaded corpus and its bibliographies, cross-checked against current
SOTA, and positioned the paper across three streams. New references added to the
bib and woven into Background and the enriched Conclusion.

## Stream 1 — Behavioural game theory & equilibrium
- **QRE / regular QRE**: McKelvey & Palfrey (1995); Goeree, Holt & Palfrey (2005, *Exp. Econ.*, axiomatic regular QRE). The paper's choice map is a quantal response; rest points are QRE fixed points.
- **Anomalies motivating stochastic-choice GT**: Goeree & Holt (2001, *AER*, "Ten Little Treasures").
- **Set-valued equilibrium**: Goeree & Louis (2021, *AER*, M-equilibrium) — nests regular QRE; our fairness-polarisation is an M-equilibrium (beliefs not consequentially unbiased across communities).
- **Nonequilibrium / level-k**: Nagel (1995); Camerer–Ho–Chong (2004); Crawford, Costa-Gomes & Iriberri (2013, *JEL*). Grounds benchmark misperception and the persistence of initial (early-lead) responses.
- **Learning in games**: Erev & Roth (1998); Camerer & Ho (1999, EWA). Our gated update is a reinforcement/experience-weighted learner; the mean-field is its stochastic-approximation ODE limit (Borkar 2008).
- *Our addition:* the regimes are **selected by a gated learning dynamic**, characterised completely (existence/dichotomy + transcritical collapse) — dynamics the equilibrium concepts above leave open.

## Stream 2 — Fairness, endogenous preferences & norms
- **Reference-dependent fairness**: Rabin (1993); Fehr & Schmidt (1999); Bolton & Ockenfels (2000); Charness & Rabin (2002).
- **Heterogeneous fairness ideals** (the empirical basis for divergent benchmarks): Cappelen, Hole, Sørensen & Tungodden (2007, *AER*); André (2024, *ReStud*, "Shallow Meritocracy").
- **Endogenous preferences / culturally transmitted norms**: Bowles (1998, *JEL*); Tyler (2006).
- **Self-fulfilling, path-dependent disagreement**: Bouchaud & Farmer (2023, *JPE*) — "quasi-non-ergodicity," agents "disagree forever." Our fairness polarisation is the social analogue: history, not payoffs, selects the legitimacy regime.
- *Our addition:* the fairness **reference is a transmissible state**, so grievance diffuses on the network and polarises legitimacy under identical payoffs.

## Stream 3 — Neuroeconomics & cognitive neuroscience
- **What OFC/PFC computes (the live debate)**: explicit economic value — Padoa-Schioppa & Conen (2017, *Neuron*) — versus latent task-state / cognitive map — Hayden & Niv (2021, *Behav. Neurosci.*). Legitimacy is a **state-like context variable** of the second kind.
- **Reversal learning**: Izquierdo, Brigman, Radke, Rudebeck & Holmes (2017, *Neuroscience*). Defensive-wrong consensus = socially-induced failure of reversal learning (not lesion-induced).
- **Other-referenced / gated prediction error**: Joiner et al. (2017); plus prefrontal RPE-readout gating (2025, arXiv) from the SOTA web scout.
- *Our addition:* the gate is stated as a functional analogy (no implementation claim), turned into model-based regressors and lab predictions.

## Where we are (one line)
The field treats the **authority of evidence as fixed**; this paper makes it an
**endogenous, learnable, transmissible state**, derives the resulting behavioural
equilibria and their collapse bifurcation, and yields the policy dissociation:
restoring *information* fails where only restoring *legitimacy* reopens learning.

## Corpus items not directly cited (noted for completeness)
- Baron, *Second-order probabilities and belief functions* — supports reading $\hat M$ as a second-order belief / ambiguity object (future extension).
- Network-theory notes — community geometry (already used).
- W10 Pattern Mining, Lectures.pdf — not directly relevant to this paper.
