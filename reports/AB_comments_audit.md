# AB-comments spirit applied to Paper II (Legitimacy-gated learning)

Same rubric as Paper I. Status: **✓ addressed** · **◐ scoped extension** ·
**△ added/strengthened this pass**. Locations refer to the 16-page build.

| # | AB concern | Status | Where / how |
|---|-----------|--------|-------------|
| 1 | What is genuinely new | ✓ | *Contributions* + *Background/positioning*: reward feedback teaches only through an endogenous **legitimacy gate**, fairness judged against a **transmissible modal-agent benchmark** with **asymmetric discomfort**; positioned explicitly against the Paper I credibility backbone ("there, whose action is believed; here, whether feedback is admitted as evidence at all"). |
| 2 | Neuroscience as inspiration, direct not defensive | ✓ / △ | *Background*: the gate is a modulation of the evidential weight of reward prediction errors; self- vs other-referenced PE up/down-weighting (Joiner et al. 2017). Already direct; a scan confirmed **no over-defensive hedging** ("we stress / do not claim / …" absent). |
| 3 | Latent, inferred variables (not observed) | ✓ / △ | Legitimacy $L$ and the perceived benchmark $\hat M$ are **latent by construction**; confidence is "communicated or behaviourally inferred, not objective accuracy." **New:** testable prediction (v) treats $L$ and $\hat M$ as latent estimands recovered from ratings/RT/reports. |
| 4 | Channels may interact | ◐ | Credibility and benchmark channels are defined separately (re-used backbone); interaction flagged with the other extensions. |
| 5 | Static vs adaptive networks | ◐ | *Limitations*: fixed networks; endogenous rewiring named as an extension. |
| 6 | Latent task states / cognitive maps | ◐ | Model tracks action values + legitimacy + benchmark, not latent task states; consistent with Paper I's scoping. |
| 7 | Gate affects only learning, not exploration | ◐ | Legitimacy gates the **private prediction error**; extending the gate to exploration is a natural addition (*Limitations*). |
| 8 | Normative benchmark | △ | **New "A normative benchmark" paragraph:** the ungated learner ($L\equiv1$) is the epistemic ideal; at the collapse point, removing the gate lifts optimal-action mass $0.01\to0.91$ and cuts defensive-wrong $P=1.00\to0.07$ (Table 6) — a ~90-point welfare cost of illegitimacy, with the gate's protective role noted. |
| 9 | Parameters with cognitive meaning | △ | **New "Cognitive interpretation" paragraph:** $\phi^->\phi^+$ loss-averse self-serving fairness; $\tau^F$ tolerance for unfairness; $\nu_F,\nu_R$ delegitimation/repair speed; $\psi$ susceptibility to transmitted grievance; $\hat M$ a possibly-distorted entitlement belief; $L$ the evidential weight granted to feedback. |
| 10 | Lab-testable predictions | △ | **New "Testable predictions" paragraph** with five predictions (i)–(v): fairness-without-payoff changes learning rate; loss/gain asymmetry in legitimacy erosion vs repair; hysteretic threshold collapse; benchmark-transmission polarisation; latent-legitimacy model-based estimation. Each mapped to a reward-learning / public-goods paradigm with a fairness manipulation. |
| minor | Early hypothesis | ✓ | Headline stated in abstract and intro ("legitimacy gating makes wrong consensus persistent"). |
| minor | Schematic figure | ✓ | Fig. 1 (mechanism, TikZ) + regime-map (Fig. 3). |
| minor | Reads like a spec / proofs inline | △ | **Four proofs moved to Appendix A ("Proofs")**; statements kept inline with a pointer, matching Paper I. |
| minor | Fewer, clearer hypotheses | ✓ | Results organised as II.1–II.6 + two phase diagrams + ablations, each a single claim. |

**Bottom line.** Comments 1, 2, 3 were already met; 4–7 are scoped extensions
(consistent with Paper I); **8, 9, 10 and the proofs-placement minor are added or
strengthened this pass** — the same three gaps that Paper I had. Paper II now
carries a normative benchmark, cognitive parameter readings, five lab-testable
predictions, and an appendix Proofs section.

## Lab program (the "latent" story for Paper II)
- **Manipulate fairness, hold payoffs fixed** (shift the communicated modal-agent
  benchmark) → measure a reduced learning rate from identical reward feedback
  under low perceived legitimacy.
- **Asymmetry test:** below-benchmark outcomes should erode legitimacy faster
  than equal above-benchmark outcomes restore it ($\phi^->\phi^+$).
- **Threshold/hysteresis:** legitimacy stable under small unfairness, sharp drop
  past $\tau^F$.
- **Transmission → polarisation:** two weakly connected groups seeded with
  different benchmarks diverge in legitimacy (and choice) under identical payoffs;
  divergence controlled by permeability.
- **Latent estimation:** recover $L,\hat M$ per trial from acceptance ratings, RT
  and fairness reports; test the gated-update model against an ungated learner.

## Plots
Figures 1–7 + regime map were checked: consistent serif/CM style, Wilson and
paired-bootstrap intervals shown, legends legible, TikZ mechanism clean. No
over-augmentation. (fig\_regimemap.pdf was regenerated from its `.tex`.)
Left as-is; quality matches Paper I.

## Not done (flagged, not silently expanded)
A generative model in which agents *infer* others' legitimacy/benchmarks from
observable cues (rather than reading $\hat M$ directly) is a modelling step, not a
text change — flagged for a future revision, mirroring the Paper I decision.
