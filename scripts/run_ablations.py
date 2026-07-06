"""Paper II ablations A1-A7 under common random numbers (Figure 7).

Operating point: the legitimacy-gated wrong-consensus setting (defensive wrong).
Each variant disables one component; the same seed fixes the exogenous
environment across variants, so the full-vs-ablation paired difference identifies
that component's causal effect on defensive wrong consensus.
"""
import numpy as np
import common
from legitimacy import scenarios, metrics, uncertainty as unc
from legitimacy.model import simulate
from legitimacy.config import SCALE

CRN_SEED = 20260618
ORDER = ["full", "A1_no_gate", "A2_true_benchmark", "A3_no_psi",
         "A4_symmetric_phi", "A5_no_adv_concern", "A6_no_lambda", "A7_no_eta"]


def variants(scale):
    b = scenarios.wrong_base(scale).replace(legitimacy_gate=True)
    return {
        "full": b,
        "A1_no_gate": b.replace(legitimacy_gate=False),
        "A2_true_benchmark": b.replace(pin_modal_true=True),
        "A3_no_psi": b.replace(psi=0.0),
        "A4_symmetric_phi": b.replace(phi_plus=b.phi_minus),
        "A5_no_adv_concern": b.replace(phi_plus=0.0),
        "A6_no_lambda": b.replace(lam=0.0),
        "A7_no_eta": b.replace(eta=0.0),
    }


def main():
    scale = common.get_scale()
    reps = min(SCALE[scale]["reps"], 300)
    V = variants(scale)
    per = {}
    for nm in ORDER:
        out = simulate(V[nm], reps, CRN_SEED)
        a = out["a_star"]
        codes = metrics.classify_regime(out["term_m"], out["term_L"], a)
        per[nm] = dict(wrong=(codes == metrics.DEFENSIVE).astype(float),
                       mstar=out["term_m"][:, :, a].mean(axis=1),
                       L=out["term_L"].mean(axis=1))
    rows, contr = {}, {}
    for nm in ORDER:
        d = per[nm]
        pw = unc.proportion_summary(d["wrong"])
        rows[nm] = dict(p_defwrong=pw, mean_mstar=unc.bootstrap_mean(d["mstar"], seed=1),
                        mean_L=unc.bootstrap_mean(d["L"], seed=2))
        if nm != "full":
            contr[nm] = unc.paired_bootstrap_diff(per["full"]["wrong"], d["wrong"], seed=3)
        print(f"[{nm:18s}] P(def-wrong)={pw['p']:.2f} [{pw['lo']:.2f},{pw['hi']:.2f}] "
              f"m*={rows[nm]['mean_mstar']['mean']:.2f} L={rows[nm]['mean_L']['mean']:.2f}")

    store = dict(order=np.array(ORDER), R=reps,
                 p_defwrong=np.array([rows[n]["p_defwrong"]["p"] for n in ORDER]),
                 p_lo=np.array([rows[n]["p_defwrong"]["lo"] for n in ORDER]),
                 p_hi=np.array([rows[n]["p_defwrong"]["hi"] for n in ORDER]),
                 mstar=np.array([rows[n]["mean_mstar"]["mean"] for n in ORDER]),
                 meanL=np.array([rows[n]["mean_L"]["mean"] for n in ORDER]),
                 d_diff=np.array([contr[n]["diff"] if n in contr else 0.0 for n in ORDER]),
                 d_lo=np.array([contr[n]["lo"] if n in contr else 0.0 for n in ORDER]),
                 d_hi=np.array([contr[n]["hi"] if n in contr else 0.0 for n in ORDER]))
    np.savez_compressed(f"{common.RESULTS}/ablations_{scale}.npz", **store)
    common.dump_json(f"{common.RESULTS}/ablations_{scale}.json",
                     {"crn_seed": CRN_SEED, "R": reps, "variants": rows, "contrasts_vs_full": contr})
    print(f"saved ablations_{scale}.npz")


if __name__ == "__main__":
    main()
