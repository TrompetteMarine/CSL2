"""Run Paper II experiments II.1-II.6 and save results (figures/tables read these).

II.1 advantageous vs disadvantageous unfairness    II.4 legitimacy-gated wrong consensus
II.2 acceptability-threshold sweep                 II.5 network geometry
II.3 modal-agent misrepresentation                 II.6 correct-but-illegitimate consensus
"""
import numpy as np
import common
from legitimacy import scenarios, metrics
from legitimacy.model import simulate, legitimacy_response
from legitimacy.config import SCALE

SEED = 7


def main():
    scale = common.get_scale()
    reps = SCALE[scale]["reps"]
    store, summ = {}, {}

    # II.1 -- asymmetric legitimacy response (deterministic fairness probe)
    dp = scenarios.discomfort_params()
    xs = [0.2, 0.4, 0.6]
    for x in xs:
        store[f"ii1_disadv_{x}"] = legitimacy_response(-x, dp, n_steps=120)
        store[f"ii1_adv_{x}"] = legitimacy_response(+x, dp, n_steps=120)
    summ["II.1"] = {f"x={x}": dict(drop_disadv=float(dp.L0 - store[f"ii1_disadv_{x}"][-1]),
                                   drop_adv=float(dp.L0 - store[f"ii1_adv_{x}"][-1])) for x in xs}
    store["ii1_xs"] = np.array(xs)

    # II.2 -- acceptability-threshold sweep
    base2 = scenarios.threshold_base(scale)
    taus = [0.05, 0.10, 0.20, 0.30, 0.40, 0.55, 0.70]
    meanL = [float(simulate(base2.replace(tau_F=t), reps, SEED)["term_L"].mean()) for t in taus]
    store["ii2_tau"] = np.array(taus); store["ii2_meanL"] = np.array(meanL)
    summ["II.2"] = dict(zip([f"tau={t}" for t in taus], meanL))

    # II.3 -- modal-agent misrepresentation -> fairness polarisation
    o3 = simulate(scenarios.modal_misrep(scale), reps, SEED, record_traj=True)
    store["ii3_L"] = o3["traj_L"].mean(axis=0)            # (T+1, M)
    store["ii3_Mhat"] = o3["traj_Mhat"].mean(axis=0)
    store["ii3_GL"] = metrics.legit_polarisation(o3["traj_L"])
    summ["II.3"] = dict(termL=o3["term_L"].mean(0).tolist(),
                        termMhat=o3["term_Mhat"].mean(0).tolist(),
                        GL_terminal=float(metrics.legit_polarisation(o3["traj_L"])[-1]),
                        regime=metrics.regime_frequencies(o3["term_m"], o3["term_L"], o3["a_star"]))

    # II.4 -- legitimacy-gated wrong consensus (no-gate / gate / gate+misrep)
    summ["II.4"] = {}
    for nm, p in scenarios.wrong_variants(scale).items():
        o = simulate(p, reps, SEED, record_traj=True); a = o["a_star"]
        store[f"ii4_{nm}_mstar"] = o["traj_m"][:, :, :, a].mean(axis=0)   # (T+1, M)
        store[f"ii4_{nm}_L"] = o["traj_L"].mean(axis=0)
        summ["II.4"][nm] = dict(m_star=float(o["term_m"][:, :, a].mean()),
                                meanL=float(o["term_L"].mean()),
                                regime=metrics.regime_frequencies(o["term_m"], o["term_L"], a))

    # II.5 -- network geometry
    base5, gammas = scenarios.geometry_grid(scale)
    gL, gGL = [], []
    for G in gammas:
        o = simulate(base5.replace(b_self=1 - G), reps, SEED)
        gL.append(float(o["term_L"].mean()))
        gGL.append(float(abs(o["term_L"].mean(0)[0] - o["term_L"].mean(0)[1])))
    store["ii5_gamma"] = np.array(gammas); store["ii5_meanL"] = np.array(gL); store["ii5_GL"] = np.array(gGL)
    summ["II.5"] = dict(gamma=list(gammas), meanL=gL, GL=gGL)

    # II.6 -- correct-but-illegitimate consensus
    o6 = simulate(scenarios.correct_illegit(scale), reps, SEED, record_traj=True, record_obs=True)
    a = o6["a_star"]
    store["ii6_mstar"] = o6["traj_m"][:, :, :, a].mean(axis=0)
    store["ii6_meanL"] = o6["obs"]["meanL"]
    summ["II.6"] = dict(m_star=float(o6["term_m"][:, :, a].mean()),
                        meanL=float(o6["term_L"].mean()),
                        regime=metrics.regime_frequencies(o6["term_m"], o6["term_L"], a))

    store["scale"] = scale; store["reps"] = reps
    np.savez_compressed(f"{common.RESULTS}/experiments_{scale}.npz", **store)
    common.dump_json(f"{common.RESULTS}/experiments_{scale}.json", summ)
    print("II.4:", {k: (round(v["m_star"], 2), round(v["meanL"], 2)) for k, v in summ["II.4"].items()})
    print("II.6:", round(summ["II.6"]["m_star"], 2), round(summ["II.6"]["meanL"], 2))
    print("II.3 GL_terminal:", round(summ["II.3"]["GL_terminal"], 2))
    print(f"saved experiments_{scale}.npz")


if __name__ == "__main__":
    main()
