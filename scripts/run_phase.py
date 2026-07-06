"""Paper II phase diagrams (resumable, budgeted).

Main: acceptability threshold tau_F (x) x initial benchmark disagreement
|Mhat_1,0 - Mhat_2,0| (y) -> terminal regime.
Secondary: permeability Gamma (x) x modal-agent transmission psi (y) -> legitimacy gap G^L.

Writes results/phase_main_<scale>.npz and phase_secondary_<scale>.npz; re-run
until ALL DONE (set ECSL_BUDGET seconds to chunk).
"""
import os
import time
import numpy as np
import common
from legitimacy import metrics
from legitimacy.model import simulate
from legitimacy.config import Params, SCALE

BUDGET = float(os.environ.get("ECSL_BUDGET", "1e9"))


def _main_base(scale):
    n = SCALE[scale]["N_phase"]
    # learnable optimum, low permeability so a benchmark gap persists
    return Params(sizes=(n, n), T=220, mu=(0.60, 0.40), lam=0.40, eta=0.20,
                  b_self=0.95, psi=0.04, nu_F=0.05, nu_R=0.01,
                  phi_minus=1.0, phi_plus=0.25)


def run_main(scale):
    base = _main_base(scale)
    g = SCALE[scale]["grid"]; reps = SCALE[scale]["reps_phase"]
    taus = np.linspace(0.05, 0.70, g)
    diss = np.linspace(0.0, 0.60, g)
    path = f"{common.RESULTS}/phase_main_{scale}.npz"
    if os.path.exists(path):
        d = np.load(path); P = d["P"].copy(); done = d["done"].copy()
    else:
        P = np.full((g, g, 5), np.nan); done = np.zeros(g, dtype=bool)
    t0 = time.time()
    for yi, dd in enumerate(diss):
        if done[yi]:
            continue
        for xi, tau in enumerate(taus):
            p = base.replace(tau_F=float(tau), Mhat0=(0.5 - dd / 2, 0.5 + dd / 2))
            out = simulate(p, reps, 1000 + yi * 100 + xi)
            fr = metrics.regime_frequencies(out["term_m"], out["term_L"], out["a_star"])
            P[yi, xi] = [fr[r] for r in metrics.REGIMES]
        done[yi] = True
        np.savez_compressed(path, P=P, done=done, taus=taus, diss=diss)
        print(f"main row {yi+1}/{g} done")
        if time.time() - t0 > BUDGET and not done.all():
            print("[budget] pausing main"); return False
    print("main phase COMPLETE"); return True


def run_secondary(scale):
    n = SCALE[scale]["N_phase"]; g = SCALE[scale]["grid"]; reps = SCALE[scale]["reps_phase"]
    base = Params(sizes=(n, n), T=220, mu=(0.60, 0.40), lam=0.40, eta=0.20,
                  Mhat0=(0.25, 0.75), nu_F=0.05, nu_R=0.01, tau_F=0.30,
                  phi_minus=1.0, phi_plus=0.25)
    gammas = np.linspace(0.02, 0.50, g)
    psis = np.linspace(0.0, 0.30, g)
    path = f"{common.RESULTS}/phase_secondary_{scale}.npz"
    if os.path.exists(path):
        d = np.load(path); GL = d["GL"].copy(); done = d["done"].copy()
    else:
        GL = np.full((g, g), np.nan); done = np.zeros(g, dtype=bool)
    t0 = time.time()
    for yi, ps in enumerate(psis):
        if done[yi]:
            continue
        for xi, G in enumerate(gammas):
            out = simulate(base.replace(b_self=float(1 - G), psi=float(ps)), reps, 2000 + yi * 100 + xi)
            L = out["term_L"].mean(axis=0)
            GL[yi, xi] = abs(L[0] - L[1])
        done[yi] = True
        np.savez_compressed(path, GL=GL, done=done, gammas=gammas, psis=psis)
        print(f"secondary row {yi+1}/{g} done")
        if time.time() - t0 > BUDGET and not done.all():
            print("[budget] pausing secondary"); return False
    print("secondary phase COMPLETE"); return True


def main():
    scale = common.get_scale()
    d1 = run_main(scale)
    if not d1:
        print("PHASE INCOMPLETE (main)"); return
    d2 = run_secondary(scale)
    print("ALL PHASE DONE" if d2 else "PHASE INCOMPLETE (secondary)")


if __name__ == "__main__":
    main()
