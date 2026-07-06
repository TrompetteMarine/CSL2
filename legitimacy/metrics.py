"""Observables and the legitimacy-aware regime classifier.

Regimes (terminal): legitimate consensus, correct-but-illegitimate consensus,
defensive wrong consensus, fairness polarisation, unresolved.
"""
from __future__ import annotations

import numpy as np

REGIMES = ["legitimate", "illegitimate_correct", "defensive_wrong",
           "fairness_polarised", "unresolved"]
LEGIT, ILLEGIT, DEFENSIVE, POLARISED, UNRESOLVED = 0, 1, 2, 3, 4


def classify_regime(term_m, term_L, a_star, theta=0.9, L_hi=0.6, G_thr=0.2):
    """Classify each replication. term_m (R,M,2), term_L (R,M). Returns codes (R,)."""
    term_m = np.asarray(term_m, dtype=float)
    term_L = np.asarray(term_L, dtype=float)
    R, M, _ = term_m.shape
    a_other = 1 - a_star
    m_star = term_m[:, :, a_star]; m_other = term_m[:, :, a_other]
    meanL = term_L.mean(axis=1)
    # legitimacy polarisation = max pairwise community gap
    GL = np.zeros(R)
    for c in range(M):
        for d in range(c + 1, M):
            GL = np.maximum(GL, np.abs(term_L[:, c] - term_L[:, d]))
    all_correct = np.all(m_star >= theta, axis=1)
    all_wrong = np.all(m_other >= theta, axis=1)
    any_correct = np.any(m_star >= theta, axis=1)
    any_wrong = np.any(m_other >= theta, axis=1)
    action_split = any_correct & any_wrong
    polarised = action_split | (GL >= G_thr)

    code = np.full(R, UNRESOLVED, dtype=int)
    code[all_correct & (meanL >= L_hi)] = LEGIT
    code[all_correct & (meanL < L_hi)] = ILLEGIT
    code[all_wrong] = DEFENSIVE
    code[polarised] = POLARISED                    # polarisation takes precedence
    return code


def regime_frequencies(term_m, term_L, a_star, **kw):
    code = classify_regime(term_m, term_L, a_star, **kw)
    return {name: float(np.mean(code == i)) for i, name in enumerate(REGIMES)}


def legit_polarisation(traj_L):
    """|Lbar_1 - Lbar_2| over time, mean over reps. traj_L (R,T+1,M)."""
    traj_L = np.asarray(traj_L)
    if traj_L.shape[-1] < 2:
        return np.zeros(traj_L.shape[1])
    return np.abs(traj_L[:, :, 0] - traj_L[:, :, 1]).mean(axis=0)


def modal_polarisation(traj_Mhat):
    traj_Mhat = np.asarray(traj_Mhat)
    if traj_Mhat.shape[-1] < 2:
        return np.zeros(traj_Mhat.shape[1])
    return np.abs(traj_Mhat[:, :, 0] - traj_Mhat[:, :, 1]).mean(axis=0)
