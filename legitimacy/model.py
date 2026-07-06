"""Legitimacy-gated social-learning simulator (Paper II core).

Synchronous trials with a simultaneous post-outcome update. Within a trial:
build the anticipatory credibility field; form augmented values; draw the
process-based choice and endogenous confidence; draw rewards; evaluate fairness
discrepancy, discomfort and the legitimacy update from the pre-update state;
revise values using the **legitimacy-gated** private prediction error plus the
retrospective social prediction error; and update the perceived modal agent.
Arms are indexed 0 (arm 1, upper boundary) and 1 (arm 2).
"""
from __future__ import annotations

import numpy as np
from scipy.special import expit

from .config import Params
from .network import make_membership, community_onehot, community_means, field_block
from . import confidence as confmod
from . import fairness as fair
from .modal import modal_update


def simulate(params: Params, n_reps: int, seed: int,
             record_traj: bool = False, record_obs: bool = False):
    rng = np.random.default_rng(seed)
    p = params
    sizes = np.asarray(p.sizes, dtype=int)
    M, N, R = p.M, p.N, n_reps
    g = make_membership(sizes)
    Csel = community_onehot(g, M)
    B = p.coupling()
    bc_over_Nc = (np.diag(B) / sizes)[g]                  # per-agent self-weight
    mu = np.asarray(p.mu, dtype=float)
    a_star = int(np.argmax(mu)); mu_star = mu.max()
    arms = (0, 1)

    Q = np.repeat(p.q_init_array()[g][None], R, axis=0).copy()      # (R,N,2)
    L = np.full((R, N), p.L0 if p.legitimacy_gate else 1.0)
    if p.pin_modal_true:                                            # ablation A2
        Mhat = np.full((R, N), p.M_true)
    else:
        Mhat = np.repeat(p.mhat0_array()[g][None], R, axis=0).copy()    # (R,N)
    A_prev = np.full((R, N), -1, dtype=int)
    C_prev = np.zeros((R, N))
    regret = np.zeros(R)

    if record_traj:
        tm = np.zeros((R, p.T + 1, M, 2)); tL = np.zeros((R, p.T + 1, M))
        tMh = np.zeros((R, p.T + 1, M)); tL[:, 0] = community_means(L, Csel, sizes)
        tMh[:, 0] = community_means(Mhat, Csel, sizes)
    if record_obs:
        ob = {k: np.zeros(p.T) for k in
              ("meanL", "m_star", "DF", "accept_rate", "discount_gap", "GL")}

    for t in range(p.T):
        # 1. anticipatory credibility field --------------------------------
        if t == 0:
            S = np.zeros((R, N, 2))
        else:
            cw = np.stack([(A_prev == k).astype(float) * C_prev for k in arms], axis=-1)
            phi = np.stack([community_means(cw[:, :, k], Csel, sizes) for k in arms], axis=-1)
            S = field_block(phi, B, g)
        # 2. augmented value, drift ----------------------------------------
        V = Q + p.lam * S
        if p.fairness_in_choice and p.zeta_F:                       # optional, off in v1
            exp_df = fair.discomfort(mu[None, None, :] - Mhat[:, :, None],
                                     p.phi_minus, p.phi_plus)
            V = V - p.zeta_F * exp_df
        v = p.beta * (V[:, :, 0] - V[:, :, 1])
        # 3. choice + confidence -------------------------------------------
        p_up = expit(2.0 * v * p.a_thr / (p.sigma ** 2))
        up = rng.random((R, N)) < p_up
        A = np.where(up, 0, 1)
        C = confmod.balance_of_evidence(p_up)
        # 4. reward ---------------------------------------------------------
        Rdraw = (rng.random((R, N)) < mu[A]).astype(float)
        # 5. fairness + legitimacy (from pre-update state) -----------------
        rF = fair.fair_benchmark(A, Mhat, p)
        dF = fair.fairness_discrepancy(Rdraw, rF)
        DF = fair.discomfort(dF, p.phi_minus, p.phi_plus)
        L_pre = L
        L_new = fair.legitimacy_update(L, DF, p.tau_F, p.nu_F, p.nu_R) \
            if p.legitimacy_gate else np.ones((R, N))
        # 6. legitimacy-gated private prediction error ---------------------
        Q_chosen = np.take_along_axis(Q, A[:, :, None], axis=2)[:, :, 0]
        delta = Rdraw - Q_chosen
        delta_eff = fair.gated_prediction_error(L_pre, delta)
        inc = np.zeros((R, N, 2))
        for k in arms:
            inc[:, :, k] += (A == k) * p.alpha * delta_eff
        # 7. retrospective social channel (self excluded) ------------------
        Ind = np.stack([(A == k).astype(float) for k in arms], axis=-1)
        m_now = np.stack([community_means(Ind[:, :, k], Csel, sizes) for k in arms], axis=-1)
        rbar = np.stack([community_means(Ind[:, :, k] * Rdraw, Csel, sizes) for k in arms], axis=-1)
        cmass = np.stack([community_means(Ind[:, :, k] * C, Csel, sizes) for k in arms], axis=-1)
        Sf = field_block(m_now, B, g); Rf = field_block(rbar, B, g); Cf = field_block(cmass, B, g)
        for k in arms:
            self_ind = (A == k).astype(float)
            wk = Sf[:, :, k] - bc_over_Nc * self_ind
            rk = Rf[:, :, k] - bc_over_Nc * self_ind * Rdraw
            ck = Cf[:, :, k] - bc_over_Nc * self_ind * C
            dsoc = rk - wk * Q[:, :, k]
            Cbar = ck / (wk + p.eps_soc)
            inc[:, :, k] += p.eta * p.gamma * np.power(np.clip(Cbar, 0, 1), p.omega) * dsoc
        # 8. simultaneous updates ------------------------------------------
        Q = np.clip(Q + inc, 0.0, 1.0)
        if not p.pin_modal_true:
            Mhat = modal_update(Mhat, B, g, Csel, sizes, p.psi, rng, p.xi_sd)
        L = L_new
        regret += (mu_star - mu[A]).sum(axis=1)
        A_prev, C_prev = A, C
        # bookkeeping -------------------------------------------------------
        if record_traj:
            tm[:, t + 1] = m_now
            tL[:, t + 1] = community_means(L, Csel, sizes)
            tMh[:, t + 1] = community_means(Mhat, Csel, sizes)
        if record_obs:
            ob["meanL"][t] = L.mean(); ob["m_star"][t] = (A == a_star).mean()
            ob["DF"][t] = DF.mean(); ob["accept_rate"][t] = (DF <= p.tau_F).mean()
            ob["discount_gap"][t] = (np.abs(delta) - np.abs(delta_eff)).mean()
            Lc = community_means(L, Csel, sizes)
            ob["GL"][t] = np.abs(Lc[:, 0] - Lc[:, 1]).mean() if M >= 2 else 0.0

    term_m = m_now
    term_L = community_means(L, Csel, sizes)
    term_Mhat = community_means(Mhat, Csel, sizes)
    out = dict(term_m=term_m, term_L=term_L, term_Mhat=term_Mhat,
               regret=regret, a_star=a_star, mu_star=mu_star)
    if record_traj:
        tm[:, 0] = tm[:, 1]
        out.update(traj_m=tm, traj_L=tL, traj_Mhat=tMh)
    if record_obs:
        out.update(obs=ob)
    return out


def legitimacy_response(delta_f, params: Params, n_steps=200, L0=None):
    """Deterministic legitimacy trajectory under a fixed fairness discrepancy.

    Drives ΔF constant (a controlled fairness probe) and iterates the legitimacy
    update; used by Experiment II.1 to expose the discomfort asymmetry.
    """
    p = params
    L = float(p.L0 if L0 is None else L0)
    df = fair.discomfort(np.array(delta_f), p.phi_minus, p.phi_plus)
    traj = np.empty(n_steps + 1); traj[0] = L
    for t in range(n_steps):
        L = float(fair.legitimacy_update(np.array(L), df, p.tau_F, p.nu_F, p.nu_R))
        traj[t + 1] = L
    return traj
