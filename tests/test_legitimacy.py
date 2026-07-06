"""Verification suite for the legitimacy-gating model (Paper II).

Covers the prescribed checks: legitimacy bounds; effective prediction error;
fairness-discomfort asymmetry; acceptability threshold; modal-agent
misperception; network/quotient updates; regime classifier; seed
reproducibility; plus bounded values and row-stochasticity.
"""
import numpy as np

from legitimacy import metrics, network, fairness as fair
from legitimacy.config import Params
from legitimacy.model import simulate, legitimacy_response
from legitimacy import scenarios
from legitimacy.network import balanced_block_W, community_onehot, community_means, field_block


# 1 -----------------------------------------------------------------------
def test_legitimacy_and_value_bounds():
    p = Params(sizes=(40, 40), T=80, Mhat0=(0.7, 0.7), nu_F=0.1)
    out = simulate(p, n_reps=6, seed=1, record_traj=True)
    assert out["traj_L"].min() >= -1e-9 and out["traj_L"].max() <= 1 + 1e-9
    assert out["term_m"].min() >= 0 and np.allclose(out["term_m"].sum(axis=2), 1.0)


# 2 -----------------------------------------------------------------------
def test_effective_prediction_error():
    L = np.array([0.0, 0.5, 1.0]); d = np.array([0.4, -0.2, 0.9])
    assert np.allclose(fair.gated_prediction_error(L, d), L * d)
    # gate off keeps legitimacy at 1 throughout, so delta_eff == delta
    p = Params(sizes=(30, 30), T=40, legitimacy_gate=False, Mhat0=(0.9, 0.9))
    out = simulate(p, n_reps=3, seed=2, record_traj=True)
    assert np.allclose(out["traj_L"], 1.0)


# 3 -----------------------------------------------------------------------
def test_fairness_discomfort_asymmetry():
    p = scenarios.discomfort_params()           # phi_minus=1.0 > phi_plus=0.25
    x = 0.6
    L_disadv = legitimacy_response(-x, p, n_steps=60)   # DF = 0.60 > tau -> falls
    L_adv = legitimacy_response(+x, p, n_steps=60)      # DF = 0.15 <= tau -> stays
    drop_disadv = p.L0 - L_disadv[-1]
    drop_adv = p.L0 - L_adv[-1]
    assert drop_disadv > drop_adv
    assert drop_adv <= 1e-9                              # advantageous does not erode legitimacy


# 4 -----------------------------------------------------------------------
def test_acceptability_threshold():
    p = scenarios.discomfort_params()
    L0 = np.array(0.5)
    L_acc = fair.legitimacy_update(L0, np.array(0.2), p.tau_F, p.nu_F, p.nu_R)    # DF<=tau
    L_unacc = fair.legitimacy_update(L0, np.array(0.6), p.tau_F, p.nu_F, p.nu_R)  # DF>tau
    assert L_acc > 0.5 and L_unacc < 0.5


# 5 -----------------------------------------------------------------------
def test_modal_agent_misperception():
    # low permeability preserves a benchmark gap; full mixing erodes it.
    p_iso = scenarios.modal_misrep("preview", b_self=0.98).replace(T=100, xi_sd=0.0)
    p_mix = p_iso.replace(b_self=0.50)
    g_iso = abs(np.diff(simulate(p_iso, 4, 1)["term_Mhat"], axis=1)).mean()
    g_mix = abs(np.diff(simulate(p_mix, 4, 1)["term_Mhat"], axis=1)).mean()
    assert g_iso > 0.1
    assert g_mix < g_iso


# 6 -----------------------------------------------------------------------
def test_network_quotient_field_exact():
    sizes = (8, 6, 5)
    B = np.array([[0.6, 0.25, 0.15], [0.2, 0.7, 0.1], [0.3, 0.1, 0.6]])
    W, g = balanced_block_W(sizes, B)
    assert network.is_row_stochastic(W) and network.is_row_stochastic(B)
    Csel = community_onehot(g, len(sizes)); sizes_arr = np.asarray(sizes)
    rng = np.random.default_rng(0)
    Mhat = rng.random((4, sum(sizes)))
    Mbar = community_means(Mhat, Csel, sizes_arr)
    fb = field_block(Mbar, B, g)            # community quotient field
    fd = Mhat @ W.T                          # explicit balanced-W average
    assert np.allclose(fb, fd, atol=1e-12)


# 7 -----------------------------------------------------------------------
def test_regime_classifier():
    correct = np.array([[[0.95, 0.05], [0.93, 0.07]]])
    wrong = np.array([[[0.05, 0.95], [0.07, 0.93]]])
    assert metrics.classify_regime(correct, np.array([[0.8, 0.85]]), 0)[0] == metrics.LEGIT
    assert metrics.classify_regime(correct, np.array([[0.2, 0.25]]), 0)[0] == metrics.ILLEGIT
    assert metrics.classify_regime(wrong, np.array([[0.2, 0.2]]), 0)[0] == metrics.DEFENSIVE
    assert metrics.classify_regime(correct, np.array([[0.85, 0.2]]), 0)[0] == metrics.POLARISED


# 8 -----------------------------------------------------------------------
def test_seed_reproducibility():
    p = Params(sizes=(50, 50), T=60, Mhat0=(0.6, 0.6))
    a = simulate(p, 8, 123); b = simulate(p, 8, 123); c = simulate(p, 8, 124)
    assert np.array_equal(a["term_m"], b["term_m"]) and np.allclose(a["term_L"], b["term_L"])
    assert not np.array_equal(a["term_m"], c["term_m"])


# 9 -- modal-agent bounds (incl. idiosyncratic noise) ----------------------
def test_modal_agent_bounds():
    p = Params(sizes=(40, 40), T=80, Mhat0=(0.1, 0.9), psi=0.2, xi_sd=0.1)
    out = simulate(p, 5, 1, record_traj=True)
    assert out["traj_Mhat"].min() >= -1e-9 and out["traj_Mhat"].max() <= 1 + 1e-9


# 10 -- no-gate ablation restores correction (A1) --------------------------
def test_no_gate_ablation():
    b = scenarios.wrong_base("preview").replace(legitimacy_gate=True)
    a = simulate(b, 40, 5)
    gate_m = a["term_m"][:, :, a["a_star"]].mean()
    ng = simulate(b.replace(legitimacy_gate=False), 40, 5)
    nogate_m = ng["term_m"][:, :, ng["a_star"]].mean()
    assert nogate_m - gate_m > 0.4          # removing the gate corrects the wrong lead


# 11 -- true-modal-benchmark ablation pins Mhat (A2) -----------------------
def test_true_modal_benchmark_ablation():
    p = Params(sizes=(30, 30), T=50, Mhat0=(0.2, 0.8), M_true=0.5,
               pin_modal_true=True, psi=0.3)
    out = simulate(p, 4, 1, record_traj=True)
    assert np.allclose(out["traj_Mhat"], 0.5)
    # effort-mode benchmark differs across arms
    eff = Params(benchmark_mode="effort", b0=0.1, b1=0.3, b2=1.0, high_effort_arm=0)
    r_high = fair.fair_benchmark(np.array([0]), np.array([0.4]), eff)
    r_low = fair.fair_benchmark(np.array([1]), np.array([0.4]), eff)
    assert np.isclose(float((r_high - r_low).item()), 0.3)


# 12 -- mean-field reduction: bounds, collapse threshold, basins -----------
def test_meanfield_bounds_and_collapse():
    from legitimacy import meanfield as mf
    p = Params()
    qs, qo, L = mf.step(0.55, 0.45, 1.0, p, 0.5)
    assert 0.0 <= L <= 1.0 and 0.0 <= qs <= 1.0 and 0.0 <= qo <= 1.0
    # legitimacy is preserved below and collapses above the analytic threshold
    a = int(np.argmax(p.mu)); mu_a = p.mu[a]
    Mc = (p.tau_F + p.nu_R * mu_a / (p.nu_F * (1 - mu_a))) / p.phi_minus
    L_lo = mf.terminal_legitimacy(p, [Mc - 0.15])[0]
    L_hi = mf.terminal_legitimacy(p, [Mc + 0.15])[0]
    assert L_lo > 0.9 and L_hi < 0.1


def test_meanfield_basins():
    from legitimacy import meanfield as mf
    p = Params()
    dv = np.linspace(-0.8, 0.8, 25); lv = np.linspace(0.05, 1.0, 15)
    R_lo = mf.basin_map(p, 0.2, dv, lv)     # modest benchmark -> all legitimate
    R_hi = mf.basin_map(p, 0.7, dv, lv)     # high benchmark -> illegitimate present
    assert (R_lo == 0).mean() > 0.9
    assert set(np.unique(R_hi)).issubset({1, 2}) and (R_hi == 2).any()


def test_meanfield_cusp_bistability():
    """Motivated-cognition legitimacy: interior single attractor at the extremes,
    bistable (3 interior equilibria) in an intermediate benchmark band."""
    from legitimacy import meanfield as mf
    p = Params()
    a = int(np.argmax(p.mu)); p1 = mf.choice_prob_star(p.mu[a] - p.mu[1 - a], p)
    eq_lo = mf.legit_equilibria(0.20, p, p1=p1)
    eq_mid = mf.legit_equilibria(0.55, p, p1=p1)
    eq_hi = mf.legit_equilibria(0.90, p, p1=p1)
    assert len(eq_lo) == 1 and len(eq_hi) == 1
    assert len([e for e in eq_mid if e[1]]) == 2      # two stable wells
    assert any(not e[1] for e in eq_mid)              # and a saddle
    assert all(0.0 < L < 1.0 for L, _ in eq_mid)      # all interior
