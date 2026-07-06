"""Deterministic mean-field reduction of the legitimacy-gated model.

Large-population limit for one community with a scalar fairness benchmark
``Mhat`` treated as a parameter.  The state collapses to the two chosen-arm
values (equivalently the value contrast ``Delta = Q(a*) - Q(other)``) and the
shared legitimacy ``L``.  Choices follow the exact DDM logistic; rewards, the
gated value update and the legitimacy update are taken in expectation.  This is
the game-theoretic companion to the stochastic model: its rest points are the
(choice, legitimacy) equilibria and its flow gives the phase portrait.
"""
from __future__ import annotations

import numpy as np
from scipy.special import expit


def choice_prob_star(Delta, p):
    """P(choose the optimal arm a*) under the DDM logistic, contrast Delta."""
    v = p.beta * Delta
    return expit(2.0 * v * p.a_thr / p.sigma ** 2)


def _legit_drift(p1, L, Mhat, p):
    """Expected per-trial legitimacy change E[dL] at (choice p1, benchmark Mhat)."""
    mu = np.asarray(p.mu, float)
    a = int(np.argmax(mu)); o = 1 - a
    d0 = p.phi_minus * Mhat            # discomfort on bad news  (R=0, disadvantageous)
    d1 = p.phi_plus * (1.0 - Mhat)     # discomfort on good news (R=1, advantageous)
    tau = p.tau_F

    def terms(mu_k):
        exc = (1 - mu_k) * max(d0 - tau, 0.0) + mu_k * max(d1 - tau, 0.0)
        acc = (1 - mu_k) * (1.0 if d0 <= tau else 0.0) + mu_k * (1.0 if d1 <= tau else 0.0)
        return exc, exc if False else acc
    exc_s, acc_s = terms(mu[a]); exc_o, acc_o = terms(mu[o])
    exc = p1 * exc_s + (1 - p1) * exc_o
    acc = p1 * acc_s + (1 - p1) * acc_o
    return -p.nu_F * exc + p.nu_R * acc


def step(qs, qo, L, p, Mhat):
    """One mean-field update of (Q(a*), Q(other), L) at fixed benchmark Mhat."""
    mu = np.asarray(p.mu, float)
    a = int(np.argmax(mu)); o = 1 - a
    Delta = qs - qo
    p1 = choice_prob_star(Delta, p)                       # P(choose a*)
    qs_n = np.clip(qs + p.alpha * L * p1 * (mu[a] - qs), 0.0, 1.0)
    qo_n = np.clip(qo + p.alpha * L * (1 - p1) * (mu[o] - qo), 0.0, 1.0)
    L_n = np.clip(L + _legit_drift(p1, L, Mhat, p), 0.0, 1.0)
    return qs_n, qo_n, L_n


def field(Delta, L, p, Mhat):
    """Increment (dDelta, dL) on the sum=1 slice Q(a*)=1/2+Delta/2, Q(o)=1/2-Delta/2."""
    D = np.asarray(Delta, float); Lg = np.asarray(L, float)
    qs = 0.5 + D / 2.0; qo = 0.5 - D / 2.0
    qs_n, qo_n, L_n = step(qs, qo, Lg, p, Mhat)
    return (qs_n - qo_n) - D, L_n - Lg


def run_to_fixed_point(qs, qo, L, p, Mhat, iters=4000, tol=1e-10):
    for _ in range(iters):
        qs2, qo2, L2 = step(qs, qo, L, p, Mhat)
        if abs(qs2 - qs) + abs(qo2 - qo) + abs(L2 - L) < tol:
            break
        qs, qo, L = qs2, qo2, L2
    return qs, qo, L


def regime_of(Delta, L, tol=0.05, L_hi=0.5):
    """0 legitimate (correct, L high); 1 correct-but-illegitimate (correct, L low);
    2 defensive-wrong (wrong, L low); 3 credulous (wrong, L high)."""
    if Delta > tol:
        return 0 if L >= L_hi else 1
    if Delta < -tol:
        return 3 if L >= L_hi else 2
    return 0 if L >= L_hi else 1


##########################################################################
# Richer legitimacy dynamics: motivated cognition as a bistable attitude.
# Legitimacy is a self-reinforcing state on a tilted double well (cusp normal
# form): x = L - 1/2 obeys  x_dot = r x - s x^3 + h,  with the tilt
# h = -mu_tilt (Dbar(Mhat, choice) - tau^F) increasing in perceived unfairness.
# For |h| < h_c = 2 s (r/(3s))^{3/2} the well is bistable (two interior stable
# legitimacy levels, one unstable) -> legitimacy hysteresis and a fold in Mhat.
##########################################################################
def _disc_mean(Mhat, p1, p):
    """Expected fairness discomfort given the choice distribution p1 (= P(a*))."""
    mu = np.asarray(p.mu, float); a = int(np.argmax(mu)); o = 1 - a
    def d(mu_k):
        return mu_k * p.phi_plus * (1 - Mhat) + (1 - mu_k) * p.phi_minus * Mhat
    return p1 * d(mu[a]) + (1 - p1) * d(mu[o])


def _legit_birthdeath(Mhat, p1, p):
    """Birth-death legitimacy target L = nu_R a / (nu_R a + nu_F e) from model rates:
    a = P(discomfort acceptable), e = E[excess discomfort]."""
    mu = np.asarray(p.mu, float); a = int(np.argmax(mu)); o = 1 - a
    d0 = p.phi_minus * Mhat; d1 = p.phi_plus * (1 - Mhat); tau = p.tau_F
    def ae(mu_k):
        acc = (1 - mu_k) * (d0 <= tau) + mu_k * (d1 <= tau)
        exc = (1 - mu_k) * max(d0 - tau, 0.0) + mu_k * max(d1 - tau, 0.0)
        return acc, exc
    acc = p1 * ae(mu[a])[0] + (1 - p1) * ae(mu[o])[0]
    exc = p1 * ae(mu[a])[1] + (1 - p1) * ae(mu[o])[1]
    return p.nu_R * acc / (p.nu_R * acc + p.nu_F * exc + 1e-12)


def calibrate_cusp(p, window=0.18):
    """Calibrate the motivated-cognition cusp entirely from model quantities.

    Center: the collapse threshold M* where mean discomfort equals tau^F.
    Wells: the model's birth-death legitimacy at the fair/unfair window edges
    M* -/+ window (from the erosion/recovery rates nu_F, nu_R).
    Tilt gain mu_t: set so the two folds sit at M* -/+ window.
    Only the self-reinforcement is expressed through the (fixed) cubic scale s;
    the fold locations, centre and wells are model-derived.
    """
    from scipy.optimize import brentq
    a = int(np.argmax(p.mu)); p1 = choice_prob_star(p.mu[a] - p.mu[1 - a], p)
    f = lambda M: _disc_mean(M, p1, p) - p.tau_F
    Mstar = brentq(f, 0.01, 0.99)
    L_hi = _legit_birthdeath(max(Mstar - window, 0.0), p1, p)
    L_lo = _legit_birthdeath(min(Mstar + window, 1.0), p1, p)
    Lc = 0.5 * (L_hi + L_lo); delta = max(0.5 * (L_hi - L_lo), 1e-3)
    s = 8.0; r = s * delta ** 2
    hc = 2 * s * (r / (3 * s)) ** 1.5
    dD = abs(_disc_mean(min(Mstar + window, 1.0), p1, p) - p.tau_F) + 1e-9
    mu_tilt = hc / dD
    rate = 0.03 / (r + 1e-9)                       # near-well timescale ~ 30 steps
    return dict(r=r, s=s, mu_tilt=mu_tilt, rate=rate, Lc=Lc,
                Mstar=Mstar, L_hi=L_hi, L_lo=L_lo, window=window)


from .config import Params as _Params
RICH = calibrate_cusp(_Params())


def legit_tilt(Mhat, p1, p, mu_tilt=None):
    mu_tilt = RICH["mu_tilt"] if mu_tilt is None else mu_tilt
    return -mu_tilt * (_disc_mean(Mhat, p1, p) - p.tau_F)


def legit_drift_rich(L, Mhat, p1, p, **kw):
    c = {**RICH, **kw}
    x = L - c["Lc"]
    return c["rate"] * (c["r"] * x - c["s"] * x ** 3 + legit_tilt(Mhat, p1, p, c["mu_tilt"]))


def legit_equilibria(Mhat, p, p1=None, **kw):
    """Interior legitimacy fixed points L* (roots of the cusp), with stability."""
    c = {**RICH, **kw}
    if p1 is None:
        a = int(np.argmax(p.mu)); p1 = choice_prob_star(p.mu[a] - p.mu[1 - a], p)
    h = legit_tilt(Mhat, p1, p, c["mu_tilt"])
    roots = np.roots([1.0, 0.0, -c["r"] / c["s"], -h / c["s"]])
    out = []
    for z in roots:
        if abs(z.imag) < 1e-9:
            L = c["Lc"] + z.real
            if -0.05 <= L <= 1.05:
                stable = (c["r"] - 3 * c["s"] * z.real ** 2) < 0
                out.append((float(np.clip(L, 0, 1)), stable))
    return sorted(out)


def step_rich(qs, qo, L, p, Mhat, **kw):
    """Mean-field update with the richer (cusp) legitimacy dynamics."""
    mu = np.asarray(p.mu, float); a = int(np.argmax(mu)); o = 1 - a
    Delta = qs - qo; p1 = choice_prob_star(Delta, p)
    qs_n = np.clip(qs + p.alpha * L * p1 * (mu[a] - qs), 0.0, 1.0)
    qo_n = np.clip(qo + p.alpha * L * (1 - p1) * (mu[o] - qo), 0.0, 1.0)
    L_n = np.clip(L + legit_drift_rich(L, Mhat, p1, p, **kw), 0.0, 1.0)
    return qs_n, qo_n, L_n


def field_rich(Delta, L, p, Mhat, **kw):
    D = np.asarray(Delta, float); Lg = np.asarray(L, float)
    qs = 0.5 + D / 2.0; qo = 0.5 - D / 2.0
    qs_n, qo_n, L_n = step_rich(qs, qo, Lg, p, Mhat, **kw)
    return (qs_n - qo_n) - D, L_n - Lg


def basin_map(p, Mhat, dv, lv, steps=1500, tol=0.05, L_hi=0.5):
    """Terminal-regime label over the (Delta, L) initial grid (vectorised)."""
    D, L = np.meshgrid(np.asarray(dv, float), np.asarray(lv, float))
    qs = 0.5 + D / 2.0; qo = 0.5 - D / 2.0; Lg = L.copy()
    for _ in range(steps):
        qs, qo, Lg = step(qs, qo, Lg, p, Mhat)
    Delta = qs - qo
    R = np.zeros(Delta.shape, dtype=int)
    hiL = Lg >= L_hi
    R[(Delta > tol) & hiL] = 0            # legitimate
    R[(Delta > tol) & ~hiL] = 1           # correct-but-illegitimate
    R[(Delta < -tol) & ~hiL] = 2          # defensive wrong
    R[(Delta < -tol) & hiL] = 3           # credulous (wrong, legitimate)
    R[(np.abs(Delta) <= tol) & ~hiL] = 1
    return R


def terminal_legitimacy(p, Mhats, D0=0.0, L0=1.0):
    """Terminal legitimacy vs benchmark Mhat (from a near-neutral, legitimate start)."""
    out = np.empty(len(Mhats))
    for k, M in enumerate(Mhats):
        _, _, L = run_to_fixed_point(0.5 + D0 / 2, 0.5 - D0 / 2, L0, p, M)
        out[k] = L
    return out
