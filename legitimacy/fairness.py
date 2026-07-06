"""Fairness discrepancy, asymmetric discomfort, and legitimacy dynamics.

The reward rule is treated as evidence only to the extent that it is legitimate.
Legitimacy ``L`` falls when accumulated fairness discomfort exceeds an
acceptability threshold, and recovers slowly when outcomes are acceptable.
Discomfort is asymmetric: disadvantageous unfairness hurts more than
advantageous unfairness (``phi_minus > phi_plus``).
"""
from __future__ import annotations

import numpy as np


def fair_benchmark(action, mhat, params):
    """Perceived fair reward benchmark r^F(a; Mhat).

    Scalar mode: r^F = Mhat. Effort mode:
    r^F = b0 + b1*1{a = high-effort arm} + b2*Mhat, so a high-effort action is
    expected to deserve more reward.
    """
    mhat = np.asarray(mhat, dtype=float)
    if params.benchmark_mode == "effort":
        eff = (np.asarray(action) == params.high_effort_arm).astype(float)
        return params.b0 + params.b1 * eff + params.b2 * mhat
    return mhat


def fairness_discrepancy(reward, benchmark):
    """ΔF = R − r^F. ΔF < 0: disadvantageous; ΔF > 0: advantageous unfairness."""
    return np.asarray(reward, dtype=float) - np.asarray(benchmark, dtype=float)


def discomfort(delta_f, phi_minus, phi_plus):
    """Asymmetric fairness discomfort DF = φ⁻[−ΔF]_+ + φ⁺[ΔF]_+ (φ⁻ ≥ φ⁺ ≥ 0)."""
    delta_f = np.asarray(delta_f, dtype=float)
    return phi_minus * np.maximum(-delta_f, 0.0) + phi_plus * np.maximum(delta_f, 0.0)


def legitimacy_update(L, df, tau_F, nu_F, nu_R):
    """L_{t+1} = clip( L − ν_F (DF − τ^F)_+ + ν_R·1{DF ≤ τ^F}, 0, 1 )."""
    L = np.asarray(L, dtype=float)
    df = np.asarray(df, dtype=float)
    excess = np.maximum(df - tau_F, 0.0)
    acceptable = (df <= tau_F).astype(float)
    return np.clip(L - nu_F * excess + nu_R * acceptable, 0.0, 1.0)


def gated_prediction_error(L, delta):
    """Legitimacy-gated prediction error δ^eff = L · δ."""
    return np.asarray(L, dtype=float) * np.asarray(delta, dtype=float)
