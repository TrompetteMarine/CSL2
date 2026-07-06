"""Endogenous confidence from the choice process (balance of evidence).

Confidence is the balance of evidence read off the exact two-boundary choice
probability: ``C = |2 P(A=1) - 1|`` in (0,1). It is zero at indifference and
rises with drift magnitude. Bounded, endogenous, and not equal to correctness.
"""
from __future__ import annotations

import numpy as np

_EPS = 1e-9


def balance_of_evidence(p_up):
    p_up = np.asarray(p_up, dtype=float)
    return np.clip(np.abs(2.0 * p_up - 1.0), _EPS, 1.0 - _EPS)
