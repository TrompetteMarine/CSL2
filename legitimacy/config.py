"""Parameters for the legitimacy-gating model (Paper II, self-contained)."""
from __future__ import annotations

from dataclasses import dataclass, replace, asdict
from typing import Tuple
import numpy as np


@dataclass(frozen=True)
class Params:
    # environment
    sizes: Tuple[int, ...] = (150, 150)
    T: int = 300
    mu: Tuple[float, float] = (0.60, 0.40)        # Bernoulli arm means

    # network coupling (row-stochastic, symmetric unless B given)
    b_self: float = 0.85
    B: Tuple[Tuple[float, ...], ...] = None

    # credibility transmission (Paper-I backbone, re-implemented)
    lam: float = 0.50                              # anticipatory weight
    eta: float = 0.20                              # retrospective weight
    gamma: float = 0.50                            # social learning-rate scale
    omega: float = 1.0
    eps_soc: float = 1e-3

    # process-based choice
    beta: float = 6.0
    sigma: float = 1.0
    a_thr: float = 1.0

    # private learning
    alpha: float = 0.20                            # constant private learning rate

    # legitimacy
    legitimacy_gate: bool = True                   # if False, L is fixed at 1
    L0: float = 1.0
    nu_F: float = 0.05                             # legitimacy loss rate
    nu_R: float = 0.01                             # legitimacy recovery rate
    tau_F: float = 0.30                            # acceptability threshold

    # fairness discomfort (asymmetric: phi_minus > phi_plus)
    phi_minus: float = 1.00
    phi_plus: float = 0.25

    # perceived modal-agent fairness benchmark
    Mhat0: Tuple[float, ...] = None               # per-community initial Mhat (else 0.5)
    psi: float = 0.10                              # modal-agent social transmission
    xi_sd: float = 0.0                            # idiosyncratic modal noise
    M_true: float = 0.50                          # true modal-agent benchmark
    pin_modal_true: bool = False                  # ablation A2: agents perceive M_true exactly
    # fair benchmark r^F(a) = b0 + b1*1{a=high-effort} + b2*Mhat  (effort mode);
    # scalar mode uses r^F = Mhat (b0=0,b1=0,b2=1).
    benchmark_mode: str = "scalar"                # "scalar" | "effort"
    b0: float = 0.0
    b1: float = 0.0
    b2: float = 1.0
    high_effort_arm: int = 0                       # which arm is "high effort"

    # fairness-in-choice term (behind a flag; off in v1)
    fairness_in_choice: bool = False
    zeta_F: float = 0.0

    # initial values
    q_init: Tuple[Tuple[float, float], ...] = None

    # regime thresholds
    theta: float = 0.90                            # action-mass threshold
    L_hi: float = 0.60                             # high-legitimacy cutoff
    L_lo: float = 0.40                             # low-legitimacy cutoff
    G_thr: float = 0.20                            # legitimacy-polarisation cutoff

    @property
    def M(self) -> int:
        return len(self.sizes)

    @property
    def N(self) -> int:
        return int(sum(self.sizes))

    def coupling(self) -> np.ndarray:
        if self.B is not None:
            return np.asarray(self.B, dtype=float)
        M = self.M
        if M == 1:
            return np.array([[1.0]])
        Bm = np.full((M, M), (1.0 - self.b_self) / (M - 1))
        np.fill_diagonal(Bm, self.b_self)
        return Bm

    def q_init_array(self) -> np.ndarray:
        if self.q_init is None:
            return np.full((self.M, 2), 0.5)
        return np.asarray(self.q_init, dtype=float)

    def mhat0_array(self) -> np.ndarray:
        if self.Mhat0 is None:
            return np.full(self.M, 0.5)
        return np.asarray(self.Mhat0, dtype=float)

    def replace(self, **kw) -> "Params":
        return replace(self, **kw)

    def as_dict(self) -> dict:
        return asdict(self)


SCALE = {
    "preview":  dict(N_per=80,  reps=80,  N_phase=50, reps_phase=40, grid=9),
    "standard": dict(N_per=150, reps=200, N_phase=80, reps_phase=60, grid=11),
}
