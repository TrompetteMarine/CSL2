"""Canonical scenarios for the Paper II experiments (single source of truth)."""
from __future__ import annotations

from .config import Params, SCALE


def _n(scale):
    return SCALE[scale]["N_per"]


def base(scale="standard"):
    n = _n(scale)
    return Params(sizes=(n, n))


# II.1 fairness probe uses legitimacy_response on this parameter set
def discomfort_params():
    return Params(phi_minus=1.0, phi_plus=0.25, tau_F=0.30, nu_F=0.05, nu_R=0.01, L0=1.0)


# II.2 acceptability-threshold sweep: persistent unfairness, vary tau_F
def threshold_base(scale="standard"):
    n = _n(scale)
    return Params(sizes=(n, n), T=300, mu=(0.60, 0.40), lam=0.40, eta=0.20, b_self=0.85,
                  Mhat0=(0.60, 0.60), nu_F=0.05, nu_R=0.01,
                  phi_minus=1.0, phi_plus=0.25, psi=0.05)


# II.3 modal-agent misrepresentation: opposed entitlements, low permeability
def modal_misrep(scale="standard", b_self=0.98):
    n = _n(scale)
    # opposed entitlements; slow benchmark transmission so the gap persists
    return Params(sizes=(n, n), T=300, mu=(0.60, 0.40), lam=0.40, eta=0.20,
                  b_self=b_self, Mhat0=(0.25, 0.75), psi=0.04,
                  nu_F=0.05, nu_R=0.01, tau_F=0.30, phi_minus=1.0, phi_plus=0.25)


# II.4 legitimacy-gated wrong consensus (Paper-I-style wrong lead)
def wrong_base(scale="standard"):
    n = _n(scale)
    # moderate transmission with an early wrong lead: the ungated model
    # self-corrects, so legitimacy gating's effect on persistence is visible
    # (not saturated into wrong consensus as in Paper I's strong-lambda regime).
    return Params(sizes=(n, n), T=300, mu=(0.55, 0.45), lam=0.42, eta=0.12, b_self=0.70,
                  alpha=0.20, q_init=((0.44, 0.72), (0.44, 0.72)),
                  Mhat0=(0.50, 0.50), nu_F=0.12, nu_R=0.01, tau_F=0.20,
                  phi_minus=1.0, phi_plus=0.25, psi=0.05)


def wrong_variants(scale="standard"):
    b = wrong_base(scale)
    return {
        "no-gate": b.replace(legitimacy_gate=False),
        "gate": b.replace(legitimacy_gate=True),
        "gate+misrep": b.replace(legitimacy_gate=True, Mhat0=(0.45, 0.65), b_self=0.90),
    }


# II.5 network geometry: vary cohesion/permeability on the misrep setup
def geometry_grid(scale="standard"):
    gammas = [0.02, 0.06, 0.12, 0.20, 0.30, 0.45]      # permeability = 1 - b_self
    return modal_misrep(scale), gammas


# II.6 correct-but-illegitimate consensus: learnable optimum, high entitlement
def correct_illegit(scale="standard"):
    n = _n(scale)
    return Params(sizes=(n, n), T=300, mu=(0.65, 0.35), lam=0.40, eta=0.20, b_self=0.85,
                  q_init=((0.5, 0.5), (0.5, 0.5)), Mhat0=(0.80, 0.80),
                  nu_F=0.05, nu_R=0.005, tau_F=0.30, phi_minus=1.0, phi_plus=0.25, psi=0.05)
