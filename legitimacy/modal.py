"""Modal-agent social learning.

Each agent carries a perceived modal-agent fairness benchmark ``Mhat_i,t``. It
is socially transmitted toward the community-coupled average, so that high
cohesion preserves local benchmarks and high permeability spreads them:

    Mhat_{i,t+1} = clip( (1-ψ) Mhat_i,t + ψ Σ_d B_{g(i),d} Mbar_{d,t} + ξ_i,t , 0, 1 ).
"""
from __future__ import annotations

import numpy as np
from .network import community_means, field_block


def modal_update(Mhat, B, g, Csel, sizes, psi, rng=None, xi_sd=0.0):
    Mhat = np.asarray(Mhat, dtype=float)
    Mbar = community_means(Mhat, Csel, sizes)          # (R, M)
    field = field_block(Mbar, B, g)                    # (R, N) exact quotient
    out = (1.0 - psi) * Mhat + psi * field
    if xi_sd > 0 and rng is not None:
        out = out + rng.normal(0.0, xi_sd, size=Mhat.shape)
    return np.clip(out, 0.0, 1.0)
