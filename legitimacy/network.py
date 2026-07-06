"""Community networks and the quotient coupling matrix (self-contained).

Balanced block weights: for i in community c and j in community d,
``W_ij = B_cd / N_d``. Under this geometry both the credibility field and the
modal-agent field are exact community-level objects.
"""
from __future__ import annotations

import numpy as np


def make_membership(sizes):
    sizes = np.asarray(sizes, dtype=int)
    return np.repeat(np.arange(len(sizes)), sizes)


def community_onehot(g, M):
    g = np.asarray(g)
    C = np.zeros((g.shape[0], int(M)))
    C[np.arange(g.shape[0]), g] = 1.0
    return C


def community_means(x, Csel, sizes):
    """Per-community mean of an (R, N) array -> (R, M)."""
    return (x @ Csel) / sizes


def balanced_block_W(sizes, B):
    sizes = np.asarray(sizes, dtype=int)
    B = np.asarray(B, dtype=float)
    M = len(sizes)
    g = make_membership(sizes)
    N = sizes.sum()
    starts = np.concatenate([[0], np.cumsum(sizes)])
    W = np.zeros((N, N))
    for c in range(M):
        rows = slice(starts[c], starts[c + 1])
        for d in range(M):
            W[rows, starts[d]:starts[d + 1]] = B[c, d] / sizes[d]
    return W, g


def is_row_stochastic(M, tol=1e-10):
    M = np.asarray(M, dtype=float)
    return bool(np.all(M >= -tol) and np.allclose(M.sum(axis=1), 1.0, atol=tol))


def spectral_gap(B):
    eig = np.sort(np.abs(np.linalg.eigvals(np.asarray(B, dtype=float))))[::-1]
    return float(1.0 - eig[1]) if len(eig) > 1 else 1.0


def field_block(comm_vals, B, g):
    """Exact agent field from community values.

    ``comm_vals`` may be (R, M) or (R, M, K). Returns the per-agent field
    ``F[:, i] = sum_d B[g_i, d] comm_vals[:, d]`` gathered by membership.
    """
    comm_vals = np.asarray(comm_vals, dtype=float)
    if comm_vals.ndim == 2:                       # (R, M)
        F = comm_vals @ B.T                       # (R, M)
        return F[:, g]                            # (R, N)
    F = np.einsum("cd,rdk->rck", B, comm_vals)    # (R, M, K)
    return F[:, g, :]                             # (R, N, K)
