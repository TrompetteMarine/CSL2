"""Monte Carlo uncertainty: Wilson intervals and (paired) bootstrap (self-contained)."""
from __future__ import annotations

import numpy as np

Z95 = 1.959963984540054


def wilson_interval(k, n, z=Z95):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1.0 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (float(max(0.0, c - h)), float(min(1.0, c + h)))


def proportion_summary(indicator, z=Z95):
    x = np.asarray(indicator, dtype=float)
    n = x.size
    k = float(x.sum())
    lo, hi = wilson_interval(k, n, z)
    return dict(p=(k / n if n else float("nan")), lo=lo, hi=hi, n=int(n))


def bootstrap_mean(x, n_boot=4000, alpha=0.05, seed=0):
    x = np.asarray(x, dtype=float); x = x[np.isfinite(x)]
    n = x.size
    if n == 0:
        return dict(mean=float("nan"), lo=float("nan"), hi=float("nan"), se=float("nan"), n=0)
    rng = np.random.default_rng(seed)
    boots = x[rng.integers(0, n, (n_boot, n))].mean(axis=1)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    se = x.std(ddof=1) / np.sqrt(n) if n > 1 else float("nan")
    return dict(mean=float(x.mean()), lo=float(lo), hi=float(hi), se=float(se), n=int(n))


def paired_bootstrap_diff(a, b, n_boot=4000, alpha=0.05, seed=0):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    d = a[m] - b[m]; n = d.size
    if n == 0:
        return dict(diff=float("nan"), lo=float("nan"), hi=float("nan"), n=0)
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, n, (n_boot, n))].mean(axis=1)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return dict(diff=float(d.mean()), lo=float(lo), hi=float(hi), n=int(n))
