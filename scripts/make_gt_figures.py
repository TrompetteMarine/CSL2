"""Game-theoretic / dynamical-systems figures for Paper II, from the mean-field
reduction (legitimacy.meanfield):

  fig8_phase_portrait.pdf   (Delta, L) flow + basins, legitimate vs collapse
  fig9_equilibrium.pdf      legitimacy-collapse bifurcation + QRE choice response
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D
import common
import plotstyle
from legitimacy import meanfield as mf
from legitimacy.config import Params

plotstyle.use()
P = plotstyle.PALETTE
GREEN, PURPLE, RED, GREY, BLUE, INK = P["green"], P["purple"], P["red"], P["grey"], P["blue"], P["ink"]


def _tint(hexc, f=0.55):
    import matplotlib.colors as mc
    r, g, b = mc.to_rgb(hexc)
    return (r + (1 - r) * f, g + (1 - g) * f, b + (1 - b) * f)


def _fixed_points(p, Mhat):
    seeds, fps = [], []
    for D0 in (-0.7, 0.0, 0.7):
        for L0 in (0.05, 0.5, 1.0):
            qs, qo, L = mf.run_to_fixed_point(0.5 + D0 / 2, 0.5 - D0 / 2, L0, p, Mhat)
            seeds.append((qs - qo, L))
    for s in seeds:
        if not any(abs(s[0] - f[0]) < 2e-2 and abs(s[1] - f[1]) < 2e-2 for f in fps):
            fps.append(s)
    return fps


def fig_phase_portrait():
    p = Params()
    dv = np.linspace(-1.0, 1.0, 67)
    lv = np.linspace(0.0, 1.0, 50)
    gq = np.linspace(-0.97, 0.97, 26)
    gl = np.linspace(0.03, 0.99, 22)
    GD, GL = np.meshgrid(gq, gl)
    bcmap = ListedColormap([_tint(GREEN), _tint(PURPLE), _tint(RED), _tint(GREY)])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], bcmap.N)

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.4))
    from matplotlib.patches import Rectangle
    for ax, Mhat, tag in [
            (axes[0], 0.20, r"(a) modest entitlement $\hat M=0.2$: learning stays legitimate"),
            (axes[1], 0.70, r"(b) high entitlement $\hat M=0.7$: legitimacy collapses")]:
        R = mf.basin_map(p, Mhat, dv, lv)
        ax.pcolormesh(dv, lv, R, cmap=bcmap, norm=norm, shading="nearest", rasterized=True, zorder=0)
        dD, dL = mf.field(GD, GL, p, Mhat)
        ax.streamplot(gq, gl, dD, dL, color=INK, density=0.8, linewidth=0.6,
                      arrowsize=0.75, zorder=1)
        for f in _fixed_points(p, Mhat):
            reg = mf.regime_of(f[0], f[1])
            col = [GREEN, PURPLE, RED, GREY][reg]
            ax.plot(f[0], f[1], "o", ms=8, mfc=col, mec="white", mew=1.2, zorder=6)
        # thin border delineating the physical state space [-1,1] x [0,1]
        ax.add_patch(Rectangle((-1, 0), 2, 1, fill=False, ec="0.55", lw=0.8, zorder=5))
        ax.axvline(0, color=INK, lw=0.5, alpha=0.28)
        ax.set_xlabel(r"value contrast $\Delta=Q(a^\star)-Q(\text{other})$", fontsize=11)
        if ax is axes[0]:
            ax.set_ylabel(r"legitimacy $L$", fontsize=11)
        else:
            ax.set_yticklabels([])
        ax.set_title(tag, fontsize=11)
        ax.tick_params(labelsize=9.5)
        ax.set_xticks([-1, -0.5, 0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
        ax.set_xlim(-1.55, 1.55); ax.set_ylim(-0.35, 1.35); ax.grid(False)
    handles = [Line2D([0], [0], marker="s", ls="", mfc=_tint(GREEN), mec="0.6", ms=9, label="legitimate consensus"),
               Line2D([0], [0], marker="s", ls="", mfc=_tint(PURPLE), mec="0.6", ms=9, label="correct-but-illegitimate"),
               Line2D([0], [0], marker="s", ls="", mfc=_tint(RED), mec="0.6", ms=9, label="defensive wrong")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.subplots_adjust(left=0.07, right=0.985, top=0.9, bottom=0.19, wspace=0.08)
    fig.suptitle(r"Phase portrait of legitimacy-gated learning in the $(\Delta,L)$ plane", y=0.99)
    for ext in ("pdf", "png"):
        fig.savefig(f"{common.FIGDIR}/fig8_phase_portrait.{ext}")
    plt.close(fig)
    print("wrote fig8_phase_portrait")


def fig_equilibrium():
    p = Params()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.4, 3.4))

    # (a) equilibrium correspondence: L=1 and L=0 branches with stability,
    # exchanging stability transcritically at Mhat*
    a = int(np.argmax(p.mu)); mu_a = p.mu[a]
    Mc = (p.tau_F + p.nu_R * mu_a / (p.nu_F * (1 - mu_a))) / p.phi_minus
    Mlo = np.linspace(0, Mc, 80); Mhi = np.linspace(Mc, 1, 80)
    # legitimate branch L*=1: stable below Mc, unstable above
    a1.plot(Mlo, np.ones_like(Mlo), color=GREEN, lw=2.8, solid_capstyle="round", label="stable equilibrium")
    a1.plot(Mhi, np.ones_like(Mhi), color="0.6", lw=1.6, ls=(0, (4, 3)), label="unstable equilibrium")
    # collapsed branch L*=0: unstable below Mc, stable above
    a1.plot(Mlo, np.zeros_like(Mlo), color="0.6", lw=1.6, ls=(0, (4, 3)))
    a1.plot(Mhi, np.zeros_like(Mhi), color=RED, lw=2.8, solid_capstyle="round")
    a1.plot([Mc], [1.0], "o", ms=7, mfc="white", mec=INK, mew=1.3, zorder=6)
    a1.plot([Mc], [0.0], "o", ms=7, mfc="white", mec=INK, mew=1.3, zorder=6)
    a1.axvline(Mc, color=INK, ls=":", lw=1.0, alpha=0.6)
    a1.text(Mc, 1.09, r"$\hat M^\star$ (transcritical)", color=INK, fontsize=8, ha="center")
    a1.axvspan(0, Mc, color=GREEN, alpha=0.05); a1.axvspan(Mc, 1, color=RED, alpha=0.05)
    a1.set_xlabel(r"perceived benchmark $\hat M$ (entitlement)")
    a1.set_ylabel(r"equilibrium legitimacy $L^\star$")
    a1.set_title("(a) behavioural-equilibrium correspondence")
    a1.set_xlim(0, 1); a1.set_ylim(-0.08, 1.16)
    a1.text(0.12, 0.86, "legitimate", color="#14601f", fontsize=8, weight="bold")
    a1.text(0.70, 0.10, "illegitimate", color="#7a0f17", fontsize=8, weight="bold")
    a1.legend(loc="center left", fontsize=7.4, frameon=True, framealpha=0.9, edgecolor="0.8")

    # (b) QRE choice response with the joint (choice, value) rest points marked
    D = np.linspace(-0.6, 0.6, 300)
    pstar = mf.choice_prob_star(D, p)
    a2.plot(D, pstar, color=BLUE, lw=2.4, label=r"QRE response $P(a^\star)=\Lambda(2\beta a\Delta/\sigma^2)$")
    a2.axhline(0.5, color=INK, lw=0.5, alpha=0.3); a2.axvline(0, color=INK, lw=0.5, alpha=0.3)
    Dstar = p.mu[a] - p.mu[1 - a]                    # correct rest point Q->reward means
    a2.plot(Dstar, mf.choice_prob_star(Dstar, p), "o", ms=9, mfc=GREEN, mec="white", mew=1.1,
            label="legitimate rest point ($L>0$)")
    a2.plot(-Dstar, mf.choice_prob_star(-Dstar, p), "o", ms=9, mfc=RED, mec="white", mew=1.1,
            label="frozen wrong ($L\\to0$)")
    a2.annotate("", xy=(Dstar - 0.02, 0.62), xytext=(0.05, 0.62),
                arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.4))
    a2.text(0.10, 0.66, r"gate open: $\Delta\!\uparrow$", color=GREEN, fontsize=7.5)
    a2.text(-0.55, 0.14, r"gate shut: frozen", color=RED, fontsize=7.5)
    a2.set_xlabel(r"value contrast $\Delta$")
    a2.set_ylabel(r"choice probability $P(a^\star)$")
    a2.set_title("(b) quantal response; legitimacy moves the state along it")
    a2.set_xlim(-0.6, 0.6); a2.set_ylim(0, 1.02)
    a2.legend(loc="lower right", fontsize=7.2, frameon=True, framealpha=0.9, edgecolor="0.8")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{common.FIGDIR}/fig9_equilibrium.{ext}")
    plt.close(fig)
    print("wrote fig9_equilibrium")


def _basin_rich(p, Mhat, dv, lv, steps=2600, tol=0.05, L_hi=None):
    if L_hi is None:
        L_hi = mf.RICH["Lc"]
    D, L = np.meshgrid(np.asarray(dv, float), np.asarray(lv, float))
    qs = 0.5 + D / 2.0; qo = 0.5 - D / 2.0; Lg = L.copy()
    for _ in range(steps):
        qs, qo, Lg = mf.step_rich(qs, qo, Lg, p, Mhat)
    Delta = qs - qo
    R = np.zeros(Delta.shape, dtype=int)          # 0 legitimate
    R[(Delta > tol) & (Lg < L_hi)] = 1            # correct-but-illegitimate
    R[(Delta < -tol)] = 2                         # (rare) still-wrong
    return R


def fig_phase_portrait_rich():
    from matplotlib.patches import Rectangle
    p = Params()
    dv = np.linspace(-1.0, 1.0, 61); lv = np.linspace(0.0, 1.0, 46)
    gq = np.linspace(-0.95, 0.95, 22); gl = np.linspace(0.03, 0.99, 22)
    GD, GL = np.meshgrid(gq, gl)
    bcmap = ListedColormap([_tint(GREEN), _tint(PURPLE), _tint(RED)])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], bcmap.N)
    panels = [(0.20, r"(a) fair rule $\hat M=0.2$: one legitimate attractor"),
              (0.55, r"(b) intermediate $\hat M=0.55$: bistable (legitimacy hysteresis)"),
              (0.90, r"(c) unfair rule $\hat M=0.9$: one illegitimate attractor")]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.7))
    for ax, (Mhat, tag) in zip(axes, panels):
        ax.pcolormesh(dv, lv, _basin_rich(p, Mhat, dv, lv), cmap=bcmap, norm=norm,
                      shading="nearest", rasterized=True, zorder=0)
        dD, dL = mf.field_rich(GD, GL, p, Mhat)
        ax.streamplot(gq, gl, dD, dL, color=INK, density=0.85, linewidth=0.6,
                      arrowsize=0.75, zorder=1)
        a = int(np.argmax(p.mu)); Dstar = p.mu[a] - p.mu[1 - a]
        for Lp, stable in mf.legit_equilibria(Mhat, p, p1=mf.choice_prob_star(Dstar, p)):
            col = GREEN if (stable and Lp >= 0.5) else (PURPLE if stable else "white")
            ax.plot(Dstar, Lp, "o", ms=9, mfc=col, mec=INK, mew=1.3, zorder=6)
        ax.add_patch(Rectangle((-1, 0), 2, 1, fill=False, ec="0.55", lw=0.8, zorder=5))
        ax.axvline(0, color=INK, lw=0.5, alpha=0.28)
        ax.set_xlabel(r"value contrast $\Delta$", fontsize=10.5)
        ax.set_ylabel(r"legitimacy $L$", fontsize=10.5) if ax is axes[0] else ax.set_yticklabels([])
        ax.set_title(tag, fontsize=9.8)
        ax.set_xticks([-1, 0, 1]); ax.set_yticks([0, 0.5, 1])
        ax.set_xlim(-1.45, 1.45); ax.set_ylim(-0.28, 1.28); ax.grid(False)
    handles = [Line2D([0], [0], marker="o", ls="", mfc=GREEN, mec=INK, ms=8, label="legitimate attractor"),
               Line2D([0], [0], marker="o", ls="", mfc=PURPLE, mec=INK, ms=8, label="illegitimate attractor"),
               Line2D([0], [0], marker="o", ls="", mfc="white", mec=INK, ms=8, label="saddle (unstable)")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, -0.01))
    fig.subplots_adjust(left=0.06, right=0.99, top=0.9, bottom=0.17, wspace=0.08)
    fig.suptitle(r"Phase portrait with motivated-cognition legitimacy: interior attractors and a fold", y=0.99)
    for ext in ("pdf", "png"):
        fig.savefig(f"{common.FIGDIR}/fig8_phase_portrait.{ext}")
    plt.close(fig)
    print("wrote fig8_phase_portrait (rich)")


def fig_hysteresis_qre():
    p = Params()
    a = int(np.argmax(p.mu)); Dstar = p.mu[a] - p.mu[1 - a]
    p1 = mf.choice_prob_star(Dstar, p)
    Ms = np.linspace(0.0, 1.0, 260)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 3.5))
    # (a) hysteresis: L* branches vs Mhat
    xs_s, ys_s, xs_u, ys_u = [], [], [], []
    for M in Ms:
        for Lp, stable in mf.legit_equilibria(M, p, p1=p1):
            (xs_s if stable else xs_u).append(M)
            (ys_s if stable else ys_u).append(Lp)
    a1.scatter(xs_s, ys_s, s=4, color=BLUE, label="stable equilibrium")
    a1.scatter(xs_u, ys_u, s=4, color="0.6", marker="s", label="unstable (saddle)")
    a1.set_xlabel(r"perceived benchmark $\hat M$ (entitlement)")
    a1.set_ylabel(r"equilibrium legitimacy $L^\star$")
    a1.set_title("(a) legitimacy hysteresis (cusp fold)")
    a1.set_xlim(0, 1); a1.set_ylim(0, 1)
    # folds at M* +/- window (model-calibrated); label each in clear space
    Ms_f, w = mf.RICH["Mstar"], mf.RICH["window"]
    def _Lat(M, hi):
        eq = [L for L, s in mf.legit_equilibria(M, p, p1=p1) if s]
        return (max(eq) if hi else min(eq)) if eq else 0.5
    a1.annotate("lower fold", xy=(Ms_f - w, _Lat(Ms_f - w + 0.01, False)),
                xytext=(0.50, 0.20), fontsize=8, color=INK,
                arrowprops=dict(arrowstyle="->", color="0.4", lw=0.9))
    a1.annotate("upper fold", xy=(Ms_f + w, _Lat(Ms_f + w - 0.01, True)),
                xytext=(0.44, 0.90), fontsize=8, color=INK,
                arrowprops=dict(arrowstyle="->", color="0.4", lw=0.9))
    a1.legend(loc="lower left", fontsize=7.2, frameon=True, framealpha=0.92,
              edgecolor="0.8", bbox_to_anchor=(0.0, 0.0))
    # (b) QRE choice response
    D = np.linspace(-0.6, 0.6, 300)
    a2.plot(D, mf.choice_prob_star(D, p), color=BLUE, lw=2.4,
            label=r"QRE response $\Lambda(2\beta a\Delta/\sigma^2)$")
    a2.axhline(0.5, color=INK, lw=0.5, alpha=0.3); a2.axvline(0, color=INK, lw=0.5, alpha=0.3)
    a2.plot(Dstar, p1, "o", ms=9, mfc=GREEN, mec="white", mew=1.1, label="correct rest point")
    a2.set_xlabel(r"value contrast $\Delta$"); a2.set_ylabel(r"choice probability $P(a^\star)$")
    a2.set_title("(b) quantal response (unchanged by $L$)")
    a2.set_xlim(-0.6, 0.6); a2.set_ylim(0, 1.02)
    a2.legend(loc="lower right", fontsize=7.4, frameon=True, framealpha=0.9, edgecolor="0.8")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{common.FIGDIR}/fig9_equilibrium.{ext}")
    plt.close(fig)
    print("wrote fig9_equilibrium (hysteresis)")


if __name__ == "__main__":
    fig_phase_portrait_rich()
    fig_hysteresis_qre()
