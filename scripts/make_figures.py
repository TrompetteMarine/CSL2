"""Render Paper II figures 2-5 from saved experiment results."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import common
import plotstyle

plotstyle.use()
_P = plotstyle.PALETTE
C1, C2 = _P["blue"], _P["orange"]
GREEN, RED, PURPLE = _P["green"], _P["red"], _P["purple"]


def _load():
    global SCALE
    return np.load(f"{common.RESULTS}/experiments_{SCALE}.npz", allow_pickle=True)


def _save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(f"{common.FIGDIR}/{name}.{ext}", dpi=300)
    plt.close(fig); print(f"wrote {name}")


def fig2_asymmetry(d):
    xs = d["ii1_xs"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.8))
    reds = ["#fb8d61", "#e34a33", "#99000d"]
    blues = ["#9ecae1", "#4292c6", "#08519c"]
    for i, x in enumerate(xs):
        t = np.arange(len(d[f"ii1_disadv_{x}"]))
        a1.plot(t, d[f"ii1_disadv_{x}"], color=reds[i], lw=1.7, label=f"disadv. $|\\Delta F|={x}$")
        a1.plot(t, d[f"ii1_adv_{x}"], color=blues[i], lw=1.4, ls="--", label=f"adv. $|\\Delta F|={x}$")
    a1.set_xlabel("trial"); a1.set_ylabel("legitimacy $L$"); a1.set_ylim(-0.02, 1.02)
    a1.set_title("(a) Legitimacy response"); a1.legend(fontsize=6, frameon=False, ncol=1)
    drop_d = [1 - d[f"ii1_disadv_{x}"][-1] for x in xs]
    drop_a = [1 - d[f"ii1_adv_{x}"][-1] for x in xs]
    w = 0.35; xi = np.arange(len(xs))
    a2.bar(xi - w / 2, drop_d, w, color=RED, label="disadvantageous")
    a2.bar(xi + w / 2, drop_a, w, color=C1, label="advantageous")
    a2.set_xticks(xi); a2.set_xticklabels([f"{x}" for x in xs]); a2.set_xlabel(r"$|\Delta F|$")
    a2.set_ylabel(r"legitimacy loss $|\Delta L|$"); a2.set_title("(b) Asymmetric discomfort")
    a2.legend(frameon=False)
    fig.suptitle("II.1  Agents mind unfairness more when it harms them", y=1.03, fontsize=9.5)
    fig.tight_layout(); _save(fig, "fig2_asymmetry")


def fig3_fairness(d):
    L = d["ii3_L"]; Mh = d["ii3_Mhat"]; t = np.arange(L.shape[0])
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 2.8))
    ax = axes[0]
    ax.plot(t, L[:, 0], color=C1, lw=1.8, label="community 1 (low entitlement)")
    ax.plot(t, L[:, 1], color=C2, lw=1.8, label="community 2 (high entitlement)")
    ax.plot(t, d["ii3_GL"], color="k", lw=1.0, ls=":", label=r"gap $G^L$")
    ax.set_xlabel("trial"); ax.set_ylabel("legitimacy $\\bar L_c$"); ax.set_ylim(-0.02, 1.02)
    ax.set_title("(a) II.3 fairness polarisation"); ax.legend(fontsize=6.5, frameon=False)
    ax = axes[1]
    ax.plot(t, Mh[:, 0], color=C1, lw=1.8); ax.plot(t, Mh[:, 1], color=C2, lw=1.8)
    ax.axhline(0.5, color="k", lw=0.6, ls=":")
    ax.set_xlabel("trial"); ax.set_ylabel(r"modal benchmark $\bar{\hat M}_c$"); ax.set_ylim(0, 1)
    ax.set_title("(b) II.3 modal benchmarks")
    ax = axes[2]
    ax.plot(d["ii2_tau"], d["ii2_meanL"], "-o", color=PURPLE, ms=4)
    ax.set_xlabel(r"acceptability threshold $\tau^F$"); ax.set_ylabel("terminal mean $L$")
    ax.set_ylim(-0.02, 1.02); ax.set_title("(c) II.2 acceptability threshold")
    fig.tight_layout(); _save(fig, "fig3_fairness")


def fig4_regimes(d):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 2.9))
    cols = {"no-gate": GREEN, "gate": RED, "gate+misrep": PURPLE}
    for nm in ("no-gate", "gate", "gate+misrep"):
        ms = d[f"ii4_{nm}_mstar"].mean(axis=1)            # population mean over communities
        a1.plot(np.arange(len(ms)), ms, color=cols[nm], lw=1.8, label=nm)
    a1.axhline(0.5, color="k", lw=0.6, ls=":")
    a1.set_xlabel("trial"); a1.set_ylabel(r"mass on optimal arm $m_t(a^\star)$"); a1.set_ylim(-0.02, 1.02)
    a1.set_title("(a) II.4 legitimacy-gated wrong consensus"); a1.legend(frameon=False)
    ms6 = d["ii6_mstar"].mean(axis=1); L6 = d["ii6_meanL"]
    t6 = np.arange(len(ms6))
    a2.plot(t6, ms6, color=GREEN, lw=1.8, label=r"$m_t(a^\star)$ (action)")
    a2.plot(np.arange(len(L6)), L6, color=PURPLE, lw=1.8, label=r"mean legitimacy $L$")
    a2.set_xlabel("trial"); a2.set_ylim(-0.02, 1.02)
    a2.set_title("(b) II.6 correct but illegitimate"); a2.legend(frameon=False)
    fig.tight_layout(); _save(fig, "fig4_regimes")


def fig5_geometry(d):
    g = d["ii5_gamma"]
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    ax.plot(g, d["ii5_meanL"], "-o", color=PURPLE, ms=4, label="mean legitimacy $\\bar L$")
    ax.plot(g, d["ii5_GL"], "-s", color="#444", ms=4, label="legitimacy gap $G^L$")
    ax.set_xlabel(r"cross-community permeability $\Gamma=1-B_{cc}$")
    ax.set_ylabel("terminal value"); ax.set_ylim(-0.02, 1.02)
    ax.set_title("II.5 network geometry and legitimacy")
    ax.legend(frameon=False)
    fig.tight_layout(); _save(fig, "fig5_geometry")


REG_COLOR = {"legitimate": "#2a7a36", "illegitimate_correct": "#6a51a3",
             "defensive_wrong": "#b2182b", "fairness_polarised": "#d2691e",
             "unresolved": "#9e9e9e"}
REG_NAMES = ["legitimate", "illegitimate_correct", "defensive_wrong",
             "fairness_polarised", "unresolved"]


def fig6_phase():
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    dm = np.load(f"{common.RESULTS}/phase_main_{SCALE}.npz")
    ds = np.load(f"{common.RESULTS}/phase_secondary_{SCALE}.npz")
    P, taus, diss = dm["P"], dm["taus"], dm["diss"]
    reg = P.argmax(axis=2)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.2, 3.2))
    cmap = ListedColormap([REG_COLOR[n] for n in REG_NAMES])
    a1.imshow(reg, origin="lower", aspect="auto", cmap=cmap, vmin=0, vmax=4,
              extent=[taus.min(), taus.max(), diss.min(), diss.max()])
    a1.set_xlabel(r"acceptability threshold $\tau^F$")
    a1.set_ylabel(r"initial benchmark disagreement $|\Delta\hat M_0|$")
    a1.set_title("(a) Main phase diagram"); a1.grid(False)
    present = sorted(set(reg.ravel()))
    a1.legend(handles=[Patch(facecolor=REG_COLOR[REG_NAMES[k]], label=REG_NAMES[k].replace("_", " "))
                       for k in present], loc="upper center", bbox_to_anchor=(0.5, -0.22),
              ncol=2, frameon=False, fontsize=6.5)
    GL, gammas, psis = ds["GL"], ds["gammas"], ds["psis"]
    im = a2.imshow(GL, origin="lower", aspect="auto", cmap="magma",
                   extent=[gammas.min(), gammas.max(), psis.min(), psis.max()])
    a2.set_xlabel(r"permeability $\Gamma=1-B_{cc}$")
    a2.set_ylabel(r"modal transmission $\psi$")
    a2.set_title(r"(b) Legitimacy gap $G^L$"); a2.grid(False)
    cb = fig.colorbar(im, ax=a2, fraction=0.046, pad=0.04); cb.set_label(r"$G^L$")
    fig.tight_layout(); _save(fig, "fig6_phase")


SHORT_ABL = {"full": "full", "A1_no_gate": "A1 $-$gate",
             "A2_true_benchmark": "A2 true $\\hat M$", "A3_no_psi": r"A3 $\psi{=}0$",
             "A4_symmetric_phi": r"A4 $\phi^-{=}\phi^+$", "A5_no_adv_concern": r"A5 $\phi^+{=}0$",
             "A6_no_lambda": r"A6 $\lambda{=}0$", "A7_no_eta": r"A7 $\eta{=}0$"}


def fig7_ablations():
    import json
    d = np.load(f"{common.RESULTS}/ablations_{SCALE}.npz", allow_pickle=True)
    order = [str(x) for x in d["order"]]; R = int(d["R"])
    labels = [SHORT_ABL.get(o, o) for o in order]
    x = np.arange(len(order))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.6, 3.2))
    p, lo, hi = d["p_defwrong"], d["p_lo"], d["p_hi"]
    a1.bar(x, p, color=["#222" if o == "full" else "#8c8c8c" for o in order], width=0.7)
    a1.errorbar(x, p, yerr=[np.clip(p - lo, 0, None), np.clip(hi - p, 0, None)],
                fmt="none", ecolor="k", capsize=2, lw=1)
    a1.set_xticks(x); a1.set_xticklabels(labels, rotation=25, ha="right", fontsize=7)
    a1.set_ylabel("P(defensive wrong)"); a1.set_ylim(0, 1.05)
    a1.set_title("(a) Regime probability (Wilson 95%)")
    names = [o for o in order if o != "full"]
    xi = np.arange(len(names))
    diff = np.array([d["d_diff"][order.index(n)] for n in names])
    dl = np.array([d["d_lo"][order.index(n)] for n in names])
    dh = np.array([d["d_hi"][order.index(n)] for n in names])
    cols = ["#2a7a36" if v > 0.5 else ("#6a51a3" if v > 0.1 else "#9e9e9e") for v in diff]
    a2.barh(xi, diff, color=cols, height=0.62)
    a2.errorbar(diff, xi, xerr=[np.clip(diff - dl, 0, None), np.clip(dh - diff, 0, None)],
                fmt="none", ecolor="k", capsize=2, lw=1)
    a2.axvline(0, color="k", lw=0.8)
    a2.set_yticks(xi); a2.set_yticklabels([SHORT_ABL[n] for n in names], fontsize=7)
    a2.set_xlabel(r"$\Delta$P(def-wrong) $=$ full $-$ ablation")
    a2.set_title("(b) Counterfactual effect (paired)")
    fig.suptitle(f"II ablations under common random numbers ($R={R}$)", y=1.02, fontsize=9.5)
    fig.tight_layout(); _save(fig, "fig7_ablations")


if __name__ == "__main__":
    SCALE = common.get_scale()
    d = _load()
    fig2_asymmetry(d); fig3_fairness(d); fig4_regimes(d); fig5_geometry(d)
    fig6_phase(); fig7_ablations()
    print("all Paper II figures done")
