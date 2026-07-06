"""Emit Paper II LaTeX tables from saved results + config (reproducible)."""
import json
import os
import common
from legitimacy.config import Params


def write_results_table(scale):
    s = json.load(open(f"{common.RESULTS}/experiments_{scale}.json"))
    g = s["II.4"]
    L = ["% Auto-generated -- experiment outcomes.",
         r"\begin{table}[t]\centering\small",
         r"\caption{Experiment outcomes (standard scale). $m(a^\star)$ is terminal "
         r"optimal-arm mass; $\bar L$ is terminal mean legitimacy; $G^L$ is the "
         r"between-community legitimacy gap. Scenario parameters: "
         r"\texttt{code/legitimacy/scenarios.py}.}",
         r"\label{tab:results}",
         r"\begin{tabular}{lll}\toprule",
         r"Experiment & Manipulation & Outcome \\ \midrule",
         r"II.1 adv.\ vs disadv. & $\Delta^F=\pm x$, $|x|$ fixed & "
         f"legitimacy loss {s['II.1']['x=0.6']['drop_disadv']:.2f} (disadv.) vs "
         f"{s['II.1']['x=0.6']['drop_adv']:.2f} (adv.) \\\\",
         r"II.2 acceptability & sweep $\tau^F$ & "
         r"mean $L$ rises $0.05\!\to\!1.00$ across $\tau^F$ \\",
         r"II.3 modal misrep. & $\hat M_1\neq\hat M_2$ & "
         f"fairness polarisation, $G^L={s['II.3']['GL_terminal']:.2f}$ \\\\",
         r"II.4 no gate & legitimacy off & "
         f"$m(a^\\star)={g['no-gate']['m_star']:.2f}$, legitimate consensus \\\\",
         r"\quad\ gate & legitimacy on & "
         f"$m(a^\\star)={g['gate']['m_star']:.2f}$, $\\bar L={g['gate']['meanL']:.2f}$: "
         r"\textbf{defensive wrong} \\",
         r"II.6 correct/illegit. & high entitlement & "
         f"$m(a^\\star)={s['II.6']['m_star']:.2f}$, $\\bar L={s['II.6']['meanL']:.2f}$ \\\\",
         r"\bottomrule\end{tabular}", r"\end{table}"]
    open(f"{common.TABDIR}/results_table.tex", "w").write("\n".join(L) + "\n")


def write_param_table():
    p = Params()
    rows = [
        (r"$\mu_1,\mu_2$", f"{p.mu[0]},{p.mu[1]}", "arm reward means", "mu"),
        (r"$\lambda$", p.lam, "anticipatory weight", "lam"),
        (r"$\eta$", p.eta, "retrospective weight", "eta"),
        (r"$\beta,\sigma,a$", f"{p.beta},{p.sigma},{p.a_thr}", "DDM choice", "beta,sigma,a\\_thr"),
        (r"$\alpha$", p.alpha, "private learning rate", "alpha"),
        (r"$\nu_F,\nu_R$", f"{p.nu_F},{p.nu_R}", "legitimacy loss / recovery", "nu\\_F,nu\\_R"),
        (r"$\tau^F$", p.tau_F, "acceptability threshold", "tau\\_F"),
        (r"$\phi^-,\phi^+$", f"{p.phi_minus},{p.phi_plus}", "discomfort weights ($\\phi^->\\phi^+$)", "phi\\_minus,phi\\_plus"),
        (r"$\psi$", p.psi, "modal-agent transmission", "psi"),
        (r"$L_0$", p.L0, "initial legitimacy", "L0"),
    ]
    L = ["% Auto-generated -- default parameters.",
         r"\begin{table}[t]\centering\small",
         r"\caption{Global default parameters (\texttt{code/legitimacy/config.py}); "
         r"per-experiment values are in \texttt{scenarios.py}. Values are projected to "
         r"$[0,1]$; the fairness-in-choice term is off by default.}",
         r"\label{tab:params}",
         r"\begin{tabular}{llll}\toprule",
         r"Symbol & Default & Role & Config key \\ \midrule"]
    for sym, val, role, key in rows:
        L.append(f"{sym} & {val} & {role} & \\texttt{{{key}}} \\\\")
    L += [r"\bottomrule\end{tabular}", r"\end{table}"]
    open(f"{common.TABDIR}/param_table.tex", "w").write("\n".join(L) + "\n")


def write_ablation_table(scale):
    a = json.load(open(f"{common.RESULTS}/ablations_{scale}.json"))
    av, con = a["variants"], a["contrasts_vs_full"]; R = a["R"]
    pretty = {"full": "full", "A1_no_gate": "A1 no gate", "A2_true_benchmark": "A2 true $\\hat M$",
              "A3_no_psi": "A3 $\\psi{=}0$", "A4_symmetric_phi": "A4 $\\phi^-{=}\\phi^+$",
              "A5_no_adv_concern": "A5 $\\phi^+{=}0$", "A6_no_lambda": "A6 $\\lambda{=}0$",
              "A7_no_eta": "A7 $\\eta{=}0$"}
    order = ["full", "A1_no_gate", "A2_true_benchmark", "A3_no_psi", "A4_symmetric_phi",
             "A5_no_adv_concern", "A6_no_lambda", "A7_no_eta"]
    L = ["% Auto-generated -- ablations.",
         r"\begin{table}[t]\centering\small",
         r"\caption{Ablation effects at the legitimacy-gated wrong-consensus operating "
         r"point, $R=" + str(R) + r"$ under common random numbers. P(def.-wrong) carries a "
         r"Wilson 95\% interval; $\Delta$ is the paired counterfactual effect "
         r"full$-$ablation with a paired-bootstrap 95\% interval.}",
         r"\label{tab:ablation}",
         r"\begin{tabular}{lccc}\toprule",
         r"Variant & P(def.-wrong) [95\%] & $m(a^\star)$ & $\Delta$P(def.-wrong) [95\%] \\ \midrule"]
    for n in order:
        v = av[n]; pw = v["p_defwrong"]; ms = v["mean_mstar"]["mean"]
        cell = f"{pretty[n]} & {pw['p']:.2f} [{pw['lo']:.2f}, {pw['hi']:.2f}] & {ms:.2f}"
        if n == "full":
            cell += r" & --- \\"
        else:
            c = con[n]; cell += f" & ${c['diff']:+.2f}$ [{c['lo']:+.2f}, {c['hi']:+.2f}] \\\\"
        L.append(cell)
    L += [r"\bottomrule\end{tabular}", r"\end{table}"]
    open(f"{common.TABDIR}/ablation_table.tex", "w").write("\n".join(L) + "\n")


def write_comparison_table():
    L = [r"\begin{table}[t]\centering\small",
         r"\caption{Positioning relative to adjacent model families.}",
         r"\label{tab:comparison}",
         r"\begin{tabular}{p{3.0cm}p{3.3cm}p{2.7cm}p{3.4cm}}\toprule",
         r"Tradition & Typical mechanism & Feedback authority & What this paper changes \\ \midrule",
         r"Social/RL learning & reward-driven value update & fixed (always valid) & legitimacy gate on evidence \\",
         r"Inequity aversion & reference-dependent utility & one-shot, no learning & fairness inside a learning loop \\",
         r"Legitimacy / proc.\ justice & rule acceptance & exogenous & endogenous, socially learned \\",
         r"Opinion dynamics & weighted averaging & no reward rule & reward legitimacy as state \\",
         r"Credibility (Paper~I) & confidence-weighted transmission & valid evidence & feedback can be delegitimised \\",
         r"\bottomrule\end{tabular}", r"\end{table}"]
    open(f"{common.TABDIR}/comparison_table.tex", "w").write("\n".join(L) + "\n")


def write_variables_table():
    rows = [
        (r"$Q_{i,t}(a)$", "latent action value"),
        (r"$C_{i,t}$", "endogenous confidence (balance of evidence)"),
        (r"$S_{i,t}(a)$", "credibility-weighted social signal"),
        (r"$L_{i,t}$", "legitimacy of the reward rule (gates evidence)"),
        (r"$\hat M_{i,t}$", "perceived modal-agent fairness benchmark"),
        (r"$M$", "true modal-agent benchmark"),
        (r"$\Delta^F_{i,t}$", "fairness discrepancy $R-r^F$"),
        (r"$D^F_{i,t}$", "asymmetric fairness discomfort"),
        (r"$\delta^{\mathrm{eff}}_{i,t}$", "legitimacy-gated prediction error $L\\,\\delta$"),
    ]
    L = [r"\begin{table}[t]\centering\small",
         r"\caption{Model variables and interpretations.}", r"\label{tab:variables}",
         r"\begin{tabular}{ll}\toprule Symbol & Interpretation \\ \midrule"]
    for s, d in rows:
        L.append(f"{s} & {d} \\\\")
    L += [r"\bottomrule\end{tabular}", r"\end{table}"]
    open(f"{common.TABDIR}/variables_table.tex", "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    scale = common.get_scale()
    write_results_table(scale)
    write_param_table()
    write_ablation_table(scale)
    write_comparison_table()
    write_variables_table()
    print(f"wrote tables to {common.TABDIR}")
