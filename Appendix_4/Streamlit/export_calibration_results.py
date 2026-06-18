from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model_logic import DESENLACES, TIPOS_SECUESTRADOR, ModeloSecuestro
from rational_behavior import (
    bayesian_posterior_update,
    mechanism_competitive_hazards_at_t,
)
from figuras_plot import (
    annotate_tau_in_legend,
    draw_tau_desenlace_band,
    terminal_tau_from_frame,
)


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "figuras_calibracion"
TEX_FIG_DIR = ROOT / "Final_tex" / "Identification_of_rational_types"
APA_PDF_NAMES = (
    "fig_mu_dc_farc.pdf",
    "fig_alpha_dc_farc.pdf",
    "fig_gamma_dc_farc.pdf",
    "fig_iota_alpha_dc_farc.pdf",
    "fig_iota_gamma_dc_farc.pdf",
    "fig_deltaH_dc_farc.pdf",
)
TYPES = ("DC", "FARC")
T_MAX = 80
SEED = 123


PARAMS = {
    "presion_S": 0.45,
    "z_region": "Metropolitana",
    "v_victim": "Privado",
    "zeta_alpha": 0.34,
    "zeta_gamma": 0.62,
    "zeta_d": 0.22,
    "zeta_R": 0.28,
    "t_mad": 14.0,
    "lambda4": 0.002,
    "eta0": -2.15,
    "eta1": 1.10,
    "eta2": 1.35,
    "omega_pay": 10.0,
    "omega_death": 42.0,
    "omega_rescue": 8.0,
    "omega_release": 3.0,
    "chi_alpha": 2.2,
    "chi_gamma": 2.6,
    "info_weight": 7.5,
}


def _entropy(mu: dict[str, float]) -> float:
    vals = np.array([max(1e-12, float(mu.get(th, 0.0))) for th in TIPOS_SECUESTRADOR])
    return float(-(vals * np.log(vals)).sum())


def _p_det(alpha: float, gamma: float) -> float:
    x = PARAMS["eta0"] + PARAMS["eta1"] * alpha + PARAMS["eta2"] * gamma
    return float(1.0 / (1.0 + np.exp(-x)))


def _factors(modelo: ModeloSecuestro, theta: str, t: int, alpha: float, gamma: float) -> dict[str, Any]:
    return mechanism_competitive_hazards_at_t(
        modelo,
        theta,
        t,
        presion_S=PARAMS["presion_S"],
        z_region=PARAMS["z_region"],
        v_victim=PARAMS["v_victim"],
        alpha=alpha,
        gamma=gamma,
        p_det=_p_det(alpha, gamma),
        zeta_alpha=PARAMS["zeta_alpha"],
        zeta_gamma=PARAMS["zeta_gamma"],
        zeta_d=PARAMS["zeta_d"],
        zeta_R=PARAMS["zeta_R"],
        estado_rescata=False,
        t_mad=PARAMS["t_mad"],
        lambda4=PARAMS["lambda4"],
        atilde_K="Continuar",
        atilde_F="Cooperar",
        atilde_S="Negociar",
    )


def _likelihoods(modelo: ModeloSecuestro, t: int, alpha: float, gamma: float, outcome: str) -> dict[str, float]:
    return {
        th: float(max(1e-300, _factors(modelo, th, t, alpha, gamma)["h_daily"].get(outcome, 1e-300)))
        for th in TIPOS_SECUESTRADOR
    }


def _predictive_outcome_probs(modelo: ModeloSecuestro, mu: dict[str, float], t: int, alpha: float, gamma: float) -> dict[str, float]:
    acc = {m: 0.0 for m in DESENLACES}
    for th in TIPOS_SECUESTRADOR:
        hd = _factors(modelo, th, t, alpha, gamma)["h_daily"]
        for m in DESENLACES:
            acc[m] += float(mu.get(th, 0.0)) * float(hd.get(m, 0.0))
    s = sum(acc.values())
    if s <= 1e-15:
        return {m: 1.0 / len(DESENLACES) for m in DESENLACES}
    return {m: float(v / s) for m, v in acc.items()}


def _info_gain(modelo: ModeloSecuestro, mu: dict[str, float], t: int, alpha: float, gamma: float) -> float:
    h0 = _entropy(mu)
    pred = _predictive_outcome_probs(modelo, mu, t, alpha, gamma)
    eh = 0.0
    for m, pm in pred.items():
        mu_post, _ = bayesian_posterior_update(mu, _likelihoods(modelo, t, alpha, gamma, m))
        eh += float(pm) * _entropy(mu_post)
    return float(max(0.0, h0 - eh))


def _policy_loss(
    modelo: ModeloSecuestro,
    mu: dict[str, float],
    t: int,
    alpha: float,
    gamma: float,
    *,
    branch: str,
    include_info: bool,
) -> float:
    pred = _predictive_outcome_probs(modelo, mu, t, alpha, gamma)
    base = (
        PARAMS["omega_pay"] * pred["Pago"] * (1.0 - alpha)
        + PARAMS["omega_death"] * pred["Muerte"]
        + PARAMS["omega_rescue"] * pred["Rescate"]
        + PARAMS["omega_release"] * pred["Liberación"]
    )
    if branch == "R":
        ref_alpha, ref_gamma = 0.72, 0.78
        branch_cost = 0.55 * (1.0 - pred["Rescate"])
    else:
        ref_alpha, ref_gamma = 0.34, 0.38
        branch_cost = 0.35 * pred["Pago"]
    inst_cost = (
        PARAMS["chi_alpha"] * (alpha - ref_alpha) ** 2
        + PARAMS["chi_gamma"] * (gamma - ref_gamma) ** 2
        + 0.35 * alpha * gamma
    )
    info = PARAMS["info_weight"] * _info_gain(modelo, mu, t, alpha, gamma) if include_info else 0.0
    return float(base + branch_cost + inst_cost - info)


def _argmin_grid(modelo: ModeloSecuestro, mu: dict[str, float], t: int, branch: str, include_info: bool) -> tuple[float, float, float]:
    grid = np.linspace(0.05, 0.95, 15)
    best = (0.5, 0.5, np.inf)
    for alpha in grid:
        for gamma in grid:
            loss = _policy_loss(modelo, mu, t, float(alpha), float(gamma), branch=branch, include_info=include_info)
            if loss < best[2]:
                best = (float(alpha), float(gamma), float(loss))
    return best


def simulate_type(theta_true: str, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    modelo = ModeloSecuestro()
    mu = {"DC": 0.25, "PAR": 0.20, "ELN": 0.20, "FARC": 0.35}
    rows: list[dict[str, Any]] = []
    tau_hyp: int | None = None
    m_tau = "Continuar"

    for t in range(T_MAX + 1):
        a_r, g_r, l_r = _argmin_grid(modelo, mu, max(1, t), "R", include_info=False)
        a_n, g_n, l_n = _argmin_grid(modelo, mu, max(1, t), "N", include_info=False)
        branch = "R" if l_r <= l_n else "N"
        a_star, g_star, _ = _argmin_grid(modelo, mu, max(1, t), branch, include_info=True)
        iota = max(mu.values())
        dh = _info_gain(modelo, mu, max(1, t), a_star, g_star)
        true_probs = _factors(modelo, theta_true, max(1, t), a_star, g_star)["h_daily"]
        p = np.array([float(true_probs[m]) for m in DESENLACES], dtype=float)
        p = p / p.sum()
        m_draw = str(rng.choice(DESENLACES, p=p))
        if tau_hyp is None and m_draw != "Continuar":
            tau_hyp = t
            m_tau = m_draw
        rows.append(
            {
                "theta_true": theta_true,
                "t": t,
                "mu_true": float(mu[theta_true]),
                **{f"mu_{th}": float(mu[th]) for th in TIPOS_SECUESTRADOR},
                "alpha": a_star,
                "gamma": g_star,
                "alpha_R": a_r,
                "gamma_R": g_r,
                "alpha_N": a_n,
                "gamma_N": g_n,
                "iota": iota,
                "Delta_H": dh,
                "branch": branch,
                "m_draw": m_draw,
                "tau_hyp": -1 if tau_hyp is None else tau_hyp,
                "m_tau": m_tau,
            }
        )
        mu, _ = bayesian_posterior_update(mu, _likelihoods(modelo, max(1, t), a_star, g_star, m_draw))

    if tau_hyp is None:
        m_tau = "Continuar"
        tau_hyp_store = -1
    else:
        tau_hyp_store = int(tau_hyp)
    out = pd.DataFrame(rows)
    out["tau_hyp"] = tau_hyp_store
    return out


def _setup_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", color="0.88", linewidth=0.6)
    ax.tick_params(axis="both", labelsize=8)


def _paired_time_figure(
    df: pd.DataFrame,
    filename: str,
    ycols: list[tuple[str, str, str, str]],
    ylabel: str,
    ylim: tuple[float, float] | None = None,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.05), sharex=True)
    for ax, th, letter in zip(axes, TYPES, ("A", "B")):
        d = df[df["theta_true"] == th].copy()
        tau_t, m_tau = terminal_tau_from_frame(
            d, tau_col="t", m_col="m_draw", tau_hyp_col="tau_hyp", m_tau_col="m_tau"
        )
        for col, label, linestyle, color in ycols:
            ax.plot(d["t"], d[col], linestyle=linestyle, color=color, linewidth=1.15, label=label)
        _setup_axes(ax)
        ax.set_title(f"({letter.lower()}) {th}", fontsize=9, loc="left", pad=9)
        ax.set_xlabel("Periodo $t$", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_xlim(0, T_MAX)
        ax.set_xticks(np.arange(0, T_MAX + 1, 10))
        ax.set_xticks(np.arange(0, T_MAX + 1, 1), minor=True)
        if ylim is not None:
            ax.set_ylim(*ylim)
        if tau_t is not None:
            draw_tau_desenlace_band(ax, tau_t, m_tau)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(3, len(labels)), frameon=False, fontsize=7)
    fig.tight_layout(rect=(0, 0.17, 1, 0.98), w_pad=2.1)
    fig.savefig(OUT_DIR / filename, bbox_inches="tight")
    plt.close(fig)


def _paired_iota_figure(df: pd.DataFrame, filename: str, ycol: str, ylabel: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.05), sharex=True, sharey=True)
    for ax, th, letter in zip(axes, TYPES, ("A", "B")):
        d = df[df["theta_true"] == th].copy()
        tau_t, m_tau = terminal_tau_from_frame(
            d, tau_col="t", m_col="m_draw", tau_hyp_col="tau_hyp", m_tau_col="m_tau"
        )
        ax.plot(d["iota"], d[ycol], color="0.0", linewidth=1.0, marker="o", markersize=2.2, label="Trayectoria")
        if tau_t is not None:
            d_tau = d[d["t"] == tau_t]
            if not d_tau.empty:
                ax.plot(
                    d_tau["iota"],
                    d_tau[ycol],
                    marker="o",
                    markersize=6.2,
                    markerfacecolor="white",
                    markeredgecolor="0.0",
                    linestyle="None",
                    label=annotate_tau_in_legend(m_tau),
                )
        _setup_axes(ax)
        ax.set_title(f"({letter.lower()}) {th}", fontsize=9, loc="left", pad=9)
        ax.set_xlabel(r"Precisión posterior $\iota_t$", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks(np.arange(0, 1.01, 0.2))
        ax.set_yticks(np.arange(0, 1.01, 0.2))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=7)
    fig.tight_layout(rect=(0, 0.17, 1, 0.98), w_pad=2.1)
    fig.savefig(OUT_DIR / filename, bbox_inches="tight")
    plt.close(fig)


def make_figures(df: pd.DataFrame) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.edgecolor": "0.15",
            "axes.linewidth": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    _paired_time_figure(
        df,
        "fig_mu_dc_farc.pdf",
        [("mu_true", r"$\mu_t(\theta_K^*)$", "-", "0.0")],
        "Posterior",
        (0.0, 1.02),
    )
    _paired_time_figure(
        df,
        "fig_alpha_dc_farc.pdf",
        [
            ("alpha", r"$\alpha_t^*$", "--", "0.0"),
            ("alpha_R", r"$\alpha^R$", ":", "0.35"),
            ("alpha_N", r"$\alpha^N$", "-.", "0.35"),
        ],
        r"Bloqueo financiero $\alpha$",
        (0.0, 1.0),
    )
    _paired_time_figure(
        df,
        "fig_gamma_dc_farc.pdf",
        [
            ("gamma", r"$\gamma_t^*$", "-", "0.0"),
            ("gamma_R", r"$\gamma^R$", ":", "0.35"),
            ("gamma_N", r"$\gamma^N$", "-.", "0.35"),
        ],
        r"Presión operativa $\gamma$",
        (0.0, 1.0),
    )
    _paired_iota_figure(
        df,
        "fig_iota_alpha_dc_farc.pdf",
        "alpha",
        r"Bloqueo financiero $\alpha_t^*$",
    )
    _paired_iota_figure(
        df,
        "fig_iota_gamma_dc_farc.pdf",
        "gamma",
        r"Presión operativa $\gamma_t^*$",
    )
    _paired_time_figure(
        df,
        "fig_deltaH_dc_farc.pdf",
        [("Delta_H", r"$\Delta H_t$", "-", "0.0")],
        "Ganancia de entropía",
        None,
    )


def make_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for th in TYPES:
        d = df[df["theta_true"] == th].copy()
        tau_t, m_tau = terminal_tau_from_frame(
            d, tau_col="t", m_col="m_draw", tau_hyp_col="tau_hyp", m_tau_col="m_tau"
        )
        rows.append(
            {
                "theta": th,
                "mu0": d["mu_true"].iloc[0],
                "muT": d["mu_true"].iloc[-1],
                "alpha_bar": d["alpha"].mean(),
                "gamma_bar": d["gamma"].mean(),
                "iota_bar": d["iota"].mean(),
                "deltaH_bar": d["Delta_H"].mean(),
                "tau_hyp": -1 if tau_t is None else int(tau_t),
                "m_tau": m_tau,
            }
        )
    return pd.DataFrame(rows)


def write_latex_table(tab: pd.DataFrame) -> None:
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Resumen de calibración dinámica en modo Continuar}",
        r"\label{tab:calibracion-resumen}",
        r"\small",
        r"\begin{tabular}{lrrrrrrrl}",
        r"\toprule",
        r"$\theta_K^\ast$ & $\mu_0$ & $\mu_T$ & $\bar{\alpha}^\ast$ & $\bar{\gamma}^\ast$ & $\bar{\iota}$ & $\overline{\Delta H}$ & $\tau^{hyp}$ & $m_{\tau^{hyp}}$ \\",
        r"\midrule",
    ]
    for _, r in tab.iterrows():
        lines.append(
            f"{r['theta']} & "
            f"{r['mu0']:.3f} & "
            f"{r['muT']:.3f} & "
            f"{r['alpha_bar']:.3f} & "
            f"{r['gamma_bar']:.3f} & "
            f"{r['iota_bar']:.3f} & "
            f"{r['deltaH_bar']:.4f} & "
            f"{int(r['tau_hyp']) if int(r['tau_hyp']) >= 0 else '—'} & "
            f"{r['m_tau']} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )
    (OUT_DIR / "tabla_resumen_calibracion.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def sync_figures_to_tex_dir() -> None:
    """Copia los PDF del manuscrito junto a Identification_of_rational_types_esp.tex."""
    TEX_FIG_DIR.mkdir(parents=True, exist_ok=True)
    for name in APA_PDF_NAMES:
        src = OUT_DIR / name
        if src.is_file():
            shutil.copy2(src, TEX_FIG_DIR / name)
    print(f"Figuras APA copiadas a: {TEX_FIG_DIR}")


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    frames = [simulate_type(th, SEED + i * 1000) for i, th in enumerate(TYPES)]
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(OUT_DIR / "calibration_results.csv", index=False)
    tab = make_table(df)
    tab.to_csv(OUT_DIR / "tabla_resumen_calibracion.csv", index=False)
    write_latex_table(tab)
    make_figures(df)
    sync_figures_to_tex_dir()
    print(tab.to_string(index=False))


if __name__ == "__main__":
    os.environ.setdefault("MPLBACKEND", "Agg")
    main()
