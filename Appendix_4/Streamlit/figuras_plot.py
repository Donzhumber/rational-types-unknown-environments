"""Utilidades compartidas para figuras APA en figuras_calibracion/."""
from __future__ import annotations

from typing import Any, Optional

import pandas as pd

# Banda vertical clara (mismo tono que Mechanism_2 / esp.tex)
TAU_BAND_FACE = "#E8E8E8"
TAU_BAND_ALPHA = 0.92


def is_continuar_desenlace(m: Any) -> bool:
    s = str(m or "").strip().lower()
    if not s or s in {"—", "-", "nan", "none"}:
        return True
    if s.startswith("continuar"):
        return True
    if s in {"cont", "continuar (a_cont)"}:
        return True
    return False


def terminal_tau_from_frame(
    df: pd.DataFrame,
    *,
    tau_col: str = "tau",
    m_col: str = "m",
    tau_hyp_col: str = "tau_hyp",
    m_tau_col: str = "m_tau",
) -> tuple[Optional[int], str]:
    """Un solo τ^{hyp}: el primer desenlace m ≠ Continuar (absorción hipotética).

    En modo Continuar puede haber sorteos m_draw terminales posteriores; se ignoran.
    Prioridad: columnas tau_hyp/m_tau, luego el primer m observable distinto de Continuar.
    """
    if df is None or df.empty:
        return None, "Continuar"

    d = df.sort_values(tau_col)

    if tau_hyp_col in d.columns:
        hyp = pd.to_numeric(d[tau_hyp_col], errors="coerce")
        valid = hyp[hyp >= 0]
        if len(valid):
            tau_star = int(valid.min())
            m_star = "Continuar"
            if m_tau_col in d.columns:
                at = d.loc[d[tau_col] == tau_star, m_tau_col]
                if not at.empty:
                    m_star = str(at.iloc[0])
                else:
                    m_star = str(d.loc[hyp >= 0, m_tau_col].iloc[0])
            if not is_continuar_desenlace(m_star):
                return tau_star, m_star

    if m_col in d.columns:
        for _, row in d.iterrows():
            m_val = str(row[m_col])
            if not is_continuar_desenlace(m_val):
                return int(row[tau_col]), m_val

    if "m_draw" in d.columns and m_col not in d.columns:
        for _, row in d.iterrows():
            m_val = str(row["m_draw"])
            if not is_continuar_desenlace(m_val):
                return int(row[tau_col]), m_val

    return None, "Continuar"


def draw_tau_desenlace_band(
    ax,
    tau: int,
    m_tau: Any,
    *,
    label: bool = True,
) -> bool:
    """Una sola banda gris en τ del primer desenlace terminal. Retorna False si m es Continuar."""
    if is_continuar_desenlace(m_tau):
        return False
    for artist in list(ax.patches):
        if artist.get_label() == "_tau_hyp_band":
            artist.remove()
    ax.axvspan(
        tau - 0.5,
        tau + 0.5,
        facecolor=TAU_BAND_FACE,
        alpha=TAU_BAND_ALPHA,
        linewidth=0,
        zorder=0,
        label="_tau_hyp_band",
    )
    if label:
        ax.text(
            tau,
            ax.get_ylim()[1],
            r"$\tau^{\mathrm{hyp}}$",
            ha="center",
            va="bottom",
            fontsize=7,
            color="0.25",
            clip_on=False,
        )
    return True


def annotate_tau_in_legend(m_tau: str) -> str:
    if is_continuar_desenlace(m_tau):
        return r"$\tau^{\mathrm{hyp}}$"
    return rf"$\tau^{{\mathrm{{hyp}}}}$ ({m_tau})"
