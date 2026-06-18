"""Informe PDF e infografía para la pestaña 6 (corrida dinámica)."""
from __future__ import annotations

import io
from datetime import datetime
from typing import Any, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

TIPOS_DEFAULT = ("DC", "PAR", "ELN", "FARC")


def _fig_to_png(fig: go.Figure, width: int = 900, height: int = 520) -> Optional[bytes]:
    try:
        import plotly.io as pio

        return pio.to_image(fig, format="png", width=width, height=height, scale=2)
    except Exception:
        return None


def _safe_mean(s: pd.Series) -> float:
    v = pd.to_numeric(s, errors="coerce")
    return float(v.mean()) if v.notna().any() else float("nan")


def _fmt_pct(v: float) -> str:
    return f"{v:.0f}%" if np.isfinite(v) else "n.d."


def _compact_note(text: str, max_len: int = 92) -> str:
    text = " ".join(str(text).replace("**", "").split())
    return text if len(text) <= max_len else text[: max_len - 1].rstrip() + "…"


def build_dyn_infographic(
    df_cyc: pd.DataFrame,
    df_mu_dyn: pd.DataFrame,
    *,
    tipo_real: str,
    tipo_colors: dict[str, str],
    tipos: tuple[str, ...] = TIPOS_DEFAULT,
    tau_stop: Optional[int] = None,
    th_dom0: str = "",
    th_domf: str = "",
    mu_t0: Optional[dict[str, float]] = None,
    mu_tf: Optional[dict[str, float]] = None,
    n_cyc: int = 1,
    meta: Optional[dict[str, Any]] = None,
    narrative_sections: Optional[list[tuple[str, list[str]]]] = None,
) -> go.Figure:
    """Panel resumen (estilo infografía) en un solo gráfico Plotly."""
    mu_t0 = mu_t0 or {}
    mu_tf = mu_tf or {}
    meta = meta or {}
    narrative_sections = narrative_sections or []
    fig = make_subplots(
        rows=3,
        cols=3,
        subplot_titles=(
            "1. Creencias μ",
            "2. α* Estado",
            "3. γ* Estado",
            "4. MDG",
            "5. Voz y detección",
            "6. ΔH",
            "7. IR/IC K y F",
            "8. γ* vs ι",
            "9. α* vs ι",
        ),
        vertical_spacing=0.22,
        horizontal_spacing=0.08,
    )

    if not df_mu_dyn.empty:
        for th in tipos:
            if th not in df_mu_dyn.columns:
                continue
            fig.add_trace(
                go.Scatter(
                    x=df_mu_dyn["tau"],
                    y=df_mu_dyn[th],
                    mode="lines+markers",
                    name=th,
                    line=dict(color=tipo_colors.get(th, "#64748b"), width=2),
                    legendgroup="mu",
                    showlegend=True,
                ),
                row=1,
                col=1,
            )
        if tau_stop is not None:
            fig.add_vline(
                x=tau_stop, line_dash="dot", line_color="#dc2626",
                row=1, col=1,
            )

    if not df_cyc.empty:
        fig.add_trace(
            go.Scatter(
                x=df_cyc["tau"], y=df_cyc["alpha"], mode="lines+markers",
                name="α*", line=dict(color="#1d4ed8", width=2),
                legendgroup="a",
            ),
            row=1, col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=df_cyc["tau"], y=df_cyc["alpha_R"], mode="lines",
                name="α^R", line=dict(color="#60a5fa", dash="dash"),
                legendgroup="a",
            ),
            row=1, col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=df_cyc["tau"], y=df_cyc["alpha_N"], mode="lines",
                name="α^N", line=dict(color="#f59e0b", dash="dot"),
                legendgroup="a",
            ),
            row=1, col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=df_cyc["tau"], y=df_cyc["gamma"], mode="lines+markers",
                name="γ*", line=dict(color="#15803d", width=2),
                legendgroup="g",
            ),
            row=1, col=3,
        )
        fig.add_trace(
            go.Scatter(
                x=df_cyc["tau"], y=df_cyc["gamma_R"], mode="lines",
                name="γ^R", line=dict(color="#86efac", dash="dash"),
                legendgroup="g",
            ),
            row=1, col=3,
        )
        fig.add_trace(
            go.Scatter(
                x=df_cyc["tau"], y=df_cyc["gamma_N"], mode="lines",
                name="γ^N", line=dict(color="#fb923c", dash="dot"),
                legendgroup="g",
            ),
            row=1, col=3,
        )

        _match_pct = [
            100.0 * float(df_cyc["match_S"].mean()),
            100.0 * float(df_cyc["match_F"].mean()),
            100.0 * float(df_cyc["match_K"].mean()),
        ]
        fig.add_trace(
            go.Bar(
                x=["S", "F", "K"], y=_match_pct,
                marker_color=["#1d4ed8", "#7c3aed", "#b45309"],
                text=[f"{v:.0f}%" for v in _match_pct],
                textposition="auto",
                showlegend=False,
            ),
            row=2, col=1,
        )

        _vc_v = df_cyc["V"].value_counts().reindex([0, 1], fill_value=0)
        _vc_d = df_cyc["d"].value_counts().reindex([0, 1], fill_value=0)
        _nn = max(1, n_cyc)
        fig.add_trace(
            go.Bar(
                x=["V=0", "V=1", "d=0", "d=1"],
                y=[
                    100.0 * float(_vc_v.get(0, 0)) / _nn,
                    100.0 * float(_vc_v.get(1, 0)) / _nn,
                    100.0 * float(_vc_d.get(0, 0)) / _nn,
                    100.0 * float(_vc_d.get(1, 0)) / _nn,
                ],
                marker_color=["#94a3b8", "#2563eb", "#cbd5e1", "#f59e0b"],
                showlegend=False,
            ),
            row=2, col=2,
        )

        fig.add_trace(
            go.Bar(
                x=df_cyc["tau"], y=df_cyc["Delta_H"],
                marker_color="#7c3aed", showlegend=False,
            ),
            row=2, col=3,
        )

        _ir_labels = ["IR^K", "IC^K", "IR^F", "IC^F"]
        _ir_cols = ["IR_K", "IC_K", "IR_F", "IC_F"]
        _ir_y = [100.0 * float(df_cyc[c].mean()) for c in _ir_cols]
        fig.add_trace(
            go.Bar(
                x=_ir_labels,
                y=_ir_y,
                marker_color=px.colors.qualitative.Set2[:4],
                text=[f"{v:.0f}%" for v in _ir_y],
                textposition="auto",
                showlegend=False,
            ),
            row=3, col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=df_cyc["gamma"], y=df_cyc["iota"], mode="markers",
                customdata=df_cyc["tau"],
                hovertemplate="γ*=%{x:.4f}<br>ι=%{y:.4f}<br>τ=%{customdata}<extra></extra>",
                marker=dict(size=9, color="#15803d"), showlegend=False,
            ),
            row=3, col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=df_cyc["alpha"], y=df_cyc["iota"], mode="markers",
                customdata=df_cyc["tau"],
                hovertemplate="α*=%{x:.4f}<br>ι=%{y:.4f}<br>τ=%{customdata}<extra></extra>",
                marker=dict(size=9, color="#1d4ed8"), showlegend=False,
            ),
            row=3, col=3,
        )

    _a_mean = _safe_mean(df_cyc["alpha"]) if not df_cyc.empty else float("nan")
    _g_mean = _safe_mean(df_cyc["gamma"]) if not df_cyc.empty else float("nan")
    _dh_mean = _safe_mean(df_cyc["Delta_H"]) if not df_cyc.empty else float("nan")
    _mu0s = mu_t0.get(th_dom0, 0.0) if th_dom0 else 0.0
    _mufs = mu_tf.get(th_domf, 0.0) if th_domf else 0.0
    _match_s = float(meta.get("match_S", float("nan")))
    _match_f = float(meta.get("match_F", float("nan")))
    _match_k = float(meta.get("match_K", float("nan")))
    _pct_v1 = float(meta.get("pct_v1", float("nan")))
    _pct_d1 = float(meta.get("pct_d1", float("nan")))
    _dh_max = float(meta.get("dh_max", float("nan")))
    _ir_min = min(_ir_y) if not df_cyc.empty and "_ir_y" in locals() and _ir_y else float("nan")
    _narr_tail = ""
    if narrative_sections:
        try:
            _first_block = narrative_sections[0][1]
            _narr_tail = _compact_note(_first_block[0], 76) if _first_block else ""
        except Exception:
            _narr_tail = ""
    _analysis = [
        _compact_note(f"Lectura: domina {th_dom0 or 'n.d.'} al inicio y {th_domf or 'n.d.'} al cierre; θ*={tipo_real}."),
        _compact_note(f"α* media={_a_mean:.3f}; se contrasta con referencias R y N sin incertidumbre."),
        _compact_note(f"γ* media={_g_mean:.3f}; muestra la intensidad operativa elegida por el Estado."),
        _compact_note(
            f"Coincidencia S/F/K={_fmt_pct(_match_s)}/{_fmt_pct(_match_f)}/{_fmt_pct(_match_k)}."
        ),
        _compact_note(f"Voz V=1={_fmt_pct(_pct_v1)}; detección d=1={_fmt_pct(_pct_d1)}."),
        _compact_note(f"ΔH media={_dh_mean:.4f}; máximo={_dh_max:.4f}. Mide aprendizaje esperado."),
        _compact_note(f"IR/IC de K y F; mínimo observado={_fmt_pct(_ir_min)}."),
        _compact_note("Pendiente visual: cómo cambia la precisión ι cuando sube o baja γ*."),
        _compact_note("Pendiente visual: cómo cambia la precisión ι cuando sube o baja α*."),
    ]
    if _narr_tail:
        _analysis[0] = _compact_note(_analysis[0] + " " + _narr_tail, 108)

    fig.update_layout(
        title=dict(
            text=(
                f"<b>Infografía de la corrida dinámica del mecanismo</b><br>"
                f"<sup>θ*={tipo_real} · {n_cyc} ciclos · "
                f"μ₀: {th_dom0} ({_mu0s:.2f}) → μ_T: {th_domf} ({_mufs:.2f}) · "
                f"prom(α*)={_a_mean:.3f} · prom(γ*)={_g_mean:.3f} · prom(ΔH)={_dh_mean:.3f}</sup>"
            ),
            x=0.02,
            xanchor="left",
        ),
        height=1320,
        margin=dict(t=175, b=185, l=82, r=150),
        legend=dict(
            title_text="Series",
            orientation="v",
            yanchor="top",
            y=0.98,
            xanchor="left",
            x=1.015,
            bgcolor="rgba(255,255,255,0.88)",
            bordercolor="rgba(148,163,184,0.55)",
            borderwidth=1,
        ),
        template="plotly_white",
    )
    for (x, y), txt in zip(
        (
            (0.155, 0.625), (0.500, 0.625), (0.845, 0.625),
            (0.155, 0.275), (0.500, 0.275), (0.845, 0.275),
            (0.155, -0.125), (0.500, -0.125), (0.845, -0.125),
        ),
        _analysis,
    ):
        fig.add_annotation(
            x=x,
            y=y,
            xref="paper",
            yref="paper",
            text=f"<span style='font-size:9px'><b>Análisis:</b> {txt}</span>",
            showarrow=False,
            align="center",
            xanchor="center",
            yanchor="bottom",
            bgcolor="rgba(255,255,255,0.92)",
            bordercolor="rgba(148,163,184,0.45)",
            borderwidth=1,
            borderpad=3,
        )
    fig.update_xaxes(title_text="", row=1, col=1)
    fig.update_yaxes(title_text="Creencia μτ(θ)", row=1, col=1)
    fig.update_xaxes(title_text="", row=1, col=2)
    fig.update_yaxes(title_text="Nivel α ∈ [0,1]", row=1, col=2)
    fig.update_xaxes(title_text="", row=1, col=3)
    fig.update_yaxes(title_text="Nivel γ ∈ [0,1]", row=1, col=3)
    fig.update_xaxes(title_text="", row=2, col=1)
    fig.update_yaxes(title_text="% de coincidencia", row=2, col=1)
    fig.update_xaxes(title_text="", row=2, col=2)
    fig.update_yaxes(title_text="% de ciclos", row=2, col=2)
    fig.update_xaxes(title_text="", row=2, col=3)
    fig.update_yaxes(title_text="Ganancia ΔH", row=2, col=3)
    fig.update_xaxes(title_text="Restricción", row=3, col=1)
    fig.update_yaxes(title_text="% de ciclos cumplidos", row=3, col=1)
    fig.update_xaxes(title_text="γ*", row=3, col=2)
    fig.update_yaxes(title_text="Precisión posterior ι", row=3, col=2)
    fig.update_xaxes(title_text="α*", row=3, col=3)
    fig.update_yaxes(title_text="Precisión posterior ι", row=3, col=3)
    for r, c in ((1, 1), (1, 2), (1, 3), (3, 2), (3, 3)):
        fig.update_yaxes(range=[0, 1.05], row=r, col=c)
    fig.update_yaxes(range=[0, 105], row=2, col=1)
    fig.update_yaxes(range=[0, 105], row=2, col=2)
    fig.update_yaxes(range=[0, 105], row=3, col=1)
    fig.update_xaxes(automargin=True, title_standoff=12, tickfont=dict(size=10))
    fig.update_yaxes(automargin=True, title_standoff=12, tickfont=dict(size=10), title_font=dict(size=10))
    for ann in fig.layout.annotations:
        if ann.text and not str(ann.text).startswith("<span"):
            ann.font = dict(size=11)
    return fig


def _pdf_styles():
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="DynTitle", parent=styles["Heading1"],
        fontSize=16, spaceAfter=10, alignment=TA_CENTER,
    ))
    styles.add(ParagraphStyle(
        name="DynH2", parent=styles["Heading2"],
        fontSize=12, spaceBefore=12, spaceAfter=6,
    ))
    styles.add(ParagraphStyle(
        name="DynBody", parent=styles["Normal"],
        fontSize=9, leading=12, alignment=TA_JUSTIFY,
    ))
    styles.add(ParagraphStyle(
        name="DynSmall", parent=styles["Normal"],
        fontSize=8, leading=10, textColor="#444444",
    ))
    return styles


def _pdf_table_from_df(df: pd.DataFrame, max_rows: int = 40) -> Any:
    from reportlab.lib import colors
    from reportlab.platypus import Table, TableStyle

    _df = df.head(max_rows).copy()
    data = [_df.columns.astype(str).tolist()] + [
        [str(x)[:28] for x in row] for row in _df.values.tolist()
    ]
    tbl = Table(data, repeatRows=1)
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e3a5f")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 7),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return tbl


def build_dyn_pdf_report(
    df_cyc: pd.DataFrame,
    df_mu_dyn: pd.DataFrame,
    figures: list[tuple[str, go.Figure]],
    *,
    meta: dict[str, Any],
    tipo_real: str,
    th_dom0: str,
    th_domf: str,
    iric_labels: list[str],
    iric_pct: list[float],
    narrative_sections: list[tuple[str, list[str]]],
) -> bytes:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        Image as RLImage,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        PageBreak,
    )

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=letter,
        leftMargin=0.65 * inch, rightMargin=0.65 * inch,
        topMargin=0.65 * inch, bottomMargin=0.65 * inch,
    )
    styles = _pdf_styles()
    story: list[Any] = []

    story.append(Paragraph("Informe de corrida dinámica del mecanismo", styles["DynTitle"]))
    story.append(Spacer(1, 8))
    _ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    story.append(Paragraph(
        f"<b>Fecha:</b> {_ts} &nbsp;|&nbsp; <b>Tipo generador θ*:</b> {tipo_real} &nbsp;|&nbsp; "
        f"<b>Ciclos:</b> {meta.get('n_cyc', '—')} &nbsp;|&nbsp; "
        f"<b>Semilla:</b> {meta.get('semilla', '—')}",
        styles["DynBody"],
    ))
    if meta.get("tau_stop") is not None:
        story.append(Paragraph(
            f"<b>Parada:</b> τ={meta.get('tau_stop')} · m={meta.get('m_stop', '—')} · "
            f"motivo={meta.get('motivo_stop', '—')}",
            styles["DynBody"],
        ))
    story.append(Paragraph(
        f"<b>Creencias:</b> modal τ=0 → <b>{th_dom0}</b>; cierre → <b>{th_domf}</b>. "
        f"Referencia teórica: <i>Mechanism.tex</i> (Estado por pisos, MDG, filtro bayesiano, IR/IC).",
        styles["DynBody"],
    ))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Resumen numérico", styles["DynH2"]))
    _kpis = [
        ("α* promedio", f"{meta.get('alpha_mean', float('nan')):.4f}"),
        ("γ* promedio", f"{meta.get('gamma_mean', float('nan')):.4f}"),
        ("ΔH promedio", f"{meta.get('dh_mean', float('nan')):.4f}"),
        ("ΔH máximo", f"{meta.get('dh_max', float('nan')):.4f}"),
        ("Coincidencia S/F/K (%)",
         f"{meta.get('match_S', 0):.0f} / {meta.get('match_F', 0):.0f} / {meta.get('match_K', 0):.0f}"),
        ("Voz V=1 (%)", f"{meta.get('pct_v1', 0):.1f}"),
        ("Detección d=1 (%)", f"{meta.get('pct_d1', 0):.1f}"),
        ("R (rescate)", f"{meta.get('R_escala', 0):,.0f}"),
    ]
    for lab, val in _kpis:
        story.append(Paragraph(f"• <b>{lab}:</b> {val}", styles["DynBody"]))
    _ir_txt = ", ".join(f"{lb} {pc:.0f}%" for lb, pc in zip(iric_labels, iric_pct))
    story.append(Paragraph(f"• <b>IR/IC:</b> {_ir_txt}", styles["DynBody"]))
    story.append(Spacer(1, 10))

    story.append(Paragraph("Lectura por bloques (alineada con Mechanism.tex)", styles["DynH2"]))
    for title, lines in narrative_sections:
        story.append(Paragraph(f"<b>{title}</b>", styles["DynBody"]))
        for ln in lines[:3]:
            story.append(Paragraph(str(ln).replace("**", ""), styles["DynSmall"]))
        story.append(Spacer(1, 4))

    story.append(PageBreak())
    story.append(Paragraph("Tabla agregada por ciclo (τ)", styles["DynH2"]))
    _tab_cols = [
        "tau", "alpha", "gamma", "Delta_H", "iota", "V", "d",
        "a_S", "a_S_t", "a_F", "a_F_t", "a_K", "a_K_t", "m",
    ]
    _tab_cols = [c for c in _tab_cols if c in df_cyc.columns]
    if _tab_cols:
        story.append(_pdf_table_from_df(df_cyc[_tab_cols].round(4)))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Creencias μ_τ(θ)", styles["DynH2"]))
    _mu_cols = ["tau"] + [c for c in df_mu_dyn.columns if c != "tau"]
    if len(_mu_cols) > 1:
        story.append(_pdf_table_from_df(df_mu_dyn[_mu_cols].round(4)))

    story.append(PageBreak())
    story.append(Paragraph("Gráficas de la corrida", styles["DynH2"]))
    for title, fig in figures:
        png = _fig_to_png(fig, width=820, height=460)
        story.append(Paragraph(f"<b>{title}</b>", styles["DynBody"]))
        story.append(Spacer(1, 4))
        if png:
            img = RLImage(io.BytesIO(png), width=6.8 * inch, height=3.8 * inch)
            story.append(img)
        else:
            story.append(Paragraph(
                "(Figura no exportada: instale <i>kaleido</i> para incrustar gráficas Plotly en el PDF.)",
                styles["DynSmall"],
            ))
        story.append(Spacer(1, 10))

    doc.build(story)
    return buf.getvalue()


def collect_report_figures(
    df_cyc: pd.DataFrame,
    df_mu_dyn: pd.DataFrame,
    *,
    tipo_colors: dict[str, str],
    tipos: tuple[str, ...] = TIPOS_DEFAULT,
    tau_stop: Optional[int] = None,
) -> list[tuple[str, go.Figure]]:
    """Subconjunto de figuras para el PDF (evita PDF excesivamente largo)."""
    out: list[tuple[str, go.Figure]] = []

    if not df_mu_dyn.empty:
        _long = df_mu_dyn.melt(
            id_vars="tau", value_vars=[t for t in tipos if t in df_mu_dyn.columns],
            var_name="Tipo", value_name="mu",
        )
        _fm = px.line(
            _long,
            x="tau",
            y="mu",
            color="Tipo",
            markers=True,
            color_discrete_map=tipo_colors,
            labels={"tau": "τ (ciclo)", "mu": "Creencia μτ(θ)", "Tipo": "Tipo θ"},
        )
        _fm.update_yaxes(range=[0, 1.05], title="Creencia μτ(θ)")
        _fm.update_xaxes(title="τ (ciclo)")
        if tau_stop is not None:
            _fm.add_vline(x=tau_stop, line_dash="dot", line_color="#dc2626")
        _fm.update_layout(margin=dict(t=45, b=45), title="1 · Creencias posteriores por tipo")
        out.append(("1 · Creencias μ_τ(θ)", _fm))

    if not df_cyc.empty:
        _fa = go.Figure()
        _fa.add_trace(go.Scatter(x=df_cyc["tau"], y=df_cyc["alpha"], mode="lines+markers", name="α*"))
        _fa.add_trace(go.Scatter(x=df_cyc["tau"], y=df_cyc["alpha_R"], mode="lines", name="α^R", line=dict(dash="dash")))
        _fa.add_trace(go.Scatter(x=df_cyc["tau"], y=df_cyc["alpha_N"], mode="lines", name="α^N", line=dict(dash="dot")))
        _fa.update_xaxes(title="τ (ciclo)")
        _fa.update_yaxes(range=[0, 1.05], title="Nivel α ∈ [0,1]")
        _fa.update_layout(margin=dict(t=45, b=45), title="2 · Bloqueo financiero óptimo α*")
        out.append(("2 · α* vs pisos R y N", _fa))

        _fg = go.Figure()
        _fg.add_trace(go.Scatter(x=df_cyc["tau"], y=df_cyc["gamma"], mode="lines+markers", name="γ*"))
        _fg.add_trace(go.Scatter(x=df_cyc["tau"], y=df_cyc["gamma_R"], mode="lines", name="γ^R", line=dict(dash="dash")))
        _fg.add_trace(go.Scatter(x=df_cyc["tau"], y=df_cyc["gamma_N"], mode="lines", name="γ^N", line=dict(dash="dot")))
        _fg.update_xaxes(title="τ (ciclo)")
        _fg.update_yaxes(range=[0, 1.05], title="Nivel γ ∈ [0,1]")
        _fg.update_layout(margin=dict(t=45, b=45), title="3 · Presión operativa óptima γ*")
        out.append(("3 · γ* vs pisos R y N", _fg))

        _fir = pd.DataFrame({
            "Restricción": ["IR^K", "IC^K", "IR^F", "IC^F"],
            "Pct": [100.0 * float(df_cyc[c].mean()) for c in ("IR_K", "IC_K", "IR_F", "IC_F")],
        })
        _fic = px.bar(_fir, x="Restricción", y="Pct", color="Restricción")
        _fic.update_xaxes(title="Restricción")
        _fic.update_yaxes(range=[0, 105], title="% de ciclos cumplidos")
        _fic.update_layout(margin=dict(t=45, b=45), showlegend=False, title="11 · Restricciones IR/IC de K y F")
        out.append(("11 · IR / IC de K y F (% ciclos)", _fic))

    return out


def build_full_dyn_report(
    df_cyc: pd.DataFrame,
    df_mu_dyn: pd.DataFrame,
    *,
    tipo_real: str,
    tipo_colors: dict[str, str],
    tipos: tuple[str, ...] = TIPOS_DEFAULT,
    tau_stop: Optional[int] = None,
    th_dom0: str,
    th_domf: str,
    mu_t0: dict[str, float],
    mu_tf: dict[str, float],
    meta: dict[str, Any],
    iric_labels: list[str],
    iric_pct: list[float],
    narrative_sections: list[tuple[str, list[str]]],
) -> tuple[bytes, go.Figure, Optional[bytes]]:
    """Devuelve (pdf_bytes, figura_infografía, png_infografía opcional)."""
    n_cyc = int(meta.get("n_cyc", max(1, len(df_cyc))))
    fig_info = build_dyn_infographic(
        df_cyc, df_mu_dyn,
        tipo_real=tipo_real,
        tipo_colors=tipo_colors,
        tipos=tipos,
        tau_stop=tau_stop,
        th_dom0=th_dom0,
        th_domf=th_domf,
        mu_t0=mu_t0,
        mu_tf=mu_tf,
        n_cyc=n_cyc,
        meta=meta,
        narrative_sections=narrative_sections,
    )
    figs = collect_report_figures(
        df_cyc, df_mu_dyn,
        tipo_colors=tipo_colors,
        tipos=tipos,
        tau_stop=tau_stop,
    )
    pdf_bytes = build_dyn_pdf_report(
        df_cyc, df_mu_dyn, figs,
        meta=meta,
        tipo_real=tipo_real,
        th_dom0=th_dom0,
        th_domf=th_domf,
        iric_labels=iric_labels,
        iric_pct=iric_pct,
        narrative_sections=narrative_sections,
    )
    png_info = _fig_to_png(fig_info, width=1400, height=1320)
    return pdf_bytes, fig_info, png_info
