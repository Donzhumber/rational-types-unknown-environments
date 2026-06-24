import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import copy
import hashlib
import html
import json
import os
import re
import unicodedata
from typing import Any, Dict, Optional, Tuple
from model_logic import ModeloSecuestro, TIPOS_SECUESTRADOR, DESENLACES
from dynamic_report import build_full_dyn_report
from rational_behavior import (
    absorption_posterior_check,
    apply_kidnapper_scale_calibration,
    apply_kidnapper_scale_calibration_cached,
    blend_hazards,
    build_kidnapper_params_df,
    build_mechanism_mu_trajectory,
    KIDNAPPER_FARC_R_ESCALA_FIXED,
    validate_tab15_all_types,
    build_t0_bayesian_posterior_report,
    format_belief_update_display_df,
    generate_incident_voice_scenario,
    resolve_observed_desenlace,
    cmh_alive_and_kill_shares,
    compute_cmh_beta_calibration_tables,
    compute_voice_likelihood_trajectory,
    compute_family_table,
    compute_kidnapper_by_type_tables,
    compute_state_table,
    compute_state_VR_VN,
    derive_kidnapper_structural_params,
    family_calibrated_vs_endogenous,
    family_institutional_cost_e,
    hybrid_temperature,
    kidnapper_backward_induction_k_table,
    kidnapper_backward_tau1_switch_fast,
    kidnapper_cost_c,
    kidnapper_tab15_flow_rev_col9,
    kidnapper_util_df_from_param_df,
    kidnapper_V_cont_branch,
    load_cmh_outcome_moments,
    mahalanobis_diagonal_loglik,
    maturation_filter,
    mechanism_competitive_hazards_at_t,
    mdg_execution_noise,
    optimize_state_instruments,
    recursive_equilibrium_path,
    refresh_kidnapper_endogenous_columns,
    shannon_entropy,
    state_calibrated_vs_endogenous,
    trajectory_entropy_series,
)


# Horizonte trayectoria bayesiana Tabla 14 / inducción secuestrador (K)
_TAB14_TRAJ_TMAX = 300
_TAB14_LIKELIHOOD_VERSION = 8
_TAB15_SWITCH_TARGETS = {
    "DC": 40,
    "PAR": 80,
    "ELN": 160,
    "FARC": 250,
}
_TAB15_FIXED_COST_COEFFS = {
    # C(γ_t, θ_K)=ϕ(θ_K) exp(κ_c(θ_K) γ_t)+ν(θ_K).
    # Calibrado v12: R común se inicializa en 20.000.000 y luego puede editarse en Tabla 12.
    "DC":   {"phi": 33.00, "kappa_c": 2.61, "nu":  0.750},
    "PAR":  {"phi": 35.00, "kappa_c": 2.63, "nu":  0.500},
    "ELN":  {"phi": 38.00, "kappa_c": 2.70, "nu":  0.250},
    "FARC": {"phi": 40.00, "kappa_c": 2.70, "nu":  0.000},
}
_TAB15_CALIB_VERSION = 12
_PSI8_CMH_CALIB_VERSION = 13
_MDG7_ALIGNMENT_CALIB_VERSION = 1
_STRUCTURAL_LAMBDA_SIGNATURE_VERSION = "theta_signature_v2"


def _stable_json_signature(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


def _data_file_signature(filename: str) -> tuple[str, float, int]:
    path = os.path.join(os.path.dirname(__file__), filename)
    try:
        stat = os.stat(path)
        return (filename, float(stat.st_mtime), int(stat.st_size))
    except OSError:
        return (filename, 0.0, 0)


@st.cache_data(show_spinner=False)
def _compute_cmh_beta_calibration_tables_cached(file_sig: tuple[str, float, int], version: int) -> dict:
    _ = file_sig, version
    return compute_cmh_beta_calibration_tables()


@st.cache_data(show_spinner=False)
def _cmh_outcome_moments_for_mdg() -> dict[str, Any]:
    """Momentos empíricos de Data_CMH para calibrar Tabla 8 / Eq. 28-29."""
    mapping = {
        "Fuga o Liberación": "Liberación",
        "Muerte": "Muerte",
        "Pago": "Pago",
        "Rescate": "Rescate",
    }
    labels = ["Liberación", "Rescate", "Pago", "Muerte"]
    path = os.path.join(os.path.dirname(__file__), "Data_CMH.csv")
    counts = {k: 0 for k in labels}
    n_cases = 0
    try:
        df = pd.read_csv(path)
        if "IDCaso" in df.columns:
            df = df.drop_duplicates("IDCaso")
        y = df.get("Y_Resultado", pd.Series(dtype=object)).map(mapping)
        vc = y.value_counts()
        counts = {k: int(vc.get(k, 0)) for k in labels}
        n_cases = int(sum(counts.values()))
    except Exception:
        counts = {"Liberación": 100, "Rescate": 319, "Pago": 398, "Muerte": 17}
        n_cases = int(sum(counts.values()))
    # Continuar no es un resultado final observado en Data_CMH; se deja con
    # pseudo-masa pequeña para que el mecanismo dinámico conserve soporte.
    cont_pseudo = max(1, int(round(0.08 * max(1, n_cases))))
    counts_full = {
        "Liberación": counts["Liberación"],
        "Rescate": counts["Rescate"],
        "Pago": counts["Pago"],
        "Muerte": counts["Muerte"],
        "Continuar": cont_pseudo,
    }
    total_full = float(sum(counts_full.values()))
    probs_full = {k: float(v / total_full) for k, v in counts_full.items()}
    total_terminal = float(max(1, n_cases))
    probs_terminal = {k: float(counts[k] / total_terminal) for k in labels}
    return {
        "counts_terminal": counts,
        "probs_terminal": probs_terminal,
        "counts_full": counts_full,
        "probs_full": probs_full,
        "n_cases": int(n_cases),
        "continuar_pseudo": int(cont_pseudo),
    }


def _default_cal_psi_params_from_cmh() -> dict[int, dict[str, Any]]:
    """Calibración inicial de Tabla 8 con orden base solicitado."""
    p = {
        "Liberación": 0.025,
        "Rescate": 0.020,
        "Pago": 0.045,
        "Muerte": 0.010,
        "Continuar": 0.900,
    }
    eps = 1e-9
    return {
        # j=1 Liberación: tercera masa base; sube si K ejecuta Liberar.
        1: {
            "delta": float(np.log(max(eps, p["Liberación"]))),
            "gamma_K": 0.55,
            "gamma_S": 0.05,
            "gamma_F": 0.05,
            "phi_gamma": -0.05,
            "phi_theta": [0.00, 0.04, -0.08, -0.02],
            "kappa": 0.15,
        },
        # j=2 Rescate: cuarta masa base; sube si S ejecuta Rescatar.
        2: {
            "delta": float(np.log(max(eps, p["Rescate"]))),
            "gamma_K": 0.05,
            "gamma_S": 0.55,
            "gamma_F": -0.10,
            "phi_gamma": 0.18,
            "phi_theta": [0.02, 0.03, -0.04, 0.02],
            "kappa": 0.20,
        },
        # j=3 Pago: segunda masa base; sube si F ejecuta Coludir.
        3: {
            "delta": float(np.log(max(eps, p["Pago"]))),
            "gamma_K": 0.05,
            "gamma_S": -0.08,
            "gamma_F": 0.55,
            "phi_gamma": -0.08,
            "phi_theta": [-0.02, -0.05, 0.10, 0.00],
            "kappa": 0.10,
        },
        # j=4 Muerte: menor masa base; sube fuertemente si K ejecuta Matar.
        4: {
            "delta": float(np.log(max(eps, p["Muerte"]))),
            "gamma_K": 0.65,
            "gamma_S": 0.00,
            "gamma_F": 0.00,
            "phi_gamma": 0.12,
            "phi_theta": [0.02, 0.05, 0.00, 0.14],
            "kappa": 0.15,
        },
        # j=5 Continuar: mayor masa base por criterio dinámico del usuario.
        5: {
            "delta": float(np.log(max(eps, p["Continuar"]))),
            "gamma_K": 0.00,
            "gamma_S": 0.00,
            "gamma_F": 0.00,
            "phi_gamma": -0.02,
            "phi_theta": [0.00, 0.00, 0.00, 0.00],
            "kappa": 1.05,
        },
    }


def _rb_family_phi_kappa_nu(f_capa: str):
    """Mapea `rb_e0` / `rb_e1` y la capacidad de pago a ``(φ_F, κ_F, ν_F)`` (ec. family-institutional-cost)."""
    e0 = float(st.session_state.get("rb_e0", 3.0))
    e1 = float(st.session_state.get("rb_e1", 12.0))
    nu = float(max(0.0, min(e0 * 0.4, e0 - 0.05)))
    phi = float(max(0.05, e0 - nu))
    tgt = float(max(1e-6, e0 + e1 - nu))
    kappa0 = float(min(5.0, max(0.05, np.log(max(1.0001, tgt / phi)))))
    if "Alta" in str(f_capa):
        kappa = float(min(5.0, max(0.05, kappa0 * 1.12)))
    else:
        kappa = float(min(5.0, max(0.05, kappa0 * 0.92)))
    return phi, kappa, nu

# Rasgos estructurales de θ_K (Vector de "ADN" del secuestrador)
_THETA_K_LABELS = [
    "Disciplina Militar",
    "Logística / Suministros",
    "Impaciencia Financiera",
    "Letalidad / Agresividad",
]

THETA_K_MAP = {
    "DC":   [0.20, 0.30, 0.90, 0.50],
    "PAR":  [0.85, 0.75, 0.40, 0.95],
    "ELN":  [0.90, 0.80, 0.30, 0.60],
    "FARC": [1.00, 0.90, 0.20, 0.80],
}

_VOZ_RASGO_LABELS = (
    "Mean pitch f₀",
    "Pitch variance",
    "Pause rate",
    "Formality / speech aggression",
)


def _default_cal_voz_params() -> dict:
    """Prior por tipo: vector de referencia x̄ ∈ R^4 y σ_L,i, σ_S,i > 0 (diagonal de Σ_L, Σ_S)."""
    specs = {
        "DC": (
            (158.0, 0.22, 0.17, 0.62),
            (8.5, 0.052, 0.032, 0.085),
            (6.0, 0.042, 0.022, 0.065),
        ),
        "PAR": (
            (171.0, 0.30, 0.11, 0.53),
            (7.8, 0.049, 0.029, 0.078),
            (5.3, 0.037, 0.021, 0.056),
        ),
        "ELN": (
            (167.0, 0.28, 0.12, 0.51),
            (7.5, 0.047, 0.027, 0.074),
            (5.5, 0.036, 0.020, 0.059),
        ),
        "FARC": (
            (176.0, 0.37, 0.13, 0.43),
            (7.0, 0.044, 0.025, 0.070),
            (4.9, 0.033, 0.018, 0.051),
        ),
    }
    return {
        t: {
            "x": [float(v) for v in specs[t][0]],
            "sigma_L": [float(v) for v in specs[t][1]],
            "sigma_S": [float(v) for v in specs[t][2]],
        }
        for t in TIPOS_SECUESTRADOR
    }


def _cal_sample_voice_bundle(th: str) -> dict[str, list[float]]:
    """Muestra $\bar x$, $\varepsilon_L$, $\varepsilon_S$ por separado (priors diagonales en sesión)."""
    vp = st.session_state.cal_voz_params[th]
    xb = np.array(vp["x"], dtype=float)
    sL = np.array(vp["sigma_L"], dtype=float)
    sS = np.array(vp["sigma_S"], dtype=float)
    eL = np.random.normal(0.0, sL, size=4)
    eS = np.random.normal(0.0, sS, size=4)
    return {"xb": xb.tolist(), "eL": eL.tolist(), "eS": eS.tolist()}


def _default_cal_pcap_params() -> dict[str, dict[str, float]]:
    """Prior del logit de $p_{\mathrm{cap}}$ por tipo $\theta_K$: $c_0,c_\alpha,c_\gamma$.

    Calibración ilustrativa (heterogeneidad entre DC, PAR, ELN, FARC); $\delta_a$ y $c_S$ siguen globales en sesión.
    """
    specs = {
        "DC": (-1.35, 1.40, 1.10),
        "PAR": (-1.00, 1.55, 1.25),
        "ELN": (-1.15, 1.45, 1.15),
        "FARC": (-1.20, 1.50, 1.20),
    }
    return {
        t: {"c0": float(specs[t][0]), "c_alpha": float(specs[t][1]), "c_gamma": float(specs[t][2])}
        for t in TIPOS_SECUESTRADOR
    }


def _default_cal_voz_pi_call() -> dict[str, float]:
    """Prior de urgencia logística (probabilidad de llamada diaria) por tipo."""
    return {"DC": 0.20, "PAR": 0.20, "ELN": 0.20, "FARC": 0.20}


_VOICE_PI_FREQ_HIGH = {"DC": 0.65, "PAR": 0.55, "ELN": 0.45, "FARC": 0.35}
_VOICE_PI_FREQ_LOW = {"DC": 0.13, "PAR": 0.11, "ELN": 0.09, "FARC": 0.07}


def _build_cal_voz_extra_df(pi_val: float, omega: float) -> pd.DataFrame:
    """Tabla auxiliar para frecuencia de contacto y peso de aprendizaje de voz."""
    rows = [
        {
            "#": 1,
            "Término": "Call probability / Urgency",
            "Coeficiente": r"\pi_{\mathrm{call}}(\theta_K)",
            "Valor": round(pi_val, 2),
            "Origen del valor": "Structural Prior",
            "Valor_KaTeX": r"\pi_{\mathrm{call}}",
            "Clase_tab7": "Observed",
        },
        {
            "#": 2,
            "Término": "Voice learning weight (moderator)",
            "Coeficiente": r"\omega_{\mathrm{voz}}",
            "Valor": round(omega, 2),
            "Origen del valor": "Calibration",
            "Valor_KaTeX": r"\omega_{\mathrm{voz}}",
            "Clase_tab7": "Observed",
        },
    ]
    return pd.DataFrame(rows)


def _build_cal_pcap_tabla6_df(prow: dict, delta_a: float, c_S: float) -> pd.DataFrame:
    """Table 6: parameters of the technical capture probability (logit)."""
    rows = [
        {
            "#": 1,
            "Término": _ui_text("Impact of executed action", "Impacto acción ejecutada"),
            "Coeficiente": r"\delta_a",
            "Valor": round(delta_a, 2),
            "Origen del valor": _ui_text("Structural (S)", "Estructural (S)"),
            "Valor_KaTeX": r"\delta_a",
            "Clase_tab7": _ui_text("Parameter", "Parámetro"),
        },
        {
            "#": 2,
            "Término": _ui_text("Baseline type heterogeneity", "Heterogeneidad basal tipo"),
            "Coeficiente": r"c_0(\theta_K)",
            "Valor": round(prow["c0"], 2),
            "Origen del valor": _ui_text("Structural (K)", "Estructural (K)"),
            "Valor_KaTeX": r"c_0",
            "Clase_tab7": _ui_text("Parameter", "Parámetro"),
        },
        {
            "#": 3,
            "Término": _ui_text("Blockade sensitivity α*", "Sensibilidad bloqueo α*"),
            "Coeficiente": r"c_\alpha(\theta_K)",
            "Valor": round(prow["c_alpha"], 2),
            "Origen del valor": _ui_text("Structural (K)", "Estructural (K)"),
            "Valor_KaTeX": r"c_\alpha",
            "Clase_tab7": _ui_text("Parameter", "Parámetro"),
        },
        {
            "#": 4,
            "Término": _ui_text("Pressure sensitivity γ*", "Sensibilidad presión γ*"),
            "Coeficiente": r"c_\gamma(\theta_K)",
            "Valor": round(prow["c_gamma"], 2),
            "Origen del valor": _ui_text("Structural (K)", "Estructural (K)"),
            "Valor_KaTeX": r"c_\gamma",
            "Clase_tab7": _ui_text("Parameter", "Parámetro"),
        },
        {
            "#": 5,
            "Término": _ui_text("Institutional capacity S", "Capacidad institucional S"),
            "Coeficiente": r"c_S(\theta_S)",
            "Valor": round(c_S, 2),
            "Origen del valor": _ui_text("Structural (S)", "Estructural (S)"),
            "Valor_KaTeX": r"c_S",
            "Clase_tab7": _ui_text("Parameter", "Parámetro"),
        },
    ]
    return pd.DataFrame(rows)


def _sync_cal_pcap_from_session_widgets(th: str, prow: dict) -> None:
    """Alinea `cal_pcap_params[th]` con los `number_input` del popover Tabla 6."""
    for stem, field in (
        ("pcap_c0", "c0"),
        ("pcap_c_alpha", "c_alpha"),
        ("pcap_c_gamma", "c_gamma"),
    ):
        k = f"{stem}_{th}"
        if k in st.session_state:
            try:
                prow[field] = float(st.session_state[k])
            except (TypeError, ValueError):
                pass


def _sync_cal_voz_from_session_widgets(th: str, vp: dict) -> None:
    """Alinea `cal_voz_params[th]` con los `number_input` del popover (persisten en sesión).

    Si el popover va **después** de la tabla KaTeX, hay que llamar esto **antes** de
    construir el DataFrame para no mostrar valores desfasados un rerun.
    """
    for i in range(4):
        kx = f"voz_x_{th}_{i}"
        kL = f"voz_sL_{th}_{i}"
        kS = f"voz_sS_{th}_{i}"
        if kx in st.session_state:
            try:
                vp["x"][i] = float(st.session_state[kx])
            except (TypeError, ValueError):
                pass
        if kL in st.session_state:
            try:
                vp["sigma_L"][i] = float(st.session_state[kL])
            except (TypeError, ValueError):
                pass
        if kS in st.session_state:
            try:
                vp["sigma_S"][i] = float(st.session_state[kS])
            except (TypeError, ValueError):
                pass


def _sync_cal_voz_extra_from_session_widgets(th: str) -> None:
    """Alinea pi_call (tipo-específico) y omega (global) con widgets."""
    k_pi = f"voz_pi_{th}"
    k_om = "voz_omega"
    if k_pi in st.session_state:
        st.session_state.cal_voz_pi_call[th] = float(st.session_state[k_pi])
    if k_om in st.session_state:
        st.session_state.cal_voz_omega = float(st.session_state[k_om])


def _resolve_voice_tab2_params() -> tuple[float, dict, dict]:
    """
    $\\omega_{\\mathrm{voz}}$, $\\pi_{\\mathrm{call}}(\\cdot)$ y Tabla 5 (§ Medición de voz)
    desde pestaña 2: prioriza widgets ``voz_omega`` / ``voz_pi_{θ}`` si existen en sesión.
    """
    if "voz_omega" in st.session_state:
        st.session_state.cal_voz_omega = float(st.session_state["voz_omega"])
    omega = float(st.session_state.get("cal_voz_omega", 0.2))
    if "cal_voz_pi_call" not in st.session_state:
        st.session_state.cal_voz_pi_call = _default_cal_voz_pi_call()
    pi = {str(k): float(v) for k, v in st.session_state.cal_voz_pi_call.items()}
    for th in TIPOS_SECUESTRADOR:
        k_pi = f"voz_pi_{th}"
        if k_pi in st.session_state:
            pi[str(th)] = float(st.session_state[k_pi])
            st.session_state.cal_voz_pi_call[str(th)] = float(st.session_state[k_pi])
    if "cal_voz_params" not in st.session_state:
        st.session_state.cal_voz_params = _default_cal_voz_params()
    voz = copy.deepcopy(st.session_state.cal_voz_params)
    return omega, pi, voz


def _df_term_value(df: pd.DataFrame, term: str) -> Optional[float]:
    """Lee «Valor» de un término en Tabla 1 (pestaña 2)."""
    if df is None or df.empty or "Término" not in df.columns:
        return None
    sub = df[df["Término"].astype(str) == str(term)]
    if sub.empty:
        return None
    try:
        return float(sub["Valor"].iloc[0])
    except (TypeError, ValueError):
        return None


def _tab2_structural_bundle_for_theta(
    theta_k: str,
    *,
    z_region: str,
    v_victim: str,
    f_capa: str,
    s_tipo: str,
) -> dict:
    """
    β_{K,j}, λ_{j0} y ζ_{·,j} efectivos de Tabla 1 (pestaña 2), incl. overrides Prior
    en ``focus_cov_store``, para alinear ℒ_H con Tabla 10/14.
    """
    _ensure_focus_cov_store_in_session()
    store = st.session_state.focus_cov_store
    betas: dict[str, float] = {}
    lams: dict[str, float] = {}
    zeta_by_j: dict[str, dict[str, float]] = {}
    for j_mech, cause, basal_term in (
        (1, "Pago", "Riesgo basal (pago)"),
        (2, "Muerte", "Riesgo basal (muerte)"),
        (3, "Rescate", "Riesgo basal (rescate)"),
    ):
        _, _, bkey, _, _ = _cal_focus_row(j_mech)
        lj = "Liberación" if bkey == "Liberación" else bkey
        pk = _focus_cov_profile_key(j_mech, theta_k, z_region, v_victim, f_capa, s_tipo)
        zp = _focus_cmh_endogenous_tentatives(theta_k)
        df0 = _build_focus_covariate_table(
            j_mech=j_mech,
            theta_k=theta_k,
            z_region=z_region,
            v_victim=v_victim,
            f_capa=f_capa,
            s_tipo=s_tipo,
            lambdas_dict=st.session_state.cal_lambdas_dict,
            betas_dict=st.session_state.cal_betas_dict,
            M_t=0.0,
            presion_S=float(st.session_state.cal_presion_S),
            h_j_numeric=0.0,
            tipo_incidente_p1=theta_k,
            zeta_phi=zp,
        )
        df = _apply_focus_cov_saved_values(df0, pk, store, theta_k)
        v_beta = _df_term_value(df, "Tipo secuestrador (θ_K)")
        if v_beta is not None:
            betas[str(bkey)] = float(v_beta)
        v_lam = _df_term_value(df, basal_term)
        if v_lam is not None:
            lams[str(lj)] = float(v_lam)
        za = _df_term_value(df, "Instrumento α (bloqueo)")
        if za is None:
            za = _df_term_value(df, "Instrumento α")
        zg = _df_term_value(df, "Instrumento γ (presión)")
        if zg is None:
            zg = _df_term_value(df, "Instrumento γ")
        zd = _df_term_value(df, "Detección p_det")
        z_block = {
            "alpha": float(za if za is not None else zp.get("zeta_alpha", 0.1)),
            "gamma": float(zg if zg is not None else zp.get("zeta_gamma", 0.1)),
            "d": float(zd if zd is not None else zp.get("zeta_d", 0.1)),
        }
        for term, key in (
            ("Capacidad de pago alta (θ_F)", "beta_F"),
            ("Víctima perfil público (θ_V)", "beta_V"),
            ("Estado laxo (θ_S)", "beta_S"),
            ("Heterogeneidad geográfica", "beta_z"),
            ("Familia paga (MDG)", "phi_F"),
            ("K continúa (MDG)", "phi_K_cont"),
            ("K mata (MDG)", "phi_K_kill"),
        ):
            value = _df_term_value(df, term)
            if value is not None:
                z_block[key] = float(value)
        if cause == "Rescate":
            z_r = _df_term_value(df, "Estado rescata (MDG)")
            z_block["R"] = float(z_r if z_r is not None else zp.get("zeta_R", 0.1))
        zeta_by_j[str(cause)] = z_block
    return {"betas": betas, "lambdas_0": lams, "zeta_by_j": zeta_by_j}


def _tab2_bundles_all_types(
    *,
    z_region: str,
    v_victim: str,
    f_capa: str,
    s_tipo: str,
) -> dict[str, dict]:
    _ensure_focus_cov_store_in_session()
    sig = _stable_json_signature({
        "version": _STRUCTURAL_LAMBDA_SIGNATURE_VERSION,
        "z_region": str(z_region),
        "v_victim": str(v_victim),
        "f_capa": str(f_capa),
        "s_tipo": str(s_tipo),
        "cal_betas_dict": st.session_state.get("cal_betas_dict", {}),
        "cal_lambdas_dict": st.session_state.get("cal_lambdas_dict", {}),
        "cal_presion_S": float(st.session_state.get("cal_presion_S", 0.5)),
        "focus_cov_store": st.session_state.get("focus_cov_store", {}),
    })
    cache = st.session_state.setdefault("_tab2_bundles_all_types_cache", {})
    if cache.get("sig") == sig and isinstance(cache.get("value"), dict):
        return copy.deepcopy(cache["value"])

    value = {
        str(th): _tab2_structural_bundle_for_theta(
            str(th),
            z_region=z_region,
            v_victim=v_victim,
            f_capa=f_capa,
            s_tipo=s_tipo,
        )
        for th in TIPOS_SECUESTRADOR
    }
    st.session_state["_tab2_bundles_all_types_cache"] = {
        "sig": sig,
        "value": copy.deepcopy(value),
    }
    return copy.deepcopy(value)


def _scale_policy_zeta_bundles(bundles: dict[str, dict], multiplier: float) -> dict[str, dict]:
    """Escala solo sensibilidad de instrumentos continuos en zeta_by_j."""
    mult = float(max(0.0, multiplier))
    if mult == 1.0:
        return bundles
    out = copy.deepcopy(bundles)
    for bundle in out.values():
        zeta_by_j = bundle.get("zeta_by_j")
        if not isinstance(zeta_by_j, dict):
            continue
        for block in zeta_by_j.values():
            if not isinstance(block, dict):
                continue
            for key in ("alpha", "gamma"):
                if key in block:
                    try:
                        block[key] = float(block[key]) * mult
                    except (TypeError, ValueError):
                        pass
    return out


def _invalidate_tab1415_caches() -> None:
    """Invalida Tabla 14 / 15 (p. ej. al cambiar θ* del incidente)."""
    for _k in (
        "rb_mu_traj_sig",
        "rb_mu_traj_snapshot",
        "tab15_mu_snapshot",
        "tab15_k_params_calibrated",
        "tab15_last_validation",
        "tab15_ransom_sig",
        "tab15_T_cached",
        "tab15_theta",
        "tab15_mu_sig",
    ):
        st.session_state.pop(_k, None)
    try:
        _run_kidnapper_backward_induction_cached.clear()
    except Exception:
        pass


def _invalidate_voice_dependent_state() -> None:
    """Descarta objetos que dependen de la trayectoria de voz / θ* del incidente."""
    _invalidate_tab1415_caches()
    for _k in ("tab3_mu_calib_sig",):
        st.session_state.pop(_k, None)


def _clear_dynamic_cycles_only() -> None:
    """Limpia solo ciclos dinámicos τ>=1; conserva escenario/ciclo base."""
    for _k in (
        "first_cycle_requested",
        "first_cycle_tau1_52",
        "first_cycle_table52",
        "first_cycle_diag52",
        "first_cycle_post54",
        "first_cycle_voice_meta",
        "dynamic_cycles_requested",
        "dynamic_cycles52",
        "dynamic_cycles_diag52",
        "dynamic_cycles_stop52",
        "dynamic_cycles_run_meta52",
    ):
        st.session_state.pop(_k, None)


def _generate_and_store_incident_voice() -> None:
    """Genera $\\tilde{\\pi}_{\\mathrm{call}}$ y $(V_t,x^{obs})$ para $\\theta^\\ast$ y los guarda en sesión."""
    omega, pi_prior, voz = _resolve_voice_tab2_params()
    theta = str(st.session_state.get("global_tipo_real", TIPOS_SECUESTRADOR[0]))
    kappa = float(st.session_state.get("incident_voice_kappa", 30.0))
    t_max = int(max(1, int(st.session_state.get("tab15_T_horizon", _TAB14_TRAJ_TMAX))))
    seed = int(st.session_state.get("global_semilla_rng", 123))
    pi_prior_map, pi_tilde, path, meta = generate_incident_voice_scenario(
        theta,
        pi_prior,
        voz,
        t_max=t_max,
        kappa=kappa,
        seed=seed,
    )
    st.session_state.incident_pi_call_prior = pi_prior_map
    st.session_state.incident_pi_call_realized = pi_tilde
    st.session_state.incident_voice_path = path
    st.session_state.incident_voice_meta = meta
    st.session_state.incident_voice_theta = theta
    st.session_state.incident_voice_seed = seed
    st.session_state.incident_voice_omega = omega
    st.session_state.incident_voice_likelihood_df = compute_voice_likelihood_trajectory(
        path,
        theta,
        omega_voz=omega,
        pi_call_by_theta=pi_prior_map,
        voz_params_by_theta=voz,
    )
    _invalidate_voice_dependent_state()


def _mdg_implementation_logit_probs(actions: list[str], a_star: str, T: float) -> dict[str, float]:
    """$\mathbb P_{\mathrm I}(\tilde a=a\mid a^\ast)\propto \exp(\mathbf 1\{a=a^\ast\}/T)$ (ec. 26)."""
    T = float(max(T, 1e-12))
    exps = [np.exp((1.0 if str(a) == str(a_star) else 0.0) / T) for a in actions]
    s = float(sum(exps))
    return {a: float(e / s) for a, e in zip(actions, exps)}


def rb_katex_title(markdown: str) -> None:
    """Título con fórmulas ($...$) encima de un widget o tabla.

    Streamlit **no** interpreta LaTeX en ``st.slider``/``number_input`` ni en
    ``column_config``; hay que usar ``st.markdown`` o ``st.latex`` aparte.
    """
    st.markdown(markdown)


_KATEX_VER = "0.16.11"
_KATEX_BASE = f"https://cdn.jsdelivr.net/npm/katex@{_KATEX_VER}/dist"

# Encabezados alineados con columnas (mismo orden que el DataFrame / editor).
RB_LATEX_HEADER_K_PARAMS = [
    r"\theta_K",
    r"\kappa_{\mathrm{rel}}",
    r"\eta",
    r"F_{\mathrm{cap}}",
    r"\phi",
    r"\kappa_c",
    r"\nu",
    r"\tilde{p}_{\mathrm{cap}}",
    r"\tilde{p}_{\mathrm{pay}}",
    r"C(\gamma,\theta)",
]
RB_LATEX_HEADER_K_UTIL = [
    r"\theta_K",
    r"U_{\mathrm{rel}}^K",
    r"U_{\mathrm{kill}}^K",
    r"V_{\mathrm{cont},t}^K",
    r"\text{rama óptima}",
    r"\text{tipo panel}",
]
RB_LATEX_HEADER_F_CAL = [
    r"\text{Parámetro}",
    r"\text{Valor}",
    r"\text{Nivel}",
]
RB_LATEX_HEADER_F_END = [
    r"\text{Objeto}",
    r"\text{Valor}",
    r"\text{Nivel}",
    r"\text{Nota}",
]
RB_LATEX_HEADER_F_EU = [r"\text{Rama}", r"\mathrm{EU}", r"\text{Ref.}"]
RB_LATEX_HEADER_S_CAL = RB_LATEX_HEADER_F_CAL
RB_LATEX_HEADER_S_END = RB_LATEX_HEADER_F_END
RB_LATEX_HEADER_S_BRANCH = [r"\text{Rama}", r"\text{Pérdida}", r"\text{Ref.}"]
RB_LATEX_HEADER_IR = [r"\text{Restricción}", r"\text{Cumple}", r"\text{Nota}"]
RB_LATEX_HEADER_CFG_PARAMS = [
    r"\theta_K",
    r"\varpi_\theta\ \text{(base)}",
    r"\eta_{\theta,z}\ \text{(región)}",
    r"\xi_{\theta,v}\ \text{(perfil)}",
    r"\text{Score}",
    r"\mu_0\ (\%)",
]
RB_LATEX_HEADER_H_BLEND = [r"\text{Desenlace}", r"h_j\ \text{(mezclado)}"]
RB_LATEX_HEADER_H_TIPO = [r"\theta_K", r"\text{Rescate}", r"\text{Pago}", r"\text{Muerte}"]
RB_LATEX_HEADER_TABLA7 = [r"\#", r"\text{Término}", r"\text{Símbolo}", r"\text{Valor}"]
RB_LATEX_TAB14_BY_COL: Dict[str, str] = {
    "t": r"t",
    "m_t": r"m_t",
    "d_t": r"d_t",
    "ω_voz": r"\omega_{\mathrm{voz}}",
    "V_t": r"V_t",
    "ℒ_H^{cont}": r"\mathcal{L}_H^{\mathrm{cont}}",
    "q(t)": r"q(t)",
    "M(t)": r"M(t)",
    "ℒ_{I,t}": r"\mathcal{L}_{I,t}",
    "ℒ_d": r"\mathcal{L}_d",
    "ℒ_voz": r"\mathcal{L}_{\mathrm{voz}}",
    "ℒ_{F,t}": r"\mathcal{L}_{F,t}",
    "ℒ_{C,t}": r"\mathcal{L}_{C,t}",
    "ℒ_F·ℒ_C": r"\mathcal{L}_F\cdot\mathcal{L}_C",
    "ℒ_t(DC)": r"\mathcal{L}_t(\mathrm{DC})",
    "ℒ_t(PAR)": r"\mathcal{L}_t(\mathrm{PAR})",
    "ℒ_t(ELN)": r"\mathcal{L}_t(\mathrm{ELN})",
    "ℒ_t(FARC)": r"\mathcal{L}_t(\mathrm{FARC})",
    "Z_t": r"Z_t",
    "p_{Cont,t}": r"p_{\mathrm{Cont},t}",
    "μ(DC)": r"\mu(\mathrm{DC})",
    "μ(PAR)": r"\mu(\mathrm{PAR})",
    "μ(ELN)": r"\mu(\mathrm{ELN})",
    "μ(FARC)": r"\mu(\mathrm{FARC})",
    "α_t": r"\alpha_t",
    "γ_t": r"\gamma_t",
    "p_det,t": r"p_{\mathrm{det},t}",
    "ι_t": r"\iota_t",
}
RB_LATEX_HEADER_TAB15 = [
    r"\tau",
    r"\mu(\theta^\ast)",
    r"\mu(\mathrm{DC})",
    r"\mu(\mathrm{PAR})",
    r"\mu(\mathrm{ELN})",
    r"\mu(\mathrm{FARC})",
    r"U_{\mathrm{kill}}^K",
    r"U_{\mathrm{rel}}^K",
    r"\tilde{p}_{\mathrm{pay}}\,R\,(1-\alpha_\tau)",
    r"-C_t",
    r"-\tilde{p}_{\mathrm{cap}}\,F_{\mathrm{cap}}",
    r"\sum_\theta \mu\,\beta(1-\tilde{p}_{\mathrm{cap}})\,V_{\mathrm{cont}}(\tau{+}1)",
    r"\bar{V}_{\mathrm{cont}}",
    r"a_K^\ast",
]
RB_LATEX_K12_EDITOR = {
    "kappa_rel": r"\kappa_{\mathrm{rel}}",
    "phi": r"\phi",
    "eta": r"\eta",
    "kappa_c": r"\kappa_c",
    "F_cap": r"F_{\mathrm{cap}}",
    "nu": r"\nu",
    "p_cap": r"\tilde{p}_{\mathrm{cap}}",
    "R_escala": r"R_{\mathrm{escala}}",
}
RB_LATEX_K12_EDITOR_CAPTION = {
    "kappa_rel": "desutil. liberación",
    "phi": "escala costo",
    "eta": "benef. reputacional kill",
    "kappa_c": "sensibilidad presión γ",
    "F_cap": "costo captura",
    "nu": "costo fijo cautiverio",
    "p_cap": "prob. captura",
    "R_escala": "rescate (Tabla 12 → Tabla 15, col. 9)",
}


def rb_latex_headers_tab14(columns: list, tipo_real: str) -> list[str]:
    """Encabezados KaTeX alineados con las columnas visibles de Tabla 14."""
    out: list[str] = []
    for col in columns:
        cs = str(col)
        if cs in RB_LATEX_TAB14_BY_COL:
            out.append(RB_LATEX_TAB14_BY_COL[cs])
        elif "Q^Cap" in cs or "pcap" in cs.lower():
            out.append(
                rf"\hat{{\mathbb{{E}}}}\bigl[p_{{\mathrm{{cap}}}}\mid\theta^\ast={tipo_real}\bigr]"
            )
        elif "Q^Cont" in cs or "pay" in cs.lower():
            out.append(
                rf"\hat{{\mathbb{{E}}}}\bigl[P_{{\mathrm{{pay}}}}\mid\theta^\ast={tipo_real}\bigr]"
            )
        else:
            out.append(cs)
    return out


def _katex_table_cell_html(val: Any) -> str:
    """Celda de tabla: HTML con ``.math`` sin escapar; texto plano escapado."""
    s = str(val)
    if "<span" in s and "class=" in s:
        return s
    return html.escape(s, quote=False)


def _katex_table_term_html(term: str) -> str:
    """Columna «Término»: convierte ``$...$`` inline a KaTeX (p. ej. presión $\\gamma$)."""
    s = str(term).strip()
    if "$" not in s:
        return html.escape(s, quote=False)
    parts: list[str] = []
    last = 0
    for m in re.finditer(r"\$([^$]+)\$", s):
        if m.start() > last:
            plain = s[last : m.start()]
            parts.append(rf"\text{{{plain}}}")
        parts.append(str(m.group(1)).strip())
        last = m.end()
    if last < len(s):
        parts.append(rf"\text{{{s[last:]}}}")
    latex = "".join(parts) if parts else s
    return f'<span class="math">{html.escape(latex, quote=False)}</span>'


_TAB14_COL_TIPS: Dict[str, str] = {
    "t": "Período de cautiverio.",
    "m_t": "Desenlace observado en t (Continuar en la trayectoria de equilibrio).",
    "d_t": "Indicador de detección de colusión Familia-Secuestrador en t.",
    "ω_voz": "Peso de la señal de voz en la verosimilitud (ω_voz).",
    "V_t": "Valor de la señal de voz emitida en t.",
    "ℒ_H^{cont}": "Verosimilitud física condicional en m_t = Continuar.",
    "q(t)": "Componente de aprendizaje q(t) en la trayectoria.",
    "M(t)": "Multiplicador de madurez M(t) = min(1, (t/T_mad)²).",
    "ℒ_{I,t}": "Verosimilitud de implementación MDG de la tripleta ejecutada: producto de P_I^F, P_I^K y P_I^S.",
    "ℒ_d": "Verosimilitud de detección (componente d_t).",
    "ℒ_voz": "Verosimilitud de la señal de voz.",
    "ℒ_{F,t}": "Verosimilitud física total en t: ℒ_I · ℒ_H · ℒ_d.",
    "ℒ_{C,t}": "Verosimilitud de colusión en t.",
    "ℒ_F·ℒ_C": "Verosimilitud conjunta: ℒ_{F,t} · ℒ_{C,t}.",
    "ℒ_t(DC)": "Verosimilitud total usada para actualizar μ_t(DC) a μ_{t+1}(DC).",
    "ℒ_t(PAR)": "Verosimilitud total usada para actualizar μ_t(PAR) a μ_{t+1}(PAR).",
    "ℒ_t(ELN)": "Verosimilitud total usada para actualizar μ_t(ELN) a μ_{t+1}(ELN).",
    "ℒ_t(FARC)": "Verosimilitud total usada para actualizar μ_t(FARC) a μ_{t+1}(FARC).",
    "Z_t": "Constante de normalización bayesiana Σ_θ μ_t(θ) ℒ_t(θ).",
    "p_{Cont,t}": "Probabilidad de continuar en t.",
    "μ(DC)": "Creencia posterior μ_t(DC) sobre tipo Delincuencia Común.",
    "μ(PAR)": "Creencia posterior μ_t(PAR) sobre tipo Paramilitares.",
    "μ(ELN)": "Creencia posterior μ_t(ELN) sobre tipo ELN.",
    "μ(FARC)": "Creencia posterior μ_t(FARC) sobre tipo FARC.",
    "α_t": "Instrumento de bloqueo financiero α_t en t.",
    "γ_t": "Instrumento de presión operativa γ_t en t.",
    "p_det,t": "Probabilidad de detección de colusión p_det,t(θ_K) = Λ(η₀(θ_K) + η₁α + η₂γ); intercepto tipo-específico.",
    "ι_t": "Índice de precisión ι_t = max_θ μ_t(θ).",
}


def render_tab14_mu_katex_table(df: pd.DataFrame, tipo_real: str) -> None:
    """Tabla 14: encabezados KaTeX en ``<th>`` y cuerpo desplazable."""
    if df is None or df.empty:
        return
    headers = rb_latex_headers_tab14(list(df.columns), str(tipo_real))
    tips: list[str] = []
    for col in df.columns:
        cs = str(col)
        if cs in _TAB14_COL_TIPS:
            tips.append(_TAB14_COL_TIPS[cs])
        elif "Q^Cap" in cs or "pcap" in cs.lower():
            tips.append(f"Prob. esperada de captura dado Q^Cap, θ*={tipo_real}: Ê[p_cap | Q^Cap].")
        elif "Q^Cont" in cs or "pay" in cs.lower():
            tips.append(f"Prob. esperada de pago dado Q^Cont, θ*={tipo_real}: Ê[P_E(pay) | Q^Cont].")
        else:
            tips.append("")
    _text_cols = {"m_t"}
    disp = pd.DataFrame(index=df.index)
    for col in df.columns:
        if col in _text_cols:
            disp[col] = df[col].astype(str)
        else:
            disp[col] = df[col].apply(
                lambda x: (
                    ""
                    if pd.isna(x)
                    else (
                        f"{float(x):.4g}"
                        if isinstance(x, (int, float, np.floating))
                        else str(x)
                    )
                )
            )
    render_generic_katex_table(
        disp,
        headers,
        compact=True,
        tight_spacing=True,
        header_nowrap=True,
        body_max_height_px=500,
        header_tooltips=tips,
        header_tooltips_open_up=True,
    )


_TAB15_COL_TIPS = [
    "Col. 1 — Período de inducción hacia atrás τ (de T a 0).",
    "Col. 2 — Creencia μ_τ(θ*) sobre el tipo real θ* en el período τ.",
    "Col. 3 — Creencia μ_τ(DC) sobre Delincuencia Común en τ.",
    "Col. 4 — Creencia μ_τ(PAR) sobre Paramilitares en τ.",
    "Col. 5 — Creencia μ_τ(ELN) sobre ELN en τ.",
    "Col. 6 — Creencia μ_τ(FARC) sobre FARC en τ.",
    "Col. 7 — Utilidad de matar para θ*: U_kill = η·log(1+p̃_cap) − F_cap·p̃_cap.",
    "Col. 8 — Utilidad de liberar para θ*: U_rel = V_L − κ_rel·R.",
    "Col. 9 — Ingreso esperado por pago de rescate: p̃_pay · R · (1 − α). Usa Tab. 14 y Tab. 12.",
    "Col. 10 — Costo operativo negativo: −C_t(γ_t, φ, κ_c, ν) = −(φ·exp(κ_c·γ_t) + ν).",
    "Col. 11 — Costo de captura negativo: −p̃_cap · F_cap para θ*.",
    "Col. 12 — Valor futuro esperado: Σ_θ μ_τ(θ) β(θ) (1−p̃_cap(θ)) V_cont(θ, τ+1).",
    "Col. 13 — Valor total de continuar: cols. 9+10+11+12 = V̄_cont(τ).",
    "Col. 14 — Acción óptima: argmax{col.7 (matar), col.8 (liberar), col.13 (continuar)}.",
]


def render_tab15_backward_katex_table(df: pd.DataFrame) -> None:
    """Tabla 15: encabezados KaTeX en ``<th>`` (mismo esquema que Tabla 14)."""
    if df is None or df.empty:
        return
    headers = list(RB_LATEX_HEADER_TAB15)
    _text_cols = {"14. a_K*"}
    disp = pd.DataFrame(index=df.index)
    for col in df.columns:
        if col in _text_cols:
            disp[col] = df[col].astype(str)
        else:
            disp[col] = df[col].apply(
                lambda x: (
                    ""
                    if pd.isna(x)
                    else (
                        f"{float(x):.4g}"
                        if isinstance(x, (int, float, np.floating))
                        else str(x)
                    )
                )
            )
    render_generic_katex_table(
        disp,
        headers,
        compact=True,
        tight_spacing=True,
        body_max_height_px=440,
        header_tooltips=_TAB15_COL_TIPS,
    )


def _tab15_branch_short_label(opt: str) -> str:
    s = str(opt)
    if "Continuar" in s:
        return "Continuar"
    if "Liberar" in s:
        return "Liberar"
    if "Matar" in s:
        return "Matar"
    return s


def build_tab15_incident_switch_summary(
    df_ia: pd.DataFrame,
    df_mu: pd.DataFrame,
    tipo_real: str,
    *,
    alpha_fallback: float,
    gamma_fallback: float,
) -> dict[str, Any]:
    """Primer τ en que col. 14 deja Continuar, con μ(θ*), α_t y γ_t en ese período."""
    out: dict[str, Any] = {
        "theta_star": str(tipo_real),
        "tau_cambio": None,
        "rama_anterior": "Continuar",
        "rama_nueva": "—",
        "mu_theta_star": float("nan"),
        "alpha_t": float(alpha_fallback),
        "gamma_t": float(gamma_fallback),
    }
    if df_ia is None or df_ia.empty or "t" not in df_ia.columns:
        return out
    _df_bw = df_ia.sort_values("t", ascending=True)
    _row_sw = None
    for _, rr in _df_bw.iterrows():
        if str(rr.get("opcion_BW", "")) != "Continuar (a_cont)":
            _row_sw = rr
            break
    if _row_sw is None:
        out["rama_nueva"] = "Continuar (sin cambio en el horizonte)"
        if not _df_bw.empty and "mu_star" in _df_bw.columns:
            out["mu_theta_star"] = float(_df_bw.iloc[-1]["mu_star"])
        return out
    tau_sw = int(_row_sw["t"])
    out["tau_cambio"] = tau_sw
    out["rama_nueva"] = _tab15_branch_short_label(_row_sw.get("opcion_BW", ""))
    if "mu_star" in _row_sw.index:
        out["mu_theta_star"] = float(_row_sw["mu_star"])
    _mu_col = f"mu_{str(tipo_real)}"
    if not np.isfinite(out["mu_theta_star"]) and _mu_col in _row_sw.index:
        out["mu_theta_star"] = float(_row_sw[_mu_col])
    if df_mu is not None and not df_mu.empty and "t" in df_mu.columns:
        _rm = df_mu.loc[df_mu["t"].astype(int) == int(tau_sw)]
        if not _rm.empty:
            _r0 = _rm.iloc[0]
            if "alpha_t" in _r0.index:
                out["alpha_t"] = float(_r0["alpha_t"])
            if "gamma_t" in _r0.index:
                out["gamma_t"] = float(_r0["gamma_t"])
    return out


def render_tab15_switch_summary_katex(summary: dict[str, Any]) -> None:
    """Tabla resumen (una fila) tras Tabla 15 para el θ* del incidente."""
    tau = summary.get("tau_cambio")
    tau_s = str(int(tau)) if tau is not None else "—"
    mu_v = summary.get("mu_theta_star", float("nan"))
    mu_s = f"{float(mu_v):.4f}" if np.isfinite(float(mu_v)) else "—"
    df_show = pd.DataFrame(
        [
            {
                "c0": str(summary.get("theta_star", "—")),
                "c1": tau_s,
                "c2": str(summary.get("rama_anterior", "Continuar")),
                "c3": str(summary.get("rama_nueva", "—")),
                "c4": mu_s,
                "c5": f"{float(summary.get('alpha_t', 0.0)):.4f}",
                "c6": f"{float(summary.get('gamma_t', 0.0)):.4f}",
            }
        ]
    )
    render_generic_katex_table(
        df_show,
        [
            r"\theta^\ast",
            r"\tau_{\mathrm{cambio}}",
            r"\text{De rama}",
            r"\text{A rama}",
            r"\mu_t(\theta^\ast)",
            r"\alpha_t",
            r"\gamma_t",
        ],
        height=88,
        compact=True,
        tight_spacing=True,
        header_tooltips=[
            "Tipo panel (Grupo secuestrador, incidente).",
            "Primer período en que la col. 14 de Tabla 15 ≠ Continuar.",
            "Rama óptima inmediatamente antes del cambio.",
            "Nueva rama óptima en τ_cambio.",
            "Creencia sobre θ* en τ_cambio (col. 2 de Tabla 15).",
            "Instrumento α en τ_cambio (trayectoria μ_t, Tabla 14).",
            "Instrumento γ en τ_cambio (trayectoria μ_t, Tabla 14).",
        ],
    )


def _katex_label_html(expr: str, element_id: str = "lbl") -> str:
    """HTML seguro: LaTeX en ``textContent`` (evita que ``\\t`` de ``\\text`` rompa en JS)."""
    safe = html.escape(_translate_latex_expression(str(expr)), quote=False)
    return (
        f'<span class="math" id="{element_id}">{safe}</span>'
        f"<script>\n"
        f"(function() {{\n"
        f'  const el = document.getElementById({json.dumps(element_id)});\n'
        f"  if (!el) return;\n"
        f"  try {{ katex.render(el.textContent, el, {{ displayMode: false, throwOnError: false }}); }}\n"
        f"  catch (e) {{}}\n"
        f"}})();\n"
        f"</script>"
    )


def _blank_dataframe_column_config(
    df: pd.DataFrame,
    base: Optional[dict] = None,
    *,
    text_cols: Optional[set] = None,
) -> dict:
    """Etiquetas vacías en ``st.dataframe`` cuando los títulos van en KaTeX arriba."""
    cfg = dict(base or {})
    _text = set(text_cols or ())
    for col in df.columns:
        if col in cfg:
            continue
        if col in _text:
            cfg[col] = st.column_config.TextColumn(label=" ")
        else:
            cfg[col] = st.column_config.NumberColumn(label=" ", format="%.4g")
    return cfg


def st_dataframe_katex_headers(
    df: pd.DataFrame,
    latex_headers: list,
    *,
    header_height: Optional[int] = None,
    header_wide: bool = False,
    column_config: Optional[dict] = None,
    text_cols: Optional[set] = None,
    **dataframe_kwargs,
) -> None:
    """``st.dataframe`` con fila de encabezados KaTeX (``rb_katex_grid_header``)."""
    if df is None or df.empty:
        return
    n = len(df.columns)
    hdrs = list(latex_headers)
    if len(hdrs) < n:
        hdrs.extend([str(c) for c in df.columns[len(hdrs) :]])
    elif len(hdrs) > n:
        hdrs = hdrs[:n]
    if header_wide:
        h = int(header_height or max(92, min(128, 72 + max(0, n - 10) * 2)))
    else:
        h = int(header_height or max(52, min(100, 46 + max(0, n - 6) * 3)))
    rb_katex_grid_header(hdrs, height=h, wide=bool(header_wide))
    _cfg = _blank_dataframe_column_config(df, column_config, text_cols=text_cols)
    st.dataframe(df, column_config=_cfg, **dataframe_kwargs)


def rb_katex_grid_header(
    labels: list,
    *,
    height: int = 64,
    width=None,
    wide: bool = False,
) -> None:
    """Fila de encabezados con KaTeX (CDN), alineada visualmente con la tabla de debajo.

    Usa un iframe (`components.html`). Requiere **conexión** la primera vez para cargar KaTeX.
    Si falla, se muestra un resumen en markdown como respaldo.

    ``wide=True``: tabla al 100 % del ancho, tipografía reducida (Tabla 14 y similares).
    """
    if not labels:
        return
    try:
        payload = json.dumps([_translate_latex_expression(str(x)) for x in labels])
    except (TypeError, ValueError):
        return
    if wide:
        html = f"""<!DOCTYPE html><html><head><meta charset="utf-8"/>
<link rel="stylesheet" href="{_KATEX_BASE}/katex.min.css" crossorigin="anonymous"/>
<script src="{_KATEX_BASE}/katex.min.js" crossorigin="anonymous"></script>
<style>
html,body{{margin:0;padding:0;overflow-x:auto;overflow-y:visible;background:transparent;}}
#rbkh{{width:100%;border-collapse:collapse;table-layout:fixed;}}
#rbkh td{{vertical-align:top;text-align:center;padding:4px 2px;font-size:0.6rem;line-height:1.2;overflow:visible;word-wrap:break-word;white-space:normal;}}
#rbkh td .katex{{font-size:0.92em;max-width:100%;}}
#rbkh td .katex-display{{margin:0;padding:0;}}
</style>
</head><body>
<table id="rbkh"><tr></tr></table>
<script>
(function() {{
  const L = {payload};
  const row = document.querySelector('#rbkh tr');
  const n = Math.max(1, L.length);
  const pct = (100 / n).toFixed(4) + '%';
  for (let i = 0; i < L.length; i++) {{
    const td = document.createElement('td');
    td.style.width = pct;
    try {{
      katex.render(L[i], td, {{ displayMode: false, throwOnError: false }});
    }} catch (e) {{ td.textContent = L[i]; }}
    row.appendChild(td);
  }}
}})();
</script>
</body></html>"""
        _scroll = True
    else:
        html = f"""<!DOCTYPE html><html><head><meta charset="utf-8"/>
<link rel="stylesheet" href="{_KATEX_BASE}/katex.min.css" crossorigin="anonymous"/>
<script src="{_KATEX_BASE}/katex.min.js" crossorigin="anonymous"></script>
<style>html,body{{margin:0;padding:0;overflow:hidden;background:transparent;}}</style>
</head><body>
<div id="rbkh" style="display:flex;align-items:flex-end;justify-content:stretch;gap:3px;padding:3px 4px 2px;box-sizing:border-box;"></div>
<script>
(function() {{
  const L = {payload};
  const r = document.getElementById('rbkh');
  for (let i = 0; i < L.length; i++) {{
    const c = document.createElement('div');
    c.style.cssText = 'flex:1 1 0;text-align:center;min-width:0;font-size:0.8rem;line-height:1.22;';
    try {{ katex.render(L[i], c, {{ displayMode: false, throwOnError: false }}); }}
    catch (e) {{ c.textContent = L[i]; }}
    r.appendChild(c);
  }}
}})();
</script>
</body></html>"""
        _scroll = False
    try:
        components.html(html, height=height, width=width, scrolling=_scroll)
    except Exception:
        rb_katex_title(
            "**Encabezados (KaTeX no disponible):** "
            + " · ".join(rf"${str(x)}$" for x in labels)
        )


# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="Identification of Rational Types",
    page_icon="⚖️", 
    layout="wide"
)

# FIX: Forzar visibilidad de gráficos Plotly y evitar el "freezeo" visual de la UI (Golden Edition)
st.markdown("""
<style>
    .js-plotly-plot { visibility: visible !important; }
    .stPlotlyChart { overflow: hidden !important; max-width: 100% !important; }
    .stPlotlyChart > div { width: 100% !important; overflow: hidden !important; }
    /* Solo iframes de gráficos: evita forzar el 100% en todos los embeds (tablas KaTeX). */
    [data-testid="stPlotlyChart"] iframe {
        border: none !important;
        width: 100% !important;
    }
    [data-testid="stPlotlyChart"] { overflow: hidden !important; }
    /* Párrafo explicativo bajo λ_j: ancho justificado (columna ecuación, pestaña 2) */
    div.mech-eq-blurb {
        text-align: justify;
        text-justify: inter-word;
        hyphens: auto;
    }
    div.mech-eq-blurb p {
        text-align: justify !important;
        text-justify: inter-word;
    }
    /* Tablas Streamlit: tamaño medio, sin forzar tipografía extrema */
    div[data-testid="stDataFrame"] [role="grid"],
    div[data-testid="stDataFrame"] [role="row"],
    div[data-testid="stDataFrame"] [role="columnheader"],
    div[data-testid="stDataFrame"] [role="gridcell"] {
        font-size: 0.8rem !important;
        line-height: 1.26 !important;
    }
    div[data-testid="stDataEditor"] [role="grid"],
    div[data-testid="stDataEditor"] [role="columnheader"],
    div[data-testid="stDataEditor"] [role="gridcell"] {
        font-size: 0.8rem !important;
        line-height: 1.26 !important;
    }
    div[data-testid="stTable"] table {
        font-size: 0.8rem !important;
        line-height: 1.28 !important;
    }
    div[data-testid="stTable"] th,
    div[data-testid="stTable"] td {
        padding: 0.24rem 0.38rem !important;
    }
    /* Barra de pestañas por encima del contenido (evita solapamiento con iframes). */
    div[data-testid="stTabs"] {
        position: relative;
        z-index: 5;
    }
    /* Métricas más compactas */
    div[data-testid="stMetric"] label {
        font-size: 0.78rem !important;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        font-size: 1.4rem !important;
    }
    /* Popover h₀: espacio cómodo sin forzar anchos rígidos */
    div[data-testid="stPopoverBody"] [data-testid="stCaption"] {
        line-height: 1.5 !important;
        margin-bottom: 0.4rem !important;
    }
    /* Ecuaciones largas en expanders: scroll horizontal, sin solapar texto */
    div[data-testid="stExpander"] div[data-testid="stLatex"] {
        overflow-x: auto !important;
        overflow-y: visible !important;
        max-width: 100%;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


# --- IDIOMA / LANGUAGE ---
_LANG_OPTIONS = ["English", "Español"]
if "app_language" not in st.session_state:
    st.session_state["app_language"] = "English"
if st.session_state.get("app_language") not in _LANG_OPTIONS:
    st.session_state["app_language"] = "English"


def _ui_text(en: str, es: str) -> str:
    """Texto de la interfaz mínima controlada por la app."""
    return en if st.session_state.get("app_language", "English") == "English" else es


def _inject_page_translator() -> None:
    """Traduce el DOM completo entre español e inglés con el widget de Google Translate.

    La app original está escrita mayoritariamente en español. Esta capa conserva el
    contenido fuente y aplica traducción de página cuando el usuario selecciona inglés.
    """
    target_lang = "en" if st.session_state.get("app_language", "English") == "English" else "es"
    components.html(
        f"""
<script>
(function() {{
  const targetLang = {json.dumps(target_lang)};
  const parentWindow = window.parent;
  const parentDocument = parentWindow.document;

  function setCookie(name, value) {{
    const expires = 'expires=' + new Date(Date.now() + 365*24*60*60*1000).toUTCString();
    const host = parentWindow.location.hostname;
    parentDocument.cookie = name + '=' + value + ';' + expires + ';path=/';
    if (host && host.indexOf('.') >= 0) {{
      parentDocument.cookie = name + '=' + value + ';' + expires + ';path=/;domain=' + host;
    }}
  }}

  setCookie('googtrans', '/es/' + targetLang);

  let root = parentDocument.getElementById('google_translate_element');
  if (!root) {{
    root = parentDocument.createElement('div');
    root.id = 'google_translate_element';
    root.style.cssText = 'position:fixed;left:-9999px;top:-9999px;width:1px;height:1px;overflow:hidden;';
    parentDocument.body.appendChild(root);
  }}

  const styleId = 'app-language-google-translate-css';
  if (!parentDocument.getElementById(styleId)) {{
    const style = parentDocument.createElement('style');
    style.id = styleId;
    style.textContent = `
      .goog-te-banner-frame.skiptranslate,
      iframe.goog-te-banner-frame,
      .goog-te-balloon-frame,
      #goog-gt-tt {{ display:none !important; visibility:hidden !important; }}
      body {{ top:0 !important; }}
    `;
    parentDocument.head.appendChild(style);
  }}

  function applySelection() {{
    protectMath();
    const combo = parentDocument.querySelector('.goog-te-combo');
    if (!combo) return false;
    const desired = targetLang === 'es' ? 'es' : 'en';
    if (combo.value !== desired) {{
      combo.value = desired;
      combo.dispatchEvent(new Event('change'));
    }}
    return true;
  }}

  function protectMath() {{
    parentDocument
      .querySelectorAll('.katex, .katex-html, .katex-mathml, .math, [data-katex], [data-latex], script, style')
      .forEach(function(el) {{
        el.classList && el.classList.add('notranslate');
        el.setAttribute && el.setAttribute('translate', 'no');
      }});
  }}

  if (!parentWindow.__rationalTypesMathObserver) {{
    parentWindow.__rationalTypesMathObserver = new MutationObserver(function() {{
      protectMath();
    }});
    parentWindow.__rationalTypesMathObserver.observe(parentDocument.body, {{
      childList: true,
      subtree: true
    }});
  }}

  function keepApplying() {{
    let attempts = 0;
    const timer = parentWindow.setInterval(function() {{
      attempts += 1;
      applySelection();
      if (attempts >= 12) parentWindow.clearInterval(timer);
    }}, 1000);
  }}

  parentWindow.googleTranslateElementInit = function() {{
    new parentWindow.google.translate.TranslateElement({{
      pageLanguage: 'es',
      includedLanguages: 'en,es',
      autoDisplay: false,
      multilanguagePage: true
    }}, 'google_translate_element');
    setTimeout(applySelection, 300);
    setTimeout(applySelection, 900);
    keepApplying();
  }};

  if (!parentDocument.getElementById('google-translate-script')) {{
    const script = parentDocument.createElement('script');
    script.id = 'google-translate-script';
    script.src = 'https://translate.google.com/translate_a/element.js?cb=googleTranslateElementInit';
    parentDocument.head.appendChild(script);
  }} else {{
    if (parentWindow.google && parentWindow.google.translate && !parentDocument.querySelector('.goog-te-combo')) {{
      parentWindow.googleTranslateElementInit();
    }}
    applySelection();
    setTimeout(applySelection, 500);
    keepApplying();
  }}
}})();
</script>
        """,
        height=0,
        width=0,
    )


_ES_EN_REPLACEMENTS = [
    # ── Table 1 · full-phrase pairs (must come first to avoid partial mangling) ──
    ("Riesgo basal (pago)", "Baseline risk (payment)"),
    ("Riesgo basal (muerte)", "Baseline risk (death)"),
    ("Riesgo basal (rescate)", "Baseline risk (ransom)"),
    ("Capacidad de pago alta (θ_F)", "Payment capacity high (θ_F)"),
    ("Víctima perfil público (θ_V)", "Public profile victim (θ_V)"),
    ("Estado laxo (θ_S)", "Lax state (θ_S)"),
    ("Estado rescata (MDG)", "State rescues (MDG)"),
    ("Tipo secuestrador (θ_K)", "Kidnapper type (θ_K)"),
    ("Heterogeneidad geográfica", "Geographic heterogeneity"),
    ("Instrumento α (bloqueo)", "Instrument α (blockade)"),
    ("Instrumento γ (presión)", "Instrument γ (pressure)"),
    ("Instrumento α", "Instrument α"),
    ("Instrumento γ", "Instrument γ"),
    ("Detección p_det", "Detection p_det"),
    ("Familia paga (MDG)", "Family pays (MDG)"),
    ("K continúa (MDG)", "K continues (MDG)"),
    ("K mata (MDG)", "K kills (MDG)"),
    # ── Sub-phrase pairs for KaTeX \text{…} blocks (most-specific first) ──────
    ("Capacidad de pago alta", "Payment capacity high"),
    ("Víctima perfil público", "Public profile victim"),
    ("Estado laxo", "Lax state"),
    ("Tipo secuestrador", "Kidnapper type"),
    ("Instrumento", "Instrument"),
    ("Detección", "Detection"),
    # ─────────────────────────────────────────────────────────────────────────────
    ("Español", "Spanish"),
    ("Sistema de Análisis Dinámico de Mecanismos", "Dynamic Mechanism Analysis System"),
    ("Identificación de tipos racionales", "Identification of Rational Types"),
    ("Elaborado por", "Prepared by"),
    ("en Economía", "in Economics"),
    ("Simulación e Incidente", "Simulation and Incident"),
    ("Simulación diaria y proceso MDG", "Daily Simulation and MDG Process"),
    ("Configuración e Inicio", "Setup and Start"),
    ("Distribución ex-ante", "Ex-ante Distribution"),
    ("Fundamentación", "Rationale"),
    ("Parámetros para la selección actual", "Parameters for the Current Selection"),
    ("Selección de priors", "Prior Selection"),
    ("Configuración Manual de Probabilidades", "Manual Probability Configuration"),
    ("Mapa regional por municipio", "Regional Map by Municipality"),
    ("Probabilidades efectivas", "Effective Probabilities"),
    ("Probabilidad de supervivencia", "Survival Probability"),
    ("Probabilidad de captura", "Capture Probability"),
    ("Medición de voz", "Voice Measurement"),
    ("Trayectoria de voz del incidente", "Incident Voice Trajectory"),
    ("Verosimilitudes de voz", "Voice Likelihoods"),
    ("Verosimilitud física", "Physical Likelihood"),
    ("Verosimilitud conjunta observable", "Observable Joint Likelihood"),
    ("Actualización de creencias", "Belief Updating"),
    ("Comportamiento racional", "Rational Behavior"),
    ("Conjuntos de Información y Espacios del Modelo", "Information Sets and Model Spaces"),
    ("Espacios de Acción y Resultados", "Action and Outcome Spaces"),
    ("Estructura del mecanismo", "Mechanism Structure"),
    ("Problema de los 3 jugadores", "Three-Player Problem"),
    ("Optimización formal y valores calibrados", "Formal Optimization and Calibrated Values"),
    ("Solución Mecanismo", "Mechanism Solution"),
    ("Visualización dinámica del mecanismo", "Dynamic Mechanism Visualization"),
    ("Gráficas dinámica", "Dynamic Charts"),
    ("Creencias", "Beliefs"),
    ("Política óptima", "Optimal Policy"),
    ("Frecuencia de decisiones óptimas", "Frequency of Optimal Decisions"),
    ("Frecuencia de la señal de voz", "Voice Signal Frequency"),
    ("Frecuencia de la señal de detección", "Detection Signal Frequency"),
    ("Ganancia esperada de información", "Expected Information Gain"),
    ("Verificación IR / IC", "IR / IC Verification"),
    ("Informe PDF e infografía", "PDF Report and Infographic"),
    ("Bitácora de semillas dinámicas", "Dynamic Seed Log"),
    ("Riesgos competitivos y maduración", "Competing Risks and Maturation"),
    ("Sorteo MDG", "MDG Draw"),
    ("Voz y silencio", "Voice and Silence"),
    ("Aprendizaje y convergencia", "Learning and Convergence"),
    ("Justificación empírica", "Empirical Justification"),
    ("Resultado posterior", "Posterior Result"),
    ("Iniciar proceso", "Start Process"),
    ("Avanzar ciclos", "Advance Cycles"),
    ("Sortear m", "Draw m"),
    ("Aplicar parámetros", "Apply Parameters"),
    ("Guardar urgencia", "Save Urgency"),
    ("Guardar captura", "Save Capture"),
    ("Restablecer captura", "Reset Capture"),
    ("Generar informe e infografía", "Generate Report and Infographic"),
    ("Descargar infografía", "Download Infographic"),
    ("Exportar figuras APA", "Export APA Figures"),
    ("Cargar gráficas", "Load Charts"),
    ("Simular trayectoria", "Simulate Trajectory"),
    ("GENERAR SORTEO", "GENERATE DRAW"),
    ("Regla de Cierre", "Closure Rule"),
    ("Grupo secuestrador", "Kidnapper Group"),
    ("Capacidad de pago", "Payment Capacity"),
    ("Tipo de Estado", "State Type"),
    ("Región de cautiverio", "Captivity Region"),
    ("Perfil de la víctima", "Victim Profile"),
    ("Desenlace focal", "Focal Outcome"),
    ("bloqueo financiero", "financial blockade"),
    ("presión operativa", "operational pressure"),
    ("bloqueo", "blockade"),
    ("presión", "pressure"),
    ("calibrado (prior)", "calibrated (prior)"),
    ("calibrado", "calibrated"),
    ("Liberar", "Release"),
    ("Matar", "Kill"),
    ("Cooperar", "Cooperate"),
    ("Negociar", "Negotiate"),
    ("Continuar", "Continue"),
    ("Rescate", "Rescue"),
    ("Colusión", "Collusion"),
    ("incidente", "incident"),
    ("Intercepto", "Intercept"),
    ("Número de corridas", "Number of Runs"),
    ("Horizonte máximo", "Maximum Horizon"),
    ("Semillas", "Seeds"),
    ("Semilla visible", "Visible Seed"),
    ("Semilla efectiva", "Effective Seed"),
    ("Corrida dinámica actual", "Current Dynamic Run"),
    ("corrida dinámica", "dynamic run"),
    ("ciclos dinámicos", "dynamic cycles"),
    ("Presione", "Press"),
    ("Seleccione", "Select"),
    ("Edite", "Edit"),
    ("parámetros actuales", "current parameters"),
    ("parámetros", "parameters"),
    ("mecanismo", "mechanism"),
    ("incidente", "incident"),
    ("trayectoria", "trajectory"),
    ("señal", "signal"),
    ("voz", "voice"),
    ("silencio", "silence"),
    ("detección", "detection"),
    ("colusión", "collusion"),
    ("familia", "family"),
    ("secuestrador", "kidnapper"),
    ("Estado", "State"),
    ("estado", "state"),
    ("víctima", "victim"),
    ("cautiverio", "captivity"),
    ("desenlace", "outcome"),
    ("desenlaces", "outcomes"),
    ("resultado", "result"),
    ("resultados", "results"),
    ("acción ejecutada", "executed action"),
    ("acción óptima", "optimal action"),
    ("acciones", "actions"),
    ("probabilidad", "probability"),
    ("probabilidades", "probabilities"),
    ("creencia", "belief"),
    ("creencias", "beliefs"),
    ("verosimilitud", "likelihood"),
    ("verosimilitudes", "likelihoods"),
    ("implementación", "implementation"),
    ("materialización", "materialization"),
    ("maduración", "maturation"),
    ("supervivencia", "survival"),
    ("captura", "capture"),
    ("rescate", "rescue"),
    ("muerte", "death"),
    ("pago", "payment"),
    ("liberación", "release"),
    ("continuar", "continue"),
    ("Continuar", "Continue"),
    ("Rescate", "Rescue"),
    ("Muerte", "Death"),
    ("Pago", "Payment"),
    ("Liberación", "Release"),
    ("Tabla", "Table"),
    ("tabla", "table"),
    ("Pestaña", "Tab"),
    ("pestaña", "tab"),
    ("Gráfica", "Chart"),
    ("gráfica", "chart"),
    ("gráficas", "charts"),
    ("dinámica", "dynamic"),
    ("dinámico", "dynamic"),
    ("dinámicas", "dynamic"),
    ("óptima", "optimal"),
    ("óptimo", "optimal"),
    ("teórico", "theoretical"),
    ("teórica", "theoretical"),
    ("empírica", "empirical"),
    ("empírico", "empirical"),
    ("ilustrativo", "illustrative"),
    ("actual", "current"),
    ("guardada", "saved"),
    ("guardado", "saved"),
    ("borrada", "deleted"),
    ("borrado", "deleted"),
    ("Guardar", "Save"),
    ("Borrar", "Delete"),
    ("Restablecer", "Reset"),
    ("Cargar", "Load"),
    ("Generar", "Generate"),
    ("Simular", "Simulate"),
    ("No hay", "There are no"),
    ("Valores", "Values"),
    ("Valor", "Value"),
    ("Término", "Term"),
    ("Coeficiente", "Coefficient"),
    ("Origen del valor", "Value Source"),
    ("Parámetro", "Parameter"),
    ("Observado", "Observed"),
    ("Período", "Period"),
    ("Día", "Day"),
    ("Motivo", "Reason"),
    ("Familia-Secuestrador", "Family-Kidnapper"),
    ("Familia", "Family"),
    ("Secuestrador", "Kidnapper"),
    ("Duro", "Hard"),
    ("Laxo", "Soft"),
    ("Alta Riqueza", "High Wealth"),
    ("Baja Riqueza", "Low Wealth"),
    ("Público", "Public"),
    ("Sí", "Yes"),
    ("No", "No"),
    ("Comunicación", "Communication"),
    ("comunicación", "communication"),
    ("Capturar", "Capture"),
    ("Grupo", "Group"),
    ("Nueva semilla", "New Seed"),
    ("Parámetros aplicados", "Parameters Applied"),
    ("Valores Prior", "Prior Values"),
    ("Valores Observado", "Observed Values"),
    ("Trayectoria", "Trajectory"),
    ("Señal", "Signal"),
    ("señales públicas", "public signals"),
    ("Voz", "Voice"),
    ("Silencio", "Silence"),
    ("Acción", "Action"),
    ("Implementación", "Implementation"),
    ("Materialización", "Materialization"),
    ("Maximización", "Maximization"),
    ("Minimización", "Minimization"),
    ("Bloque", "Block"),
    ("bloque", "block"),
    ("Pulse", "Click"),
    ("Use", "Use"),
    ("parámetro", "parameter"),
    ("sesión", "session"),
    ("restablecidos", "reset"),
    ("restablecido", "reset"),
    ("guardados", "saved"),
    ("guardado", "saved"),
    ("aplicada", "applied"),
    ("aplicado", "applied"),
    ("activa", "active"),
    ("ajustado", "adjusted"),
    ("ajustará", "will be adjusted"),
    ("recalcular", "recalculate"),
    ("recalcula", "recalculates"),
    ("visualizarlo", "view it"),
    ("acumulados", "accumulated"),
    ("municipios", "municipalities"),
    ("municipio", "municipality"),
    ("datos", "data"),
    ("Dato", "Data"),
    ("Causa", "Cause"),
    ("Covariables foco", "Focus Covariates"),
    ("foco", "focus"),
    ("Propensión", "Propensity"),
    ("referencia archivada", "archived reference"),
    ("referencia", "reference"),
    ("Frecuencia alta", "High Frequency"),
    ("Frecuencia baja", "Low Frequency"),
    ("frecuencia", "frequency"),
    ("Urgencia", "Urgency"),
    ("urgencia", "urgency"),
    ("Osciloscopio", "Oscilloscope"),
    ("visor", "viewer"),
    ("ilustración", "illustration"),
    ("rasgo", "feature"),
    ("alta", "high"),
    ("baja", "low"),
    ("por tipo", "by type"),
    ("tipo", "type"),
    ("tecnología", "technology"),
    ("Pesos", "Weights"),
    ("pesos", "weights"),
    ("Peso", "Weight"),
    ("piso de ruido", "noise floor"),
    ("temperatura base", "base temperature"),
    ("decaimiento", "decay"),
    ("modelo teórico", "theoretical model"),
    ("transformada inversa", "inverse transform"),
    ("panel superior", "top panel"),
    ("resultado de", "result of"),
    ("Historia pública inicial", "Initial Public History"),
    ("historia pública", "public history"),
    ("información", "information"),
    ("Información", "Information"),
    ("espacios", "spaces"),
    ("estructuras", "structures"),
    ("Rama", "Branch"),
    ("rama", "branch"),
    ("cooperación", "cooperation"),
    ("preferida", "preferred"),
    ("utilidades esperadas", "expected utilities"),
    ("fila", "row"),
    ("filas", "rows"),
    ("No se pudo construir", "Could not build"),
    ("compartido", "shared"),
    ("arriba", "above"),
    ("columna", "column"),
    ("alimenta", "feeds"),
    ("convergencia", "convergence"),
    ("Bitácora", "Log"),
    ("última corrida", "last run"),
    ("terminal observado", "observed terminal state"),
    ("umbral", "threshold"),
    ("pruebe otra semilla", "try another seed"),
    ("más días", "more days"),
    ("masa", "mass"),
    ("supera", "exceeds"),
    ("alcanza", "is reached"),
    ("actualización", "updating"),
    ("normalización", "normalization"),
    ("restricción", "constraint"),
    ("restricciones", "constraints"),
    ("cumple", "satisfied"),
    ("nota", "note"),
    ("pérdida", "loss"),
    ("centros bayesianos", "Bayesian centers"),
    ("penalización", "penalty"),
    ("ganancia informacional", "information gain"),
    ("programa global", "global program"),
    ("regla discreta", "discrete rule"),
    ("factible", "feasible"),
    ("auditoría", "audit"),
    ("posición", "position"),
    ("presión operacional", "operational pressure"),
    ("informe", "report"),
    ("infografía", "infographic"),
    ("Construyendo", "Building"),
    ("Descargar", "Download"),
    ("instale", "install"),
    ("disponible", "available"),
    ("Activar cálculo de log-verosimilitud", "Enable log-likelihood calculation"),
    ("cálculo", "calculation"),
    ("log-verosimilitud", "log-likelihood"),
    ("desenlaces absorbentes", "absorbing outcomes"),
    ("Cierre ciclo base", "Base Cycle Closure"),
    # --- short phrases missing from earlier entries ---
    ("fuerza prior Beta", "prior Beta strength"),
    ("Llamada", "Call"),
    ("Sí", "Yes"),
    ("con el mismo", "with the same"),
    ("para recalcular", "to recalculate"),
    ("Controles actuales", "Current controls"),
    ("aleatoria", "random"),
    ("únicamente al presionar", "only by pressing"),
    ("semilla", "seed"),
    ("realizada", "realized"),
    ("pestaña 2", "tab 2"),
    ("modelo vs manual", "model vs manual"),
    ("origen de los Priors para la simulación", "source of Priors for the simulation"),
    ("Selecciona el", "Select the"),
    ("Ingresa los valores para los primeros 3 grupos", "Enter the values for the first 3 groups"),
    ("se ajustará automáticamente para que la suma sea 100", "will be automatically adjusted so the sum is 100"),
    ("Todos los valores deben ser estrictamente mayores a 0", "All values must be strictly greater than 0"),
    ("ya es el 100% o más", "is already 100% or more"),
    ("lo cual no es válido", "which is not valid"),
    ("Reduce los valores", "Reduce the values"),
    ("Configuración Manual Activa", "Manual Configuration Active"),
    ("se ha ajustado a", "has been adjusted to"),
    ("Cada polígono es un", "Each polygon is a"),
    ("el color muestra la", "the color shows the"),
    ("provienen de la misma base de casos que usa la aplicación", "come from the same case database used by the application"),
    ("no cargado en el inicio para acelerar la app", "not loaded at startup to speed up the app"),
    ("Use el botón para visualizarlo", "Use the button to view it"),
    ("Total de secuestros acumulados", "Total accumulated kidnappings"),
    ("suma de todos los municipios con datos", "sum of all municipalities with data"),
    ("Cobertura temporal del panel municipal", "Temporal coverage of the municipal panel"),
    ("Primer año con registro", "First year with records"),
    ("Último año con registro", "Last year with records"),
    ("Total Secuestros", "Total Kidnappings"),
    ("Primer Año", "First Year"),
    ("Último Año", "Last Year"),
    ("Total de secuestros", "Total kidnappings"),
    ("Departamento", "Department"),
    ("No se pudo cargar el mapa municipal", "Could not load the municipal map"),
    ("o no hay datos para graficar", "or there are no data to plot"),
    ("Comprueba que el archivo", "Check that the file"),
    ("esté en la misma carpeta que", "is in the same folder as"),
    ("tenga registros", "has records"),
    ("luego reinicia la aplicación", "then restart the application"),
    ("Desenlace focal en Tabla 1", "Focal Outcome in Table 1"),
    ("Editar valores Prior", "Edit Prior Values"),
    ("guardados en la sesión", "saved in the session"),
    ("Vuelve a los valores base", "Resets to base values"),
    ("restablecidos", "reset"),
    ("Mide la probabilidad técnica de captura dado el entorno y las políticas aplicadas", "Measures the technical capture probability given the environment and applied policies"),
    ("Es fundamental para la verosimilitud de supervivencia del captor", "It is fundamental for the captor's survival likelihood"),
    ("Editar captura", "Edit Capture"),
    ("Frecuencia alta", "High Frequency"),
    ("Frecuencia baja", "Low Frequency"),
    ("Aplica", "Applies"),
    ("Escala aplicada en ambos modos", "Scale applied in both modes"),
    ("La frecuencia baja es el 20% de la alta", "Low frequency is 20% of the high"),
    ("Pulse", "Press"),
    ("para generar una señal de ejemplo", "to generate a sample signal"),
    ("Agente a analizar", "Agent to analyze"),
    ("Resetear al modelo teórico", "Reset to theoretical model"),
    ("Desenlace físico a calibrar", "Physical outcome to calibrate"),
    ("Pesos del vector de tecnología", "Technology vector weights"),
    ("Materialización", "Materialization"),
    ("Arquitectura estocástica del", "Stochastic architecture of the"),
    ("Transforma la", "Transforms the"),
    ("intención estratégica", "strategic intention"),
    ("realizaciones observables", "observable realizations"),
    ("Fase 1", "Phase 1"),
    ("Fase 2", "Phase 2"),
    ("Implementación de la Intención", "Implementation of Intention"),
    ("Materialización del Desenlace", "Outcome Materialization"),
    ("los intervalos se construyen con la ley física activa", "the intervals are built with the active physical law"),
    ("Los corchetes en negrita indican la caída del sorteo", "Bold brackets indicate the draw outcome"),
    ("Historia pública y conjuntos de información", "Public history and information sets"),
    ("Edita la fila inicial", "Edit the initial row"),
    ("comprueba el resultado en la tabla pública", "check the result in the public table"),
    ("Acciones iniciales", "Initial actions"),
    ("Historia pública inicial", "Initial public history"),
    ("Estado ($S$): minimización en", "State ($S$): minimization in"),
    ("Los valores calibrados provienen del", "The calibrated values come from the"),
    ("Abra la pestaña", "Open tab"),
    ("desplácese hasta", "scroll to"),
    ("para generar los valores calibrados", "to generate the calibrated values"),
    ("Pestaña 6", "Tab 6"),
    ("Visualización dinámica del mecanismo", "Dynamic mechanism visualization"),
    ("Paneles construidos a partir de", "Panels built from"),
    ("Cada gráfica va seguida de una lectura breve", "Each chart is followed by a brief reading"),
    ("anclada en los números de la corrida", "anchored in the run numbers"),
    ("No hay ciclos dinámicos", "No dynamic cycles"),
    ("tras", "after"),
    ("Cargar gráficas de pestaña 6", "Load tab 6 charts"),
    ("Renderiza las gráficas usando los ciclos ya calculados", "Renders the charts using the already-computed cycles"),
    ("No recalcula ni modifica los resultados del mecanismo", "Does not recalculate or modify the mechanism results"),
    ("Para acelerar el flujo después de", "To speed up the workflow after"),
    ("las gráficas pesadas no se renderizan automáticamente", "the heavy charts are not rendered automatically"),
    ("Use el botón para ver la pestaña 6 completa", "Use the button to view the full tab 6"),
    ("con los mismos resultados guardados", "with the same saved results"),
    ("Creencias", "Beliefs"),
    ("Política óptima", "Optimal policy"),
    ("frente a benchmarks de rescate", "vs. rescue benchmarks"),
    ("y negociación", "and negotiation"),
    ("Frecuencia de decisiones óptimas", "Frequency of optimal decisions"),
    ("vs. ejecución MDG", "vs. MDG execution"),
    ("Incluir", "Include"),
    ("largo plazo", "long term"),
    ("corto plazo", "short term"),
    ("Nota:", "Note:"),
    ("según", "according to"),
]


def _translate_text_to_english(text: str) -> str:
    if st.session_state.get("app_language", "English") != "English":
        return text
    out = str(text)
    for src, dst in _ES_EN_REPLACEMENTS:
        pattern = re.escape(src)
        if re.match(r"\w", src, flags=re.UNICODE):
            pattern = r"(?<!\w)" + pattern
        if re.search(r"\w$", src, flags=re.UNICODE):
            pattern = pattern + r"(?!\w)"
        out = re.sub(pattern, dst, out, flags=re.IGNORECASE if src.islower() else 0)
    return out


def _translate_latex_expression(expr: str) -> str:
    """Translate only human text inside LaTeX text blocks, leaving commands intact."""
    if st.session_state.get("app_language", "English") != "English":
        return expr

    def repl(match: re.Match) -> str:
        return r"\text{" + _translate_text_to_english(match.group(1)) + "}"

    return re.sub(r"\\text\{([^{}]*)\}", repl, str(expr))


def _translate_display_value(value: Any) -> Any:
    if st.session_state.get("app_language", "English") != "English":
        return value
    if isinstance(value, str):
        return _translate_text_to_english(value)
    if isinstance(value, list):
        return [_translate_display_value(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_translate_display_value(v) for v in value)
    if isinstance(value, dict):
        return {_translate_display_value(k): _translate_display_value(v) for k, v in value.items()}
    if isinstance(value, pd.DataFrame):
        df = value.copy()
        df.columns = [_translate_text_to_english(str(c)) for c in df.columns]
        obj_cols = df.select_dtypes(include=["object", "string"]).columns
        for col in obj_cols:
            df[col] = df[col].map(lambda x: _translate_text_to_english(x) if isinstance(x, str) else x)
        return df
    return value


def _patch_streamlit_translation_layer() -> None:
    if getattr(st, "_rational_types_translation_patched", False):
        return

    def wrap_first_arg(name: str) -> None:
        original = getattr(st, name)

        def wrapped(*args, **kwargs):
            if args:
                args = (_translate_display_value(args[0]),) + args[1:]
            if "label" in kwargs:
                kwargs["label"] = _translate_display_value(kwargs["label"])
            return original(*args, **kwargs)

        setattr(st, name, wrapped)

    for _name in (
        "title", "header", "subheader", "markdown", "caption", "info", "success",
        "warning", "error", "button", "slider", "number_input", "text_input",
        "form_submit_button", "expander", "metric", "checkbox", "download_button",
    ):
        if hasattr(st, _name):
            wrap_first_arg(_name)

    if hasattr(st, "latex"):
        original_latex = st.latex

        def latex_wrapped(body, *args, **kwargs):
            return original_latex(_translate_latex_expression(str(body)), *args, **kwargs)

        st.latex = latex_wrapped

    original_tabs = st.tabs

    def tabs_wrapped(labels, *args, **kwargs):
        return original_tabs(_translate_display_value(labels), *args, **kwargs)

    st.tabs = tabs_wrapped

    def wrap_select_like(name: str) -> None:
        original = getattr(st, name)

        def wrapped(label, options, *args, **kwargs):
            label = _translate_display_value(label)
            user_format = kwargs.get("format_func")

            def translated_format(option):
                shown = user_format(option) if user_format else option
                return _translate_display_value(shown)

            kwargs["format_func"] = translated_format
            return original(label, options, *args, **kwargs)

        setattr(st, name, wrapped)

    for _name in ("selectbox", "radio"):
        if hasattr(st, _name):
            wrap_select_like(_name)

    for _name in ("dataframe", "table", "data_editor"):
        if hasattr(st, _name):
            original = getattr(st, _name)

            def wrapped_data(data=None, *args, _original=original, **kwargs):
                return _original(_translate_display_value(data), *args, **kwargs)

            setattr(st, _name, wrapped_data)

    if hasattr(st, "plotly_chart"):
        original_plotly_chart = st.plotly_chart

        def translate_plotly_value(value):
            if isinstance(value, str):
                return _translate_text_to_english(value)
            if isinstance(value, list):
                return [translate_plotly_value(v) for v in value]
            if isinstance(value, tuple):
                return tuple(translate_plotly_value(v) for v in value)
            if isinstance(value, dict):
                return {k: translate_plotly_value(v) for k, v in value.items()}
            return value

        def plotly_chart_wrapped(figure_or_data, *args, **kwargs):
            if st.session_state.get("app_language", "English") == "English" and hasattr(figure_or_data, "to_dict"):
                try:
                    figure_or_data = go.Figure(translate_plotly_value(figure_or_data.to_dict()))
                except Exception:
                    pass
            return original_plotly_chart(figure_or_data, *args, **kwargs)

        st.plotly_chart = plotly_chart_wrapped

    st._rational_types_translation_patched = True


_patch_streamlit_translation_layer()


# --- UTILIDADES BÁSICAS ---


def _st_table_row_count(data) -> int:
    """Número de filas de un DataFrame o Styler (para altura del grid Arrow)."""
    if data is None:
        return 0
    inner = getattr(data, "data", None)
    if isinstance(inner, pd.DataFrame):
        return int(len(inner.index))
    try:
        return int(len(data))
    except Exception:
        return 0


def _glide_full_height_px(
    n_rows: int,
    *,
    row_px: int = 36,
    header_px: int = 52,
    slack_px: int = 42,
    min_px: int = 130,
    cap_px: int = 9600,
) -> int:
    """Altura en px para mostrar todas las filas del dataframe/editor sin scroll interno (aprox.)."""
    if n_rows <= 0:
        return min_px
    return int(min(cap_px, max(min_px, header_px + n_rows * row_px + slack_px)))


def normalize_name(value):
    if value is None: return ""
    text = unicodedata.normalize("NFKD", str(value).upper())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return text.replace(",", "").replace(".", "").strip()


def _h0_d_select_label(x) -> str:
    """Etiqueta legible para d₀ ∈ {0,1} (popover historia inicial)."""
    return {"—": "—", "0": "0 · no detección", "1": "1 · detección"}.get(str(x), str(x))


# --- CONSTANTES GEOGRÁFICAS ---

DPTO_REGION_FALLBACK_RAW = {
    # Andina
    "ANTIOQUIA": "Andina", "BOYACÁ": "Andina", "CALDAS": "Andina", 
    "CUNDINAMARCA": "Andina", "HUILA": "Andina", "QUINDÍO": "Andina", 
    "RISARALDA": "Andina", "SANTANDER": "Andina", "TOLIMA": "Andina",
    "BOGOTÁ, D.C.": "Andina",
    
    # Caribe
    "ATLÁNTICO": "Caribe", "BOLÍVAR": "Caribe", "CESAR": "Caribe", 
    "CÓRDOBA": "Caribe", "LA GUAJIRA": "Caribe", "MAGDALENA": "Caribe", 
    "SUCRE": "Caribe", "SAN ANDRÉS, PROVIDENCIA Y SANTA CATALINA": "Caribe",
    "ARCHIPIELAGO DE SAN ANDRES, PROVIDENCIA Y SANTA CATALINA": "Caribe",
    
    # Pacífica / Zona Roja
    "CAUCA": "Pacífica / Zona Roja", "CHOCÓ": "Pacífica / Zona Roja", 
    "NARIÑO": "Pacífica / Zona Roja", "NORTE DE SANTANDER": "Pacífica / Zona Roja", 
    "VALLE DEL CAUCA": "Pacífica / Zona Roja",
    
    # Oriente / Selva
    "AMAZONAS": "Oriente / Selva", "ARAUCA": "Oriente / Selva", 
    "CAQUETÁ": "Oriente / Selva", "CASANARE": "Oriente / Selva", 
    "GUAINÍA": "Oriente / Selva", "GUAVIARE": "Oriente / Selva", 
    "META": "Oriente / Selva", "PUTUMAYO": "Oriente / Selva", 
    "VAUPÉS": "Oriente / Selva", "VICHADA": "Oriente / Selva"
}

DPTO_REGION_FALLBACK = {
    normalize_name(departamento): region
    for departamento, region in DPTO_REGION_FALLBACK_RAW.items()
}

METROS_LIST = [
    "BELLO", "BOGOTA DC", "BUCARAMANGA", "CARTAGENA DE INDIAS", "CUCUTA", 
    "ENVIGADO", "MANIZALES", "MEDELLIN", "PEREIRA", "SABANETA", 
    "SANTIAGO DE CALI", "SOACHA"
]

def get_corrected_region(muni_name, dpto_name):
    """
    Corrige la región de un municipio basándose en su departamento,
    dejando fijos únicamente los metropolitanos.
    """
    norm_muni = normalize_name(muni_name)
    norm_dpto = normalize_name(dpto_name)
    
    # 1. Metropolitanos son fijos
    if norm_muni in METROS_LIST:
        return "Metropolitana"
    
    # 2. San Andrés es Caribe
    if "SAN ANDRES" in norm_dpto:
        return "Caribe"
        
    # 3. Los demás se ajustan a la región de su departamento (Corrección Estricta)
    return DPTO_REGION_FALLBACK.get(norm_dpto, "Sin región")

# --- CONSTANTES DE CALIBRACIÓN ---
COEF_DELTA = {
    "FARC": 0.0,      # Referencia (aprox 35% casos CMH)
    "ELN": -0.63,     # ln(19/36)
    "PAR": -0.03,     # ln(35/36)
    "DC": -1.25       # ln(10/36)
}

COEF_ETA = {
    # Valores UNIFORMES para todos los tipos (θ_K) según restricción estructural.
    # Calibrados como ln(RRR) de main.tex (Ref: Metropolitana)
    "Metropolitana":        {"DC": 0.00,  "PAR": 0.00,  "ELN": 0.00,  "FARC": 0.00}, # Referencia
    "Andina":               {"DC": -0.45, "PAR": -0.45, "ELN": -0.45, "FARC": -0.45}, # ln(0.637)
    "Caribe":               {"DC": -0.70, "PAR": -0.70, "ELN": -0.70, "FARC": -0.70}, # ln(0.496)
    "Pacífica / Zona Roja": {"DC": -0.20, "PAR": -0.20, "ELN": -0.20, "FARC": -0.20}, # Estimado
    "Oriente / Selva":      {"DC": -0.32, "PAR": -0.32, "ELN": -0.32, "FARC": -0.32}  # ln(0.727)
}

COEF_XI = {
    # Valores UNIFORMES según main.tex (ln-RRR para Liberación)
    "Público": {"DC": 1.36, "PAR": 1.36, "ELN": 1.36, "FARC": 1.36}, # ln(3.917)
    "Privado": {"DC": 0.00, "PAR": 0.00, "ELN": 0.00, "FARC": 0.00}  # Referencia
}

REGIONES = list(COEF_ETA.keys())

REGION_COLORS = {
    "Metropolitana": "#4C78A8",
    "Andina": "#59A14F",
    "Caribe": "#F28E2B",
    "Pacífica / Zona Roja": "#E15759",
    "Oriente / Selva": "#76B7B2",
    "Sin región": "#D9D9D9"
}

REGION_MAP_INV = {
    "Metrópolis": "Metropolitana",
    "Andina": "Andina",
    "Caribe": "Caribe",
    "Pacífico/Roja": "Pacífica / Zona Roja",
    "Oriente/Selva": "Oriente / Selva"
}

# Tres renglones LaTeX (tooltip «Prior») — se renderizan con KaTeX en `_render_focus_covariate_katex_table`.
_FOCUS_CAL_TIP_LAM = (
    r"\lambda_{j0}\ \text{según}\ \texttt{cal\_lambdas\_dict}\ \text{(inicial ModeloSecuestro; editable pestaña 2).}",
    r"\text{Riesgo basal \textit{cause-specific}:}\ \textbf{Mechanism.tex},\ \textbf{main.tex}.",
    r"\text{Agregados:}\ \texttt{Data\_CMH}\text{ / CNMH en otras vistas; aquí calibración estructural.}",
)
_FOCUS_CAL_TIP_BK = (
    r"\beta_{K,j}(\theta_{K})\ \text{desde}\ \texttt{cal\_betas\_dict}\ \text{(ModeloSecuestro; editable P2).}",
    r"\text{Matriz por tipo}\ \theta_{K}\ \text{y causa}\ j\ \text{; alineada al mecanismo del texto.}",
    r"\text{Fuente:}\ \texttt{model\_logic}\ \text{+ app; no contador incidente a incidente.}",
)
_FOCUS_CAL_TIP_Z = (
    r"\texttt{COEF\_ETA}[z,\theta_{K}]\text{: misma pieza que }\eta\ \text{en softmax de }\mu_0\ \text{(pestaña 1).}",
    r"\texttt{app.py}\text{:}\ \texttt{COEF\_DELTA},\ \texttt{COEF\_XI}\text{; guías \textbf{main.tex} / CNMH.}",
    r"\text{Par}\ (z,\theta_{K})\text{: región}\ Z\ \text{en P1, tipo en P2 (fila activa).}",
)
# Tooltip «Prior» (ζ, φ): tentativo desde Data_CMH — misma esquina superior derecha de la celda.
_FOCUS_CAL_TIP_INST_MDG = (
    r"\text{Valor tentativo: mezcla 50\%-50\% entre marginal}\ \hat{p}_j^{\mathrm{CMH}}\ \text{y}\ \hat{p}_j(\theta_K)\ \text{en}\ \texttt{Data\_CMH}.",
    r"\text{Escala en }[0.04,\,0.92]\ \text{(exponente; \textbf{main.tex} / Mechanism).}",
    r"\text{En simulación MDG el estado puede actualizarlos; ancla descriptiva CMH, no MLE.}",
)


@st.cache_data(show_spinner=False)
def _focus_cmh_endogenous_tentatives(theta_k: str) -> dict:
    """ζ, φ tentativos a partir de proporciones CMH (marginal + fila θ_K)."""
    marg = {"Liberación": 0.50, "Rescate": 0.10, "Pago": 0.15, "Muerte": 0.12}
    ths = {k: marg[k] for k in marg}
    tbl = None
    try:
        tbl = _compute_cmh_beta_calibration_tables_cached(
            _data_file_signature("Data_CMH.csv"),
            _PSI8_CMH_CALIB_VERSION,
        )
    except Exception:
        tbl = None
    if tbl and isinstance(tbl.get("marginal_outcome"), dict):
        mo = tbl["marginal_outcome"]
        for k in marg:
            if k in mo:
                try:
                    v = float(mo[k])
                    if np.isfinite(v):
                        marg[k] = float(np.clip(v, 1e-4, 0.995))
                except (TypeError, ValueError):
                    pass
    if tbl is not None:
        bst = tbl.get("shares_by_theta")
        if bst is not None and hasattr(bst, "loc"):
            try:
                ser = bst.loc[str(theta_k)]
                for k in marg:
                    try:
                        v = float(ser[k])
                        if np.isfinite(v):
                            ths[k] = float(np.clip(v, 1e-4, 0.995))
                    except (TypeError, ValueError, KeyError):
                        pass
            except Exception:
                pass
    mix = {k: 0.5 * marg[k] + 0.5 * ths[k] for k in marg}
    pL, pR, pP, pM = mix["Liberación"], mix["Rescate"], mix["Pago"], mix["Muerte"]
    den = pL + pR + pP + pM + 1e-9

    def _cl(x: float, lo: float = 0.04, hi: float = 0.92) -> float:
        return float(max(lo, min(hi, x)))

    z_a = _cl(0.10 + 0.42 * (pM / den))
    z_g = _cl(0.09 + 0.48 * (pR / den))
    # Sensibilidad directa de las lambdas a la politica ejecutada
    # (alpha_star_used, gamma_star_used). No usa multiplicadores adicionales: estos
    # valores reemplazan las magnitudes base de zeta_alpha/zeta_gamma.
    _lambda_policy_sensitivity = {
        "DC": {"zeta_alpha": 0.24092264250100145, "zeta_gamma": 0.5450200031654301},
        "PAR": {"zeta_alpha": 0.21828464252363947, "zeta_gamma": 0.5848120031256381},
        "ELN": {"zeta_alpha": 0.21005264253187145, "zeta_gamma": 0.5532280031572221},
        "FARC": {"zeta_alpha": 0.20870864253321547, "zeta_gamma": 0.6197560030906941},
    }
    _sens = _lambda_policy_sensitivity.get(str(theta_k))
    if isinstance(_sens, dict):
        z_a = float(_sens["zeta_alpha"])
        z_g = float(_sens["zeta_gamma"])
    z_d = _cl(0.08 + 0.44 * ((pM + 0.35 * pR) / den))
    ph_f = _cl(0.11 + 0.52 * (pL / den))
    ph_kc = _cl(0.09 + 0.46 * ((pL + 0.5 * pR) / den))
    ph_kk = _cl(0.07 + 0.58 * (pM / den))
    z_R = _cl(0.12 + 0.50 * (pR / den))
    return {
        "zeta_alpha": z_a,
        "zeta_gamma": z_g,
        "zeta_d": z_d,
        "phi_F": ph_f,
        "phi_K_cont": ph_kc,
        "phi_K_kill": ph_kk,
        "zeta_R": z_R,
    }


# Cita alineada con references.bib (@online{cnmh_datos, ...}); markdown para enlace clicable en la app.
_CITA_CMH_URL = (
    "https://micrositios.centrodememoriahistorica.gov.co/observatorio/portal-de-datos/el-conflicto-en-cifras/"
)
def _cita_biblio_cmh() -> str:
    return _ui_text(
        f"**Source (CNMH, 2026):** National Centre for Historical Memory. "
        f"[Data Portal: The conflict in figures]({_CITA_CMH_URL}) "
        "(Observatory of Memory and Conflict).",
        f"**Fuente (CNMH, 2026):** Centro Nacional de Memoria Histórica. "
        f"[Portal de Datos: El conflicto en cifras]({_CITA_CMH_URL}) "
        "(Observatorio de Memoria y Conflicto).",
    )


CITA_BIBLIO_CMH = (
    "**Fuente (CNMH, 2026):** Centro Nacional de Memoria Histórica. "
    f"[Portal de Datos: El conflicto en cifras]({_CITA_CMH_URL}) "
    "(Observatorio de Memoria y Conflicto)."
)

# Desenlace focal en pestaña 2 (j en Mechanism.tex): Pago, Muerte, Rescate.
CAL_FOCUS_OUTCOMES = [
    (1, "j=1 · Pago", "Pago", "Pago", -0.1),
    (2, "j=2 · Muerte", "Muerte", "Muerte", -0.2),
    (3, "j=3 · Rescate", "Rescate", "Rescate", 0.5),
]

# Texto corto en botón «Guardar DC … · θ_K» (DC = desenlace focal / causa j).
_FOCUS_OUTCOME_BTN_SHORT = {1: "Pago", 2: "Muerte", 3: "Rescate"}
_FOCUS_LABEL_TO_J = {v: k for k, v in _FOCUS_OUTCOME_BTN_SHORT.items()}
_CAL_FOCUS_SELECT_OPTIONS = ["Pago", "Muerte", "Rescate"]

# Soporte completo para MDG (Materialización y Eq. 28-29)
_MDG_OUTCOME_LABELS = {
    1: "Liberación",
    2: "Rescate",
    3: "Pago",
    4: "Muerte",
    5: "Continuar"
}

_MECH_LATEX = {
    1: r"""\begin{aligned}
\lambda_1(t \mid \cdot) &= \lambda_{10}(t) \exp \Bigl(
+\beta_F\,\mathbf{1}\{\theta_F=\text{Alta}\}
-\beta_V\,\mathbf{1}\{\theta_V=\text{Público}\}
+\beta_{S,1}\,\mathbf{1}\{\theta_S=\text{Laxo}\}
+\beta_{K,1}(\theta_K)
+\beta_{z,1}(z) \\
&\quad
-\zeta_{\alpha,1}\alpha_t^\ast - \zeta_{\gamma,1}\gamma_t^\ast - \zeta_{d,1}p_{\mathrm{det},t}
+\varphi_{F,1}\,\mathbf{1}\{\tilde{a}_t^F=\text{Pagar}\}
+\varphi_{K,1}\,\mathbf{1}\{\tilde{a}_t^K=\mathrm{cont}\}
\Bigr)
\end{aligned}""",
    2: r"""\begin{aligned}
\lambda_2(t \mid \cdot) &= \lambda_{20}(t) \exp\!\Bigl(
+\beta_{S,2}\,\mathbf{1}\{\theta_S=\text{Laxo}\}
+\beta_{K,2}(\theta_K)
+\beta_{z,2}(z)
+\zeta_{\alpha,2}\alpha_t^\ast + \zeta_{\gamma,2}\gamma_t^\ast
-\varphi_{F,2}\,\mathbf{1}\{\tilde{a}_t^F=\text{Pagar}\} \\
&\quad
-\zeta_{d,2}p_{\mathrm{det},t}
+\varphi_{K,2}^{\mathrm{kill}}\,\mathbf{1}\{\tilde{a}_t^K=\mathrm{kill}\}
+\varphi_{K,2}^{\mathrm{cont}}\,\mathbf{1}\{\tilde{a}_t^K=\mathrm{cont}\}
\Bigr)
\end{aligned}""",
    3: r"""\begin{aligned}
\lambda_3(t \mid \cdot) &= \lambda_{30}(t) \exp\!\Bigl(
-\beta_{S,3}\,\mathbf{1}\{\theta_S=\text{Laxo}\}
+\beta_{K,3}(\theta_K)
+\beta_{z,3}(z)
+\zeta_{\alpha,3}\alpha_t^\ast + \zeta_{\gamma,3}\gamma_t^\ast + \zeta_{d,3}p_{\mathrm{det},t} \\
&\quad
+\zeta_{R}\,\mathbf{1}\{\tilde{a}_t^S=\text{Rescate}\}
-\varphi_{F,3}\,\mathbf{1}\{\tilde{a}_t^F=\text{Pagar}\}
+\varphi_{K,3}\,\mathbf{1}\{\tilde{a}_t^K=\mathrm{cont}\}
\Bigr)
\end{aligned}""",
}

# Texto unificado (tono Mechanism.tex / riesgos competitivos + MDG + Bayes); varía solo el desenlace j.
_MECH_EQUATION_BLURB_ES = {
    1: (
        "La intensidad de pago $\\lambda_1$ representa la propensión marginal al cierre del episodio por la **causa "
        "$j=1$** (**pago** / liberación por pago), actuando como una **tasa instantánea** y no como una probabilidad "
        "porcentual diaria. "
        "Bajo una estructura de **riesgos proporcionales**, esta intensidad escala una línea base según el entorno, "
        "la tecnología delictiva del captor y los instrumentos de política pública aplicados. "
        "Al operar en un marco de **riesgos competitivos**, el signo algebraico de cada coeficiente determina si un "
        "factor **acelera o frena** el desenlace por la **causa $j=1$**. "
        "Así el **aprendizaje bayesiano** puede procesar lo que **realmente se ejecutó** (vía proceso MDG) para "
        "identificar la identidad estructural del captor y sustentar el ajuste del modelo en esta dimensión."
    ),
    2: (
        "La intensidad de muerte $\\lambda_2$ representa la propensión marginal al cierre del episodio por la **causa "
        "$j=2$** (**muerte** del cautivo), actuando como una **tasa instantánea** y no como una probabilidad "
        "porcentual diaria. "
        "Bajo una estructura de **riesgos proporcionales**, esta intensidad escala una línea base según el entorno, "
        "la tecnología delictiva del captor y los instrumentos de política pública aplicados. "
        "En **riesgos competitivos**, el signo algebraico de cada coeficiente determina si un factor **acelera o frena** "
        "la **causa $j=2$** frente a las demás salidas. "
        "Así el **aprendizaje bayesiano** puede procesar lo que **realmente se ejecutó** (vía proceso MDG) para "
        "identificar la identidad estructural del captor y sustentar el ajuste del modelo en esta dimensión."
    ),
    3: (
        "La intensidad de rescate $\\lambda_3$ representa la propensión marginal al cierre del episodio por la **causa "
        "$j=3$** (**rescate** / intervención estatal con ese desenlace), como **tasa instantánea** y no como probabilidad "
        "porcentual diaria. "
        "Bajo **riesgos proporcionales**, escala la línea base según el entorno, la tecnología delictiva del captor "
        "y los instrumentos de política pública —incluida la materialización de la **vía de rescate** cuando el Estado "
        "actúa en esa dirección. "
        "En **riesgos competitivos**, el signo algebraico de cada coeficiente fija si un factor **acelera o frena** "
        "el desenlace por la **causa $j=3$** frente al pago ($j=1$) o la muerte ($j=2$). "
        "Así el **aprendizaje bayesiano** puede procesar lo que **realmente se ejecutó** (vía proceso MDG) para "
        "identificar la identidad estructural del captor y sustentar el ajuste del modelo en esta dimensión."
    ),
}

_MECH_EQUATION_BLURB_EN = {
    1: (
        "The payment intensity $\\lambda_1$ represents the marginal propensity for the episode to close through "
        "**cause $j=1$** (**payment** / release by payment), acting as an **instantaneous rate** rather than a daily "
        "percentage probability. "
        "Under a **proportional hazards** structure, this intensity scales a baseline according to the environment, "
        "the captor's criminal technology, and the applied public-policy instruments. "
        "Operating within a **competing-risks** framework, the algebraic sign of each coefficient determines whether "
        "a factor **accelerates or retards** the outcome under **cause $j=1$**. "
        "Bayesian learning can thus process what was **actually executed** (via the MDG process) to identify the "
        "captor's structural type and support model calibration along this dimension."
    ),
    2: (
        "The death intensity $\\lambda_2$ represents the marginal propensity for the episode to close through "
        "**cause $j=2$** (**death** of the captive), acting as an **instantaneous rate** rather than a daily "
        "percentage probability. "
        "Under a **proportional hazards** structure, this intensity scales a baseline according to the environment, "
        "the captor's criminal technology, and the applied public-policy instruments. "
        "In **competing risks**, the algebraic sign of each coefficient determines whether a factor **accelerates or "
        "retards** **cause $j=2$** relative to the other outcomes. "
        "Bayesian learning can thus process what was **actually executed** (via the MDG process) to identify the "
        "captor's structural type and support model calibration along this dimension."
    ),
    3: (
        "The rescue intensity $\\lambda_3$ represents the marginal propensity for the episode to close through "
        "**cause $j=3$** (**rescue** / state intervention with that outcome), as an **instantaneous rate** rather "
        "than a daily percentage probability. "
        "Under **proportional hazards**, it scales the baseline according to the environment, the captor's criminal "
        "technology, and public-policy instruments — including the materialisation of the **rescue pathway** when "
        "the State acts in that direction. "
        "In **competing risks**, the algebraic sign of each coefficient fixes whether a factor **accelerates or "
        "retards** the outcome under **cause $j=3$** relative to payment ($j=1$) or death ($j=2$). "
        "Bayesian learning can thus process what was **actually executed** (via the MDG process) to identify the "
        "captor's structural type and support model calibration along this dimension."
    ),
}

_MECH_EQUATION_BLURB = _MECH_EQUATION_BLURB_ES


def _cal_focus_row(j_mech: int):
    for row in CAL_FOCUS_OUTCOMES:
        if row[0] == j_mech:
            return row
    return CAL_FOCUS_OUTCOMES[0]


# Etiquetas «Término» → LaTeX inline (KaTeX) para subíndices θ_F, μ₀, α, γ, etc.
_FOCUS_TERM_KATEX = {
    "Riesgo basal (pago)": r"\text{Riesgo basal (pago)}",
    "Capacidad de pago alta (θ_F)": r"\text{Capacidad de pago alta }(\theta_{\mathrm{F}})",
    "Víctima perfil público (θ_V)": r"\text{Víctima perfil público }(\theta_{\mathrm{V}})",
    "Estado laxo (θ_S)": r"\text{Estado laxo }(\theta_{\mathrm{S}})",
    "Tipo secuestrador (θ_K)": r"\text{Tipo secuestrador }(\theta_{K})",
    "Heterogeneidad geográfica": r"\text{Heterogeneidad geográfica}",
    "Región de cautiverio (Z; z)": (
        r"\text{Región de cautiverio }(Z;\ z)"
    ),
    "Instrumento α (bloqueo)": r"\text{Instrumento }\alpha\text{ (bloqueo)}",
    "Instrumento γ (presión)": r"\text{Instrumento }\gamma\text{ (presión)}",
    "Detección p_det": r"\text{Detección }p_{\mathrm{det}}",
    "Familia paga (MDG)": r"\text{Familia paga (MDG)}",
    "K continúa (MDG)": r"\text{K continúa (MDG)}",
    "Riesgo basal (muerte)": r"\text{Riesgo basal (muerte)}",

    "K mata (MDG)": r"\text{K mata (MDG)}",
    "Riesgo basal (rescate)": r"\text{Riesgo basal (rescate)}",

    "Estado rescata (MDG)": r"\text{Estado rescata (MDG)}",
    "Instrumento α": r"\text{Instrumento }\alpha",
    "Instrumento γ": r"\text{Instrumento }\gamma",
    "Presión en exponente (app)": r"\text{Presión en exponente (app)}",
    "Suma lineal en exp. (app)": r"\text{Suma lineal en exp. (app)}",
    "Maduración (escala h)": r"\text{Maduración (escala h)}",
    "Hazard discreto causal": r"\text{Hazard discreto causal}",
    "Prior μ₀ · región": r"\text{Prior }\mu_0\text{ · región}",
    "Prior μ₀ · perfil víctima": r"\text{Prior }\mu_0\text{ · perfil víctima}",
    "Prior μ₀ · intercepto tipo": r"\text{Prior }\mu_0\text{ · intercepto tipo}",
    "Nota panel incidente (pestaña 1)": r"\text{Nota panel incidente (pestaña 1)}",
    # Tabla 2 · intensidades efectivas (evitar «Término» con $…$: rompe \text{…} en KaTeX).
    "Umbral de maduración (texto)": r"\text{Umbral de maduración (texto)}",
    "Canal exógeno (basal)": r"\text{Canal exógeno (basal)}",
    # Tabla 3 · logit de p_det,t (Mechanism.tex) — ES y EN
    "Intercepto logit (p_det)": r"\text{Intercepto logit }(p_{\mathrm{det}})",
    "Intercept logit (p_det)": r"\text{Intercept logit }(p_{\mathrm{det}})",
    "Peso de α* en p_det": (
        r"\text{Peso de }\alpha^\ast\text{ en }p_{\mathrm{det}}"
    ),
    "Weight of α* in p_det": (
        r"\text{Weight of }\alpha^\ast\text{ in }p_{\mathrm{det}}"
    ),
    "Peso de γ* en p_det": (
        r"\text{Peso de }\gamma^\ast\text{ en }p_{\mathrm{det}}"
    ),
    "Weight of γ* in p_det": (
        r"\text{Weight of }\gamma^\ast\text{ in }p_{\mathrm{det}}"
    ),
    # Tabla 4 · supervivencia (ec. 37–38 Mechanism.tex) — ES y EN
    "α₀ rescate (Prior · ec. 38)": r"\text{Letalidad intrínseca }\alpha_0(\theta_K)",
    "β_R precisión rescate (Prior · ec. 38)": r"\text{Productividad }\beta_R",
    "Precisión modal (ec. 37)": r"\iota_t",
    # Table 4 · English keys — avoid \_  inside \text{} (KaTeX issue with _ in term strings)
    "α_leth lethality (Prior · eq. 38)": r"\text{Lethality }\alpha_{\mathrm{leth}}(\theta_K)",
    "β_R productivity (Prior · eq. 38)": r"\text{Productivity }\beta_R",
    "Modal precision (eq. 37)": r"\iota_t",
    "Match indicator (eq. 38)": r"\text{Match }\mathbf{1}\{\hat{\theta}_t=\theta_K\}",
    # Tabla 8 · Propensión Psi_j (Ec. 29)
    "Capacidad operativa (phi_j,gamma)": r"\text{Capacidad operativa }(\phi_{j,\gamma})",
    "Peso: Disciplina Militar (phi_j,1)": r"\text{Peso: Disciplina Militar }(\phi_{j,1})",
    "Peso: Logística / Suministros (phi_j,2)": r"\text{Peso: Logística / Suministros }(\phi_{j,2})",
    "Peso: Impaciencia Financiera (phi_j,3)": r"\text{Peso: Impaciencia Financiera }(\phi_{j,3})",
    "Peso: Letalidad / Agresividad (phi_j,4)": r"\text{Peso: Letalidad / Agresividad }(\phi_{j,4})",
    "Precisión kappa_j": r"\text{Precisión }\kappa_j",
}

# Tabla 5 · columna «Término»: KaTeX (evitar «·» Unicode dentro de \text: rompe como \cdotp;
# usar \,\cdot\, en modo matemático; f_{0} agrupado para no partir f y el subíndice).
_TABLA5_LAB_KATEX = (
    r"\text{1. Tono medio } f_{0}",
    r"\text{2. Varianza del tono}",
    r"\text{3. Tasa de pausas}",
    r"\text{4. Formalidad / agresión del discurso}",
)
_TABLA5_TERM_KATEX: dict[str, str] = {}
for _i5 in range(4):
    _lp = f"{_i5 + 1}. {_VOZ_RASGO_LABELS[_i5]}"
    _lk = _TABLA5_LAB_KATEX[_i5]
    for _sp, _st in (
        ("referencia acústica (Observado)", r"\text{referencia acústica (Observado)}"),
        ("error largo plazo (Observado)", r"\text{error largo plazo (Observado)}"),
        ("error corto plazo (Observado)", r"\text{error corto plazo (Observado)}"),
    ):
        _TABLA5_TERM_KATEX[f"{_lp} · {_sp}"] = _lk + r" \,\cdot\, " + _st

# Tabla 7 · MDG: «Término» con griegos, subíndices y \underline{c} (evitar Unicode c̲ dentro de \text).
# Fila η_cal: misma cadena en DataFrame y lookup; el punto medio no va dentro de \text (KaTeX lo rompe → \cdotp duplicado).
_TABLA7_TERM_ETA_CAL = "η_cal · calibrado (prior)"

_TABLA7_MDG_TERM_KATEX: dict[str, str] = {
    # Spanish keys
    "Prior marginal μ(θ_K) incidente": (
        r"\text{Prior marginal }\mu(\theta_K)\text{ incidente}"
    ),
    "Temperatura base T₀": r"\text{Temperatura base } T_0",
    "Entropía de referencia H(μ₀)": r"\text{Entropía de referencia } H(\mu_0)",
    "Ratio H(μ_t) / H(μ₀)": r"\text{Ratio } \bigl(H(\mu_t) \big/ H(\mu_0)\bigr)",
    "Piso c̲ (inferior)": r"\text{Piso }\underline{c}\text{ (inferior)}",
    _TABLA7_TERM_ETA_CAL: (
        r"\eta_{\mathrm{cal}}\,\cdotp\,\text{calibrado (prior)}"
    ),
    # English keys (used when _ui_text() returns English term strings)
    "Prior marginal μ(θ_K) incident": (
        r"\text{Prior marginal }\mu(\theta_K)\text{ incident}"
    ),
    "Base temperature T₀": r"\text{Base temperature } T_0",
    "Reference entropy H(μ₀)": r"\text{Reference entropy } H(\mu_0)",
    "Floor c̲ (lower)": r"\text{Floor }\underline{c}\text{ (lower)}",
}


def _mdg_action_label_to_katex(lbl: str) -> str:
    """Etiquetas tipo «Matar (a_kill) ★» → texto + (a_{\\mathrm{kill}}) en modo matemático."""
    s = str(lbl).strip()
    star_tex = ""
    if "★" in s:
        si = s.rfind("★")
        s = s[:si].strip()
        star_tex = r" \,\text{★}"
    m = re.match(r"^(.+?)\s*\(a_([A-Za-z0-9]+)\)\s*$", s)
    if not m:
        return r"\text{" + _escape_katex_text_fragment(s) + "}" + star_tex
    name = m.group(1).strip()
    tag = m.group(2)
    return (
        r"\text{"
        + _escape_katex_text_fragment(name)
        + r"}\,(a_{\mathrm{"
        + tag
        + r"}})"
        + star_tex
    )


def _escape_katex_text_fragment(s: str) -> str:
    """Fragmento dentro de \\text{…}; «·» Unicode no debe ir literal en un solo \\text (KaTeX → \\cdotp)."""
    return (
        str(s)
        .replace("\\", r"\textbackslash ")
        .replace("{", r"\{")
        .replace("}", r"\}")
        .replace("_", r"\_")
    )


def _focus_term_to_latex(term_plain: str) -> str:
    term_plain = unicodedata.normalize("NFC", str(term_plain))
    if term_plain in _TABLA5_TERM_KATEX:
        return _TABLA5_TERM_KATEX[term_plain]
    if term_plain in _TABLA7_MDG_TERM_KATEX:
        return _TABLA7_MDG_TERM_KATEX[term_plain]
    # Punto medio U+2219 vs U+00B7; sin esto el split « · » hace \text{η_cal} y KaTeX descompone mal.
    _tn = term_plain.replace("\u2219", "\u00b7").strip()
    if _tn == _TABLA7_TERM_ETA_CAL:
        return _TABLA7_MDG_TERM_KATEX[_TABLA7_TERM_ETA_CAL]
    if re.match(
        r"^\u03b7_cal\s*[\u00b7\u2219]\s*calibrado\s*\(prior\)\s*$",
        term_plain,
    ):
        return _TABLA7_MDG_TERM_KATEX[_TABLA7_TERM_ETA_CAL]
    if term_plain.startswith("Implementación MDG · "):
        _rest = term_plain[len("Implementación MDG · "):]
        return r"\text{Implementación MDG} \,\cdot\, " + _mdg_action_label_to_katex(_rest)
    if term_plain in _FOCUS_TERM_KATEX:
        return _FOCUS_TERM_KATEX[term_plain]
    if " · " in term_plain:
        _parts = [p for p in term_plain.split(" · ") if p]
        if len(_parts) >= 2:
            def _part_to_tex(p: str) -> str:
                if p in _FOCUS_TERM_KATEX:
                    return _FOCUS_TERM_KATEX[p]
                if p in _TABLA7_MDG_TERM_KATEX:
                    return _TABLA7_MDG_TERM_KATEX[p]
                return r"\text{" + _escape_katex_text_fragment(p) + "}"
            return r" \,\cdot\, ".join(_part_to_tex(p) for p in _parts)
    esc = _escape_katex_text_fragment(term_plain)
    return r"\text{" + esc + "}"


def _fmt_es_num(x, ndigits: int = 2) -> str:
    """Formato numérico visible: EN 1,234.56; ES 1.234,56."""
    if x is None:
        return "—"
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return str(x)
    if np.isnan(xf) or np.isinf(xf):
        return "—"
    xf = round(float(xf), int(ndigits))
    neg = xf < 0
    xf = abs(xf)
    s = f"{xf:,.{int(ndigits)}f}"
    if st.session_state.get("app_language", "English") == "English":
        body = s
    else:
        body = s.replace(",", "\x00").replace(".", ",").replace("\x00", ".")
    return ("-" if neg else "") + body


def _fmt_es_num_sigfirst(x, *, max_decimals: int = 6) -> str:
    """Formato compacto con separador decimal según idioma."""
    if x is None:
        return "—"
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return str(x)
    if np.isnan(xf) or np.isinf(xf):
        return "—"
    if xf == 0:
        return "0"
    ax = abs(xf)
    if ax < 0.1:
        dec = min(max_decimals, max(3, int(np.floor(-np.log10(ax))) + 1))
    else:
        dec = min(max_decimals, 2)
    s = _fmt_es_num(xf, dec)
    dec_sep = "." if st.session_state.get("app_language", "English") == "English" else ","
    if dec_sep in s:
        s = s.rstrip("0").rstrip(dec_sep)
    return s


def _parse_es_num(s: str):
    """Interpreta números visibles según idioma; inválido o vacío -> None."""
    s = (s or "").strip().replace(" ", "").replace("\u00a0", "")
    if not s or s in ("-", ",", "."):
        return None
    neg = s.startswith("-")
    if neg:
        s = s[1:].strip()
    if not s:
        return None
    try:
        if st.session_state.get("app_language", "English") == "English":
            if "." in s:
                left, right = s.rsplit(".", 1)
                if "," in right or "." in right:
                    return None
                left = left.replace(",", "")
                if not right.isdigit():
                    return None
                if left == "":
                    left = "0"
                if not left.isdigit():
                    return None
                v = float(f"{left}.{right}")
            else:
                s_clean = s.replace(",", "")
                if s_clean == "" or not s_clean.isdigit():
                    return None
                v = float(s_clean)
        elif "," in s:
            left, right = s.rsplit(",", 1)
            if "." in right or "," in right:
                return None
            left = left.replace(".", "")
            if not right.isdigit():
                return None
            if left == "":
                left = "0"
            if not left.isdigit():
                return None
            v = float(f"{left}.{right}")
        else:
            s_clean = s.replace(".", "")
            if s_clean == "" or not s_clean.isdigit():
                return None
            v = float(s_clean)
    except ValueError:
        return None
    return -v if neg else v


# Tabla 1 (covariables foco λ_j): valores editables guardados en JSON junto a app.py
_FOCUS_COV_JSON = os.path.join(os.path.dirname(__file__), "user_focus_covariates.json")
# Tabla 1 · «Editar valores Prior»: solo filas con # ∈ [TAB1_PRIOR_FILA_MIN, TAB1_PRIOR_FILA_MAX] (incl.).
TAB1_PRIOR_FILA_MIN = 5
TAB1_PRIOR_FILA_MAX = 11


def _focus_cov_profile_key(
    j_mech: int,
    theta_k: str,
    z_region: str,
    v_victim: str,
    f_capa: str,
    s_tipo: str,
) -> str:
    return "|".join(str(x) for x in (j_mech, theta_k, z_region, v_victim, f_capa, s_tipo))


def _load_focus_cov_store() -> dict:
    if not os.path.isfile(_FOCUS_COV_JSON):
        return {}
    try:
        with open(_FOCUS_COV_JSON, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError, json.JSONDecodeError):
        return {}


def _save_focus_cov_store(store: dict) -> None:
    with open(_FOCUS_COV_JSON, "w", encoding="utf-8") as fp:
        json.dump(store, fp, ensure_ascii=False, indent=2)


def _apply_focus_cov_saved_values(df: pd.DataFrame, profile_key: str, store: dict, th_actual: str) -> pd.DataFrame:
    # Clave para la fila de Tipo secuestrador (theta_K) - específica del tipo
    entry_actual = store.get(profile_key)
    # Clave para el resto de filas (invariantes) - forzamos referencia a FARC
    # Se asume que el separador en profile_key es "|" y theta_k es el segundo elemento
    _pk_ref = profile_key.replace(f"|{th_actual}|", "|FARC|")
    entry_ref = store.get(_pk_ref)

    if (not entry_actual or not isinstance(entry_actual, dict)) and (not entry_ref or not isinstance(entry_ref, dict)):
        return df

    out = df.copy()
    for i in out.index:
        if "#" in out.columns:
            try:
                _nr = int(round(float(out.at[i, "#"])))
            except (TypeError, ValueError):
                continue
            if _nr < TAB1_PRIOR_FILA_MIN or _nr > TAB1_PRIOR_FILA_MAX:
                continue
        if "Origen del valor" in out.columns:
            _ov = str(out.at[i, "Origen del valor"]).strip()
            if _ov == "Observado" or _ov.startswith("Observado"):
                continue

        term = str(out.at[i, "Término"])
        is_tk = "Tipo secuestrador" in term
        entry = entry_actual if is_tk else entry_ref
        
        if not entry or not isinstance(entry, dict):
            continue
        term_vals = entry.get("valores_por_termino")
        if not isinstance(term_vals, dict) or term not in term_vals:
            continue
            
        try:
            v_saved = float(term_vals[term])
            out.at[i, "Valor"] = round(v_saved, 2)
            out.at[i, "Origen del valor"] = "Editado (Memoria)"
        except (TypeError, ValueError):
            pass
    return out


def _tab1_popover_rows_solo_calibrados(df_cov: pd.DataFrame) -> pd.DataFrame:
    """Pop «Editar valores Prior · Tabla 1»: solo filas con # ∈ [TAB1_PRIOR_FILA_MIN, TAB1_PRIOR_FILA_MAX]."""
    _cols = ["#", "Término", "Coeficiente", "Valor", "Origen del valor"]
    if df_cov is None or df_cov.empty:
        return pd.DataFrame(columns=_cols)
    if "#" not in df_cov.columns:
        return pd.DataFrame(columns=_cols)
    _nums = pd.to_numeric(df_cov["#"], errors="coerce")

    def _in_range(x) -> bool:
        try:
            if pd.isna(x):
                return False
            k = int(round(float(x)))
        except (TypeError, ValueError):
            return False
        return TAB1_PRIOR_FILA_MIN <= k <= TAB1_PRIOR_FILA_MAX

    _m = _nums.apply(_in_range)
    _out = df_cov.loc[_m, _cols].copy()
    if _out.empty:
        return _out.reset_index(drop=True)
    _out["Valor"] = pd.to_numeric(_out["Valor"], errors="coerce").astype(float)
    return _out.reset_index(drop=True)


def _ensure_focus_cov_store_in_session() -> None:
    if "focus_cov_store" not in st.session_state:
        st.session_state.focus_cov_store = _load_focus_cov_store()


def _render_focus_prior_valor_inputs(
    df_prior: pd.DataFrame, *, widget_stem: str, profile_key: str
) -> dict[str, float]:
    """Devuelve un dict término → valor editado para cada fila de ``df_prior``."""
    out: dict[str, float] = {}
    if df_prior is None or df_prior.empty:
        st.info(
            _ui_text(
                f"No editable rows **# {TAB1_PRIOR_FILA_MIN}–{TAB1_PRIOR_FILA_MAX}** for this cause j "
                "(rows 1–4 are fixed here).",
                f"No hay filas **# {TAB1_PRIOR_FILA_MIN}–{TAB1_PRIOR_FILA_MAX}** editables en esta causa j "
                "(las filas 1–4 no se modifican aquí).",
            )
        )
        return out
    bump = int(st.session_state.get(f"fce_bump_{profile_key}", 0))
    for _, row in df_prior.iterrows():
        term = str(row["Término"])
        coef_compact = str(row["Coeficiente"])
        num = int(row["#"])
        try:
            v_default = float(row["Valor"])
        except (TypeError, ValueError):
            v_default = 0.0
        kid = hashlib.md5(term.encode("utf-8")).hexdigest()[:16]
        wkey = f"{widget_stem}_v_{bump}_{kid}"
        # Una sola línea horizontal: # · término · símbolo LaTeX · input
        c_n, c_t, c_k, c_v = st.columns((0.38, 1.45, 1.15, 1.12), gap="small")
        with c_n:
            st.markdown(f"{num}")
        with c_t:
            st.markdown(f"**{_translate_text_to_english(term)}**")
        with c_k:
            st.latex(coef_compact)
        with c_v:
            v = st.number_input(
                "Valor",
                value=float(v_default),
                step=0.001,
                format="%.6f",
                key=wkey,
                label_visibility="collapsed",
            )
        out[term] = float(v)
    return out


@st.cache_data(show_spinner=False)
def _build_focus_covariate_table(
    *,
    j_mech: int,
    theta_k: str,
    z_region: str,
    v_victim: str,
    f_capa: str,
    s_tipo: str,
    lambdas_dict: dict,
    betas_dict: dict,
    M_t: float,
    presion_S: float,
    h_j_numeric: float,
    tipo_incidente_p1: str,
    zeta_phi: dict,
) -> pd.DataFrame:
    """Tabla 1: solo sumandos de λ_j(t) en la ecuación mostrada (_MECH_LATEX para j=1,2,3).

    «Coeficiente» = símbolo calibrado (LaTeX corto); «Origen del valor» = expresión LaTeX del sumando
    o la etiqueta **Prior** cuando el coeficiente ζ/φ es editable y proviene del ancla CMH (mix marginal·θ_K).
    La columna **Clase** (KaTeX) solo indica **Prior** u **Observado** según el tipo de insumo.
    No incluye κS, M(t), h_j ni filas de μ₀ del panel: no forman parte del exponente escrito en esa ecuación.
    """
    _, _lab, bkey, _hkey, _ = _cal_focus_row(j_mech)
    lj = "Liberación" if bkey == "Liberación" else bkey
    beta_val = float(betas_dict[theta_k][bkey])
    lam0 = float(lambdas_dict[lj])
    ind_alta = 1 if "Alta" in str(f_capa) else 0
    ind_pub = 1 if str(v_victim) == "Público" else 0
    ind_laxo = 1 if str(s_tipo) == "Laxo" else 0
    # Coeficientes transversales (invariantes al tipo theta_K en la Tabla 1)
    # Se toma el valor de FARC como referencia común para garantizar "los demás iguales"
    eta_z = float(COEF_ETA.get(z_region, {}).get("FARC", 0.0))
    xi_v  = float(COEF_XI.get(v_victim, {}).get("FARC", 0.0))
    
    # Coeficientes estructurales fijos (main.tex)
    B_F = 0.62  # ln(1.868) - Capacidad de pago (Middle vs Upper)
    B_S = 0.33  # ln(1.390) - Intervención estatal / Laxitud
    
    # Instrumentos (zeta, phi): específicos del tipo θ_K activo en Tabla 1.
    _zp_ref = zeta_phi if isinstance(zeta_phi, dict) else _focus_cmh_endogenous_tentatives(theta_k)
    za = float(_zp_ref.get("zeta_alpha", 0.1))
    zg = float(_zp_ref.get("zeta_gamma", 0.1))
    zd = float(_zp_ref.get("zeta_d", 0.1))
    ph_f = float(_zp_ref.get("phi_F", 0.1))
    ph_kc = float(_zp_ref.get("phi_K_cont", 0.1))
    ph_kk = float(_zp_ref.get("phi_K_kill", 0.1))
    z_R = float(_zp_ref.get("zeta_R", 0.1))
    
    _PRI = "Prior"
    _OBS = "Observado"
    CAL = "Calibrado"
    def _rw(
        n: int,
        term: str,
        coef: str,
        val: object,
        proc: str,
        *,
        val_katex=None,
        origen_tipo=None,
        clase_tab7=None,
    ) -> dict:
        d = {"#": n, "Término": term, "Coeficiente": proc, "Valor": val, "Origen del valor": coef}
        if val_katex is not None:
            d["Valor_KaTeX"] = val_katex
        if origen_tipo is not None:
            d["Origen_cal_tip"] = origen_tipo
        d["Clase_tab7"] = (clase_tab7 or "").strip()
        return d

    rows: list = []
    o = 0

    if j_mech == 1:
        o += 1
        rows.append(
            _rw(
                o,
                "Riesgo basal (pago)",
                r"\lambda_{10}(t)",
                lam0,
                r"\lambda_{10}",
                clase_tab7=_PRI,
            )
        )
        o += 1
        rows.append(
            _rw(
                o,
                "Capacidad de pago alta (θ_F)",
                "Prior (Calibrado)",
                float(ind_alta) * B_F,
                r"\beta_{\mathrm{F}}",
                origen_tipo=CAL,
                clase_tab7=_PRI,
            )
        )
        o += 1
        rows.append(
            _rw(
                o,
                "Víctima perfil público (θ_V)",
                "Prior (Calibrado)",
                xi_v,
                r"\beta_{\mathrm{V}}",
                origen_tipo=CAL,
                clase_tab7=_PRI,
            )
        )
        o += 1
        rows.append(
            _rw(
                o,
                "Estado laxo (θ_S)",
                "Prior (Calibrado)",
                float(ind_laxo) * B_S,
                r"\beta_{S,1}",
                origen_tipo=CAL,
                clase_tab7=_PRI,
            )
        )
        o += 1
        rows.append(
            _rw(
                o,
                "Tipo secuestrador (θ_K)",
                "Prior (Calibrado)",
                beta_val,
                r"\beta_{K,1}",
                origen_tipo=CAL,
                clase_tab7=_PRI,
            )
        )
        o += 1
        rows.append(
            _rw(
                o,
                "Heterogeneidad geográfica",
                "Prior (Calibrado)",
                eta_z,
                r"\beta_{z,1}",
                origen_tipo=CAL,
                clase_tab7=_PRI,
            )
        )
        o += 1
        rows.append(
            _rw(
                o,
                "Instrumento α (bloqueo)",
                "Prior (Calibrado)",
                za,
                r"\zeta_{\alpha,1}",
                origen_tipo=CAL,
                clase_tab7=_PRI,
            )
        )
        o += 1
        rows.append(
            _rw(
                o,
                "Instrumento γ (presión)",
                "Prior (Calibrado)",
                zg,
                r"\zeta_{\gamma,1}",
                origen_tipo=CAL,
                clase_tab7=_PRI,
            )
        )
        o += 1
        rows.append(
            _rw(
                o,
                "Detección p_det",
                "Prior (Calibrado)",
                zd,
                r"\zeta_{d,1}",
                origen_tipo=CAL,
                clase_tab7=_PRI,
            )
        )
        o += 1
        rows.append(
            _rw(
                o,
                "Familia paga (MDG)",
                "Prior (Calibrado)",
                ph_f,
                r"\varphi_{\mathrm{F},1}",
                origen_tipo=CAL,
                clase_tab7=_PRI,
            )
        )
        o += 1
        rows.append(
            _rw(
                o,
                "K continúa (MDG)",
                "Prior (Calibrado)",
                ph_kc,
                r"\varphi_{\mathrm{K},1}",
                origen_tipo=CAL,
                clase_tab7=_PRI,
            )
        )
    elif j_mech == 2:
        o += 1
        rows.append(
            _rw(
                o,
                "Riesgo basal (muerte)",
                r"\lambda_{20}(t)",
                lam0,
                r"\lambda_{20}",
                clase_tab7=_PRI,
            )
        )
        o += 1
        rows.append(
            _rw(
                o,
                "Estado laxo (θ_S)",
                "Prior (Calibrado)",
                float(ind_laxo) * B_S,
                r"\beta_{S,2}",
                origen_tipo=CAL,
                clase_tab7=_PRI,
            )
        )
        o += 1
        rows.append(
            _rw(
                o,
                "Tipo secuestrador (θ_K)",
                "Prior (Calibrado)",
                beta_val,
                r"\beta_{K,2}",
                origen_tipo=CAL,
                clase_tab7=_PRI,
            )
        )
        o += 1
        rows.append(
            _rw(
                o,
                "Heterogeneidad geográfica",
                "Prior (Calibrado)",
                eta_z,
                r"\beta_{z,2}",
                origen_tipo=CAL,
                clase_tab7=_PRI,
            )
        )
        o += 1
        rows.append(
            _rw(
                o,
                "Instrumento α",
                "Prior (Calibrado)",
                za,
                r"\zeta_{\alpha,2}",
                origen_tipo=CAL,
                clase_tab7=_PRI,
            )
        )
        o += 1
        rows.append(
            _rw(
                o,
                "Instrumento γ",
                "Prior (Calibrado)",
                zg,
                r"\zeta_{\gamma,2}",
                origen_tipo=CAL,
                clase_tab7=_PRI,
            )
        )
        o += 1
        rows.append(
            _rw(
                o,
                "Familia paga (MDG)",
                "Prior (Calibrado)",
                ph_f,
                r"\varphi_{\mathrm{F},2}",
                origen_tipo=CAL,
                clase_tab7=_PRI,
            )
        )
        o += 1
        rows.append(
            _rw(
                o,
                "Detección p_det",
                "Prior (Calibrado)",
                zd,
                r"\zeta_{d,2}",
                origen_tipo=CAL,
                clase_tab7=_PRI,
            )
        )
        o += 1
        rows.append(
            _rw(
                o,
                "K mata (MDG)",
                "Prior (Calibrado)",
                ph_kk,
                r"\varphi_{\mathrm{K},2}^{\mathrm{kill}}",
                origen_tipo=CAL,
                clase_tab7=_PRI,
            )
        )
        o += 1
        rows.append(
            _rw(
                o,
                "K continúa (MDG)",
                "Prior (Calibrado)",
                ph_kc,
                r"\varphi_{\mathrm{K},2}^{\mathrm{cont}}",
                origen_tipo=CAL,
                clase_tab7=_PRI,
            )
        )
    else:
        o += 1
        rows.append(
            _rw(
                o,
                "Riesgo basal (rescate)",
                r"\lambda_{30}(t)",
                lam0,
                r"\lambda_{30}",
                clase_tab7=_PRI,
            )
        )
        o += 1
        rows.append(
            _rw(
                o,
                "Estado laxo (θ_S)",
                "Prior (Calibrado)",
                float(ind_laxo) * B_S,
                r"\beta_{S,3}",
                origen_tipo=CAL,
                clase_tab7=_PRI,
            )
        )
        o += 1
        rows.append(
            _rw(
                o,
                "Tipo secuestrador (θ_K)",
                "Prior (Calibrado)",
                beta_val,
                r"\beta_{K,3}",
                origen_tipo=CAL,
                clase_tab7=_PRI,
            )
        )
        o += 1
        rows.append(
            _rw(
                o,
                "Heterogeneidad geográfica",
                "Prior (Calibrado)",
                eta_z,
                r"\beta_{z,3}",
                origen_tipo=CAL,
                clase_tab7=_PRI,
            )
        )
        o += 1
        rows.append(
            _rw(
                o,
                "Instrumento α",
                "Prior (Calibrado)",
                za,
                r"\zeta_{\alpha,3}",
                origen_tipo=CAL,
                clase_tab7=_PRI,
            )
        )
        o += 1
        rows.append(
            _rw(
                o,
                "Instrumento γ",
                "Prior (Calibrado)",
                zg,
                r"\zeta_{\gamma,3}",
                origen_tipo=CAL,
                clase_tab7=_PRI,
            )
        )
        o += 1
        rows.append(
            _rw(
                o,
                "Detección p_det",
                "Prior (Calibrado)",
                zd,
                r"\zeta_{d,3}",
                origen_tipo=CAL,
                clase_tab7=_PRI,
            )
        )
        o += 1
        rows.append(
            _rw(
                o,
                "Estado rescata (MDG)",
                "Prior (Calibrado)",
                z_R,
                r"\zeta_{\mathrm{R}}",
                origen_tipo=CAL,
                clase_tab7=_PRI,
            )
        )
        o += 1
        rows.append(
            _rw(
                o,
                "Familia paga (MDG)",
                "Prior (Calibrado)",
                ph_f,
                r"\varphi_{\mathrm{F},3}",
                origen_tipo=CAL,
                clase_tab7=_PRI,
            )
        )
        o += 1
        rows.append(
            _rw(
                o,
                "K continúa (MDG)",
                "Prior (Calibrado)",
                ph_kc,
                r"\varphi_{\mathrm{K},3}",
                origen_tipo=CAL,
                clase_tab7=_PRI,
            )
        )

    return pd.DataFrame(rows)


def _build_cal_voz_tabla5_df(vp: dict) -> pd.DataFrame:
    """Tabla 5: parámetros de medición de voz (x̄, σ_L, σ_S) para los 4 rasgos acústicos."""
    rasgo_labels = [
        ("Mean pitch f₀", r"\bar{x}_1", r"\sigma_{L,1}", r"\sigma_{S,1}"),
        ("Pitch variance", r"\bar{x}_2", r"\sigma_{L,2}", r"\sigma_{S,2}"),
        ("Pause rate", r"\bar{x}_3", r"\sigma_{L,3}", r"\sigma_{S,3}"),
        ("Formality / aggression", r"\bar{x}_4", r"\sigma_{L,4}", r"\sigma_{S,4}"),
    ]
    rows = []
    n = 0
    origen_voz = "Voice calibration (session)"
    for i, (rasgo, tx, tsl, tss) in enumerate(rasgo_labels):
        n += 1
        rows.append({"#": n, "Término": f"{rasgo} — mean", "Coeficiente": tx,
                     "Valor": round(float(vp["x"][i]), 2), "Origen del valor": origen_voz,
                     "Valor_KaTeX": tx, "Clase_tab7": "Observed"})
        n += 1
        rows.append({"#": n, "Término": f"{rasgo} — long-term std.", "Coeficiente": tsl,
                     "Valor": round(float(vp["sigma_L"][i]), 2), "Origen del valor": origen_voz,
                     "Valor_KaTeX": tsl, "Clase_tab7": "Observed"})
        n += 1
        rows.append({"#": n, "Término": f"{rasgo} — short-term std.", "Coeficiente": tss,
                     "Valor": round(float(vp["sigma_S"][i]), 2), "Origen del valor": origen_voz,
                     "Valor_KaTeX": tss, "Clase_tab7": "Observed"})
    return pd.DataFrame(rows)


def _render_focus_covariate_katex_table(
    df: pd.DataFrame,
    *,
    show_origen: bool = True,
    compact_iframe_bottom: bool = False,
    font_boost_pt: float = 0.0,
    term_font_boost_pt: float = 0.0,
    term_line_height: Optional[float] = None,
    col_width_css_override: Optional[str] = None,
    iframe_slack_px: Optional[int] = None,
    collapse_gap_below: bool = False,
) -> None:
    """KaTeX (CDN): Tabla 1 puede ocultar «Origen del valor» y usar ``font_boost_pt`` (p. ej. +2 pt).

    ``iframe_slack_px`` (si no es ``None``) sustituye el margen vertical extra bajo la tabla dentro del iframe
    (útil para pegar el botón «Editar valores» al KaTeX sin franja blanca).

    ``collapse_gap_below=True`` inyecta JS que mide y elimina el hueco visual entre este iframe
    y el siguiente elemento Streamlit (p. ej. un botón popover).
    """
    if df is None or df.empty:
        return
    _fb = max(0.0, float(font_boost_pt))
    _ptx = f"{_fb:g}pt"
    _tfb = float(term_font_boost_pt)
    _tptx = f"{_tfb:g}pt"
    _tlh = float(term_line_height) if term_line_height is not None else None
    _tlh_css = f"{_tlh:g}" if _tlh is not None else ""
    rows_out = []
    has_vk = "Valor_KaTeX" in df.columns
    has_clase = "Clase_tab7" in df.columns
    has_orig_col = "Origen del valor" in df.columns
    show_orig_col = bool(show_origen and has_orig_col)
    for _, row in df.iterrows():
        val_tex = None
        if has_vk:
            v = row.get("Valor_KaTeX")
            if v is not None and pd.notna(v):
                val_tex = str(v)
        try:
            _vn = float(row["Valor"])
            if str(row.get("Coeficiente", "")).strip() == "t" and abs(_vn - round(_vn)) < 1e-6:
                _val_es = _fmt_es_num(float(int(round(_vn))), 0)
            else:
                _val_es = _fmt_es_num_sigfirst(_vn)
            val_tex = None
        except (TypeError, ValueError):
            _val_es = str(row["Valor"])
        _coef_raw = str(row.get("Coeficiente", ""))
        # Tabla 1 (y similares): «Coeficiente» = LaTeX corto del parámetro;
        # si no empieza por \\, convertir con _focus_term_to_latex. Tabla 5: pasar tal cual.
        if show_orig_col:
            _cstrip = _coef_raw.strip()
            if _cstrip.startswith("\\"):
                _coef_for_cell = _translate_latex_expression(_coef_raw)
                _coef_is_latex = True
            else:
                _coef_for_cell = _focus_term_to_latex(_translate_text_to_english(_coef_raw))
                _coef_is_latex = True
        else:
            _coef_for_cell = _translate_latex_expression(_coef_raw)
            _coef_is_latex = True
        _term_display = _translate_text_to_english(str(row["Término"]))
        rd = {
            "n": int(row["#"]),
            "term": _term_display,
            "term_tex": _translate_latex_expression(_focus_term_to_latex(str(row["Término"]))),
            "coef_is_latex": _coef_is_latex,
            "coef_display": _coef_for_cell,
            "val": _val_es,
            "val_tex": _translate_latex_expression(val_tex) if val_tex is not None else val_tex,
        }
        if show_orig_col:
            rd["orig_tex"] = _translate_text_to_english(str(row.get("Origen del valor", "")))
        if has_clase:
            rd["clase_tab7"] = _translate_text_to_english(str(row.get("Clase_tab7", "") or ""))
        rows_out.append(rd)
    try:
        payload = json.dumps(rows_out, ensure_ascii=False)
    except (TypeError, ValueError):
        _df_drop = ["Valor_KaTeX", "Origen_cal_tip", "Clase_tab7"]
        if not show_origen:
            _df_drop.append("Origen del valor")
        _fdd = df.drop(columns=_df_drop, errors="ignore")
        st.dataframe(
            _fdd,
            width="stretch",
            height=_glide_full_height_px(_st_table_row_count(_fdd)),
            hide_index=True,
        )
        return
    payload_safe = payload.replace("</", "<\\/")
    nrows = len(rows_out)
    _frag_id = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:14]
    _tb_id = f"cov_tb_{_frag_id}"
    _json_id = f"cov_j_{_frag_id}"
    _th_orig = (
        f'<th class="orig">{html.escape(_translate_text_to_english("Origen del valor"), quote=False)}</th>'
        if show_orig_col else ""
    )
    _th_clase = (
        f'<th class="prior-flag" title="{html.escape(_translate_text_to_english("Prior = coef. / riesgo calibrado; Observado = indicadores del incidente (Config.)."), quote=True)}">{html.escape(_translate_text_to_english("Clase"), quote=False)}</th>'
        if has_clase
        else ""
    )
    _js_show_orig = "true" if show_orig_col else "false"
    _js_show_clase = "true" if has_clase else "false"
    if col_width_css_override is not None:
        _col_width_css = str(col_width_css_override)
    elif show_orig_col and has_clase:
        _col_width_css = """
.cov-katex-table-root th.num,.cov-katex-table-root td.num{width:4.5%;}
.cov-katex-table-root th.term,.cov-katex-table-root td.term{width:28%;}
.cov-katex-table-root th.coef,.cov-katex-table-root td.coef{width:16%;}
.cov-katex-table-root th.val,.cov-katex-table-root td.val{width:9%;}
.cov-katex-table-root th.orig,.cov-katex-table-root td.orig{width:32%;}
.cov-katex-table-root th.prior-flag,.cov-katex-table-root td.prior-flag{width:10.5%;}
"""
    elif show_orig_col:
        _col_width_css = """
.cov-katex-table-root th.num,.cov-katex-table-root td.num{width:4.5%;}
.cov-katex-table-root th.term,.cov-katex-table-root td.term{width:28%;}
.cov-katex-table-root th.coef,.cov-katex-table-root td.coef{width:22%;}
.cov-katex-table-root th.val,.cov-katex-table-root td.val{width:9%;}
.cov-katex-table-root th.orig,.cov-katex-table-root td.orig{width:36.5%;}
"""
    elif has_clase:
        _col_width_css = """
.cov-katex-table-root th.num,.cov-katex-table-root td.num{width:5%;}
.cov-katex-table-root th.term,.cov-katex-table-root td.term{width:40%;}
.cov-katex-table-root th.coef,.cov-katex-table-root td.coef{width:24%;}
.cov-katex-table-root th.val,.cov-katex-table-root td.val{width:14%;}
.cov-katex-table-root th.prior-flag,.cov-katex-table-root td.prior-flag{width:17%;}
"""
    else:
        _col_width_css = """
.cov-katex-table-root th.num,.cov-katex-table-root td.num{width:5%;}
.cov-katex-table-root th.term,.cov-katex-table-root td.term{width:40%;}
.cov-katex-table-root th.coef,.cov-katex-table-root td.coef{width:30%;}
.cov-katex-table-root th.val,.cov-katex-table-root td.val{width:25%;}
"""
    html_fragment = f"""<div class="cov-katex-table-root">
<link rel="stylesheet" href="{_KATEX_BASE}/katex.min.css" crossorigin="anonymous"/>
<script src="{_KATEX_BASE}/katex.min.js" crossorigin="anonymous"></script>
<style>
.cov-katex-table-root{{margin:0;padding:0;background:transparent;color:inherit;max-width:100%;box-sizing:border-box;
  font-family:system-ui,-apple-system,sans-serif;font-size:calc(11.5px + {_ptx});overflow-x:hidden;line-height:1.25;}}
.cov-katex-table-root table{{width:100%;border-collapse:separate;border-spacing:0;table-layout:fixed;margin:0;}}
.cov-katex-table-root th,.cov-katex-table-root td{{border:1px solid rgba(127,127,127,.38);padding:3px 5px;line-height:1.22;vertical-align:middle;word-wrap:break-word;overflow-wrap:anywhere;}}
.cov-katex-table-root th{{background:rgba(127,127,127,.14);text-align:left;font-weight:600;font-size:calc(0.76rem + {_ptx});}}
.cov-katex-table-root th.term,.cov-katex-table-root td.term{{font-size:calc(1em + {_tptx});{f"line-height:{_tlh_css};" if _tlh_css else ""}}}
.cov-katex-table-root th.coef,.cov-katex-table-root td.coef{{hyphens:manual;-webkit-hyphens:manual;word-break:break-word;}}
.cov-katex-table-root th.num,.cov-katex-table-root td.num{{text-align:center;font-variant-numeric:tabular-nums;white-space:nowrap;}}
.cov-katex-table-root th.val,.cov-katex-table-root td.val{{text-align:right;font-variant-numeric:tabular-nums;}}
.cov-katex-table-root td.orig{{font-size:calc(0.88em + {_ptx});}}
{_col_width_css}
.cov-katex-table-root td.term .kx{{display:block;overflow:visible;white-space:normal;text-align:left;}}
.cov-katex-table-root td.term .katex{{white-space:normal;}}
.cov-katex-table-root td.coef .kx{{display:block;overflow:visible;max-width:100%;text-align:left;white-space:normal;}}
.cov-katex-table-root td.val .kx{{display:block;overflow:visible;white-space:normal;text-align:right;}}
.cov-katex-table-root td.orig .kx{{display:block;overflow:visible;max-width:100%;text-align:left;white-space:normal;}}
.cov-katex-table-root .katex{{font-size:calc(0.95em + {_ptx});}}
.cov-katex-table-root td.coef .katex{{font-size:calc(0.9em + {_ptx});}}
.cov-katex-table-root tr:nth-child(even) td{{background:rgba(127,127,127,.06);}}
.cov-katex-table-root th.prior-flag,.cov-katex-table-root td.prior-flag{{
  font-size:calc(0.74rem + {_ptx});line-height:1.22;
  text-align:center;padding:3px 4px;vertical-align:middle;font-weight:inherit;word-break:break-word;
}}
.cov-katex-table-root th.prior-flag{{font-weight:600;}}
.cov-katex-table-root td.prior-flag.clase-prior{{background:rgba(65,125,210,0.13);color:inherit;}}
.cov-katex-table-root td.prior-flag.clase-obs{{background:rgba(115,145,95,0.18);color:inherit;}}
.cov-katex-table-root td.prior-flag.clase-endog{{background:rgba(195,115,45,0.16);color:inherit;font-weight:inherit;}}
.cov-katex-table-root td.prior-flag.clase-noprior{{opacity:1;font-weight:inherit;}}
</style>
<table><thead><tr>
<th class="num">#</th>
<th class="term">{html.escape(_translate_text_to_english("Término"), quote=False)}</th>
<th class="coef">{html.escape(_translate_text_to_english("Coeficiente"), quote=False)}</th>
<th class="val">{html.escape(_translate_text_to_english("Valor"), quote=False)}</th>{_th_orig}{_th_clase}
</tr></thead><tbody id="{_tb_id}"></tbody></table>
<script id="{_json_id}" type="application/json">{payload_safe}</script>
<script>
(function() {{
  /* Convierte LaTeX básico a Unicode aproximado cuando KaTeX no está disponible. */
  function _latexApprox(s) {{
    if (!s) return s;
    var greek = {{alpha:'α',beta:'β',gamma:'γ',delta:'δ',epsilon:'ε',zeta:'ζ',eta:'η',
      theta:'θ',iota:'ι',kappa:'κ',lambda:'λ',mu:'μ',nu:'ν',xi:'ξ',pi:'π',
      rho:'ρ',sigma:'σ',tau:'τ',upsilon:'υ',phi:'φ',chi:'χ',psi:'ψ',omega:'ω',
      varphi:'φ',varepsilon:'ε',vartheta:'ϑ',cdot:'·',ldots:'…'}};
    var subD={{'0':'₀','1':'₁','2':'₂','3':'₃','4':'₄','5':'₅','6':'₆','7':'₇','8':'₈','9':'₉'}};
    var r = s;
    r = r.replace(/\\\\(?:mathrm|mathit|mathbf|mathnormal|text|textrm|textit|textbf|hat|tilde|bar|vec)\\{{([^}}]*)\\}}/g, '$1');
    r = r.replace(/\\\\([a-zA-Z]+)/g, function(m,c){{ return greek[c]||''; }});
    r = r.replace(/\\\\/g, '');
    r = r.replace(/\^\\{{([^}}]*)\\}}/g, function(m,c){{ return c; }});
    r = r.replace(/_\\{{([^}}]*)\\}}/g, function(m,c){{
      return c.replace(/[0-9]/g, function(d){{ return subD[d]||d; }});
    }});
    r = r.replace(/\\{{([^}}]*)\\}}/g,'$1');
    r = r.replace(/_([0-9]+)/g, function(m,d){{
      return d.split('').map(function(c){{return subD[c]||c;}}).join('');
    }});
    r = r.replace(/_([A-Za-z])/g,'$1');
    r = r.replace(/[\{{\\}}]/g,'');
    return r;
  }}
  const SHOW_ORIGEN = {_js_show_orig};
  const SHOW_CLASE = {_js_show_clase};
  const data = JSON.parse(document.getElementById('{_json_id}').textContent);
  const tb = document.getElementById('{_tb_id}');
  const kopts = {{ displayMode: false, throwOnError: false, strict: false }};
  const hasKatex = (typeof katex !== 'undefined');
  for (let i = 0; i < data.length; i++) {{
    const r = data[i];
    const tr = document.createElement('tr');
    const tdN = document.createElement('td'); tdN.className = 'num'; tdN.textContent = String(r.n); tr.appendChild(tdN);
    const tdT = document.createElement('td'); tdT.className = 'term';
    const spT = document.createElement('span'); spT.className = 'kx'; tdT.appendChild(spT); tr.appendChild(tdT);
    const tdC = document.createElement('td'); tdC.className = 'coef';
    const spC = document.createElement('span'); spC.className = 'kx'; tdC.appendChild(spC); tr.appendChild(tdC);
    const tdV = document.createElement('td'); tdV.className = 'val';
    const spV = document.createElement('span'); spV.className = 'kx'; tdV.appendChild(spV); tr.appendChild(tdV);
    let spO = null;
    if (SHOW_ORIGEN) {{
      const tdO = document.createElement('td'); tdO.className = 'orig';
      spO = document.createElement('span'); spO.className = 'kx'; tdO.appendChild(spO); tr.appendChild(tdO);
    }}
    if (SHOW_CLASE) {{
      const tdCl = document.createElement('td');
      tdCl.className = 'prior-flag';
      const lab = String(r.clase_tab7 || '');
      tdCl.textContent = lab;
      if (lab === 'Endógena' || lab === 'Endogenous') tdCl.classList.add('clase-endog');
      else if (lab === 'Prior') tdCl.classList.add('clase-prior');
      else if (lab === 'Observado' || lab === 'Observed') tdCl.classList.add('clase-obs');
      else tdCl.classList.add('clase-noprior');
      tr.appendChild(tdCl);
    }}
    tb.appendChild(tr);
    if (hasKatex) {{
      try {{ katex.render(r.term_tex, spT, kopts); }} catch (e) {{ spT.textContent = r.term; }}
      if (r.coef_is_latex) {{
        try {{ katex.render(r.coef_display, spC, kopts); }} catch (e) {{ spC.textContent = _latexApprox(r.coef_display); }}
      }} else {{
        spC.textContent = r.coef_display;
      }}
      if (SHOW_ORIGEN && spO) {{
        try {{ katex.render(r.orig_tex, spO, kopts); }} catch (e) {{ spO.textContent = r.orig_tex; }}
      }}
      if (r.val_tex) {{
        try {{ katex.render(r.val_tex, spV, kopts); }} catch (e) {{ spV.textContent = r.val; }}
      }} else {{
        spV.textContent = r.val;
      }}
    }} else {{
      spT.textContent = r.term;
      spC.textContent = r.coef_is_latex ? _latexApprox(r.coef_display) : r.coef_display;
      if (spO) spO.textContent = r.orig_tex || '';
      spV.textContent = r.val;
    }}
  }}
}})();
</script>
</div>"""
    if compact_iframe_bottom:
        _hdr_px = 34
        _slack_default = 16
        _body_pad = "0"
    else:
        _hdr_px = 36
        _slack_default = 30
        _body_pad = "0 0 6px 0"
    if iframe_slack_px is not None:
        _slack_px = int(max(0, int(iframe_slack_px)))
    else:
        _slack_px = _slack_default
    # Tabla 1 (compact + slack 0): menos padding vertical en el cómputo de altura → menos “aire” bajo la última fila.
    _pad_px = (
        2
        if compact_iframe_bottom
        and iframe_slack_px is not None
        and int(iframe_slack_px) == 0
        else 6
    )
    _rh_add = int(max(2, round(_fb * 1.35))) if _fb > 0 else 0
    if nrows >= 10:
        _row_px = 28.0 + _rh_add
    elif nrows > 4:
        _row_px = (32.5 if compact_iframe_bottom else 34.5) + _rh_add
    else:
        _row_px = 35.0 + _rh_add
    
    # Si hay font boost, el wrapping es más probable; añadimos un extra proporcional
    if _fb > 1.0:
        _row_px += 3.0
    _need_h = _pad_px + _hdr_px + max(0, nrows) * _row_px + _slack_px
    _iframe_h = int(min(9800, max(64, _need_h)))
    _iframe_scroll = False
    _gap_js = ""
    if collapse_gap_below:
        _gap_js = (
            "<script>(function(){"
            "function _cg(){"
            "try{"
            "var me=window.frameElement;"
            "if(!me)return;"
            "var c=me.parentElement;"
            "for(var i=0;i<10&&c;i++){"
            "var t=c.getAttribute?c.getAttribute('data-testid'):'';"
            "if(t==='stElementContainer'||t==='element-container')break;"
            "c=c.parentElement;"
            "}"
            "if(!c)return;"
            "var nx=c.nextElementSibling;"
            "if(!nx)return;"
            "var gap=Math.round(nx.getBoundingClientRect().top-c.getBoundingClientRect().bottom);"
            "if(gap>3){nx.style.setProperty('margin-top',(-gap)+'px','important');}"
            "}catch(e){}}"
            "setTimeout(_cg,0);setTimeout(_cg,250);"
            "})();</script>"
        )
    _iframe_doc = (
        '<!DOCTYPE html><html><head><meta charset="utf-8"></head>'
        f'<body style="margin:0;padding:{_body_pad}">'
        + html_fragment
        + _gap_js
        + "</body></html>"
    )
    try:
        components.html(
            _iframe_doc,
            height=_iframe_h,
            width=None,
            scrolling=_iframe_scroll,
        )
    except Exception:
        _df_drop2 = ["Valor_KaTeX", "Origen_cal_tip", "Clase_tab7"]
        if not show_origen:
            _df_drop2.append("Origen del valor")
        _fdd2 = df.drop(columns=_df_drop2, errors="ignore")
        st.dataframe(
            _fdd2,
            width="stretch",
            height=_glide_full_height_px(_st_table_row_count(_fdd2)),
            hide_index=True,
        )


def _build_kidnapper_panel_calibrated_util_df(
    row: pd.Series,
    *,
    alpha: float,
    gamma: float,
    R: float,
    beta: float,
) -> pd.DataFrame:
    """Parámetros calibrados del tipo panel para $U^K_{\mathrm{rel}}$, $U^K_{\mathrm{kill}}$, $V^K_{\mathrm{cont},t}$."""
    pc = float(row["p_cap_tilde"])
    _phi = float(row["phi"])
    _kc = float(row["kappa_c"])
    _nu = float(row["nu"])
    _C_t0 = kidnapper_cost_c(float(gamma), _phi, _kc, _nu)
    return pd.DataFrame(
        [
            {
                "#": 1,
                "Término": "Desutilidad liberación",
                "Coeficiente": r"\kappa_{\mathrm{rel}}(\theta_K)",
                "Valor": round(float(row["kappa_rel"]), 3),
            },
            {
                "#": 2,
                "Término": "Beneficio reputacional (kill)",
                "Coeficiente": r"\eta(\theta_K)",
                "Valor": round(float(row["eta"]), 3),
            },
            {
                "#": 3,
                "Término": "Penalidad captura",
                "Coeficiente": r"F_{\mathrm{cap}}(\theta_K,\theta_S)",
                "Valor": round(float(row["F_cap"]), 3),
            },
            {
                "#": 4,
                "Término": "Prob. captura efectiva",
                "Coeficiente": r"\tilde{p}_{\mathrm{cap},t}(\theta_K)",
                "Valor": round(pc, 4),
            },
            {
                "#": 5,
                "Término": "Prob. pago / liberación por pago",
                "Coeficiente": r"\tilde{p}_{\mathrm{pay},t}(\theta_K)",
                "Valor": round(float(row["h_LibPago"]), 4),
            },
            {
                "#": 6,
                "Término": "Escala costo cautiverio",
                "Coeficiente": r"\phi(\theta_K)",
                "Valor": round(float(row["phi"]), 3),
            },
            {
                "#": 7,
                "Término": r"Sensibilidad presión $\gamma$",
                "Coeficiente": r"\kappa_c(\theta_K)",
                "Valor": round(float(row["kappa_c"]), 3),
            },
            {
                "#": 8,
                "Término": "Costo fijo cautiverio",
                "Coeficiente": r"\nu(\theta_K)",
                "Valor": round(float(row["nu"]), 3),
            },
            {
                "#": 9,
                "Término": "Costo operativo (evaluado en $t=0$)",
                "Coeficiente": r"C_0(\gamma_0,\theta_K)",
                "Valor": round(_C_t0, 3),
            },
            {
                "#": 10,
                "Término": "Bloqueo financiero (inicial)",
                "Coeficiente": r"\alpha_0",
                "Valor": round(float(alpha), 3),
            },
            {
                "#": 11,
                "Término": "Presión operativa (inicial)",
                "Coeficiente": r"\gamma_0",
                "Valor": round(float(gamma), 3),
            },
            {
                "#": 12,
                "Término": "Escala rescate",
                "Coeficiente": r"R",
                "Valor": round(float(R), 2),
            },
            {
                "#": 13,
                "Término": "Factor descuento",
                "Coeficiente": r"\beta(\theta_K)",
                "Valor": round(float(beta), 3),
            },
        ]
    )


def _render_mechanism_lh_physical_equations() -> None:
    """Ecuaciones del bloque físico (Mechanism.tex: LH-compacta, LH-cont, LH-out)."""
    st.markdown("**Verosimilitud física** (código causal $c_t\\in\\{\\mathrm{Cont},1,2,3,4\\}$)")
    st.latex(
        r"\mathbb{P}_{\mathbb{E}}\!\bigl(c_t \mid \theta_K,h_t,\theta_F,\theta_V\bigr)"
        r"=p_{\mathrm{Cont},\,t\mid\theta_K}^{\,\mathbf{1}\{c_t=\mathrm{Cont}\}}"
        r"\prod_{j=1}^{4} h_j(t\mid\theta_K)^{\,\mathbf{1}\{c_t=j\}}"
        r"\qquad\text{(eq. LH-compacta)}"
    )
    st.latex(
        r"\mathcal{L}_H^{\mathrm{cont}}(t\mid\theta_K)=p_{\mathrm{Cont},\,t\mid\theta_K}"
        r"=\exp\!\Bigl(-\sum_{j=1}^{4}\tilde{\lambda}_j(t\mid\theta_K)\,\Delta t\Bigr)"
        r"\quad\text{si }c_t{=}\mathrm{Cont}\ \text{(eq. LH-cont)}"
    )
    st.latex(
        r"\mathcal{L}_H^{\mathrm{out}}(t,j^\ast\mid\theta_K)=h_{j^\ast}(t\mid\theta_K)"
        r"\quad\text{si }c_t{=}j^\ast\in\{1,2,3,4\}\ \text{(eq. LH-out)}"
    )


def _render_mechanism_bayes_likelihood_rest(omega_voz: float = 0.0) -> None:
    """Detección, verosimilitud conjunta, comunicación y Bayes (post bloque físico)."""
    st.markdown("**Detección** (no depende de $\\theta_K$; eq. Ld-bernoulli)")
    st.latex(
        r"\mathbb{P}_{\mathbb{E}}(d_t\mid\alpha_t^\ast,\gamma_t^\ast)"
        r"=p_{\mathrm{det},t}^{\,d_t}\bigl(1-p_{\mathrm{det},t}\bigr)^{1-d_t},"
        r"\quad p_{\mathrm{det},t}=\mathbb{P}_{\mathbb{E}}(d_t{=}1\mid\alpha_t^\ast,\gamma_t^\ast)"
    )
    st.markdown("**Verosimilitud conjunta observable** (eq. LH-joint)")
    st.latex(
        r"\mathcal{L}_{F,t}(\theta_K)"
        r"=\underbrace{\mathcal{L}_{I,t}(\theta_K)}_{\text{acciones ejecutadas}}"
        r"\cdot\underbrace{\mathbb{P}_{\mathbb{E}}(m_t\mid\theta_K,h_t,\theta_F,\theta_V)}_{\mathcal{L}_H}"
        r"\cdot\underbrace{\mathbb{P}_{\mathbb{E}}(d_t\mid\alpha_t^\ast,\gamma_t^\ast)}_{\mathcal{L}_d}"
    )
    st.latex(
        r"\mathcal{L}_{I,t}(\theta_K)=\prod_{i\in\{F,K,S\}}"
        r"\mathbb{P}_{I}^{i}(\tilde a_t^i\mid a_t^{i\ast}(\theta_K),X_t)"
    )
    st.markdown("**Comunicación** (eq. LC; Lvoz-diag si $V_t=1$)")
    st.latex(
        r"\mathcal{L}_{C,t}(\theta_K\mid V_t)=\begin{cases}"
        r"\bigl[\mathcal{L}_{\mathrm{voz},t}(\theta_K)\,\pi_{\mathrm{call}}(\theta_K)\bigr]^{\omega_{\mathrm{voz}}}"
        r"& V_t=1,\\[0.5em]"
        r"\bigl[1-\pi_{\mathrm{call}}(\theta_K)\bigr]^{\omega_{\mathrm{voz}}}"
        r"& V_t=0,"
        r"\end{cases}"
    )
    if float(omega_voz) <= 1e-12:
        st.caption(
            r"En la app: $\omega_{\mathrm{voz}}=0$ $\Rightarrow$ $\mathcal{L}_{C,t}(\theta_K\mid V_t)\equiv 1$ "
            r"por exponente cero. Si $\omega_{\mathrm{voz}}>0$ y no hay señal, se usa el bloque de silencio "
            r"$[1-\pi_{\mathrm{call}}(\theta_K)]^{\omega_{\mathrm{voz}}}$."
        )
    st.markdown("**Actualización de creencias** (eq. bayes-unif)")
    st.latex(
        r"\mu_{t+1}(\theta_K)"
        r"=\frac{\mu_t(\theta_K)\,\mathcal{L}_{F,t}(\theta_K)\,"
        r"\mathcal{L}_{C,t}(\theta_K\mid V_t)}"
        r"{\sum_{\tilde{\theta}_K\in\Theta_K}\mu_t(\tilde{\theta}_K)\,"
        r"\mathcal{L}_{F,t}(\tilde{\theta}_K)\,"
        r"\mathcal{L}_{C,t}(\tilde{\theta}_K\mid V_t)}"
    )


def _render_mechanism_bayes_likelihood_block(omega_voz: float = 0.0) -> None:
    """Ecuaciones completas: verosimilitud física + detección + Bayes."""
    _render_mechanism_lh_physical_equations()
    _render_mechanism_bayes_likelihood_rest(omega_voz)


def _render_belief_update_continuar_table(
    modelo: ModeloSecuestro,
    mu_0: dict,
    *,
    t_eval: int,
    d_obs: int,
    presion_S: float,
    z_region: str,
    v_victim: str,
    alpha: float,
    gamma: float,
    p_det: float,
    zeta_alpha: float,
    zeta_gamma: float,
    zeta_d: float,
    zeta_R: float,
    estado_rescata: bool,
    t_mad: float,
    lambda4: float,
    omega_voz: float = 0.0,
    pi_call_by_theta: Optional[dict] = None,
    voz_params_by_theta: Optional[dict] = None,
    V_t: Optional[int] = None,
    atilde_F: Optional[str] = None,
    atilde_K: Optional[str] = None,
    atilde_S: Optional[str] = None,
    implementation_likelihood_by_theta: Optional[dict[str, float]] = None,
    table_title: str = "Tabla 13 · Verosimilitud física y actualización de creencias (μ)",
    tab2_bundle_by_theta: Optional[dict] = None,
    aggregate_unknown_theta: bool = False,
    aggregate_lc_unknown_theta: bool = True,
) -> None:
    """Tabla numérica bayes-unif con $m_t=\\mathrm{Continuar}$ y parámetros de pestañas 1–3."""
    _df_raw, _mu1, _meta = build_t0_bayesian_posterior_report(
        modelo,
        mu_0,
        "Continuar",
        int(d_obs),
        presion_S=presion_S,
        z_region=z_region,
        v_victim=v_victim,
        alpha=alpha,
        gamma=gamma,
        p_det=p_det,
        zeta_alpha=zeta_alpha,
        zeta_gamma=zeta_gamma,
        zeta_d=zeta_d,
        zeta_R=zeta_R,
        estado_rescata=estado_rescata,
        t_mad=t_mad,
        lambda4=lambda4,
        t_eval=int(t_eval),
        omega_voz=omega_voz,
        pi_call_by_theta=pi_call_by_theta,
        voz_params_by_theta=voz_params_by_theta,
        V_t=V_t,
        atilde_F=atilde_F,
        atilde_K=atilde_K,
        atilde_S=atilde_S,
        implementation_likelihood_by_theta=implementation_likelihood_by_theta,
        tab2_bundle_by_theta=tab2_bundle_by_theta,
        aggregate_unknown_theta=bool(aggregate_unknown_theta),
        aggregate_lc_unknown_theta=bool(aggregate_lc_unknown_theta),
    )
    _t_ev = int(_meta.get("t_eval", t_eval))
    _d_ev = int(_meta.get("d_obs", d_obs))
    _m_ev = str(_meta.get("m_obs", "Continuar"))
    st.markdown(
        f"**{table_title}** "
        f"($t={_t_ev}$, $m_t={_m_ev}$, $d_t={_d_ev}$; "
        r"$\tilde{\lambda}_j$, $\mathcal{L}_{I,t}$, $\mathcal{L}_H$, $\mathcal{L}_d$, $\mathcal{L}_{F,t}$, $\mu_{t+1}$)"
    )
    _df_show = format_belief_update_display_df(_df_raw, include_hazards=True)
    st.dataframe(_df_show, hide_index=True, use_container_width=True)
    _chk = sum(float(_mu1.get(th, 0.0)) for th in TIPOS_SECUESTRADOR)
    st.caption(
        f"Comprobación: $\\sum_\\theta \\mu_{{t+1}}(\\theta)={_chk:.4f}$; "
        f"tipos con mayor $\\mathcal{{L}}_H^{{\\mathrm{{cont}}}}$ (menor $\\sum_j\\tilde\\lambda_j$) "
        f"gainan masa relativa."
    )


def _katex_strip_dollars(expr: str) -> str:
    s = str(expr).strip()
    if len(s) >= 2 and s[0] == "$" and s[-1] == "$":
        return s[1:-1]
    return s


def _tab12_probability_coef_tooltip_latex(coef: str) -> str:
    """Globo de Tabla 12 para coeficientes de probabilidad calculados."""
    s = str(coef)
    if r"\tilde{p}_{\mathrm{cap}" in s:
        return (
            r"\begin{aligned}"
            r"\tilde p_{\mathrm{cap},0}(\theta_K)"
            r"&=\mathbb E_{\tilde a_0^S\mid\mathcal Q_0^{Cap}}"
            r"\!\left[p_{\mathrm{cap}}(\tilde a_0^S,\theta_K,\alpha_0,\gamma_0)\right]\\"
            r"&=\sum_{a^S\in\{N,R\}}"
            r"P_I^S(a^S\mid\text{Tabla 10b})\,"
            r"p_{\mathrm{cap}}(a^S,\theta_K,\alpha_0,\gamma_0)."
            r"\end{aligned}"
        )
    if r"\tilde{p}_{\mathrm{pay}" in s:
        return (
            r"\begin{aligned}"
            r"\tilde p_{\mathrm{pay},0}(\theta_K)"
            r"&=\mathbb E_{\tilde A_0\mid\mathcal Q_0^{Cont}}"
            r"\!\left[\mathbb P_E(m_0=\mathrm{pay}\mid\tilde A_0,X'_0,\theta_K)\right]\\"
            r"&=\sum_{\tilde a^K,\tilde a^S,\tilde a^F}"
            r"P_I^K(\tilde a^K\mid Q^{Cont})P_I^S(\tilde a^S\mid\text{Tabla 10b})"
            r"P_I^F(\tilde a^F\mid\text{Tabla 10b})\\"
            r"&\quad\cdot h_{\mathrm{Pago}}(0\mid\tilde A_0,X'_0,\theta_K),"
            r"\quad h_{\mathrm{Pago}}\text{ viene de hazards competitivos de Tabla 10c.}"
            r"\end{aligned}"
        )
    return ""


def render_kidnapper_calibrated_params_katex(df_cal: pd.DataFrame) -> None:
    """Tabla 12: parámetros calibrados del secuestrador (KaTeX: coeficiente y valor)."""
    if df_cal is None or df_cal.empty:
        return
    rows_out: list[dict[str, str]] = []
    for _, row in df_cal.iterrows():
        coef = _katex_strip_dollars(str(row.get("Coeficiente", "")))
        term = _katex_table_term_html(str(row.get("Término", "")))
        try:
            v_f = float(row.get("Valor", 0.0))
            v_str = f"{v_f:.4f}".rstrip("0").rstrip(".")
        except (TypeError, ValueError):
            v_str = html.escape(str(row.get("Valor", "")))
        coef_tip = _tab12_probability_coef_tooltip_latex(coef)
        coef_html = f'<span class="math">{html.escape(coef, quote=False)}</span>'
        if coef_tip:
            coef_html = (
                f'<span class="math t11-cell-tip" '
                f'data-latex="{html.escape(coef_tip, quote=True)}">'
                f'{html.escape(coef, quote=False)}</span>'
            )
        rows_out.append(
            {
                "c0": html.escape(str(int(row.get("#", 0)))),
                "c1": term,
                "c2": coef_html,
                "c3": v_str,
            }
        )
    _n = len(rows_out)
    _h = max(280, 48 + 28 * (_n + 1))
    render_generic_katex_table(
        pd.DataFrame(rows_out),
        [
            r"\#",
            r"\text{Término}",
            r"\text{Coeficiente}",
            r"\text{Valor}",
        ],
        height=_h,
        compact=True,
        tight_spacing=True,
        header_font_boost_pt=1.0,
        header_tooltips=[
            "Índice.",
            "Descripción del parámetro (Tabla 9 / calibración).",
            "Símbolo en Mechanism.tex (kidnapper-kill, kidnapper-cont).",
            "Valor numérico para el tipo panel en la sesión.",
        ],
    )


def _render_kidnapper_params_katex_table(df: pd.DataFrame) -> None:
    """Renderizado estático premium para la Tabla 9 (Parámetros por tipo)."""
    if df is None or df.empty:
        return

    # Encabezados KaTeX exactos
    headers = [
        r"\theta_K", r"\kappa_{\mathrm{rel}}", r"\eta", r"F_{\mathrm{cap}}",
        r"\phi", r"\kappa_c", r"\nu", r"\tilde{p}_{\mathrm{cap}}",
        r"\tilde{p}_{\mathrm{pay}}", r"C(\gamma,\theta)"
    ]

    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8"/>
<link rel="stylesheet" href="{_KATEX_BASE}/katex.min.css" crossorigin="anonymous"/>
<script src="{_KATEX_BASE}/katex.min.js" crossorigin="anonymous"></script>
<style>
    html,body{{margin:0;padding:0;overflow:hidden;background:transparent;font-family:sans-serif;}}
    table {{ width: 100%; border-collapse: collapse; margin-top: 5px; font-size: 0.85rem; }}
    th, td {{ padding: 6px 4px; text-align: center; border-bottom: 1px solid #ddd; }}
    th {{ background-color: #f8f9fa; color: #555; font-weight: normal; }}
    .tipo {{ font-weight: bold; text-align: left; width: 10%; }}
</style>
</head><body>
<table id="t9">
    <thead><tr>"""
    for h in headers:
        html += f"<th><span class='math'>{h}</span></th>"
    html += "</tr></thead><tbody>"

    for _, row in df.iterrows():
        html += "<tr>"
        html += f"<td class='tipo'>{row['theta_K']}</td>"
        html += f"<td>{row['kappa_rel']:.2f}</td>"
        html += f"<td>{row['eta']:.2f}</td>"
        html += f"<td>{row['F_cap']:.2f}</td>"
        html += f"<td>{row['phi']:.2f}</td>"
        html += f"<td>{row['kappa_c']:.2f}</td>"
        html += f"<td>{row['nu']:.2f}</td>"
        html += f"<td>{row['p_cap_tilde']:.2f}</td>"
        html += f"<td>{row['h_LibPago']:.2f}</td>"
        html += f"<td>{row['C_gamma_theta']:.2f}</td>"
        html += "</tr>"

    html += """</tbody></table>
<script>
    document.querySelectorAll('.math').forEach(el => {
        try { katex.render(el.textContent, el, { displayMode: false, throwOnError: false }); }
        catch (e) { }
    });
</script>
</body></html>"""
    components.html(html, height=180)


def _render_compact_katex_expr(expr: str, height: int = 34) -> None:
    """Ecuación KaTeX compacta para encabezados de tablas."""
    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8"/>
<link rel="stylesheet" href="{_KATEX_BASE}/katex.min.css" crossorigin="anonymous"/>
<script src="{_KATEX_BASE}/katex.min.js" crossorigin="anonymous"></script>
<style>
    html,body{{margin:0;padding:0;overflow:hidden;background:transparent;font-family:sans-serif;}}
    #eq{{font-size:0.92rem;line-height:1.15;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}}
</style>
</head><body><div id="eq">{_katex_label_html(expr, "eq")}</div>
</body></html>"""
    components.html(html, height=height)


def _render_widget_katex_label(expr: str, caption: str = "", height: int = 36) -> None:
    """Etiqueta KaTeX compacta para controles Streamlit (símbolo + texto opcional)."""
    cap_html = (
        f'<span class="cap" style="margin-left:0.35rem;font-size:0.78rem;color:#555;">'
        f"{html.escape(str(caption), quote=False)}</span>"
        if caption
        else ""
    )
    html_doc = f"""<!DOCTYPE html><html><head><meta charset="utf-8"/>
<link rel="stylesheet" href="{_KATEX_BASE}/katex.min.css" crossorigin="anonymous"/>
<script src="{_KATEX_BASE}/katex.min.js" crossorigin="anonymous"></script>
<style>
    html,body{{margin:0;padding:0 0 6px 0;overflow:hidden;background:transparent;font-family:sans-serif;}}
    #wrap{{font-size:0.84rem;line-height:1.2;color:#31333f;display:flex;align-items:baseline;flex-wrap:wrap;gap:0.25rem;}}
</style>
</head><body><div id="wrap">{_katex_label_html(expr, "katex-lbl")}{cap_html}</div>
</body></html>"""
    components.html(html_doc, height=height)


def _render_h0_source_note(height: int = 28) -> None:
    """Nota compacta para Tabla 9a: texto HTML + m_0 renderizado en KaTeX."""
    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8"/>
<link rel="stylesheet" href="{_KATEX_BASE}/katex.min.css" crossorigin="anonymous"/>
<script src="{_KATEX_BASE}/katex.min.js" crossorigin="anonymous"></script>
<style>
    html,body{{margin:0;padding:0;overflow:hidden;background:transparent;font-family:sans-serif;}}
    #note{{font-size:0.92rem;line-height:1.2;color:#31333f;white-space:nowrap;}}
</style>
</head><body>
<div id="note">Las acciones y <span id="m0"></span> se toman de Pestaña 3 &middot; Materialización.</div>
<script>
    try {{ katex.render("m_0", document.getElementById('m0'), {{ displayMode: false, throwOnError: false }}); }}
    catch (e) {{ document.getElementById('m0').textContent = "m0"; }}
</script>
</body></html>"""
    components.html(html, height=height)


def render_generic_katex_table(
    df,
    katex_headers,
    height=120,
    compact=False,
    header_tooltips=None,
    header_tooltip_latex=None,
    header_font_boost_pt=0.0,
    tight_spacing=False,
    header_boost_by_index=None,
    relaxed_compact=False,
    header_nowrap=False,
    body_max_height_px: Optional[int] = None,
    header_tooltips_open_up: bool = False,
    header_tooltip_font_delta_pt: float = 0.0,
    header_tooltip_top_space_px: int = 0,
    row_tooltips=None,
):
    """Renderizado estático con estilo Tabla 9 para tablas genéricas.

    Con ``compact=True``, ``relaxed_compact=True`` y ``tight_spacing=False`` aumenta padding,
    ``max-width`` de encabezados y altura mínima (p. ej. Tabla 10). Si ``tight_spacing=True``,
    ``relaxed_compact`` no aplica.

    ``header_nowrap``: encabezados ``th`` en una sola línea (sin ``max-width``; desplazamiento
    horizontal si la tabla es ancha). Útil en Tabla 10 con fórmulas largas.
    """
    if df is None or df.empty:
        return
    df = _translate_display_value(df)
    katex_headers = [_translate_latex_expression(str(h)) for h in list(katex_headers)]

    n_h = len(katex_headers)
    n_r = int(len(df))
    # Altura mínima según columnas/filas (encabezados KaTeX + celdas; evita recorte).
    if compact:
        if tight_spacing:
            _auto_h = max(70, 30 + 20 * (n_r + 1) + min(30, max(0, n_h - 5) * 4))
            _body_pad = "0"
            _table_font = "0.74rem"
            _cell_pad = "2px 3px"
            _head_lh = "1.02"
            _head_max = "7.4rem"
            _td_lh = "1.05"
        elif relaxed_compact:
            _auto_h = max(
                92,
                46 + 30 * (n_r + 1) + min(56, max(0, n_h - 5) * 8),
            )
            _body_pad = "4px 10px"
            _table_font = "0.78rem"
            _cell_pad = "8px 14px"
            _head_lh = "1.24"
            _head_max = "10rem"
            _td_lh = "1.22"
        else:
            _auto_h = max(76, 38 + 25 * (n_r + 1) + min(42, max(0, n_h - 5) * 6))
            _body_pad = "0 2px"
            _table_font = "0.74rem"
            _cell_pad = "4px 4px"
            _head_lh = "1.12"
            _head_max = "7.4rem"
            _td_lh = "1.16"
    else:
        _auto_h = max(120, 56 + 36 * (n_r + 1) + min(80, max(0, n_h - 5) * 10))
        _body_pad = "4px 6px"
        _table_font = "0.82rem"
        _cell_pad = "8px 6px"
        _head_lh = "1.3"
        _head_max = "8.5rem"
        _td_lh = "1.16"
    height = int(max(height, _auto_h))
    if body_max_height_px is not None:
        height = int(min(height, int(body_max_height_px) + 12))
    _body_overflow = (
        f"max-height:{int(body_max_height_px)}px;overflow-y:auto;overflow-x:auto;"
        if body_max_height_px is not None
        else "overflow-x:auto;overflow-y:visible;"
    )
    _tips = [_translate_text_to_english(str(x)) for x in list(header_tooltips or [])]
    if len(_tips) < len(katex_headers):
        _tips.extend([""] * (len(katex_headers) - len(_tips)))
    _tips = [html.escape(str(x), quote=True) for x in _tips[:len(katex_headers)]]
    _tip_latex = [_translate_latex_expression(str(x)) for x in list(header_tooltip_latex or [])]
    if len(_tip_latex) < len(katex_headers):
        _tip_latex.extend([""] * (len(katex_headers) - len(_tip_latex)))
    _tip_latex = [html.escape(str(x), quote=True) for x in _tip_latex[:len(katex_headers)]]
    _header_boost = float(header_font_boost_pt)
    _boost_by_index = dict(header_boost_by_index or {})
    _tip_open_up_js = "true" if bool(header_tooltips_open_up) else "false"
    _tip_font_delta = float(header_tooltip_font_delta_pt)
    _tip_top_space = int(max(0, header_tooltip_top_space_px))
    if _tip_top_space:
        height = int(height + _tip_top_space)
        _body_pad = f"{_tip_top_space}px 10px 4px 10px"
    _nowrap_css = ""
    if header_nowrap:
        _nowrap_css = """    th.katex-th-nowrap { white-space: nowrap; max-width: none !important; word-wrap: normal; }
"""

    html_fragment = f"""<!DOCTYPE html><html><head><meta charset="utf-8"/>
<link rel="stylesheet" href="{_KATEX_BASE}/katex.min.css" crossorigin="anonymous"/>
<script src="{_KATEX_BASE}/katex.min.js" crossorigin="anonymous"></script>
<style>
    html,body{{margin:0;padding:{_body_pad};{_body_overflow}background:transparent;font-family:sans-serif;}}
    /* Ocultar barras de desplazamiento visualmente manteniendo funcionalidad */
    html::-webkit-scrollbar, body::-webkit-scrollbar {{ display: none; }}
    html,body {{ -ms-overflow-style: none; scrollbar-width: none; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 0; font-size: {_table_font}; table-layout: auto; }}
    th, td {{ padding: {_cell_pad}; text-align: center; vertical-align: middle; border-bottom: 1px solid #eee; }}
    th {{ background-color: #f8f9fa; color: #555; font-weight: normal; white-space: normal; word-wrap: break-word; line-height: {_head_lh}; max-width: {_head_max}; cursor: help; position: sticky; top: 0; z-index: 2; }}
    th .math .katex {{ font-size: calc(1em + var(--hboost, {_header_boost}pt)); }}
    th.has-tip:hover {{ background-color: #eef3fb; z-index: 2147483646; }}
    th .tipbox {{
        visibility: hidden;
        opacity: 0;
        position: fixed;
        left: var(--tip-left, 50vw);
        top: 4px;
        transform: translateX(-50%);
        z-index: 2147483647;
        min-width: 180px;
        width: max-content;
        max-width: calc(100vw - 16px);
        padding: 9px 10px;
        border-radius: 6px;
        background: rgba(31, 41, 55, 0.96);
        color: #fff;
        text-align: left;
        font-size: 0.72rem;
        font-weight: 400;
        line-height: 1.25;
        white-space: nowrap;
        overflow: visible;
        box-shadow: 0 8px 18px rgba(0,0,0,0.18);
        pointer-events: none;
        transition: opacity 0.08s ease-in-out;
    }}
        th.has-tip:hover .tipbox {{ visibility: visible; opacity: 1; }}
    th .tipbox .tiptext {{ margin-bottom: 4px; }}
    th .tipbox .tipmath {{ display: block; color: #fff; white-space: nowrap; }}
    th .tipbox .tipmath .katex {{ white-space: nowrap; }}
    th .tipbox .katex {{ color: #fff; font-size: calc(1em + 4pt + {_tip_font_delta}pt); }}
    th.has-tip:hover::before {{
        content: none;
    }}
{_nowrap_css}    td {{ line-height: {_td_lh}; }}
    /* Tabla 11: globo propio (iframe Streamlit); disparador en columna Parámetro para las tres probabilidades. */
    .t11-cell-tip {{
        cursor: help;
        text-decoration: underline dotted rgba(80, 80, 120, 0.55);
    }}
    .t11-cell-tip-float {{
        display: none;
        position: fixed;
        z-index: 2147483647;
        max-width: min(540px, calc(100vw - 20px));
        padding: 9px 10px;
        border-radius: 6px;
        background: rgba(31, 41, 55, 0.96);
        color: #fff;
        text-align: left;
        font-size: 0.72rem;
        font-weight: 400;
        line-height: 1.3;
        white-space: normal;
        word-wrap: break-word;
        box-shadow: 0 8px 18px rgba(0, 0, 0, 0.2);
        pointer-events: none;
    }}
    .t11-cell-tip-float.t11-show {{ display: block; }}
    .t11-cell-tip-float .katex,
    .t11-cell-tip-float .katex-html {{ color: #fff !important; }}
    .t11-cell-tip-float .katex-display {{ margin: 0.15em 0; }}
    .row-cell-tip {{ cursor: help; text-decoration: underline dotted rgba(80,80,120,0.5); }}
    .row-tip-float {{
        display: none; position: fixed; z-index: 2147483647;
        max-width: min(500px, calc(100vw - 20px));
        padding: 9px 11px; border-radius: 6px;
        background: rgba(31,41,55,0.96); color: #fff;
        text-align: left; font-size: 0.72rem; line-height: 1.3;
        white-space: normal; word-wrap: break-word;
        box-shadow: 0 8px 18px rgba(0,0,0,0.2); pointer-events: none;
    }}
    .row-tip-float.row-tip-show {{ display: block; }}
    .row-tip-float .katex, .row-tip-float .katex-html {{ color: #fff !important; }}
    .row-tip-float .katex-display {{ margin: 0.1em 0; }}
    .row-tip-float table {{ width: auto; border-collapse: collapse; margin: 4px 0; }}
    .row-tip-float th, .row-tip-float td {{ padding: 1px 6px; border: 1px solid rgba(255,255,255,0.18); font-size: 0.70rem; text-align: right; background: transparent; }}
    .row-tip-float th {{ text-align: center; font-weight: 600; background: rgba(255,255,255,0.06); }}
</style>
</head><body>
<table>
    <thead><tr>"""
    for _idx_h, (h, tip, tip_ltx) in enumerate(zip(katex_headers, _tips, _tip_latex)):
        _has_tip = bool(tip or tip_ltx)
        _th_classes = []
        if _has_tip:
            _th_classes.append("has-tip")
        if header_nowrap:
            _th_classes.append("katex-th-nowrap")
        _class_attr = f' class="{" ".join(_th_classes)}"' if _th_classes else ""
        _boost_val = float(_boost_by_index.get(_idx_h, _header_boost))
        _style_attr = f' style="--hboost: {_boost_val}pt;"'
        _tipbox = ""
        if _has_tip:
            _tip_math = f"<span class='tipmath' data-latex=\"{tip_ltx}\"></span>" if tip_ltx else ""
            _tip_text = f"<div class='tiptext'>{tip}</div>" if (tip and not tip_ltx) else ""
            _tipbox = f"<div class='tipbox'>{_tip_math or _tip_text}</div>"
        html_fragment += f"<th{_class_attr}{_style_attr}><span class='math'>{h}</span>{_tipbox}</th>"
    html_fragment += "</tr></thead><tbody>"

    _row_tips_list = [_translate_text_to_english(str(x)) for x in list(row_tooltips or [])]
    if len(_row_tips_list) < len(df):
        _row_tips_list.extend([""] * (len(df) - len(_row_tips_list)))
    for _ri, (_, row) in enumerate(df.iterrows()):
        html_fragment += "<tr>"
        _rt = str(_row_tips_list[_ri]) if _ri < len(_row_tips_list) else ""
        for _ci, val in enumerate(row):
            if _ci == 0 and _rt:
                _esc_rt = html.escape(_rt, quote=True)
                html_fragment += f'<td class="row-cell-tip" data-tip-html="{_esc_rt}">{_katex_table_cell_html(val)}</td>'
            else:
                html_fragment += f"<td>{_katex_table_cell_html(val)}</td>"
        html_fragment += "</tr>"

    html_fragment += """</tbody></table>
<script>
    document.querySelectorAll('.math').forEach(el => {
        try { katex.render(el.textContent, el, { displayMode: false, throwOnError: false }); }
        catch (e) { }
    });
    document.querySelectorAll('.tipmath').forEach(el => {
        const src = el.getAttribute('data-latex') || '';
        try { katex.render(src, el, { displayMode: false, throwOnError: false }); }
        catch (e) { el.textContent = src; }
    });
    document.querySelectorAll('th.has-tip').forEach(th => {
        const box = th.querySelector('.tipbox');
        if (!box) return;
        th.addEventListener('mouseenter', () => {
            box.style.visibility = 'hidden';
            box.style.opacity = '0';
            box.style.left = '50vw';
            requestAnimationFrame(() => {
                const thRect = th.getBoundingClientRect();
                const boxRect = box.getBoundingClientRect();
                const pad = 8;
                let left = thRect.left + thRect.width / 2;
                const half = boxRect.width / 2;
                left = Math.max(pad + half, Math.min(window.innerWidth - pad - half, left));
                box.style.left = `${left}px`;
                if (%TIP_OPEN_UP%) {
                    let top = thRect.top - boxRect.height - 8;
                    if (top < pad) top = pad;
                    box.style.top = `${top}px`;
                } else {
                    box.style.top = '4px';
                }
                box.style.visibility = 'visible';
                box.style.opacity = '1';
            });
        });
        th.addEventListener('mouseleave', () => {
            box.style.visibility = 'hidden';
            box.style.opacity = '0';
        });
    });
    (function () {
        let floatBox = null;
        function ensureBox() {
            if (!floatBox) {
                floatBox = document.createElement('div');
                floatBox.className = 't11-cell-tip-float';
                floatBox.setAttribute('aria-hidden', 'true');
                document.body.appendChild(floatBox);
            }
            return floatBox;
        }
        function hideCellTip() {
            if (floatBox) {
                floatBox.classList.remove('t11-show');
                floatBox.innerHTML = '';
            }
        }
        document.querySelectorAll('.t11-cell-tip').forEach((span) => {
            span.addEventListener('mouseenter', () => {
                const src = span.getAttribute('data-latex');
                if (!src) return;
                const box = ensureBox();
                box.innerHTML = '';
                const inner = document.createElement('span');
                inner.className = 't11-float-katex';
                box.appendChild(inner);
                try {
                    katex.render(src, inner, { displayMode: true, throwOnError: false });
                } catch (e) {
                    inner.textContent = src;
                }
                box.classList.add('t11-show');
                requestAnimationFrame(() => {
                    const r = span.getBoundingClientRect();
                    const pad = 8;
                    const br = box.getBoundingClientRect();
                    let left = r.left + r.width / 2 - br.width / 2;
                    left = Math.max(pad, Math.min(window.innerWidth - pad - br.width, left));
                    let top = Math.max(pad, r.top - br.height - 6);
                    box.style.left = `${left}px`;
                    box.style.top = `${top}px`;
                });
            });
            span.addEventListener('mouseleave', hideCellTip);
        });
    })();
    (function () {
        var rtBox = null;
        function ensureRtBox() {
            if (!rtBox) {
                rtBox = document.createElement('div');
                rtBox.className = 'row-tip-float';
                rtBox.setAttribute('aria-hidden', 'true');
                document.body.appendChild(rtBox);
            }
            return rtBox;
        }
        function hideRt() { if (rtBox) rtBox.classList.remove('row-tip-show'); }
        document.querySelectorAll('td.row-cell-tip').forEach(function(td) {
            td.addEventListener('mouseenter', function() {
                var raw = td.getAttribute('data-tip-html');
                if (!raw) return;
                var box = ensureRtBox();
                box.innerHTML = raw;
                box.querySelectorAll('[data-katex]').forEach(function(el) {
                    var src = el.getAttribute('data-katex') || '';
                    var disp = el.hasAttribute('data-katex-disp');
                    try { katex.render(src, el, { displayMode: disp, throwOnError: false }); }
                    catch(e) { el.textContent = src; }
                });
                box.classList.add('row-tip-show');
                requestAnimationFrame(function() {
                    var r = td.getBoundingClientRect();
                    var pad = 8;
                    var br = box.getBoundingClientRect();
                    var left = r.left + r.width/2 - br.width/2;
                    left = Math.max(pad, Math.min(window.innerWidth - pad - br.width, left));
                    var top = Math.max(pad, r.top - br.height - 6);
                    box.style.left = left + 'px';
                    box.style.top = top + 'px';
                });
            });
            td.addEventListener('mouseleave', hideRt);
        });
    })();
</script>
</body></html>"""
    html_fragment = html_fragment.replace("%TIP_OPEN_UP%", _tip_open_up_js)
    components.html(html_fragment, height=height)


def _tabla11_par_calibrados_latex(p: str) -> str:
    """Etiqueta LaTeX (solo math) para la columna Parámetro de Tabla 11."""
    if p == "V_L":
        return r"V_{L}"
    if p == "R":
        return r"R"
    if p == "F_col":
        return r"F_{\mathrm{col}}"
    if p == "phi_F":
        return r"\phi_{F}(\theta_F)"
    if p == "kappa_F (θ_F actual)":
        return r"\kappa_{F}(\theta_F)"
    if p == "nu_F":
        return r"\nu_{F}(\theta_F)"
    if p == "alpha_0":
        return r"\alpha_0"
    if p == "gamma_0":
        return r"\gamma_0"
    if p == "E_thetaK_I_F_tilde_psurv":
        return (
            r"\tilde{p}_{\mathrm{surv}}^{\mathrm{coop}}"
            r"=\frac{1}{2}\bigl(\mathbb{E}_{\mu}[p_{\mathrm{surv},0}]"
            r"+p_{\mathrm{CMH}}\bigr)"
        )
    if p == "E_thetaK_I_F_tilde_prel":
        return r"\tilde{p}_{\mathrm{rel}}^{\mathrm{col}}\ (\mathrm{CMH})"
    # Tres esperanzas anidadas (Mechanism.tex; mismo orden que family-utility-coop/col)
    if "Coop" in p and "s_t=1" in p:
        return (
            r"\mathbb{E}_{\theta_K \mid \mathcal{I}_t^F}\!\left["
            r"\mathbb{E}_{\tilde{A}_t \mid \mathcal{Q}_t^{\mathrm{Coop}}}\!\left["
            r"\mathbb{P}_{\mathrm{E}}(s_t{=}1\mid \gamma_t,\tilde{A}_t,\theta_K)"
            r"\right]\right]"
        )
    if "Col" in p and "m_t=rel" in p:
        return (
            r"\mathbb{E}_{\theta_K \mid \mathcal{I}_t^F}\!\left["
            r"\mathbb{E}_{\tilde{A}_t \mid \mathcal{Q}_t^{\mathrm{Col}}}\!\left["
            r"\mathbb{P}_{\mathrm{E}}(m_t{=}\mathrm{rel}\mid \tilde{A}_t,R,\theta_K)"
            r"\right]\right]"
        )
    if "d_t=1" in p and "Q_t^F" in p:
        return (
            r"\mathbb{E}_{\tilde{a}_t^F \mid \mathcal{Q}_{t}^{F}}\!\left["
            r"\mathbb{P}_{\mathrm{E}}(d_t{=}1\mid \alpha_t,\tilde{a}_t^F,\mathcal{I}_t^F)"
            r"\right]"
        )
    return r"\mathrm{" + re.sub(r"([%#_{}&])", r"\\\1", str(p)) + "}"


def _tabla11_nested_param_tip_latex(p_raw: str) -> Optional[str]:
    """LaTeX del globo (``data-latex`` + KaTeX en JS) para las tres probabilidades en columna Parámetro."""
    if "Coop" in p_raw and "s_t=1" in p_raw:
        return (
            r"\begin{aligned}"
            r"&\sum_{\theta_K}\mu_F(\theta_K)\,\tilde{p}_{\mathrm{surv},0}(\theta_K)\\[0.35em]"
            r"&\tilde{p}_{\mathrm{surv},0}(\theta_K)="
            r"\mathbb{E}_{\tilde A_0\mid\mathcal Q_0^{\mathrm{Coop}}}\!\left["
            r"\mathbb{P}_{\mathrm{E}}(s_0{=}1\mid\gamma_0,\tilde A_0,\theta_K)\right]\\"
            r"&\mathcal Q_0^{\mathrm{Coop}}:\ \tilde a_0^F\sim P_I^F(\cdot\mid a_{\mathrm{coop}}),\ "
            r"\tilde a_0^K,\tilde a_0^S\sim P_I\ \text{\scriptsize de Tabla 10b.}\\"
            r"&\mu_F\ \text{\scriptsize es la creencia bayesiana completa de Familia.}"
            r"\end{aligned}"
        )
    if "Col" in p_raw and "m_t=rel" in p_raw:
        return (
            r"\begin{aligned}"
            r"&\sum_{\theta_K}\mu_F(\theta_K)\,\tilde{p}_{\mathrm{rel},0}(\theta_K)\\[0.35em]"
            r"&\tilde{p}_{\mathrm{rel},0}(\theta_K)="
            r"\mathbb{E}_{\tilde{A}_0\mid\mathcal{Q}_0^{\mathrm{Col}}}\!\left["
            r"\mathbb{P}_{\mathrm{E}}(m_0{=}\mathrm{rel}\mid \tilde{A}_0,R,\theta_K)\right]\\"
            r"&\mathcal Q_0^{\mathrm{Col}}:\ \tilde a_0^F\sim P_I^F(\cdot\mid a_{\mathrm{col}}),\ "
            r"\tilde a_0^K,\tilde a_0^S\sim P_I\ \text{\scriptsize de Tabla 10b.}\\"
            r"&\mathbb{P}_{\mathrm{E}}(m_0{=}\mathrm{rel})"
            r"=h_{\mathrm{Liberación}}(0\mid\tilde A_0,X'_0,\theta_K)"
            r"\quad\text{\scriptsize bloque físico de Tabla 10c.}"
            r"\end{aligned}"
        )
    if "d_t=1" in p_raw and "Q_t^F" in p_raw:
        return (
            r"\begin{aligned}"
            r"&\tilde{p}_{\mathrm{det},0}"
            r"=\mathbb{E}_{\tilde{a}_0^F\mid\mathcal{Q}_0^F}\!\left["
            r"\mathbb{P}_{\mathrm{E}}(d_0{=}1\mid\alpha_0,\tilde{a}_0^F,\mathcal{I}_0^F)\right]\\[0.3em]"
            r"&\quad=\mathbb{P}_{\mathrm{I},F}(\tilde a_0^F{=}a_{\mathrm{col}})\cdot p_{\mathrm{det},0},\\"
            r"&\quad p_{\mathrm{det},0}=\Lambda(\eta_0{+}\eta_1\alpha_0^\ast{+}\eta_2\gamma_0^\ast).\\"
            r"&\quad\text{\scriptsize No lleva suma exterior sobre }\theta_K\text{\scriptsize ; }"
            r"P_{I,F}\text{\scriptsize\ usa la creencia de Tabla 10b.}"
            r"\end{aligned}"
        )
    return None


def render_tabla11_family_calibrated_katex(df_cal: pd.DataFrame) -> None:
    """Tabla 11 con KaTeX (parámetro y nivel en math; valor en texto)."""
    if df_cal is None or df_cal.empty:
        return
    rows_out = []
    for _, row in df_cal.iterrows():
        p_raw = str(row.get("Parámetro", ""))
        ltx = _tabla11_par_calibrados_latex(p_raw)
        v = row.get("Valor", "")
        try:
            v_str = f"{float(v):.3f}"
        except (TypeError, ValueError):
            v_str = html.escape(str(v))
        n_raw = str(row.get("Nivel", ""))
        _tip_ltx = _tabla11_nested_param_tip_latex(p_raw)
        _math = f'<span class="math">{html.escape(ltx, quote=False)}</span>'
        if _tip_ltx is not None:
            _dlt = html.escape(_tip_ltx, quote=True)
            c0_html = f'<span class="t11-cell-tip" data-latex="{_dlt}">{_math}</span>'
        else:
            c0_html = _math
        c1_html = html.escape(v_str)
        rows_out.append(
            {
                "c0": c0_html,
                "c1": c1_html,
                "c2": f'<span class="math">\\text{{{n_raw}}}</span>',
            }
        )
    df_show = pd.DataFrame(rows_out)
    _n = len(rows_out)
    _h = max(220, 44 + 30 * (_n + 1))
    render_generic_katex_table(
        df_show,
        [
            r"\text{Parámetro}",
            r"\text{Valor}",
            r"\text{Nivel}",
        ],
        height=_h,
        compact=True,
        tight_spacing=True,
        header_tooltips=[
            "Símbolo o esperanza (Mechanism.tex). Para las tres probabilidades anidadas (Nivel «Calculado»), pase el cursor sobre la fórmula subrayada: globo con LaTeX (Tablas 10 y 10c).",
            "Valor numérico en la sesión actual.",
            "Origen: calibrado (`rb_*`) o calculado desde Tabla 10 / 10c.",
        ],
        header_tooltips_open_up=True,
        header_tooltip_top_space_px=42,
        header_font_boost_pt=1.0,
    )


def _softmax_from_utilities(utilities: dict[str, float], T: float) -> dict[str, float]:
    """Softmax estable para probabilidades MDG de implementación."""
    if not utilities:
        return {}
    T = float(max(T, 1e-12))
    u_max = max(float(v) for v in utilities.values())
    exps = {k: float(np.exp((float(v) - u_max) / T)) for k, v in utilities.items()}
    total = float(sum(exps.values()))
    return {k: (v / total if total > 1e-12 else 0.0) for k, v in exps.items()}


def _mdg_t0_temperature(agent: str) -> float:
    """Temperatura efectiva de Tabla 7 en t=0 para el agente indicado."""
    T0 = float(st.session_state.get(f"mdg_T0_{agent}", 1.0))
    cbar = float(st.session_state.get(f"mdg_cbar_{agent}", 0.05))
    eta = float(st.session_state.get("cal_mdg_eta_cal_by_i", {}).get(agent, 0.0))
    return float(max(hybrid_temperature(1.0, T0, 1.0, eta, 0, cbar), 1e-12))


def _mdg7_session_probs(agent: str, keys: list[str], base_probs: dict[str, float]) -> dict[str, float]:
    """Aplica ediciones de Tabla 7 como pesos, sin congelar la probabilidad teórica."""
    weights = []
    any_session = False
    for i, k in enumerate(keys):
        sk = f"mdg7_p_{agent}_{i}"
        if sk in st.session_state:
            any_session = True
            weights.append(max(1e-12, float(st.session_state[sk])))
        else:
            weights.append(1.0)
    if not any_session:
        return base_probs
    adjusted = {
        k: float(max(0.0, base_probs.get(k, 0.0)) * w)
        for k, w in zip(keys, weights)
    }
    total = float(sum(adjusted.values()))
    if total <= 1e-12:
        return base_probs
    return {k: float(v / total) for k, v in adjusted.items()}


def _tab3_materialization_action_probs(agent: str, base_probs: dict[str, float]) -> dict[str, float]:
    """Prioriza las probabilidades generadas en Pestaña 3 · Transformada Inversa."""
    snap = st.session_state.get("tab3_materialization_action_probs", {})
    probs = snap.get(agent, {}) if isinstance(snap, dict) else {}
    if not probs:
        return base_probs
    if agent == "K":
        mapped = {
            "PIK_rel": float(probs.get("Liberar", base_probs.get("PIK_rel", 0.0))),
            "PIK_kill": float(probs.get("Matar", base_probs.get("PIK_kill", 0.0))),
            "PIK_cont": float(probs.get("Continuar", base_probs.get("PIK_cont", 0.0))),
        }
    elif agent == "F":
        mapped = {
            "PIF_coop": float(probs.get("Cooperar", base_probs.get("PIF_coop", 0.0))),
            "PIF_col": float(probs.get("Coludir", base_probs.get("PIF_col", 0.0))),
        }
    elif agent == "S":
        mapped = {
            "PIS_res": float(probs.get("Rescatar", base_probs.get("PIS_res", 0.0))),
            "PIS_neg": float(probs.get("No Rescatar", base_probs.get("PIS_neg", 0.0))),
        }
    else:
        return base_probs
    total = float(sum(max(0.0, v) for v in mapped.values()))
    if total <= 1e-12:
        return base_probs
    return {k: float(max(0.0, v) / total) for k, v in mapped.items()}


def _mdg_indicator_probs(agent: str, actions: list[str], intent: str) -> dict[str, float]:
    """Ley de implementación de Mechanism.tex: exp(1{a=a*}/T_t)."""
    T = _mdg_t0_temperature(agent)
    exps = [float(np.exp((1.0 if str(a) == str(intent) else 0.0) / T)) for a in actions]
    total = float(sum(exps))
    return {a: float(e / total) for a, e in zip(actions, exps)}


def _outcome_probs_for_actions(theta: str, gamma0: float, iota: float, a_k: str, a_s: str, a_f: str) -> dict[str, float]:
    """P_E(m_t=j | tilde A_t, X'_t, iota_t) para una tripleta ejecutada."""
    theta_vec = THETA_K_MAP.get(theta, [0.0, 0.0, 0.0, 0.0])
    psis = []
    for j in [1, 2, 3, 4, 5]:
        p = st.session_state.cal_psi_params[j]
        psi = float(p["delta"])
        if j == 1 and a_k == "Liberar":
            psi += float(p["gamma_K"])
        elif j == 2 and a_s in ("Rescatar", "Rescate"):
            psi += float(p["gamma_S"])
        elif j == 3 and a_f == "Coludir":
            psi += float(p["gamma_F"])
        elif j == 4 and a_k == "Matar":
            psi += float(p["gamma_K"])
        psi += float(p["phi_gamma"]) * float(gamma0)
        psi += sum(float(p["phi_theta"][i]) * float(theta_vec[i]) for i in range(4))
        psi += float(p["kappa"]) * float(iota)
        psis.append(float(np.exp(psi)))
    total = float(sum(psis))
    probs = [p / total for p in psis] if total > 1e-12 else [0.0] * 5
    return {
        "lib": probs[0],
        "res": probs[1],
        "pay": probs[2],
        "kill": probs[3],
        "cont": probs[4],
    }


def _mechanism_m_probs_for_actions(
    theta: str,
    t_eval: int,
    alpha_star_used: float,
    gamma_star_used: float,
    p_det: float,
    a_k: str,
    a_s: str,
    a_f: str,
    *,
    z_region: str,
    v_victim: str,
    f_capa: str,
    s_tipo: str,
    allow_zero_time: bool = False,
    policy_sensitivity: float = 1.0,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Ley física de m_t según Mechanism.tex: p_Cont y h_j desde lambdas competitivas."""
    theta_eff = str(theta) if str(theta) in TIPOS_SECUESTRADOR else str(TIPOS_SECUESTRADOR[0])
    modelo_eff = copy.deepcopy(modelo)
    bundle = _tab2_structural_bundle_for_theta(
        theta_eff,
        z_region=str(z_region),
        v_victim=str(v_victim),
        f_capa=str(f_capa),
        s_tipo=str(s_tipo),
    )
    bundle = _scale_policy_zeta_bundles({theta_eff: bundle}, float(policy_sensitivity)).get(theta_eff, bundle)
    if isinstance(bundle.get("betas"), dict):
        modelo_eff.betas[theta_eff].update(bundle["betas"])
    if isinstance(bundle.get("lambdas_0"), dict):
        modelo_eff.lambdas_0.update(bundle["lambdas_0"])
    zp = _focus_cmh_endogenous_tentatives(theta_eff)
    t_for_mech = int(t_eval) if bool(allow_zero_time) else max(1, int(t_eval))
    factors = mechanism_competitive_hazards_at_t(
        modelo_eff,
        theta_eff,
        t_for_mech,
        presion_S=float(gamma_star_used),
        z_region=str(z_region),
        v_victim=str(v_victim),
        alpha=float(alpha_star_used),
        gamma=float(gamma_star_used),
        p_det=float(p_det),
        zeta_alpha=float(zp.get("zeta_alpha", 0.1)),
        zeta_gamma=float(zp.get("zeta_gamma", 0.1)),
        zeta_d=float(zp.get("zeta_d", 0.1)),
        zeta_R=float(zp.get("zeta_R", 0.1)),
        estado_rescata=str(a_s).strip().lower().startswith("rescat"),
        t_mad=float(st.session_state.get("cal_T_mad", 30.0)),
        lambda4=float(st.session_state.get("cal_lambda_4", 0.0005)),
        zeta_by_j=bundle.get("zeta_by_j") if isinstance(bundle.get("zeta_by_j"), dict) else None,
        atilde_F=str(a_f),
        atilde_K=str(a_k),
        atilde_S=str(a_s),
    )
    hd = dict(factors.get("h_daily", {}))
    probs = {
        "Liberación": float(hd.get("Liberación", 0.0)),
        "Rescate": float(hd.get("Rescate", 0.0)),
        "Pago": float(hd.get("Pago", 0.0)),
        "Muerte": float(hd.get("Muerte", 0.0)),
        "Continuar": float(hd.get("Continuar", 0.0)),
    }
    total = float(sum(max(0.0, v) for v in probs.values()))
    if total > 1e-12:
        probs = {k: float(max(0.0, v) / total) for k, v in probs.items()}
    else:
        probs = {"Liberación": 0.0, "Rescate": 0.0, "Pago": 0.0, "Muerte": 0.0, "Continuar": 1.0}
    return probs, factors


def _t52_mechanism_m_tooltip_lines(
    factors: dict[str, Any],
    tau: int,
    alpha_used: float,
    gamma_used: float,
    p_det: float,
    *,
    policy_label: str = r"\ast",
) -> str:
    lam = dict(factors.get("lam", {})) if isinstance(factors, dict) else {}
    return (
        rf'<div>\(M({int(tau)})={float(factors.get("M_t", 0.0)):.6f}\), '
        rf'\(\alpha^{{{policy_label}}}={float(alpha_used):.4f}\), '
        rf'\(\gamma^{{{policy_label}}}={float(gamma_used):.4f}\), '
        rf'\(p_{{det}}={float(p_det):.4f}\).</div>'
        rf'<div>\(\tilde\lambda_1^{{Pago}}={float(lam.get("Pago", 0.0)):.8f}\), '
        rf'\(\tilde\lambda_2^{{Muerte}}={float(lam.get("Muerte", 0.0)):.8f}\), '
        rf'\(\tilde\lambda_3^{{Rescate}}={float(lam.get("Rescate", 0.0)):.8f}\), '
        rf'\(\tilde\lambda_4={float(lam.get("Exógeno", 0.0)):.8f}\).</div>'
        rf'<div>\(p_{{Cont}}=\exp[-\sum_j\tilde\lambda_j\Delta t]'
        rf'={float(factors.get("p_cont", 0.0)):.8f}\), '
        rf'\(h_j=(1-p_{{Cont}})\tilde\lambda_j/\sum_\ell\tilde\lambda_\ell\).</div>'
    )


def _expected_outcomes_over_tilde_A(
    theta: str,
    gamma0: float,
    iota: float,
    pk: dict[str, float],
    ps: dict[str, float],
    pf: dict[str, float],
) -> dict[str, float]:
    """E_{tilde A | Q}[P_E(m=j | tilde A, X', iota)]."""
    out = {k: 0.0 for k in ("lib", "res", "pay", "kill", "cont")}
    for ak, p_k in pk.items():
        for a_s, p_s in ps.items():
            for af, p_f in pf.items():
                weight = float(p_k) * float(p_s) * float(p_f)
                probs = _outcome_probs_for_actions(theta, gamma0, iota, ak, a_s, af)
                for key in out:
                    out[key] += weight * probs[key]
    return out


def _expected_outcomes_over_tilde_A_hazards(
    theta: str,
    t_eval: int,
    alpha0: float,
    gamma0: float,
    p_det: float,
    pk: dict[str, float],
    ps: dict[str, float],
    pf: dict[str, float],
    z_region: str,
    v_victim: str,
    f_capa: str,
    s_tipo: str,
) -> dict[str, float]:
    """E_{tilde A | Q}[P_E(m=j)] via hazards competitivos Eqs. (28)-(29)."""
    out = {k: 0.0 for k in ("lib", "res", "pay", "kill", "cont")}
    for ak, p_k in pk.items():
        for a_s, p_s in ps.items():
            for af, p_f in pf.items():
                weight = float(p_k) * float(p_s) * float(p_f)
                if weight <= 1e-15:
                    continue
                probs, _ = _mechanism_m_probs_for_actions(
                    theta, t_eval, alpha0, gamma0, p_det, ak, a_s, af,
                    z_region=z_region,
                    v_victim=v_victim,
                    f_capa=f_capa,
                    s_tipo=s_tipo,
                )
                out["lib"]  += weight * float(probs.get("Liberación", 0.0))
                out["res"]  += weight * float(probs.get("Rescate",    0.0))
                out["pay"]  += weight * float(probs.get("Pago",       0.0))
                out["kill"] += weight * float(probs.get("Muerte",     0.0))
                out["cont"] += weight * float(probs.get("Continuar",  0.0))
    return out


def _build_t0_family_state_mdg_probs(
    modelo: ModeloSecuestro,
    mu: dict[str, float],
    presion_S: float,
    precision_iota: float,
    alpha0: float,
    gamma0: float,
    ransom_scale: float,
    f_capa: str = "Alta Riqueza",
) -> dict[str, float]:
    """Probabilidades de implementación MDG en t=0 para Familia y Estado."""
    cmh_alive, cmh_kill = cmh_alive_and_kill_shares()
    rb = lambda k, d: float(st.session_state.get(k, d))
    _phi_f, _kap_f, _nu_f = _rb_family_phi_kappa_nu(f_capa)
    df_f, _ = compute_family_table(
        modelo,
        mu,
        presion_S,
        rb("rb_vl", 100.0),
        ransom_scale,
        gamma0,
        _phi_f,
        _kap_f,
        _nu_f,
        rb("rb_fcol", 40.0),
        rb("rb_pdet0", 0.08),
        rb("rb_pdeta", 0.35),
        alpha0,
        cmh_alive,
    )
    f_utils = {
        "PIF_coop": float(df_f.loc[df_f["Rama"] == "Cooperar (a_coop)", "EU ilustrativa"].iloc[0]),
        "PIF_col": float(df_f.loc[df_f["Rama"] == "Colusión (a_col)", "EU ilustrativa"].iloc[0]),
    }
    f_probs = _softmax_from_utilities(f_utils, _mdg_t0_temperature("F"))
    f_probs = _mdg7_session_probs("F", ["PIF_coop", "PIF_col"], f_probs)
    f_probs = _tab3_materialization_action_probs("F", f_probs)

    df_s, _ = compute_state_table(
        mu,
        modelo,
        presion_S,
        precision_iota,
        rb("rb_omk", 350.0),
        rb("rb_omp", 15.0),
        rb("rb_omg", 1.2),
        alpha0,
        gamma0,
        ransom_scale,
        (
            rb("rb_ops0", 2.0), rb("rb_ops1", 0.6), rb("rb_ops2", 0.9),
            rb("rb_ops3", 0.30), rb("rb_ops4", 0.40), rb("rb_ops5", 0.20),
        ),
        (
            rb("rb_mt0", 1.5), rb("rb_mt1", 0.45), rb("rb_mt2", 0.75),
            rb("rb_mt3", 0.25), rb("rb_mt4", 0.35), rb("rb_mt5", 0.15),
        ),
        (rb("rb_calp", 0.8), rb("rb_cgam", 0.5), rb("rb_ccross", 0.2)),
        cmh_kill,
        cmh_alive,
    )
    # El Estado minimiza pérdida; para aplicar la misma softmax exp(U/T), usamos U=-pérdida.
    s_utils = {
        "PIS_res": -float(df_s.loc[df_s["Rama"] == "Rescate", "Pérdida"].iloc[0]),
        "PIS_neg": -float(df_s.loc[df_s["Rama"] == "Negociar", "Pérdida"].iloc[0]),
    }
    s_probs = _softmax_from_utilities(s_utils, _mdg_t0_temperature("S"))
    s_probs = _mdg7_session_probs("S", ["PIS_res", "PIS_neg"], s_probs)
    s_probs = _tab3_materialization_action_probs("S", s_probs)
    return {**f_probs, **s_probs}


def _build_t0_kidnapper_mdg_probs_for_theta(
    theta: str,
    mu: dict[str, float],
    presion_S: float,
    alpha0: float,
    gamma0: float,
    ransom_scale: float,
    estado_duro: bool,
    beta_k: float,
) -> dict[str, float]:
    """Probabilidades MDG de K por tipo, con la misma base usada en Tabla 10."""
    p_cap_base = float(st.session_state.get("rb_pcap", 0.12))
    _, df_uk, _ = compute_kidnapper_by_type_tables(
        modelo,
        mu,
        presion_S,
        alpha0,
        gamma0,
        ransom_scale,
        p_cap_base,
        estado_duro,
        str(theta),
        beta_k,
    )
    row = df_uk[df_uk["theta_K"].astype(str) == str(theta)].iloc[0]
    utilities = {
        "PIK_rel": float(row["U_rel"]),
        "PIK_kill": float(row["U_kill"]),
        "PIK_cont": float(row["V_cont"]),
    }
    probs = _softmax_from_utilities(utilities, _mdg_t0_temperature("K"))
    probs = _mdg7_session_probs("K", ["PIK_rel", "PIK_kill", "PIK_cont"], probs)
    return _tab3_materialization_action_probs("K", probs)


def _build_t0_implementation_likelihood_by_theta(
    mu_for_fs: dict[str, float],
    presion_S: float,
    precision_iota: float,
    alpha0: float,
    gamma0: float,
    ransom_scale: float,
    f_capa: str,
    estado_duro: bool,
    beta_k: float,
    atilde_F: str,
    atilde_K: str,
    atilde_S: str,
) -> dict[str, float]:
    """\(\mathcal L_{I,t}\): probabilidad MDG de la tripleta ejecutada observada."""
    fs_probs = _build_t0_family_state_mdg_probs(
        modelo,
        mu_for_fs,
        presion_S,
        precision_iota,
        alpha0,
        gamma0,
        ransom_scale,
        f_capa,
    )
    f_key = "PIF_col" if str(atilde_F) == "Coludir" else "PIF_coop"
    s_key = "PIS_res" if str(atilde_S) in ("Rescatar", "Rescate") else "PIS_neg"
    k_key_by_action = {
        "Liberar": "PIK_rel",
        "Matar": "PIK_kill",
        "Continuar": "PIK_cont",
    }
    k_key = k_key_by_action.get(str(atilde_K), "PIK_cont")
    out: dict[str, float] = {}
    for theta in TIPOS_SECUESTRADOR:
        kp = _build_t0_kidnapper_mdg_probs_for_theta(
            str(theta),
            mu_for_fs,
            presion_S,
            alpha0,
            gamma0,
            ransom_scale,
            estado_duro,
            beta_k,
        )
        out[str(theta)] = float(
            max(
                1e-300,
                float(fs_probs.get(f_key, 0.0))
                * float(fs_probs.get(s_key, 0.0))
                * float(kp.get(k_key, 0.0)),
            )
        )
    return out


def _build_t0_outcome_probs(theta: str, gamma0: float, iota: float) -> dict[str, float]:
    """Probabilidades físicas del desenlace m_0 según la tripleta ejecutada inicial."""
    snap = st.session_state.get("tab3_materialization_outcome_probs", {})
    if isinstance(snap, dict) and snap:
        return {
            "PE_m_lib": float(snap.get("lib", snap.get("Liberación", 0.0))),
            "PE_m_res": float(snap.get("res", snap.get("Rescate", 0.0))),
            "PE_m_pay": float(snap.get("pay", snap.get("Pago", 0.0))),
            "PE_m_kill": float(snap.get("kill", snap.get("Muerte", 0.0))),
            "PE_m_cont": float(snap.get("cont", snap.get("Continuar", 0.0))),
        }
    a_k = str(st.session_state.get("h0_Atilde_K", "Continuar"))
    a_s_raw = str(st.session_state.get("h0_Atilde_S", "—"))
    a_f = str(st.session_state.get("h0_Atilde_F", "—"))
    a_s = "Rescatar" if a_s_raw == "Rescate" else "No Rescatar"
    probs = _outcome_probs_for_actions(theta, gamma0, iota, a_k, a_s, a_f)
    return {
        "PE_m_lib": probs["lib"],
        "PE_m_res": probs["res"],
        "PE_m_pay": probs["pay"],
        "PE_m_kill": probs["kill"],
        "PE_m_cont": probs["cont"],
    }


def _p_surv_precision_logit(theta: str, iota: float, theta_hat: str) -> float:
    """Logit de supervivencia encadenado a Tabla 4 y sensible a la precisión modal μ."""
    a0_surv = float(st.session_state.get("cal_surv_alpha0", {}).get(theta, -5.0))
    beta_R = float(st.session_state.get("cal_surv_beta_R", 7.0))
    iota_c = float(max(0.0, min(1.0, iota)))
    match = 1.0 if str(theta_hat) == str(theta) else 0.0
    u = float(a0_surv + beta_R * iota_c * match)
    return float(1.0 / (1.0 + np.exp(-u)))


def _build_t0_longitudinal_mechanism_table(
    modelo: ModeloSecuestro,
    mu: dict[str, float],
    presion_S: float,
    t_mad: float,
    lambda4: float,
    precision_iota: float,
    alpha0: float,
    gamma0: float,
    ransom_scale: float,
    estado_duro: bool,
    beta_k: float,
    t_max: int,
    z_region: str,
    v_victim: str,
    theta_true: str,
    f_capa: str = "",
    s_tipo: str = "",
    atilde_F: Optional[str] = None,
    atilde_K: Optional[str] = None,
    atilde_S: Optional[str] = None,
) -> pd.DataFrame:
    """Tabla operativa de intensidades, riesgos diarios e incidencias acumuladas."""
    def _kidnapper_intention_probs(theta: str) -> dict[str, float]:
        """Ley MDG de implementación de K en t=0: softmax sobre utilidades por rama."""
        p_cap_base = float(st.session_state.get("rb_pcap", 0.12))
        _, df_uk, _ = compute_kidnapper_by_type_tables(
            modelo,
            mu,
            presion_S,
            alpha0,
            gamma0,
            ransom_scale,
            p_cap_base,
            estado_duro,
            theta,
            beta_k,
        )
        row = df_uk[df_uk["theta_K"] == theta].iloc[0]
        utilities = {
            "PIK_rel": float(row["U_rel"]),
            "PIK_kill": float(row["U_kill"]),
            "PIK_cont": float(row["V_cont"]),
        }
        probs = _softmax_from_utilities(utilities, _mdg_t0_temperature("K"))
        probs = _mdg7_session_probs("K", ["PIK_rel", "PIK_kill", "PIK_cont"], probs)
        return _tab3_materialization_action_probs("K", probs)

    rows = []
    S_prev = 1.0
    F = {j: 0.0 for j in ("Pago", "Muerte", "Rescate", "Exógeno")}
    theta_hat = max(mu, key=lambda k: float(mu.get(k, 0.0)))
    iota = float(max(0.0, min(1.0, precision_iota)))
    # η₀(θ_K) tipo-específico: detectabilidad basal varía entre organizaciones
    eta0 = float(st.session_state.get(f"cal_eta0_pdet_{str(theta_true)}", _ETA0_PDET_DEFAULTS.get(str(theta_true), -2.0)))
    eta1 = float(st.session_state.get("cal_eta1_pdet", 1.0))
    eta2 = float(st.session_state.get("cal_eta2_pdet", 1.0))
    p_surv = _p_surv_precision_logit(theta_true, iota, theta_hat)
    # p_det,t(θ_K) = Lambda(η₀(θ_K) + η₁α_t* + η₂γ_t*)  (Mechanism.tex eq:detection)
    p_det = float(1.0 / (1.0 + np.exp(-(eta0 + eta1 * float(alpha0) + eta2 * float(gamma0)))))
    pik_t0 = _kidnapper_intention_probs(theta_true)
    t_mad = float(max(1e-9, t_mad))
    lambda4 = float(max(0.0, lambda4))

    # Coeficientes zeta para las intensidades (Mechanism.tex, bloque proporcional).
    # Se obtienen del mismo bloque endogeno CMH que usa la Tabla 1 (consistencia).
    _zp = _focus_cmh_endogenous_tentatives(theta_true)
    _za = float(_zp.get("zeta_alpha", 0.1))
    _zg = float(_zp.get("zeta_gamma", 0.1))
    _zd = float(_zp.get("zeta_d", 0.1))
    _zR = float(_zp.get("zeta_R", 0.1))
    _f_capa_eff = str(f_capa or st.session_state.get("f_capa", "Alta Riqueza"))
    _s_tipo_eff = str(s_tipo or ("Duro" if estado_duro else "Laxo"))
    _atf_eff = str(atilde_F or st.session_state.get("h0_Atilde_F", "Cooperar"))
    _atk_eff = str(atilde_K or st.session_state.get("h0_Atilde_K", "Continuar"))
    _ats_eff = str(atilde_S or st.session_state.get("h0_Atilde_S", "No Rescatar"))
    _tab2_bundles = _tab2_bundles_all_types(
        z_region=str(z_region),
        v_victim=str(v_victim),
        f_capa=str(_f_capa_eff),
        s_tipo=str(_s_tipo_eff),
    )

    # Procesa el horizonte solicitado. En el escenario base se llama con t_max=0;
    # al avanzar ciclo se usa t=1 para actualizar M(t), lambdas y riesgos.
    _t_last = max(0, int(t_max))
    for tt in range(0, _t_last + 1):
        # M(t) = min{1, (t/T_mad)^2}  (Mechanism.tex eq. maturation-filter)
        M_t = float(min(1.0, (tt / t_mad) ** 2)) if t_mad > 0 else 0.0
        # Para tt=0: M(0)=0, por lo que lambda_tilde_j(0)=0 para j=1,2,3.
        # La unica intensidad activa en t=0 es lambda_4 (canal exogeno basal).
        h = {k: 0.0 for k in DESENLACES}
        for _theta_lam, _w_lam in mu.items():
            _w_lam = float(_w_lam)
            if _w_lam <= 0.0 or str(_theta_lam) not in TIPOS_SECUESTRADOR:
                continue
            _modelo_lam = copy.deepcopy(modelo)
            _bundle_lam = _tab2_bundles.get(str(_theta_lam), {}) or {}
            if isinstance(_bundle_lam.get("betas"), dict):
                _modelo_lam.betas[str(_theta_lam)].update(_bundle_lam["betas"])
            if isinstance(_bundle_lam.get("lambdas_0"), dict):
                _modelo_lam.lambdas_0.update(_bundle_lam["lambdas_0"])
            _zeta_bj_lam = (
                _bundle_lam.get("zeta_by_j")
                if isinstance(_bundle_lam.get("zeta_by_j"), dict)
                else None
            )
            _h_lam = _modelo_lam.calcular_hazards(
                tt,
                str(_theta_lam),
                presion_S,
                maturity_mult=M_t,
                z_region=z_region,
                v_victim=v_victim,
                alpha=float(alpha0),
                gamma=float(gamma0),
                p_det=p_det,
                zeta_alpha=_za,
                zeta_gamma=_zg,
                zeta_d=_zd,
                zeta_R=_zR,
                estado_rescata=str(_ats_eff).strip().lower().startswith("rescat"),
                zeta_by_j=_zeta_bj_lam,
                atilde_F=_atf_eff,
                atilde_K=_atk_eff,
                atilde_S=_ats_eff,
            )
            for _k_lam in DESENLACES:
                h[_k_lam] += _w_lam * float(_h_lam.get(_k_lam, 0.0))
        # tilde_lambda_j(t) = M(t)*lambda_j(t|C_t) para j=1,2,3
        # tilde_lambda_4(t) = lambda_4 (basal constante, sin filtro de maduracion)
        # blend_hazards ya aplica maturity_mult=M_t dentro de calcular_hazards,
        # por lo que h["Pago"], h["Muerte"], h["Rescate"] ya son las tilde_lambda.
        lam = {
            "Pago": float(max(0.0, h["Pago"])),
            "Muerte": float(max(0.0, h["Muerte"])),
            "Rescate": float(max(0.0, h["Rescate"])),
            "Exógeno": lambda4,
        }
        lam_sum = float(sum(lam.values()))
        # p_Cont,t = exp(-sum_j tilde_lambda_j(t) * Delta_t), Delta_t=1
        p_cont = float(np.exp(-lam_sum))
        # q(t) = 1 - p_Cont,t
        q = float(1.0 - p_cont)
        # xi_j(t) = tilde_lambda_j(t) / sum_l tilde_lambda_l(t)
        xi = {j: (lam[j] / lam_sum if lam_sum > 1e-12 else 0.0) for j in F}
        # h_j(t) = q(t) * xi_j(t)
        daily_h = {j: q * xi[j] for j in F}

        # F_j(t) = F_j(t-1) + h_j(t)*S(t-1)  (Mechanism.tex, longitudinal system)
        for j in F:
            F[j] += float(daily_h[j]) * S_prev
        # S(t) = S(t-1) * p_Cont,t
        S_new = S_prev * p_cont

        rows.append(
            {
                "t": tt,
                "M": M_t,
                "lam1": lam["Pago"],
                "lam2": lam["Muerte"],
                "lam3": lam["Rescate"],
                "lam4": lam["Exógeno"],
                "Pcont": p_cont,
                "q": q,
                "xi1": xi["Pago"],
                "xi2": xi["Muerte"],
                "xi3": xi["Rescate"],
                "xi4": xi["Exógeno"],
                "Psurv": p_surv,
                "Pdet": p_det,
                **pik_t0,
                "S": S_new,
                "F1": F["Pago"],
                "F2": F["Muerte"],
                "F3": F["Rescate"],
                "F4": F["Exógeno"],
            }
        )
        S_prev = S_new

    return pd.DataFrame(rows)


def _build_t0_capture_mdg_report(
    theta: str,
    row: pd.Series,
    alpha0: float,
    gamma0: float,
    agent_probs: Optional[dict[str, float]] = None,
) -> pd.DataFrame:
    """Continuación de Tabla 10: captura técnica y desenlaces físicos m_0."""
    agent_probs = agent_probs or {}
    pcap = st.session_state.cal_pcap_params.get(theta, _default_cal_pcap_params()[theta])
    delta_a = float(st.session_state.get("cal_pcap_delta_a", 0.0))
    c_S = float(st.session_state.get("cal_pcap_c_S", 0.0))
    logit_cap = (
        delta_a
        + float(pcap["c0"])
        + float(pcap["c_alpha"]) * float(alpha0)
        + float(pcap["c_gamma"]) * float(gamma0)
        + c_S
    )
    p_cap = float(1.0 / (1.0 + np.exp(-logit_cap)))
    _tt = int(row.get("t", 0)) if isinstance(row, pd.Series) else 0
    return pd.DataFrame(
        [
            {
                "t": _tt,
                "p_cap": p_cap,
                "PIF_coop": float(agent_probs.get("PIF_coop", 0.0)),
                "PIF_col": float(agent_probs.get("PIF_col", 0.0)),
                "PIS_res": float(agent_probs.get("PIS_res", 0.0)),
                "PIS_neg": float(agent_probs.get("PIS_neg", 0.0)),
                "PE_m_lib": float(agent_probs.get("PE_m_lib", 0.0)),
                "PE_m_res": float(agent_probs.get("PE_m_res", 0.0)),
                "PE_m_pay": float(agent_probs.get("PE_m_pay", 0.0)),
                "PE_m_kill": float(agent_probs.get("PE_m_kill", 0.0)),
                "PE_m_cont": float(agent_probs.get("PE_m_cont", 0.0)),
            }
        ]
    )


def _p_cap_technical_for_executed_S(
    theta: str, alpha0: float, gamma0: float, executed_s: str
) -> float:
    """\(p_{\mathrm{cap}}(a,\theta,\theta_S,\alpha^\ast,\gamma^\ast)\) con \(a=\tilde a^S\) (eq. p-cap-tecnica).

    \(\delta_a\) global (Tabla 6) más desplazamiento por modo ejecutado (rescate vs. negociación).
    """
    pcap = st.session_state.cal_pcap_params.get(theta, _default_cal_pcap_params()[theta])
    delta_a = float(st.session_state.get("cal_pcap_delta_a", 0.0))
    c_S = float(st.session_state.get("cal_pcap_c_S", 0.0))
    ex_res = float(st.session_state.get("cal_pcap_mode_rescue", 0.2))
    ex_neg = float(st.session_state.get("cal_pcap_mode_neg", -0.1))
    mode = ex_res if str(executed_s) == "Rescatar" else ex_neg
    logit_cap = (
        delta_a
        + mode
        + float(pcap["c0"])
        + float(pcap["c_alpha"]) * float(alpha0)
        + float(pcap["c_gamma"]) * float(gamma0)
        + c_S
    )
    return float(1.0 / (1.0 + np.exp(-logit_cap)))


def _p_surv_rescue_logit_for_executed_S(
    theta: str, iota: float, theta_hat: str, executed_s: str
) -> float:
    """eq. p-surv-rescue-logit-ajustado: bonus \(\beta_R\iota\) solo si S materializa rescate."""
    if str(executed_s) == "Rescatar":
        return _p_surv_precision_logit(theta, iota, theta_hat)
    else:
        a0_surv = float(st.session_state.get("cal_surv_alpha0", {}).get(theta, -5.0))
        return float(1.0 / (1.0 + np.exp(-a0_surv)))


def _t10b_mdg_S_weights(agent: dict[str, float]) -> dict[str, float]:
    """Pesos de materialización de S encadenados a Tabla 10b (PIS_neg, PIS_res)."""
    wn = float(agent.get("PIS_neg", 0.0))
    wr = float(agent.get("PIS_res", 0.0))
    s = wn + wr
    if s <= 1e-12:
        return {"No Rescatar": 0.5, "Rescatar": 0.5}
    return {"No Rescatar": wn / s, "Rescatar": wr / s}


def _t10b_mdg_F_weights(agent: dict[str, float]) -> dict[str, float]:
    """Pesos de materialización de F encadenados a Tabla 10b (PIF_coop, PIF_col)."""
    wc = float(agent.get("PIF_coop", 0.0))
    wl = float(agent.get("PIF_col", 0.0))
    s = wc + wl
    if s <= 1e-12:
        return {"Cooperar": 0.5, "Coludir": 0.5}
    return {"Cooperar": wc / s, "Coludir": wl / s}


def _t10_traj_mdg_K_weights(row: pd.Series) -> dict[str, float]:
    """Pesos PIK_rel / PIK_kill / PIK_cont de la fila t=0 de la trayectoria (Tabla 10)."""
    pr = float(row.get("PIK_rel", 0.0))
    pk = float(row.get("PIK_kill", 0.0))
    pc = float(row.get("PIK_cont", 0.0))
    s = pr + pk + pc
    if s <= 1e-12:
        return {"Liberar": 1.0 / 3.0, "Matar": 1.0 / 3.0, "Continuar": 1.0 / 3.0}
    return {"Liberar": pr / s, "Matar": pk / s, "Continuar": pc / s}


def _mechanism_tilde_p_cap_from_t10b_S(
    theta: str, alpha0: float, gamma0: float, agent: dict[str, float]
) -> float:
    ps = _t10b_mdg_S_weights(agent)
    tot = sum(
        float(ps[a]) * _p_cap_technical_for_executed_S(theta, alpha0, gamma0, a)
        for a in ("No Rescatar", "Rescatar")
    )
    return float(max(0.0, min(1.0, tot)))


def _apply_tab12_mechanism_probabilities(
    df_params: pd.DataFrame,
    agent_mdg_probs: dict[str, float],
    *,
    alpha0: float,
    gamma0: float,
    p_det: float,
    f_capa: str,
    s_tipo: str,
    t_eval: int = 0,
) -> pd.DataFrame:
    """Vincula las probabilidades de Tabla 12 a las expectativas de Tabla 10/Mechanism.tex."""
    out = df_params.copy()
    ps = _t10b_mdg_S_weights(agent_mdg_probs)
    pf = _t10b_mdg_F_weights(agent_mdg_probs)
    k_cont = _mdg_indicator_probs("K", ["Liberar", "Matar", "Continuar"], "Continuar")
    p_det_eff = float(max(0.0, min(1.0, float(p_det))))
    for ii in out.index:
        theta = str(out.at[ii, "theta_K"])
        out.at[ii, "p_cap_tilde"] = round(
            _mechanism_tilde_p_cap_from_t10b_S(theta, alpha0, gamma0, agent_mdg_probs),
            4,
        )
        q_cont = _expected_outcomes_over_tilde_A_hazards(
            theta,
            int(t_eval),
            float(alpha0),
            float(gamma0),
            p_det_eff,
            k_cont,
            ps,
            pf,
            z_region=str(st.session_state.get("z_region", "Andina")),
            v_victim=str(st.session_state.get("v_victim", "Privado")),
            f_capa=str(f_capa),
            s_tipo=str(s_tipo),
        )
        out.at[ii, "h_LibPago"] = round(float(q_cont.get("pay", 0.0)), 4)
    return out


def _mechanism_tilde_p_surv_from_t10b_S(
    theta: str, iota: float, theta_hat: str, agent: dict[str, float]
) -> float:
    ps = _t10b_mdg_S_weights(agent)
    tot = sum(
        float(ps[a]) * _p_surv_rescue_logit_for_executed_S(theta, iota, theta_hat, a)
        for a in ("No Rescatar", "Rescatar")
    )
    return float(max(0.0, min(1.0, tot)))


def _mechanism_E_tildeA_Qcoop_PE_s1(
    theta: str,
    iota: float,
    theta_hat: str,
    pk: dict[str, float],
    ps: dict[str, float],
    pf_coop: dict[str, float],
) -> float:
    """:math:`\\mathbb{E}_{\\tilde{A}_t\\mid\\mathcal{Q}_t^{\\mathrm{Coop}}}\\bigl[\\mathbb{P}_{\\mathrm{E}}(s_t{=}1\\mid\\gamma_t,\\tilde{A}_t,\\theta_K)\\bigr]`
    con :math:`\\mathcal{Q}^{\\mathrm{Coop}}` materializada vía ley de implementación de :math:`F` centrada en **Cooperar**
    (:func:`_mdg_indicator_probs`), y pesos MDG de :math:`K,S` coherentes con Tabla 10b.

    Aquí :math:`\\mathbb{P}_{\\mathrm{E}}(s_t{=}1\\mid\\cdot)` es el logit de supervivencia focal
    :func:`_p_surv_rescue_logit_for_executed_S` según :math:`\\tilde{a}_t^S` ejecutado.
    """
    tot = 0.0
    for ak, p_k in pk.items():
        for a_s, p_s in ps.items():
            for _af, p_f in pf_coop.items():
                w = float(p_k) * float(p_s) * float(p_f)
                s_lbl = "Rescatar" if str(a_s) in ("Rescatar", "Rescate") else "No Rescatar"
                tot += w * float(
                    _p_surv_rescue_logit_for_executed_S(theta, iota, theta_hat, s_lbl)
                )
    return float(max(0.0, min(1.0, tot)))


def _build_t0_tilde_prob_report(
    modelo: ModeloSecuestro,
    mu: dict[str, float],
    theta: str,
    presion_S: float,
    alpha0: float,
    gamma0: float,
    ransom_scale: float,
    estado_duro: bool,
    beta_k: float,
    precision_iota: float,
    agent_mdg_probs: dict[str, float],
    traj_row_t0: pd.Series,
) -> pd.DataFrame:
    """Probabilidades tilde en t=0 encadenadas a Tabla 10b (PIF/PIS) y a PIK/Pdet de la fila longitudinal.

    `mu` debe ser la misma creencia que alimenta Tabla 10b (p. ej. ``mu_tab``) para \(\hat\theta_0=\arg\max_\theta\mu(\theta)\).
    Los argumentos ``modelo``, ``presion_S``, ``ransom_scale``, ``estado_duro`` y ``beta_k`` se conservan en la firma
    por compatibilidad y posibles extensiones t>0; el cálculo actual solo usa ``theta``, ``alpha0``, ``gamma0``,
    ``precision_iota``, ``agent_mdg_probs`` y ``traj_row_t0``.

    En particular, ``ptilde_det`` es la columna homónima de Tabla 10c y en código es
    ``PIF_col * Pdet``: ``PIF_col`` viene de ``agent_mdg_probs`` (Tabla 10b) y ``Pdet`` de la fila
    ``t=0`` de la trayectoria (logit Tabla 3 en ``alpha0``, ``gamma0``). Bajo la parametrización
    actual, la esperanza de Mechanism sobre ``\\tilde{a}^F`` coincide numéricamente con ese producto
    (no hay término extra de ``\\tilde{a}_t^F`` dentro del logit de detección más allá del peso MDG
    sobre coludir).
    """
    _ = (modelo, presion_S, ransom_scale, estado_duro, beta_k)
    iota0 = float(max(0.0, min(1.0, precision_iota)))
    theta_hat = max(mu, key=lambda k: float(mu.get(k, 0.0))) if mu else theta
    pk_eq = _t10_traj_mdg_K_weights(traj_row_t0)
    ps_eq = _t10b_mdg_S_weights(agent_mdg_probs)
    pf_eq = _t10b_mdg_F_weights(agent_mdg_probs)
    f_col = _mdg_indicator_probs("F", ["Cooperar", "Coludir"], "Coludir")
    s_neg = _mdg_indicator_probs("S", ["No Rescatar", "Rescatar"], "No Rescatar")
    k_cont = _mdg_indicator_probs("K", ["Liberar", "Matar", "Continuar"], "Continuar")

    _tt = int(traj_row_t0.get("t", 0)) if isinstance(traj_row_t0, pd.Series) else 0
    p_det_row = float(max(0.0, min(1.0, float(traj_row_t0.get("Pdet", 0.0)))))
    _hz_kw = dict(
        theta=theta,
        t_eval=int(_tt),
        alpha0=float(alpha0),
        gamma0=float(gamma0),
        p_det=p_det_row,
        z_region=str(st.session_state.get("z_region", "")),
        v_victim=str(st.session_state.get("v_victim", "")),
        f_capa=str(st.session_state.get("f_capa", "Alta Riqueza")),
        s_tipo=str("Duro" if estado_duro else "Laxo"),
    )
    # ptilde_rel: Q^Col — F degenerada en Coludir, K y S con pesos MDG (Eqs. 28-29)
    q_col = _expected_outcomes_over_tilde_A_hazards(
        pk=pk_eq, ps=ps_eq, pf=f_col, **_hz_kw
    )
    # ptilde_kill: Q^Neg — S degenerada en No Rescatar, K y F con pesos MDG (Eqs. 28-29)
    q_neg = _expected_outcomes_over_tilde_A_hazards(
        pk=pk_eq, ps=s_neg, pf=pf_eq, **_hz_kw
    )
    # ptilde_pay: Q^Cont — K degenerado en Continuar, S y F con pesos MDG (Eqs. 28-29)
    q_cont = _expected_outcomes_over_tilde_A_hazards(
        pk=k_cont, ps=ps_eq, pf=pf_eq, **_hz_kw
    )

    p_cap_tilde = _mechanism_tilde_p_cap_from_t10b_S(theta, alpha0, gamma0, agent_mdg_probs)
    p_surv_tilde = _mechanism_tilde_p_surv_from_t10b_S(theta, iota0, theta_hat, agent_mdg_probs)
    p_if_col = float(agent_mdg_probs.get("PIF_col", 0.0))
    p_det_tilde = float(max(0.0, min(1.0, p_if_col * p_det_row)))
    return pd.DataFrame(
        [
            {
                "t": _tt,
                "ptilde_cap": p_cap_tilde,
                "ptilde_surv": p_surv_tilde,
                "ptilde_rel": float(q_col["lib"]),
                "ptilde_pay": float(q_cont["pay"]),
                "ptilde_kill": float(q_neg["kill"]),
                "ptilde_det": p_det_tilde,
            }
        ]
    )


def _family_nested_expectations_tab10(
    modelo: ModeloSecuestro,
    mu: dict[str, float],
    presion_S: float,
    precision_iota: float,
    alpha0: float,
    gamma0: float,
    ransom_scale: float,
    estado_duro: bool,
    beta_k: float,
    f_capa: str,
    z_region: str,
    v_victim: str,
    t_mad: float,
    lambda4: float,
    *,
    mu_mdg_for_agent: Optional[dict[str, float]] = None,
    p_det_theta_focal: Optional[str] = None,
) -> tuple[float, float, float]:
    """Tres esperanzas de Mechanism.tex (coop / col / det) coherentes con Tabla 10 y 10c.

    Devuelve ``(p_rel, p_det, p_s1)`` en el orden esperado por ``f1_nested_triple`` en
    ``rational_behavior.compute_family_table``:

    - ``p_s1``: :math:`\\mathbb{E}_{\\theta_K\\mid\\mathcal{I}_t^F}[\\cdots]` con interior
      ``\\tilde{p}_{\\mathrm{surv},t}(\\theta_K)`` = ``_mechanism_tilde_p_surv_from_t10b_S``;
    - ``p_rel``: mismo peso :math:`\\mu(\\theta_K)` sobre ``ptilde_rel`` de
      ``_build_t0_tilde_prob_report`` (rama colusión / ``Q^{\\mathrm{Col}}``);
    - ``p_det``: **sin** suma :math:`\\sum_{\\theta_K}\\mu(\\theta_K)`; es un único
      :math:`\\tilde{p}_{\\mathrm{det},0}` en el **tipo panel** ``p_det_theta_focal`` (trayectoria degenerada
      en ese :math:`\\theta_K`). Los pesos ``PIF_col`` / MDG en ``agent_probs`` siguen construidos con
      ``mu_mdg_for_agent`` (p. ej. ``mu_tab`` del tablero), como en Tabla 10b. Forma cerrada
      :math:`\\mathbb{P}_{\\mathrm{I},F}(\\mathrm{col})\\cdot p_{\\mathrm{det},t}` con
      :math:`p_{\\mathrm{det},t}=\\Lambda(\\eta_0+\\eta_1\\alpha^\\ast+\\eta_2\\gamma^\\ast)` en la fila
      :math:`t{=}0`. Si ``p_det_theta_focal`` es ``None``, se mantiene la mezcla ponderada por ``mu`` en el bucle
      (comportamiento antiguo).

    Mezcla con ``\\mu`` en el bucle para ``p_rel`` y ``p_s1`` (argumento ``mu``; p. ej. ``\\mu`` degenerada en
    el tipo panel). Para cada ``\\theta_K`` con peso positivo, la fila :math:`t{=}0` de la trayectoria usa
    creencia degenerada en ese tipo.
    """
    mu_pi = mu_mdg_for_agent if mu_mdg_for_agent is not None else mu
    agent_probs = _build_t0_family_state_mdg_probs(
        modelo, mu_pi, presion_S, precision_iota, alpha0, gamma0, ransom_scale, f_capa
    )
    theta_hat = (
        max(mu_pi, key=lambda k: float(mu_pi.get(k, 0.0)))
        if mu_pi
        else TIPOS_SECUESTRADOR[0]
    )
    s_rel = 0.0
    s_det = 0.0
    s_s1 = 0.0
    tm = float(max(1e-9, t_mad))
    l4 = float(max(0.0, lambda4))
    for th in TIPOS_SECUESTRADOR:
        w = float(mu.get(th, 0.0))
        if w <= 0.0:
            continue
        _mu_theta = {t: (1.0 if t == th else 0.0) for t in TIPOS_SECUESTRADOR}
        df_traj = _build_t0_longitudinal_mechanism_table(
            modelo,
            _mu_theta,
            presion_S,
            tm,
            l4,
            precision_iota,
            alpha0,
            gamma0,
            ransom_scale,
            estado_duro,
            beta_k,
            t_max=0,
            z_region=z_region,
            v_victim=v_victim,
            theta_true=th,
            f_capa=f_capa,
            s_tipo=("Duro" if estado_duro else "Laxo"),
        )
        row0 = df_traj.iloc[0]
        p_df = _build_t0_tilde_prob_report(
            modelo,
            mu_pi,
            th,
            presion_S,
            alpha0,
            gamma0,
            ransom_scale,
            estado_duro,
            beta_k,
            precision_iota,
            agent_probs,
            row0,
        )
        s_rel += w * float(max(0.0, min(1.0, float(p_df["ptilde_rel"].iloc[0]))))
        if p_det_theta_focal is None:
            s_det += w * float(max(0.0, min(1.0, float(p_df["ptilde_det"].iloc[0]))))
        s_s1 += w * float(
            max(
                0.0,
                min(
                    1.0,
                    float(
                        _mechanism_tilde_p_surv_from_t10b_S(
                            th, float(precision_iota), theta_hat, agent_probs
                        )
                    ),
                ),
            )
        )
    if p_det_theta_focal is not None:
        th0 = str(p_det_theta_focal)
        if th0 in TIPOS_SECUESTRADOR:
            _mu_theta0 = {t: (1.0 if str(t) == th0 else 0.0) for t in TIPOS_SECUESTRADOR}
            df_traj0 = _build_t0_longitudinal_mechanism_table(
                modelo,
                _mu_theta0,
                presion_S,
                tm,
                l4,
                precision_iota,
                alpha0,
                gamma0,
                ransom_scale,
                estado_duro,
                beta_k,
                t_max=0,
                z_region=z_region,
                v_victim=v_victim,
                theta_true=th0,
                f_capa=f_capa,
                s_tipo=("Duro" if estado_duro else "Laxo"),
            )
            row00 = df_traj0.iloc[0]
            p_df0 = _build_t0_tilde_prob_report(
                modelo,
                mu_pi,
                th0,
                presion_S,
                alpha0,
                gamma0,
                ransom_scale,
                estado_duro,
                beta_k,
                precision_iota,
                agent_probs,
                row00,
            )
            s_det = float(max(0.0, min(1.0, float(p_df0["ptilde_det"].iloc[0]))))
    return (s_rel, s_det, s_s1)


def _df_to_cache_records(df: pd.DataFrame) -> Tuple[Tuple[Any, ...], ...]:
    return tuple(tuple(row) for row in df.to_numpy())


def _scale_kidnapper_r_escala_df(
    df: pd.DataFrame, old_base: float, new_base: float
) -> pd.DataFrame:
    """Compatibilidad: antes escalaba ``R_escala`` por tipo; ahora R es común."""
    if df is None or df.empty or "R_escala" not in df.columns:
        return df
    return _force_common_tab12_r(df, float(new_base))


def _force_common_tab12_r(df: pd.DataFrame, R_base: float) -> pd.DataFrame:
    """Mantiene un único R común en Tabla 12; la heterogeneidad queda en costos y β."""
    if df is None or df.empty:
        return df
    out = df.copy()
    out["R_escala"] = round(float(R_base), 2)
    if "beta_k" not in out.columns:
        out["beta_k"] = float(st.session_state.get("rb_betak", 0.92))
    else:
        out["beta_k"] = pd.to_numeric(out["beta_k"], errors="coerce").fillna(
            float(st.session_state.get("rb_betak", 0.92))
        )
    return out


def _apply_fixed_tab15_cost_params(
    df: pd.DataFrame,
    modelo: ModeloSecuestro,
    *,
    R_base: float,
    gamma_oper: float,
    p_cap_base: float,
    estado_duro: bool,
) -> pd.DataFrame:
    """Fija ϕ, κ_c y ν por tipo; usa override de sesión (tab3_phi_{th} etc.) si existe."""
    if df is None or df.empty or "theta_K" not in df.columns:
        return df
    out = _force_common_tab12_r(df, float(R_base))
    for th in TIPOS_SECUESTRADOR:
        idx = out[out["theta_K"].astype(str) == str(th)].index
        if len(idx) == 0:
            continue
        ii = int(idx[0])
        coeff = _TAB15_FIXED_COST_COEFFS.get(
            str(th), {"phi": 0.5, "kappa_c": 1.0, "nu": 0.0}
        )
        phi_val = float(st.session_state.get(f"tab3_phi_{th}", coeff["phi"]))
        kc_val  = float(st.session_state.get(f"tab3_kc_{th}",  coeff["kappa_c"]))
        nu_val  = float(st.session_state.get(f"tab3_nu_{th}",  coeff["nu"]))
        out.at[ii, "phi"]     = round(phi_val, 6)
        out.at[ii, "kappa_c"] = round(kc_val,  6)
        out.at[ii, "nu"]      = round(nu_val,  6)
    return out


def _betas_lambdas_cache_items(
    betas: dict, lambdas: dict
) -> Tuple[
    Tuple[Tuple[str, Tuple[Tuple[str, float], ...]], ...],
    Tuple[Tuple[str, float], ...],
]:
    """β por tipo (anidado); λ₀ global (plano, ver ModeloSecuestro.lambdas_0)."""
    b_items = tuple(
        (str(k), tuple(sorted((str(j), float(v)) for j, v in d.items())))
        for k, d in sorted(betas.items())
        if isinstance(d, dict)
    )
    l_items = tuple(
        (str(k), float(v))
        for k, v in sorted(lambdas.items())
        if not isinstance(v, dict)
    )
    return b_items, l_items


@st.cache_data(show_spinner="Calibrando Tabla 12 (primera vez puede tardar un poco)…")
def _run_kidnapper_scale_calibration_cached(
    df_params_records: Tuple[Tuple[Any, ...], ...],
    df_columns: Tuple[str, ...],
    betas_items: Tuple[Tuple[str, Tuple[Tuple[str, float], ...]], ...],
    lambdas_items: Tuple[Tuple[str, float], ...],
    *,
    R_base: float,
    gamma_oper: float,
    p_cap_base: float,
    estado_duro: bool,
    presion_S: float,
    alpha: float,
    beta_k: float,
    gamma_lo: float,
    gamma_hi: float,
    T_horizon: int,
    finalize: bool,
    df_mu_records: Optional[Tuple[Tuple[Any, ...], ...]] = None,
    df_mu_columns: Optional[Tuple[str, ...]] = None,
) -> pd.DataFrame:
    return apply_kidnapper_scale_calibration_cached(
        df_params_records,
        df_columns,
        betas_items,
        lambdas_items,
        R_base=float(R_base),
        gamma_oper=float(gamma_oper),
        p_cap_base=float(p_cap_base),
        estado_duro=bool(estado_duro),
        presion_S=float(presion_S),
        alpha=float(alpha),
        beta_k=float(beta_k),
        gamma_lo=float(gamma_lo),
        gamma_hi=float(gamma_hi),
        T_horizon=int(T_horizon),
        finalize=bool(finalize),
        df_mu_records=df_mu_records,
        df_mu_columns=df_mu_columns,
    )


@st.cache_data(show_spinner="Calculando Tabla 15 (inducción hacia atrás)…")
def _run_kidnapper_backward_induction_cached(
    mu_records: Tuple[Tuple[Any, ...], ...],
    mu_columns: Tuple[str, ...],
    k_params_records: Tuple[Tuple[Any, ...], ...],
    k_params_columns: Tuple[str, ...],
    betas_items: Tuple[Tuple[str, Tuple[Tuple[str, float], ...]], ...],
    lambdas_items: Tuple[Tuple[str, float], ...],
    *,
    tipo_real: str,
    beta_k: float,
    R: float,
    t_mad: float,
    T: int,
    alpha_fallback: float,
    gamma_fallback: float,
    alpha_tab12: float,
    ransom_tab12: float,
) -> Tuple[pd.DataFrame, dict]:
    df_mu = pd.DataFrame(list(mu_records), columns=list(mu_columns))
    df_k = pd.DataFrame(list(k_params_records), columns=list(k_params_columns))
    betas = {k: dict(v) for k, v in betas_items}
    lambdas = {k: float(v) for k, v in lambdas_items}
    modelo = ModeloSecuestro(betas=betas, lambdas_0=lambdas)
    df_ia, meta = kidnapper_backward_induction_k_table(
        modelo,
        df_mu,
        df_k,
        tipo_real=str(tipo_real),
        beta_k=float(beta_k),
        R=float(R),
        t_mad=float(t_mad),
        T=int(T),
        alpha_fallback=float(alpha_fallback),
        gamma_fallback=float(gamma_fallback),
        alpha_tab12=float(alpha_tab12),
        ransom_tab12=float(ransom_tab12),
    )
    return df_ia, meta


@st.cache_data(show_spinner="Verificando Tabla 15 para los 4 tipos…")
def _run_tab15_all_types_validation_cached(
    mu_records: Tuple[Tuple[Any, ...], ...],
    mu_columns: Tuple[str, ...],
    k_params_records: Tuple[Tuple[Any, ...], ...],
    k_params_columns: Tuple[str, ...],
    betas_items: Tuple[Tuple[str, Tuple[Tuple[str, float], ...]], ...],
    lambdas_items: Tuple[Tuple[str, float], ...],
    *,
    R_base: float,
    beta_k: float,
    alpha: float,
    gamma_lo: float,
    T_check: int,
) -> Dict[str, Dict[str, Any]]:
    df_mu = pd.DataFrame(list(mu_records), columns=list(mu_columns))
    df_k = pd.DataFrame(list(k_params_records), columns=list(k_params_columns))
    betas = {k: dict(v) for k, v in betas_items}
    lambdas = {k: float(v) for k, v in lambdas_items}
    modelo = ModeloSecuestro(betas=betas, lambdas_0=lambdas)
    return validate_tab15_all_types(
        df_k,
        modelo,
        df_mu,
        R_base=float(R_base),
        beta_k=float(beta_k),
        alpha=float(alpha),
        gamma_lo=float(gamma_lo),
        T_check=int(T_check),
    )


def _enforce_tab12_continuar_defaults(
    df_params: pd.DataFrame,
    modelo: ModeloSecuestro,
    *,
    R_base: float,
    gamma_oper: float,
    p_cap_base: float,
    estado_duro: bool,
) -> tuple[pd.DataFrame, bool]:
    """Eleva Tabla 12 a los pisos por tipo que hacen viable Continuar en τ=1."""
    if df_params is None or df_params.empty or "theta_K" not in df_params.columns:
        return df_params, False
    out = df_params.copy()
    if "beta_k" not in out.columns:
        out["beta_k"] = float(st.session_state.get("rb_betak", 0.92))
    changed = False
    for th in TIPOS_SECUESTRADOR:
        idx = out[out["theta_K"].astype(str) == str(th)].index
        if len(idx) == 0:
            continue
        ii = int(idx[0])
        r_target = float(R_base)
        try:
            r_current = float(out.at[ii, "R_escala"])
        except (KeyError, TypeError, ValueError):
            r_current = 0.0
        if not np.isfinite(r_current) or abs(r_current - r_target) > 1e-9:
            par = derive_kidnapper_structural_params(
                modelo,
                str(th),
                float(p_cap_base),
                bool(estado_duro),
                R_scale=float(R_base),
                gamma_oper=float(gamma_oper),
            )
            out.at[ii, "kappa_rel"] = round(float(par["kappa_rel"]), 3)
            out.at[ii, "eta"] = round(float(par["eta"]), 3)
            out.at[ii, "F_cap"] = round(float(par["F_cap"]), 3)
            out.at[ii, "phi"] = round(float(par["phi"]), 4)
            out.at[ii, "kappa_c"] = round(float(par["kappa_c"]), 3)
            out.at[ii, "nu"] = round(float(par["nu"]), 4)
            out.at[ii, "p_cap_tilde"] = round(float(par["p_cap"]), 4)
            out.at[ii, "R_escala"] = round(float(r_target), 2)
            changed = True
    if changed:
        out = refresh_kidnapper_endogenous_columns(
            out, modelo, float(gamma_oper), float(gamma_oper), alpha=float(st.session_state.get("h0_alpha", 0.0))
        )
    return out, bool(changed)


def _tab15_tau1_and_switch(
    modelo: ModeloSecuestro,
    df_mu: pd.DataFrame,
    df_k: pd.DataFrame,
    *,
    tipo_real: str,
    beta_k: float,
    R_base: float,
    t_mad: float,
    T: int,
    alpha: float,
    gamma: float,
    ransom: float,
) -> tuple[bool, Optional[int], str]:
    meta = kidnapper_backward_tau1_switch_fast(
        modelo,
        df_mu,
        df_k,
        tipo_real=str(tipo_real),
        beta_k=float(beta_k),
        R=float(R_base),
        t_mad=float(t_mad),
        T=int(T),
        alpha_fallback=float(alpha),
        gamma_fallback=float(gamma),
        alpha_tab12=float(alpha),
        ransom_tab12=float(ransom),
    )
    opt1 = str(meta.get("opcion_tau1", ""))
    sw = meta.get("primer_tau_backward")
    sw_i = int(sw) if sw is not None else None
    return bool(meta.get("ok_tau1", opt1 == "Continuar (a_cont)")), sw_i, opt1


def _tab15_t1t2_and_switch(
    modelo: ModeloSecuestro,
    df_mu: pd.DataFrame,
    df_k: pd.DataFrame,
    *,
    tipo_real: str,
    beta_k: float,
    R_base: float,
    t_mad: float,
    T: int,
    alpha: float,
    gamma: float,
    ransom: float,
) -> tuple[bool, bool, Optional[int], str, str]:
    meta = kidnapper_backward_tau1_switch_fast(
        modelo,
        df_mu,
        df_k,
        tipo_real=str(tipo_real),
        beta_k=float(beta_k),
        R=float(R_base),
        t_mad=float(t_mad),
        T=int(T),
        alpha_fallback=float(alpha),
        gamma_fallback=float(gamma),
        alpha_tab12=float(alpha),
        ransom_tab12=float(ransom),
    )
    opt1 = str(meta.get("opcion_tau1", ""))
    opt2 = str(meta.get("opcion_tau2", ""))
    sw = meta.get("primer_tau_backward")
    sw_i = int(sw) if sw is not None else None
    ok1 = bool(meta.get("ok_tau1", opt1 == "Continuar (a_cont)"))
    ok2 = bool(meta.get("ok_tau2", opt2 == "Continuar (a_cont)"))
    return ok1, ok2, sw_i, opt1, opt2


def _tab15_tau1_and_switch_full(
    modelo: ModeloSecuestro,
    df_mu: pd.DataFrame,
    df_k: pd.DataFrame,
    *,
    tipo_real: str,
    beta_k: float,
    R_base: float,
    t_mad: float,
    T: int,
    alpha: float,
    gamma: float,
    ransom: float,
) -> tuple[bool, Optional[int], str]:
    df_ia, meta = kidnapper_backward_induction_k_table(
        modelo,
        df_mu,
        df_k,
        tipo_real=str(tipo_real),
        beta_k=float(beta_k),
        R=float(R_base),
        t_mad=float(t_mad),
        T=int(T),
        alpha_fallback=float(alpha),
        gamma_fallback=float(gamma),
        alpha_tab12=float(alpha),
        ransom_tab12=float(ransom),
    )
    row1 = df_ia.loc[df_ia["t"].astype(int) == 1] if not df_ia.empty else pd.DataFrame()
    opt1 = str(row1.iloc[0]["opcion_BW"]) if not row1.empty else ""
    sw = meta.get("primer_tau_backward")
    sw_i = int(sw) if sw is not None else None
    return opt1 == "Continuar (a_cont)", sw_i, opt1


def _ensure_tab15_focal_switch_after_tau1(
    df_params: pd.DataFrame,
    modelo: ModeloSecuestro,
    df_mu: pd.DataFrame,
    *,
    tipo_real: str,
    beta_k: float,
    R_base: float,
    t_mad: float,
    T: int,
    alpha: float,
    gamma: float,
    min_switch_tau: int = 1,
    target_switch_tau: Optional[int] = None,
    max_switch_tau: int = 500,
    full_constraint: bool = False,
) -> tuple[pd.DataFrame, float, dict]:
    """Ajusta costos Tabla 12(θ*) con R común para τ=1=Continuar y cambio antes de 500."""
    out = _force_common_tab12_r(df_params, float(R_base))
    meta = {"changed": False, "ok_tau1": False, "primer_tau": None, "opcion_tau1": ""}
    idx = out[out["theta_K"].astype(str) == str(tipo_real)].index
    if len(idx) == 0 or df_mu is None or df_mu.empty:
        return out, float(R_base), meta
    ii = int(idx[0])
    try:
        phi0 = float(out.at[ii, "phi"])
    except (KeyError, TypeError, ValueError):
        phi0 = 1.0
    try:
        kc0 = float(out.at[ii, "kappa_c"])
    except (KeyError, TypeError, ValueError):
        kc0 = 1.0
    try:
        nu0 = float(out.at[ii, "nu"])
    except (KeyError, TypeError, ValueError):
        nu0 = 0.0

    switch_upper = int(max(2, min(int(max_switch_tau), int(T) + 1)))
    target_tau = int(
        np.clip(
            int(target_switch_tau if target_switch_tau is not None else switch_upper - 1),
            max(2, int(min_switch_tau) + 1),
            max(2, switch_upper - 1),
        )
    )

    def _valid_switch(sw_val: Any) -> bool:
        return sw_val is not None and max(1, int(min_switch_tau)) < int(sw_val) < int(switch_upper)

    def _set_cost_scale(cost_mult: float, kc_mult: float) -> None:
        out.at[ii, "R_escala"] = round(float(R_base), 2)
        out.at[ii, "phi"] = round(max(1e-9, phi0 * float(cost_mult)), 6)
        out.at[ii, "nu"] = round(max(0.0, nu0 * float(cost_mult)), 6)
        out.at[ii, "kappa_c"] = round(max(1e-9, kc0 * float(kc_mult)), 6)

    def _eval() -> tuple[bool, Optional[int], str]:
        _eval_fn = _tab15_tau1_and_switch_full if bool(full_constraint) else _tab15_tau1_and_switch
        return _eval_fn(
            modelo, df_mu, out,
            tipo_real=tipo_real, beta_k=beta_k, R_base=R_base, t_mad=t_mad,
            T=T, alpha=alpha, gamma=gamma, ransom=float(R_base),
        )

    best_any: tuple[float, bool, Optional[int], str, float, float] = (float("inf"), False, None, "", 1.0, 1.0)
    best_valid: Optional[tuple[float, bool, Optional[int], str, float, float]] = None

    def _consider_candidate(cmult: float, kmult: float) -> None:
        nonlocal best_any, best_valid
        _set_cost_scale(float(cmult), float(kmult))
        ok_try, sw_try, opt_try = _eval()
        if not ok_try:
            return
        sw_pen = abs(int(sw_try) - int(target_tau)) if sw_try is not None else 10_000
        score_any = float(sw_pen)
        if score_any < best_any[0]:
            best_any = (score_any, ok_try, sw_try, opt_try, float(cmult), float(kmult))
        if _valid_switch(sw_try):
            score_valid = float(abs(int(sw_try) - int(target_tau)))
            if best_valid is None or score_valid < best_valid[0]:
                best_valid = (score_valid, ok_try, sw_try, opt_try, float(cmult), float(kmult))

    # Búsqueda gruesa: cubre todo el dominio sin ejecutar cientos de BI completas.
    for cmult in np.geomspace(0.01, 24.0, num=18):
        for kmult in (0.55, 0.85, 1.0, 1.35, 2.05):
            _consider_candidate(float(cmult), float(kmult))

    # Refinamiento local alrededor del mejor candidato encontrado.
    _, _, _, _, cmid, kmid = best_valid if best_valid is not None else best_any
    c_lo = max(0.005, float(cmid) / 2.2)
    c_hi = min(36.0, float(cmid) * 2.2)
    k_lo = max(0.25, float(kmid) / 1.7)
    k_hi = min(3.00, float(kmid) * 1.7)
    for cmult in np.geomspace(c_lo, c_hi, num=11):
        for kmult in np.linspace(k_lo, k_hi, num=5):
            _consider_candidate(float(cmult), float(kmult))

    # Micro-refinamiento si ya hay una solución válida: acerca el cambio al objetivo.
    if best_valid is not None:
        _, _, _, _, cmid, kmid = best_valid
        for cmult in np.geomspace(max(0.005, float(cmid) / 1.35), min(36.0, float(cmid) * 1.35), num=7):
            for kmult in np.linspace(max(0.25, float(kmid) / 1.25), min(3.00, float(kmid) * 1.25), num=3):
                _consider_candidate(float(cmult), float(kmult))

    _, ok, sw, opt, cmult, kmult = best_valid if best_valid is not None else best_any
    _set_cost_scale(float(cmult), float(kmult))
    meta["changed"] = bool(
        ok and (abs(float(cmult) - 1.0) > 1e-6 or abs(float(kmult) - 1.0) > 1e-6)
    )
    meta.update({
        "ok_tau1": bool(ok),
        "primer_tau": sw,
        "opcion_tau1": opt,
        "target_tau": int(target_tau),
        "switch_upper": int(switch_upper),
    })
    return out, float(R_base), meta


def _tab15_calibrated_switch_summary(
    df_params: pd.DataFrame,
    modelo: ModeloSecuestro,
    df_mu: pd.DataFrame,
    *,
    beta_k: float,
    R_base: float,
    t_mad: float,
    T: int,
    alpha: float,
    gamma: float,
) -> tuple[pd.DataFrame, dict]:
    out = _force_common_tab12_r(df_params.copy(), float(R_base))
    ordered_types = ["DC", "PAR", "ELN", "FARC"]
    target_by_type = {
        th: int(min(max(2, int(T) - 1), int(_TAB15_SWITCH_TARGETS.get(th, 40))))
        for th in ordered_types
    }

    def _full_summary(df_check: pd.DataFrame) -> dict[str, dict[str, Any]]:
        checked: dict[str, dict[str, Any]] = {}
        for th in ordered_types:
            previous_switches = [
                int(checked[p]["primer_tau"])
                for p in ordered_types[:ordered_types.index(str(th))]
                if p in checked and checked[p].get("primer_tau") is not None
            ]
            min_sw = max([3] + previous_switches)
            ok_tau1, sw, opt = _tab15_tau1_and_switch_full(
                modelo,
                df_mu,
                df_check,
                tipo_real=str(th),
                beta_k=float(beta_k),
                R_base=float(R_base),
                t_mad=float(t_mad),
                T=int(T),
                alpha=float(alpha),
                gamma=float(gamma),
                ransom=float(R_base),
            )
            ok_switch = sw is not None and int(sw) > int(min_sw) and int(sw) < int(min(100, T + 1))
            checked[str(th)] = {
                "ok_tau1": bool(ok_tau1),
                "ok_switch": bool(ok_switch),
                "ok_switch_before_100": bool(ok_switch),
                "primer_tau": int(sw) if sw is not None else None,
                "opcion_tau1": "Continuar (a_cont)" if ok_tau1 else str(opt or "—"),
                "R_escala": round(float(R_base), 2),
                "tau_min_requerido": int(min_sw),
                "tau_objetivo": int(target_by_type[str(th)]),
                "verificado_con_tabla_completa": True,
                "costos_fijos_tabla12": True,
            }
        return checked

    summary_full = _full_summary(out)
    return out, summary_full


def _rb_hashable_float_seq(seq: Optional[Any]) -> Optional[Tuple[float, ...]]:
    if seq is None:
        return None
    return tuple(round(float(x), 8) for x in seq)


def _rb_voice_path_digest(path: Optional[Any]) -> str:
    if path is None:
        return ""
    try:
        return hashlib.md5(
            json.dumps(path, sort_keys=True, default=str).encode()
        ).hexdigest()
    except (TypeError, ValueError):
        return ""


def _rb_mu_traj_signature(
    *,
    t_max: int,
    mu0: dict,
    m_obs: str,
    d_obs: int,
    presion_S: float,
    z_region: str,
    v_victim: str,
    f_capa: str,
    s_tipo: str,
    alpha: float,
    gamma: float,
    p_det: float,
    zeta: Tuple[float, float, float, float],
    estado_rescata: bool,
    t_mad: float,
    lambda4: float,
    omega_voz: float,
    voice_seed: int,
    tipo_emit: str,
    voice_emit_from_mu: bool,
    voice_digest: str,
    alpha_by_t: Optional[Tuple[float, ...]],
    gamma_by_t: Optional[Tuple[float, ...]],
    epi_tag: Tuple[Any, ...],
) -> Tuple[Any, ...]:
    return (
        int(_TAB14_LIKELIHOOD_VERSION),
        int(t_max),
        tuple(sorted((str(k), round(float(v), 8)) for k, v in mu0.items())),
        str(m_obs),
        int(d_obs),
        round(float(presion_S), 8),
        str(z_region),
        str(v_victim),
        str(f_capa),
        str(s_tipo),
        round(float(alpha), 8),
        round(float(gamma), 8),
        round(float(p_det), 8),
        zeta,
        bool(estado_rescata),
        round(float(t_mad), 6),
        round(float(lambda4), 8),
        round(float(omega_voz), 8),
        int(voice_seed),
        str(tipo_emit),
        bool(voice_emit_from_mu),
        str(voice_digest),
        alpha_by_t,
        gamma_by_t,
        epi_tag,
    )


def _rb_attach_mu_traj_epi_columns(
    df_mu: pd.DataFrame,
    *,
    tipo_real: str,
    t0_gamma: float,
    t0_alpha: float,
    iota_t0: float,
    kc_k12: dict,
    ps_k12: dict,
    pf_k12: dict,
    p3_mdg_agent: dict,
) -> pd.DataFrame:
    if df_mu is None or df_mu.empty:
        return df_mu
    out = df_mu.copy()
    _th = str(tipo_real)
    _ep_pay: list[float] = []
    _ep_pcap: list[float] = []
    _ep_pay_by_theta: dict[str, list[float]] = {str(th): [] for th in TIPOS_SECUESTRADOR}
    _ep_pcap_by_theta: dict[str, list[float]] = {str(th): [] for th in TIPOS_SECUESTRADOR}
    for _, _rw in out.iterrows():
        _t_row = int(_rw.get("t", 0))
        _gamma_row = float(_rw.get("gamma_t", t0_gamma))
        _alpha_row = float(_rw.get("alpha_t", t0_alpha))
        _pdet_row = float(
            1.0
            / (
                1.0
                + np.exp(
                    -(
                        float(st.session_state.get(f"cal_eta0_pdet_{_th}", _ETA0_PDET_DEFAULTS.get(_th, -2.0)))
                        + float(st.session_state.get("cal_eta1_pdet", 1.0)) * _alpha_row
                        + float(st.session_state.get("cal_eta2_pdet", 1.0)) * _gamma_row
                    )
                )
            )
        )
        for _th_i in TIPOS_SECUESTRADOR:
            _th_i = str(_th_i)
            _q_pay_i = float(
                _expected_outcomes_over_tilde_A_hazards(
                    _th_i,
                    _t_row,
                    _alpha_row,
                    _gamma_row,
                    _pdet_row,
                    kc_k12,
                    ps_k12,
                    pf_k12,
                    z_region=str(st.session_state.get("z_region", "Andina")),
                    v_victim=str(st.session_state.get("v_victim", "Privado")),
                    f_capa=str(st.session_state.get("f_capa", "Alta Riqueza")),
                    s_tipo=str(st.session_state.get("s_tipo", "Duro")),
                )["pay"]
            )
            _pcap_i = float(
                _mechanism_tilde_p_cap_from_t10b_S(
                    _th_i, _alpha_row, _gamma_row, p3_mdg_agent
                )
            )
            _ep_pay_by_theta[_th_i].append(round(_q_pay_i, 6))
            _ep_pcap_by_theta[_th_i].append(round(_pcap_i, 6))
        _ep_pay.append(_ep_pay_by_theta.get(_th, [float("nan")])[-1])
        _ep_pcap.append(_ep_pcap_by_theta.get(_th, [float("nan")])[-1])
    out["Epi_pay_Qcont_mu"] = _ep_pay
    out["Epi_pcap_Qcap"] = _ep_pcap
    for _th_i in TIPOS_SECUESTRADOR:
        _th_i = str(_th_i)
        out[f"Epi_pay_Qcont_{_th_i}"] = _ep_pay_by_theta[_th_i]
        out[f"Epi_pcap_Qcap_{_th_i}"] = _ep_pcap_by_theta[_th_i]
    return out


# --- CARGA DE DATOS GEOGRÁFICOS ---

@st.cache_data
def load_muni_mapping():
    path = os.path.join(os.path.dirname(__file__), 'muni_mapping.json')
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return {}

MUNI_MAPPING = load_muni_mapping()

@st.cache_data
def load_zone_reference_v4():
    path = os.path.join(os.path.dirname(__file__), "Data_CMH.csv")
    if not os.path.exists(path):
        return {}

    df = pd.read_csv(path, dtype={"CódigoDANEdeMunicipio": str})
    zone_by_code = {}
    for _, row in df.dropna(subset=["CódigoDANEdeMunicipio", "Zona_Geografica"]).iterrows():
        muni = row.get("Municipio", "")
        dpto = row.get("Departamento", "")
        
        # Corrección obligatoria de la región
        corrected_zone = get_corrected_region(muni, dpto)
        
        code = str(row.get("CódigoDANEdeMunicipio", "")).strip()
        if code.isdigit():
            code = code.zfill(5)
        if len(code) == 5:
            display_zone = REGION_MAP_INV.get(corrected_zone, corrected_zone)
            zone_by_code.setdefault(code, display_zone)
    return zone_by_code

ARCHIPIELAGO_DPTO_CODE = "88"
ARCHIPIELAGO_INSET_CENTER = (-79.15, 12.35)
ARCHIPIELAGO_INSET_SCALE = 16.0

def collect_points(coords, points):
    if coords and isinstance(coords[0], (int, float)):
        points.append((coords[0], coords[1]))
        return
    for item in coords:
        collect_points(item, points)

def transform_coords(coords, source_center, target_center, scale):
    if coords and isinstance(coords[0], (int, float)):
        lon, lat = coords[:2]
        return [
            target_center[0] + (lon - source_center[0]) * scale,
            target_center[1] + (lat - source_center[1]) * scale
        ] + coords[2:]
    return [transform_coords(item, source_center, target_center, scale) for item in coords]

def geometry_center(geometry):
    points = []
    collect_points(geometry.get("coordinates", []), points)
    if not points:
        return None
    lons = [point[0] for point in points]
    lats = [point[1] for point in points]
    return ((min(lons) + max(lons)) / 2, (min(lats) + max(lats)) / 2)

def load_detailed_archipielago_features():
    path = os.path.join(os.path.dirname(__file__), "san_andres_providencia_arcgis.geojson")
    if not os.path.exists(path):
        return None

    with open(path, "r") as f:
        detailed_geojson = json.load(f)

    features = []
    for feature in detailed_geojson.get("features", []):
        props = feature.get("properties", {})
        divipola = str(props.get("divipola", ""))
        municipio = props.get("Municipios", "")
        departamento = props.get("Departamentos", "")
        features.append({
            "type": "Feature",
            "properties": {
                "DPTO_CCDGO": ARCHIPIELAGO_DPTO_CODE,
                "MPIO_CCDGO": divipola[-3:],
                "MPIO_CNMBR": municipio,
                "MPIO_CCNCT": divipola,
                "DPTO_CNMBR": departamento
            },
            "geometry": feature.get("geometry")
        })
    return features

@st.cache_data
def load_municipio_stats():
    path = os.path.join(os.path.dirname(__file__), "Data_CMH.csv")
    if not os.path.exists(path):
        return pd.DataFrame()

    df = pd.read_csv(path, dtype={"CódigoDANEdeMunicipio": str})
    df["CódigoDANEdeMunicipio"] = df["CódigoDANEdeMunicipio"].str.zfill(5)

    return df.groupby("CódigoDANEdeMunicipio").agg(
        Total=("IDCaso", "count"),
        Primero=("Año", "min"),
        Ultimo=("Año", "max"),
    ).reset_index()

@st.cache_data
def load_municipio_geojson_v4():
    path = os.path.join(os.path.dirname(__file__), "co_2018_MGN_MPIO_POLITICO.geojson")
    if not os.path.exists(path):
        return None, pd.DataFrame()

    with open(path, "r") as f:
        geojson = json.load(f)

    detailed_archipielago_features = load_detailed_archipielago_features()
    if detailed_archipielago_features:
        geojson["features"] = [
            feature for feature in geojson.get("features", [])
            if feature.get("properties", {}).get("DPTO_CCDGO") != ARCHIPIELAGO_DPTO_CODE
        ] + detailed_archipielago_features

    features = geojson.get("features", [])
    archipielago_features = [
        feature for feature in features
        if feature.get("properties", {}).get("DPTO_CCDGO") == ARCHIPIELAGO_DPTO_CODE
    ]
    archipielago_points = []
    for feature in archipielago_features:
        collect_points(feature.get("geometry", {}).get("coordinates", []), archipielago_points)

    if archipielago_points:
        lons = [point[0] for point in archipielago_points]
        lats = [point[1] for point in archipielago_points]
        source_center = ((min(lons) + max(lons)) / 2, (min(lats) + max(lats)) / 2)
        display_features = [
            feature for feature in features
            if feature.get("properties", {}).get("DPTO_CCDGO") != ARCHIPIELAGO_DPTO_CODE
        ]
        for feature in archipielago_features:
            inset_feature = copy.deepcopy(feature)
            feature_center = geometry_center(inset_feature.get("geometry", {}))
            if feature_center is None:
                continue
            target_center = (
                ARCHIPIELAGO_INSET_CENTER[0] + (feature_center[0] - source_center[0]),
                ARCHIPIELAGO_INSET_CENTER[1] + (feature_center[1] - source_center[1])
            )
            props = inset_feature["properties"]
            props["MPIO_CCNCT"] = f"INSET_{props.get('MPIO_CCNCT')}"
            props["MPIO_CNMBR"] = f"{props.get('MPIO_CNMBR')} (ampliado)"
            inset_feature["geometry"]["coordinates"] = transform_coords(
                inset_feature["geometry"]["coordinates"],
                feature_center,
                target_center,
                ARCHIPIELAGO_INSET_SCALE
            )
            display_features.append(inset_feature)
        geojson["features"] = display_features

    stats_df = load_municipio_stats()

    rows = []
    for feature in geojson.get("features", []):
        props = feature.get("properties", {})
        code = props.get("MPIO_CCNCT")
        municipio = props.get("MPIO_CNMBR", "")
        departamento = props.get("DPTO_CNMBR", "")
        region = get_corrected_region(municipio, departamento)

        muni_stats = stats_df[stats_df["CódigoDANEdeMunicipio"] == code]
        total, primero, ultimo = 0, 0, 0
        if not muni_stats.empty:
            t0 = muni_stats["Total"].iloc[0]
            p0 = muni_stats["Primero"].iloc[0]
            u0 = muni_stats["Ultimo"].iloc[0]
            try:
                total = int(t0) if pd.notna(t0) else 0
            except (ValueError, TypeError):
                total = 0
            try:
                primero = int(p0) if pd.notna(p0) else 0
            except (ValueError, TypeError):
                primero = 0
            try:
                ultimo = int(u0) if pd.notna(u0) else 0
            except (ValueError, TypeError):
                ultimo = 0

        rows.append({
            "feature_id": code,
            "Municipio": municipio.title(),
            "Departamento": departamento.title(),
            "Region": region,
            "Total Secuestros": total,
            "Primer Año": primero,
            "Último Año": ultimo,
        })

    return geojson, pd.DataFrame(rows)

def update_dynamic_priors():
    if 'z_region' in st.session_state:
        st.session_state.z_region_cal = st.session_state.z_region
    z = st.session_state.z_region
    v = st.session_state.v_victim
    scores = {}
    for t in TIPOS_SECUESTRADOR:
        d = COEF_DELTA.get(t, 0.0)
        e = COEF_ETA.get(z, {}).get(t, 0.0)
        x = COEF_XI.get(v, {}).get(t, 0.0)
        scores[t] = d + e + x
    exp_scores = {t: np.exp(s) for t, s in scores.items()}
    sum_exp = sum(exp_scores.values())
    new_priors = [ (exp_scores[t] / sum_exp) * 100 for t in TIPOS_SECUESTRADOR ]
    st.session_state.dynamic_priors = new_priors

# Inicializar sesión
if 'dynamic_priors' not in st.session_state:
    st.session_state.dynamic_priors = [25.0, 20.0, 20.0, 35.0]
if 'manual_priors' not in st.session_state:
    st.session_state.manual_priors = [25.0, 25.0, 25.0, 25.0]
if 'prior_mode' not in st.session_state:
    st.session_state.prior_mode = "Modelo"
if 'final_priors' not in st.session_state:
    st.session_state.final_priors = [25.0, 20.0, 20.0, 35.0]
if 'z_region' not in st.session_state:
    st.session_state.z_region = "Andina"
if 'v_victim' not in st.session_state:
    st.session_state.v_victim = "Privado"


def _structural_defaults_from_modelo():
    """Copias profundas de β y λ₀ ilustrativos (Mechanism.tex / ModeloSecuestro)."""
    m0 = ModeloSecuestro()
    return copy.deepcopy(m0.betas), copy.deepcopy(m0.lambdas_0)


if "cal_betas_dict" not in st.session_state:
    _bb, _ll = _structural_defaults_from_modelo()
    st.session_state.cal_betas_dict = _bb
    st.session_state.cal_lambdas_dict = _ll
if st.session_state.get("structural_lambda_signature_version") != _STRUCTURAL_LAMBDA_SIGNATURE_VERSION:
    _bb, _ll = _structural_defaults_from_modelo()
    st.session_state.cal_betas_dict = _bb
    st.session_state.cal_lambdas_dict = _ll
    st.session_state["structural_lambda_signature_version"] = _STRUCTURAL_LAMBDA_SIGNATURE_VERSION
    for _cache_key in (
        "dynamic_cycles52",
        "dynamic_cycles_diag52",
        "dynamic_cycles_stop52",
        "tab15_mu_snapshot",
        "tab15_k_params_calibrated",
        "tab15_last_validation",
        "rb_mu_traj_snapshot",
        "rb_mu_traj_sig",
    ):
        st.session_state.pop(_cache_key, None)
if "cal_presion_S" not in st.session_state:
    st.session_state.cal_presion_S = 0.5
if "cal_alpha_star" not in st.session_state:
    st.session_state.cal_alpha_star = 0.35
if "cal_gamma_star" not in st.session_state:
    st.session_state.cal_gamma_star = float(st.session_state.cal_presion_S)
if "cal_t_hazard" not in st.session_state:
    st.session_state.cal_t_hazard = 1
if "cal_rho_mat" not in st.session_state:
    st.session_state.cal_rho_mat = 0.04
if "cal_T_mad" not in st.session_state:
    st.session_state.cal_T_mad = 5.0
if st.session_state.get("cal_T_mad_default_version") != "tmad_5":
    st.session_state.cal_T_mad = 5.0
    st.session_state["cal_T_mad_default_version"] = "tmad_5"
if "cal_lambda_4" not in st.session_state:
    st.session_state.cal_lambda_4 = 0.0005
if st.session_state.get("cal_lambda_4_default_version") != "lambda4_0p0005":
    if abs(float(st.session_state.get("cal_lambda_4", 0.002)) - 0.002) < 1e-12:
        st.session_state.cal_lambda_4 = 0.0005
    st.session_state["cal_lambda_4_default_version"] = "lambda4_0p0005"
# Tabla 3 · p_det,t(θ_K) = Λ(η₀(θ_K)+η₁α*+η₂γ*) · Mechanism.tex
# η₀(θ_K) es tipo-específico: distintas organizaciones difieren en detectabilidad basal
_ETA0_PDET_DEFAULTS = {"DC": -1.5, "PAR": -2.0, "ELN": -2.5, "FARC": -2.8}
for _th_init in ("DC", "PAR", "ELN", "FARC"):
    if f"cal_eta0_pdet_{_th_init}" not in st.session_state:
        st.session_state[f"cal_eta0_pdet_{_th_init}"] = _ETA0_PDET_DEFAULTS[_th_init]


def _pdet_eta0_for_theta(theta_k: str) -> float:
    theta_key = str(theta_k)
    return float(st.session_state.get(
        f"cal_eta0_pdet_{theta_key}",
        _ETA0_PDET_DEFAULTS.get(theta_key, -2.0),
    ))


def _pdet_logit_prob(theta_k: str, alpha_v: float, gamma_v: float) -> float:
    idx = (
        _pdet_eta0_for_theta(theta_k)
        + float(st.session_state.get("cal_eta1_pdet", 1.0)) * float(alpha_v)
        + float(st.session_state.get("cal_eta2_pdet", 1.0)) * float(gamma_v)
    )
    return float(1.0 / (1.0 + np.exp(-idx)))

if "cal_eta1_pdet" not in st.session_state:
    st.session_state.cal_eta1_pdet = 1.0
if "cal_eta2_pdet" not in st.session_state:
    st.session_state.cal_eta2_pdet = 1.0
# Tabla 4 · ec. 37–38 · supervivencia bajo rescate focal (eq:iota-precision-mode, eq:p-surv-rescue-logit-ajustado)
_SURV_ALPHA0_SENSITIVE_DEFAULTS = {
    "DC": -5.25,
    "PAR": -5.15,
    "ELN": -5.05,
    "FARC": -4.95,
}
_SURV_BETA_R_SENSITIVE_DEFAULT = 7.0
if "cal_surv_alpha0" not in st.session_state:
    st.session_state.cal_surv_alpha0 = dict(_SURV_ALPHA0_SENSITIVE_DEFAULTS)
if "cal_surv_beta_R" not in st.session_state:
    st.session_state.cal_surv_beta_R = float(_SURV_BETA_R_SENSITIVE_DEFAULT)
if st.session_state.get("cal_surv_sensitivity_version") != "mu_sensitive_v2":
    st.session_state.cal_surv_alpha0 = dict(_SURV_ALPHA0_SENSITIVE_DEFAULTS)
    st.session_state.cal_surv_beta_R = float(_SURV_BETA_R_SENSITIVE_DEFAULT)
    st.session_state.cal_surv_sensitivity_version = "mu_sensitive_v2"
# Tabla 6 · p_cap = Λ(δ_a+c₀+c_α α*+c_γ γ*+c_S) · Mechanism.tex (eq:p-cap-tecnica)
if "cal_pcap_delta_a" not in st.session_state:
    st.session_state.cal_pcap_delta_a = 0.0
if "cal_pcap_c_S" not in st.session_state:
    st.session_state.cal_pcap_c_S = 0.0
if "cal_pcap_params" not in st.session_state:
    st.session_state.cal_pcap_params = _default_cal_pcap_params()
    if "cal_pcap_c0" in st.session_state:
        _leg0 = float(st.session_state.cal_pcap_c0)
        _leg_a = float(st.session_state.get("cal_pcap_c_alpha", 1.5))
        _leg_g = float(st.session_state.get("cal_pcap_c_gamma", 1.2))
        for _tk in TIPOS_SECUESTRADOR:
            st.session_state.cal_pcap_params[_tk] = {
                "c0": _leg0,
                "c_alpha": _leg_a,
                "c_gamma": _leg_g,
            }
if "cal_voz_params" not in st.session_state:
    st.session_state.cal_voz_params = _default_cal_voz_params()
if "cal_voz_pi_call" not in st.session_state:
    st.session_state.cal_voz_pi_call = _default_cal_voz_pi_call()
if "cal_voz_omega" not in st.session_state:
    st.session_state.cal_voz_omega = 0.2
if st.session_state.get("cal_voz_default_version") != "voz_base_0p20":
    st.session_state.cal_voz_pi_call = _default_cal_voz_pi_call()
    st.session_state.cal_voz_omega = 0.2
    for _th_pi0 in TIPOS_SECUESTRADOR:
        st.session_state[f"voz_pi_{_th_pi0}"] = 0.2
    st.session_state["voz_omega"] = 0.2
    st.session_state["cal_voz_default_version"] = "voz_base_0p20"
if "cal_voz_osc_bundle" not in st.session_state:
    st.session_state.cal_voz_osc_bundle = None
if "cal_voz_osc_th" not in st.session_state:
    st.session_state.cal_voz_osc_th = None
# MDG · η_cal,i por agente i ∈ {K,S,F} (Mechanism.tex · ec. (26)–(27), temperatura / calendario)
if "cal_mdg_eta_cal_by_i" not in st.session_state:
    st.session_state.cal_mdg_eta_cal_by_i = {"K": 0.048, "S": 0.056, "F": 0.043}
if st.session_state.get("cal_mdg7_alignment_version") != int(_MDG7_ALIGNMENT_CALIB_VERSION):
    st.session_state["mdg_T0_K"] = 0.32
    st.session_state["mdg_T0_S"] = 0.30
    st.session_state["mdg_T0_F"] = 0.28
    st.session_state["mdg_cbar_K"] = 0.02
    st.session_state["mdg_cbar_S"] = 0.02
    st.session_state["mdg_cbar_F"] = 0.02
    st.session_state.cal_mdg_eta_cal_by_i = {"K": 0.070, "S": 0.075, "F": 0.065}
    for _mdg_code_reset in ("K", "S", "F"):
        for _idx_reset in range(6):
            st.session_state.pop(f"mdg7_p_{_mdg_code_reset}_{_idx_reset}", None)
    st.session_state["cal_mdg7_alignment_version"] = int(_MDG7_ALIGNMENT_CALIB_VERSION)

st.title(_ui_text("⚖️ Identification of Rational Types", "⚖️ Identificación de tipos racionales"))
st.markdown(
    _ui_text(
        "Prepared by: Prof. Humberto Bernal, Ph.D. in Economics, Universidad de los Andes, Colombia",
        "Elaborado por: Prof. Humberto Bernal, Ph.D. en Economía, Universidad de los Andes, Colombia",
    )
)
_header_language_choice = st.radio(
    _ui_text("Language", "Idioma"),
    _LANG_OPTIONS,
    index=_LANG_OPTIONS.index(st.session_state.get("app_language", "English")),
    horizontal=True,
    key="app_language_header_choice",
    help=_ui_text(
        "Select English for the jury-facing version or Spanish for the original version.",
        "Seleccione inglés para la versión dirigida al jurado o español para la versión original.",
    ),
)
if _header_language_choice != st.session_state.get("app_language"):
    st.session_state["app_language"] = _header_language_choice
    st.rerun()
_inject_page_translator()

# --- SECCIÓN GLOBAL: SIMULACIÓN E INCIDENTE ---
with st.container():
    if "mechanism_started" not in st.session_state:
        st.session_state["mechanism_started"] = False
    st.markdown(
        """
        <style>
        div[data-testid="stButton"] > button[kind="primary"] {
            background-color: #16a34a !important;
            border-color: #15803d !important;
            color: white !important;
        }
        div[data-testid="stButton"] > button[kind="primary"]:hover {
            background-color: #15803d !important;
            border-color: #166534 !important;
            color: white !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    _sim_title_col, _sim_btn_col, _sim_dyn_col, _sim_m_mode_col, _sim_m0_col = st.columns(
        [4, 1, 1, 1, 1],
        vertical_alignment="center",
    )
    with _sim_title_col:
        st.markdown("### Simulación e Incidente")
    with _sim_btn_col:
        _run_full_process_now = st.button(
            "Iniciar proceso",
            key="btn_iniciar_proceso_global",
            type="primary",
            use_container_width=True,
            help=(
                "Ejecuta el incidente con los parámetros actuales y fuerza el cálculo "
                "de las tablas dependientes, incluida Tabla 14 y Tabla 15."
            ),
        )
    with _sim_dyn_col:
        _run_dynamic_blocks_now = st.button(
            "Avanzar ciclos",
            key="btn_avanzar_ciclos_global",
            type="primary",
            disabled=not bool(st.session_state.get("mechanism_started", False)),
            use_container_width=True,
            help=(
                "Genera bloques dinámicos con estructura de Tabla 5.2. En modo Sorteo "
                "se detiene cuando m != Continuar; en modo Continuar ignora esa parada."
            ),
        )
    with _sim_m_mode_col:
        _t52_m_mode = st.selectbox(
            "Modo m",
            ["Sorteo", "Continuar"],
            key="t52_m_mode",
            help=(
                "Sorteo: sortea m y detiene si m != Continuar. Continuar: sortea m igual, "
                "lo usa en la verosimilitud, pero ignora la parada por m."
            ),
        )
    with _sim_m0_col:
        _reroll_base_m0_now = st.button(
            "Sortear m τ=0",
            key="btn_reroll_base_m_tau0",
            disabled=(
                not bool(st.session_state.get("mechanism_started", False))
            ),
            use_container_width=True,
            help=(
                "Genera nuevamente solo el desenlace m de τ=0 en Tabla 5.2 "
                "y lo guarda como resultado del ciclo base del tipo actual."
            ),
        )
    if st.session_state.get("t52_m_mode_last") != str(_t52_m_mode):
        if "t52_m_mode_last" in st.session_state:
            st.session_state.pop("base_cycle_m_tau0", None)
            _clear_dynamic_cycles_only()
        st.session_state["t52_m_mode_last"] = str(_t52_m_mode)
    if _reroll_base_m0_now:
        st.session_state["base_m_tau0_reroll_counter"] = int(
            st.session_state.get("base_m_tau0_reroll_counter", 0)
        ) + 1
        st.session_state.pop("base_cycle_m_tau0", None)
        _base_m_by_theta = dict(st.session_state.get("base_cycle_m_tau0_by_theta", {}))
        _base_m_by_theta.pop(str(st.session_state.get("global_tipo_real", "")), None)
        st.session_state["base_cycle_m_tau0_by_theta"] = _base_m_by_theta
        _clear_dynamic_cycles_only()
        st.success("Nuevo sorteo de m para τ=0 solicitado. La Tabla 5.2 se actualizará en pestaña 5.")
    c1, c2 = st.columns([2, 1])
    if "dynamic_seed_reset_counts" not in st.session_state:
        st.session_state["dynamic_seed_reset_counts"] = {}
    if "dynamic_seed_run_log" not in st.session_state:
        st.session_state["dynamic_seed_run_log"] = []
    if "dynamic_saved_runs_by_seed" not in st.session_state:
        st.session_state["dynamic_saved_runs_by_seed"] = {}
    if "dynamic_run_counter_by_seed" not in st.session_state:
        st.session_state["dynamic_run_counter_by_seed"] = {}
    with c1:
        _seed_col, _save_seed_col, _delete_seed_col, _reset_col = st.columns([4, 1, 1, 1], vertical_alignment="bottom")
        with _reset_col:
            _reset_dynamic_seed_now = st.button(
                "↻",
                key="btn_reset_dynamic_seed",
                use_container_width=True,
                help=(
                    "Borra las semillas/corridas dinámicas guardadas y genera una nueva "
                    "semilla visible. El ciclo base no se toca hasta presionar Iniciar proceso."
                ),
            )
        if _reset_dynamic_seed_now:
            _old_seed52 = int(st.session_state.get("global_semilla_rng", 123))
            _new_seed52 = int(np.random.default_rng().integers(0, 1_000_000))
            while _new_seed52 == _old_seed52:
                _new_seed52 = int(np.random.default_rng().integers(0, 1_000_000))
            st.session_state["global_semilla_rng"] = int(_new_seed52)
            st.session_state["dynamic_seed_reset_counts"] = {}
            st.session_state["dynamic_seed_run_log"] = []
            st.session_state["dynamic_saved_runs_by_seed"] = {}
            st.session_state["dynamic_run_counter_by_seed"] = {}
            st.session_state.pop("dynamic_active_source", None)
            st.session_state.pop("dynamic_current_run_saved", None)
            _clear_dynamic_cycles_only()
            st.session_state["mechanism_started"] = False
            st.session_state.pop("full_process_context_sig", None)
            st.session_state.pop("incident_voice_context_sig", None)
            st.session_state.pop("incident_voice_path", None)
            st.session_state.pop("incident_voice_meta", None)
            st.session_state.pop("incident_pi_call_prior", None)
            st.session_state.pop("incident_pi_call_realized", None)
            st.success(
                f"Semillas dinámicas borradas. Nueva semilla: {_new_seed52}. "
                "Presione Iniciar proceso y luego Avanzar ciclos."
            )
        with _seed_col:
            semilla = st.number_input(
                "Semillas (RNG):",
                min_value=0,
                max_value=999999,
                value=123,
                step=1,
                key="global_semilla_rng",
                help="Fija la semilla de NumPy al cargar la app y la reproducibilidad de la optimización dinámica (3 jugadores, pestaña 4).",
            )
        with _save_seed_col:
            _save_dynamic_seed_now = st.button(
                "G",
                key="btn_save_dynamic_seed_run",
                use_container_width=True,
                help=(
                    "Guardar: conserva la corrida dinámica actual bajo la semilla visible. "
                    "Solo se guarda si presiona este botón."
                ),
            )
        with _delete_seed_col:
            _delete_dynamic_seed_now = st.button(
                "B",
                key="btn_delete_dynamic_seed_run",
                use_container_width=True,
                help=(
                    "Borrar: elimina solo la corrida guardada bajo la semilla visible. "
                    "No borra otras semillas ni el ciclo base."
                ),
            )
        _seed_key_now = str(int(st.session_state.get("global_semilla_rng", semilla)))
        if _save_dynamic_seed_now:
            _cycles_to_save = st.session_state.get("dynamic_cycles52")
            if isinstance(_cycles_to_save, list) and _cycles_to_save:
                _saved_runs52 = dict(st.session_state.get("dynamic_saved_runs_by_seed", {}))
                _run_meta_to_save = dict(st.session_state.get("dynamic_cycles_run_meta52", {}))
                _run_meta_to_save["saved"] = True
                _stop_to_save = dict(st.session_state.get("dynamic_cycles_stop52", {}))
                st.session_state["dynamic_cycles_run_meta52"] = dict(_run_meta_to_save)
                _saved_runs52[_seed_key_now] = copy.deepcopy(
                    {
                        "dynamic_cycles52": st.session_state.get("dynamic_cycles52"),
                        "dynamic_cycles_diag52": st.session_state.get("dynamic_cycles_diag52"),
                        "dynamic_cycles_stop52": st.session_state.get("dynamic_cycles_stop52"),
                        "dynamic_cycles_run_meta52": dict(_run_meta_to_save),
                        "first_cycle_tau1_52": st.session_state.get("first_cycle_tau1_52"),
                        "first_cycle_table52": st.session_state.get("first_cycle_table52"),
                        "first_cycle_diag52": st.session_state.get("first_cycle_diag52"),
                        "first_cycle_post54": st.session_state.get("first_cycle_post54"),
                        "first_cycle_voice_meta": st.session_state.get("first_cycle_voice_meta"),
                    }
                )
                st.session_state["dynamic_saved_runs_by_seed"] = _saved_runs52
                _log52 = [
                    _r for _r in list(st.session_state.get("dynamic_seed_run_log", []))
                    if str((_r or {}).get("Semilla visible", "")) != _seed_key_now
                ]
                _log52.append(
                    {
                        "Semilla visible": int(_seed_key_now),
                        "Corrida": int(_run_meta_to_save.get("run_counter", 0)),
                        "Semilla efectiva": int(_run_meta_to_save.get("seed_effective", int(_seed_key_now))),
                        "Periodo de parada": int(_stop_to_save.get("tau", 0)),
                        "τ parada": int(_stop_to_save.get("tau", 0)),
                        "m parada": str(_stop_to_save.get("m", "—")),
                        "Motivo": (
                            "desenlace terminal"
                            if str(_stop_to_save.get("motivo", "")).lower() == "desenlace"
                            else "horizonte máximo"
                        ),
                    }
                )
                st.session_state["dynamic_seed_run_log"] = _log52
                st.session_state["dynamic_current_run_saved"] = True
                st.session_state["dynamic_active_source"] = f"guardada en semilla {_seed_key_now}"
                st.success(f"Corrida guardada bajo la semilla {_seed_key_now}.")
            else:
                st.warning("No hay una corrida dinámica actual para guardar. Presione Avanzar ciclos primero.")
        if _delete_dynamic_seed_now:
            _saved_runs52 = dict(st.session_state.get("dynamic_saved_runs_by_seed", {}))
            _had_saved52 = _seed_key_now in _saved_runs52
            _saved_runs52.pop(_seed_key_now, None)
            st.session_state["dynamic_saved_runs_by_seed"] = _saved_runs52
            st.session_state["dynamic_seed_run_log"] = [
                _r for _r in list(st.session_state.get("dynamic_seed_run_log", []))
                if str((_r or {}).get("Semilla visible", "")) != _seed_key_now
            ]
            if _had_saved52:
                st.success(f"Corrida guardada de la semilla {_seed_key_now} borrada.")
            else:
                st.info(f"La semilla {_seed_key_now} no tenía corrida guardada.")
        _active_src52 = st.session_state.get("dynamic_active_source")
        if _active_src52:
            st.caption(f"Corrida dinámica actual: {_active_src52}.")
    limite_dias = int(c2.number_input(
        "Horizonte máximo (días, τ):",
        min_value=2,
        max_value=5000,
        value=200,
        step=1,
        help=(
            "Número máximo de τ que se proyectan en pestaña 5 si no aparece un desenlace "
            "distinto de m=Continuar."
        ),
    ))
    st.session_state["limite_dias"] = int(limite_dias)
    try:
        _mu0_for_iota_global = [
            max(0.0, float(v)) / 100.0
            for v in list(st.session_state.get("final_priors", []))
        ]
        precision_iota = float(max(_mu0_for_iota_global)) if _mu0_for_iota_global else 0.0
    except Exception:
        precision_iota = 0.0

    # Controles globales en formulario: evita un rerun completo por cada cambio
    # intermedio. Los cálculos del mecanismo usan solo valores aplicados.
    if "h0_alpha" not in st.session_state:
        st.session_state.h0_alpha = 0.20
    if "h0_gamma" not in st.session_state:
        st.session_state.h0_gamma = 0.90
    if "global_tipo_real" not in st.session_state:
        st.session_state["global_tipo_real"] = TIPOS_SECUESTRADOR[0]
    if "global_f_capa" not in st.session_state:
        st.session_state["global_f_capa"] = "Alta Riqueza"
    if "global_s_tipo" not in st.session_state:
        st.session_state["global_s_tipo"] = "Duro"
    if "cal_desenlace" not in st.session_state:
        st.session_state["cal_desenlace"] = _CAL_FOCUS_SELECT_OPTIONS[0]

    def _idx_or_zero(options: list, value: Any) -> int:
        try:
            return list(options).index(value)
        except ValueError:
            return 0

    with st.form("global_incident_params_form", clear_on_submit=False):
        st.caption("Edite los parámetros y presione **Aplicar parámetros** para recalcular.")
        col_a, col_b, col_c = st.columns([2, 2, 2])
        with col_a:
            _form_tipo_real = st.selectbox(
                "Grupo secuestrador (θ, incidente):",
                TIPOS_SECUESTRADOR,
                index=_idx_or_zero(TIPOS_SECUESTRADOR, st.session_state.get("global_tipo_real")),
            )
            _form_f_capa = st.selectbox(
                "Capacidad de pago (F):",
                ["Alta Riqueza", "Baja Riqueza"],
                index=_idx_or_zero(["Alta Riqueza", "Baja Riqueza"], st.session_state.get("global_f_capa")),
            )
        with col_b:
            _form_s_tipo = st.selectbox(
                "Tipo de Estado (θ_S):",
                ["Duro", "Laxo"],
                index=_idx_or_zero(["Duro", "Laxo"], st.session_state.get("global_s_tipo")),
            )
            _form_z_reg = st.selectbox(
                "Región de cautiverio (Z):",
                REGIONES,
                index=_idx_or_zero(REGIONES, st.session_state.get("z_region")),
            )
        with col_c:
            _form_v_perf = st.selectbox(
                "Perfil de la víctima (V):",
                list(COEF_XI.keys()),
                index=_idx_or_zero(list(COEF_XI.keys()), st.session_state.get("v_victim")),
            )
            _form_sel_out = st.selectbox(
                "Desenlace focal (pestaña 2):",
                _CAL_FOCUS_SELECT_OPTIONS,
                index=_idx_or_zero(_CAL_FOCUS_SELECT_OPTIONS, st.session_state.get("cal_desenlace")),
            )
        _col_ag1, _col_ag2, _col_apply = st.columns([2, 2, 1], vertical_alignment="bottom")
        with _col_ag1:
            _form_h0_alpha = st.slider(
                r"α₀ · bloqueo financiero (τ=0)",
                min_value=0.0,
                max_value=1.0,
                value=float(st.session_state.get("h0_alpha", 0.20)),
                step=0.01,
                format="%.2f",
                help=r"Bloqueo financiero inicial α₀ ∈ [0,1]. Ajustable antes de cada corrida.",
            )
        with _col_ag2:
            _form_h0_gamma = st.slider(
                r"γ₀ · presión operativa (τ=0)",
                min_value=0.0,
                max_value=1.0,
                value=float(st.session_state.get("h0_gamma", 0.90)),
                step=0.01,
                format="%.2f",
                help=r"Presión operativa inicial γ₀ ∈ [0,1]. Ajustable antes de cada corrida.",
            )
        with _col_apply:
            _apply_global_params = st.form_submit_button("Aplicar parámetros", use_container_width=True)

    if _apply_global_params:
        _old_tipo_real = str(st.session_state.get("global_tipo_real", TIPOS_SECUESTRADOR[0]))
        st.session_state["global_tipo_real"] = str(_form_tipo_real)
        st.session_state["global_f_capa"] = str(_form_f_capa)
        st.session_state["global_s_tipo"] = str(_form_s_tipo)
        st.session_state["z_region"] = str(_form_z_reg)
        st.session_state["v_victim"] = str(_form_v_perf)
        st.session_state["cal_desenlace"] = str(_form_sel_out)
        st.session_state["h0_alpha"] = float(_form_h0_alpha)
        st.session_state["h0_gamma"] = float(_form_h0_gamma)
        if _old_tipo_real != str(_form_tipo_real):
            _invalidate_tab1415_caches()
        _clear_dynamic_cycles_only()
        st.session_state["mechanism_started"] = False
        st.session_state.pop("full_process_context_sig", None)
        st.session_state.pop("incident_voice_context_sig", None)
        st.success("Parámetros aplicados. Presione **Iniciar proceso** para recalcular el mecanismo.")

    tipo_real = str(st.session_state.get("global_tipo_real", TIPOS_SECUESTRADOR[0]))
    f_capa = str(st.session_state.get("global_f_capa", "Alta Riqueza"))
    s_tipo = str(st.session_state.get("global_s_tipo", "Duro"))
    st.session_state["f_capa"] = f_capa
    st.session_state["s_tipo"] = s_tipo
    z_reg = str(st.session_state.get("z_region", REGIONES[0]))
    v_perf = str(st.session_state.get("v_victim", list(COEF_XI.keys())[0]))
    _sel_out = str(st.session_state.get("cal_desenlace", _CAL_FOCUS_SELECT_OPTIONS[0]))
    _theta_prev = st.session_state.get("tab1415_last_theta")
    if _theta_prev is not None and str(_theta_prev) != str(tipo_real):
        _invalidate_tab1415_caches()
    st.session_state["tab1415_last_theta"] = str(tipo_real)

    # Instrumentos: toman el valor de los sliders de Tab 4 (h0_alpha, h0_gamma).
    # Si Tab 4 aún no se ha renderizado en la sesión actual, conservan el valor anterior o 0.
    st.session_state.cal_alpha_star = float(st.session_state.get("h0_alpha", st.session_state.get("cal_alpha_star", 0.0)))
    st.session_state.cal_gamma_star = float(st.session_state.get("h0_gamma", st.session_state.get("cal_gamma_star", 0.0)))
    st.session_state.cal_presion_S = st.session_state.cal_gamma_star

    update_dynamic_priors()
    np.random.seed(int(semilla))
    _voice_context_sig = (
        str(tipo_real),
        str(f_capa),
        str(s_tipo),
        str(st.session_state.z_region),
        str(st.session_state.v_victim),
        int(semilla),
        round(float(st.session_state.get("incident_voice_kappa", 30.0)), 8),
        round(float(st.session_state.get("cal_voz_omega", 0.2)), 8),
        tuple(
            (str(th), round(float(st.session_state.cal_voz_pi_call.get(th, 0.0)), 8))
            for th in TIPOS_SECUESTRADOR
        ),
        tuple(
            (
                str(th),
                tuple(round(float(v), 8) for v in st.session_state.cal_voz_params.get(th, {}).get("x", [])),
                tuple(round(float(v), 8) for v in st.session_state.cal_voz_params.get(th, {}).get("sigma_L", [])),
                tuple(round(float(v), 8) for v in st.session_state.cal_voz_params.get(th, {}).get("sigma_S", [])),
            )
            for th in TIPOS_SECUESTRADOR
        ),
    )
    _voice_stored_sig = st.session_state.get("incident_voice_context_sig")
    _voice_stale = (
        st.session_state.get("incident_voice_path") is None
        or str(st.session_state.get("incident_voice_theta", "")) != str(tipo_real)
        or int(st.session_state.get("incident_voice_seed", -1)) != int(semilla)
        or _voice_stored_sig != _voice_context_sig
    )
    if _run_full_process_now:
        _invalidate_tab1415_caches()
        st.session_state.pop("rb_p3_bundle", None)
        st.session_state.pop("opt3j_result", None)
        for _c1_k52 in (
            "first_cycle_requested",
            "first_cycle_tau1_52",
            "first_cycle_table52",
            "first_cycle_diag52",
            "first_cycle_post54",
            "first_cycle_voice_meta",
            "dynamic_cycles_requested",
            "dynamic_cycles52",
            "dynamic_cycles_diag52",
            "dynamic_cycles_stop52",
        ):
            st.session_state.pop(_c1_k52, None)
        st.session_state["mechanism_started"] = True
        st.session_state["force_tab14_compute_from_start"] = True
        st.session_state["force_tab15_compute_from_start"] = True
        st.session_state["full_process_context_sig"] = _voice_context_sig
        st.session_state["base_h0_alpha"] = float(st.session_state.get("h0_alpha", 0.20))
        st.session_state["base_h0_gamma"] = float(st.session_state.get("h0_gamma", 0.90))
        _generate_and_store_incident_voice()
        st.session_state["incident_voice_context_sig"] = _voice_context_sig
        st.success(
            _ui_text(
                "Process started with current parameters. Dependent tables will be recalculated in this run.",
                "Proceso iniciado con los parámetros actuales. Se recalcularán las tablas dependientes en este recorrido.",
            )
        )
        st.rerun()
    if _run_dynamic_blocks_now and st.session_state.get("full_process_context_sig") != _voice_context_sig:
        st.session_state["mechanism_started"] = False
        for _c1_k52 in (
            "first_cycle_requested",
            "first_cycle_tau1_52",
            "first_cycle_table52",
            "first_cycle_diag52",
            "first_cycle_post54",
            "first_cycle_voice_meta",
            "dynamic_cycles_requested",
            "dynamic_cycles52",
            "dynamic_cycles_diag52",
            "dynamic_cycles_stop52",
        ):
            st.session_state.pop(_c1_k52, None)
        st.warning(
            _ui_text(
                "Current parameters do not match the base scenario. Press **Start Process** "
                "to regenerate the base scenario and then advance the first cycle.",
                "Los parámetros actuales no coinciden con el escenario base. Presione **Iniciar proceso** "
                "para regenerar el escenario base y luego avance el primer ciclo.",
            )
        )
    elif _run_dynamic_blocks_now:
        _seed_click52 = str(int(st.session_state.get("global_semilla_rng", semilla)))
        _run_counters52 = dict(st.session_state.get("dynamic_run_counter_by_seed", {}))
        _run_counters52[_seed_click52] = int(_run_counters52.get(_seed_click52, 0)) + 1
        st.session_state["dynamic_run_counter_by_seed"] = _run_counters52
        st.session_state["dynamic_current_run_counter"] = int(_run_counters52[_seed_click52])
        st.session_state["dynamic_current_run_saved"] = False
        st.session_state["dynamic_active_source"] = (
            f"temporal no guardada · semilla {_seed_click52} · corrida {_run_counters52[_seed_click52]}"
        )
        st.session_state["first_cycle_requested"] = True
        st.session_state["dynamic_cycles_requested"] = True
        st.session_state["force_tab14_compute_from_start"] = True
        st.session_state["force_tab15_compute_from_start"] = True
        for _c1_k52 in (
            "first_cycle_tau1_52",
            "first_cycle_table52",
            "first_cycle_diag52",
            "first_cycle_post54",
            "first_cycle_voice_meta",
            "dynamic_cycles52",
            "dynamic_cycles_diag52",
            "dynamic_cycles_stop52",
        ):
            st.session_state.pop(_c1_k52, None)
        st.session_state["first_cycle_pending_rerun"] = True
        st.success(
            _ui_text(
                "Cycles requested. Will advance until m differs from Continue or until the maximum horizon.",
                "Ciclos solicitados. Se avanzará hasta que m sea distinto de Continuar o hasta el horizonte máximo.",
            )
        )
    if st.session_state.get("full_process_context_sig") != _voice_context_sig:
        st.session_state["mechanism_started"] = False
        for _c1_k52 in (
            "first_cycle_requested",
            "first_cycle_tau1_52",
            "first_cycle_table52",
            "first_cycle_diag52",
            "first_cycle_post54",
            "first_cycle_voice_meta",
            "dynamic_cycles_requested",
            "dynamic_cycles52",
            "dynamic_cycles_diag52",
            "dynamic_cycles_stop52",
        ):
            st.session_state.pop(_c1_k52, None)
        st.info(
            _ui_text(
                "Adjust the incident parameters and press **Start Process** to generate the voice and calculate the mechanism.",
                "Ajuste los parámetros del incidente y presione **Iniciar proceso** para generar la voz y calcular el mecanismo.",
            )
        )

    if st.session_state.get("mechanism_started", False):
        with st.expander(
            _ui_text("Incident Voice Trajectory", "Trayectoria de voz del incidente"),
            expanded=bool(st.session_state.get("incident_voice_meta")),
        ):
            _ic1, _ic2 = st.columns([1, 2])
            with _ic1:
                st.number_input(
                    _ui_text("κ (prior Beta strength)", "κ (fuerza prior Beta)"),
                    min_value=2.0,
                    max_value=500.0,
                    value=float(st.session_state.get("incident_voice_kappa", 30.0)),
                    step=1.0,
                    key="incident_voice_kappa",
                    disabled=True,
                )
            with _ic2:
                st.caption(
                    _ui_text(
                        "The incident's random voice trajectory is generated only when pressing **Start Process**.",
                        "La trayectoria de voz aleatoria del incidente se genera únicamente al presionar **Iniciar proceso**.",
                    )
                )

            _meta_inc = st.session_state.get("incident_voice_meta")
            if _meta_inc:
                _th_stored = str(st.session_state.get("incident_voice_theta", ""))
                _seed_stored = st.session_state.get("incident_voice_seed")
                if _th_stored != str(tipo_real) or int(_seed_stored or -1) != int(semilla):
                    st.warning(
                        _ui_text(
                            f"The saved trajectory corresponds to θ*={_th_stored}, seed={_seed_stored}. "
                            f"Current controls: θ*={tipo_real}, seed={int(semilla)}. "
                            "Press **Start Process** to recalculate.",
                            f"La trayectoria guardada corresponde a θ*={_th_stored}, semilla={_seed_stored}. "
                            f"Controles actuales: θ*={tipo_real}, semilla={int(semilla)}. "
                            "Presione **Iniciar proceso** para recalcular.",
                        )
                    )
                _pi_prior_s = st.session_state.get("incident_pi_call_prior", {})
                _pi_tilde_s = st.session_state.get("incident_pi_call_realized", {})
                _df_pi_inc = pd.DataFrame(
                    {
                        "θ": TIPOS_SECUESTRADOR,
                        _ui_text("π_call (prior, tab 2)", "π_call (prior, pestaña 2)"): [
                            round(float(_pi_prior_s.get(th, 0.0)), 4) for th in TIPOS_SECUESTRADOR
                        ],
                        _ui_text("π̃_call (realized, incident)", "π̃_call (realizada, incidente)"): [
                            round(float(_pi_tilde_s.get(th, 0.0)), 4) for th in TIPOS_SECUESTRADOR
                        ],
                    }
                )
                st.dataframe(_df_pi_inc, hide_index=True, use_container_width=True)
                _path_inc = st.session_state.get("incident_voice_path", [])
                if _path_inc:
                    _rows_path = []
                    _call_col = _ui_text("Call", "Llamada")
                    for _s in _path_inc:
                        _x = _s.get("x_obs")
                        _x_str = (
                            ", ".join(f"{v:.3f}" for v in _x)
                            if _x is not None
                            else "—"
                        )
                        _rows_path.append(
                            {
                                "t": _s.get("t"),
                                "V_t": _s.get("V_t"),
                                _call_col: _ui_text("Yes", "Sí") if int(_s.get("V_t", 0)) == 1 else "No",
                                "x_obs": _x_str,
                                _ui_text("sender", "emisor"): _s.get("emisor_voz"),
                            }
                        )
                    st.dataframe(pd.DataFrame(_rows_path), hide_index=True, use_container_width=True)
                    _df_vlik = st.session_state.get("incident_voice_likelihood_df")
                    if isinstance(_df_vlik, pd.DataFrame) and not _df_vlik.empty:
                        st.markdown(
                            _ui_text(
                                "**Voice Likelihoods (Table 14)** · "
                                r"$\mathcal{L}_{C,t}$, $\mathcal{L}_{\mathrm{voz},t}$ with the same $(V_t,x^{\mathrm{obs}})$",
                                "**Verosimilitudes de voz (Tabla 14)** · "
                                r"$\mathcal{L}_{C,t}$, $\mathcal{L}_{\mathrm{voz},t}$ con el mismo $(V_t,x^{\mathrm{obs}})$",
                            )
                        )
                        _show_vlik = _df_vlik.rename(
                            columns={
                                "t": "t",
                                "V_t": "V_t",
                                "Llamada": _call_col,
                                "L_voz": "ℒ_voz",
                                "L_C": "ℒ_{C,t}",
                            }
                        )
                        _vlik_cols = [
                            c
                            for c in [
                                "t",
                                "V_t",
                                _call_col,
                                "ℒ_voz",
                                "ℒ_{C,t}",
                            ]
                            if c in _show_vlik.columns
                        ]
                        _show_vlik_render = _show_vlik[_vlik_cols].copy()
                        for _cn in ("t", "V_t", "ℒ_voz", "ℒ_{C,t}"):
                            if _cn in _show_vlik_render.columns:
                                _show_vlik_render[_cn] = pd.to_numeric(
                                    _show_vlik_render[_cn], errors="coerce"
                                )
                        if _call_col in _show_vlik_render.columns:
                            _show_vlik_render[_call_col] = _show_vlik_render[_call_col].astype(str)
                        st.dataframe(
                            _show_vlik_render,
                            hide_index=True,
                            use_container_width=True,
                        )

    if not st.session_state.get("mechanism_started", False):
        st.markdown("---")
        st.info(
            _ui_text(
                "You can review the base model tabs and tables. The incident voice and calculated "
                "mechanism results will be generated when pressing **Start Process**.",
                "Puede revisar las pestañas y tablas base del modelo. La voz del incidente y los "
                "resultados calculados del mecanismo se generarán al presionar **Iniciar proceso**.",
            )
        )

st.markdown("---")


tab_cfg, tab_cal, tab_mdg, tab_rb, tab_mech_sol, tab_dyn = st.tabs(
    [
        _ui_text("1 · Setup and Start", "1 · Configuración e Inicio"),
        _ui_text("2 · Probabilities", "2 · Probabilidades"),
        "3 · MDG",
        _ui_text("4 · Family-Kidnapper", "4 · Familia-Secuestrador"),
        _ui_text("5 · Mechanism Solution", "5 · Solución Mecanismo"),
        _ui_text("6 · Dynamic Charts", "6 · Gráficas dinámica"),
    ]
)


with tab_cfg:
    st.markdown(_ui_text("## Setup and Start", "## Configuración e Inicio"))
    st.caption(
        _ui_text(
            "Global controls (Simulation, Incident and Instruments) are now at the top for easy access from any tab.",
            "Los controles globales (Simulación, Incidente e Instrumentos) ahora se encuentran en la parte superior para facilitar su acceso desde cualquier pestaña.",
        )
    )

    st.markdown(_ui_text("### Ex-ante Distribution (priors)", "### Distribución ex-ante (priors)"))
    st.markdown(_ui_text("#### Rationale", "#### Fundamentación"))
    st.latex(r"\mu_0(\theta \mid z, \theta_V) = \frac{\exp(\varpi_\theta + \eta_{\theta, z} + \xi_{\theta, v})}{\sum_{\theta' \in \Theta} \exp(\varpi_{\theta'} + \eta_{\theta', z} + \xi_{\theta', v})}")
    st.markdown(
        _ui_text(
            "The initial probability follows a **softmax** over $\\varpi$ (base frequency), $\\eta$ (region **Z**) and $\\xi$ (profile **V**), "
            "according to **Mechanism.tex** (eq. 1269).",
            "La probabilidad inicial sigue una **softmax** sobre $\\varpi$ (frecuencia base), $\\eta$ (región **Z**) y $\\xi$ (perfil **V**), "
            "según **Mechanism.tex** (ec. 1269).",
        )
    )

    st.markdown(_ui_text("#### Parameters for the Current Selection (Z, V)", "#### Parámetros para la selección actual (Z, V)"))
    z = st.session_state.z_region
    v = st.session_state.v_victim

    param_data = []
    for t in TIPOS_SECUESTRADOR:
        d = COEF_DELTA.get(t, 0.0)
        e = COEF_ETA.get(z, {}).get(t, 0.0)
        x = COEF_XI.get(v, {}).get(t, 0.0)
        score = d + e + x
        param_data.append({
            _ui_text("Type", "Tipo"): t,
            "varpi": round(d, 2),
            "eta_z": round(e, 2),
            "xi_v": round(x, 2),
            "Score": round(score, 2),
            "μ₀ (%)": round(st.session_state.dynamic_priors[TIPOS_SECUESTRADOR.index(t)], 2),
        })

    rb_katex_grid_header(RB_LATEX_HEADER_CFG_PARAMS, height=52)
    _df_cfg = pd.DataFrame(param_data)
    st.dataframe(
        _df_cfg,
        width="stretch",
        height=_glide_full_height_px(_st_table_row_count(_df_cfg)),
        hide_index=True,
    )

    st.markdown(_ui_text("#### Prior Selection (model vs manual)", "#### Selección de priors (modelo vs manual)"))
    _prior_opts = [_ui_text("Model", "Modelo"), "Manual"]
    mode = st.radio(
        _ui_text("Select the source of Priors for the simulation:", "Selecciona el origen de los Priors para la simulación:"),
        _prior_opts,
        horizontal=True,
        key="prior_mode_selector",
    )
    st.session_state.prior_mode = mode

    _prior_model_col = _ui_text("Prior (Model %)", "Prior (Modelo %)")
    _prior_manual_col = _ui_text("Prior (Manual %)", "Prior (Manual %)")
    _group_col = _ui_text("Group", "Grupo")
    comparison_data = pd.DataFrame({
        _group_col: TIPOS_SECUESTRADOR,
        _prior_model_col: [round(p, 2) for p in st.session_state.dynamic_priors],
        _prior_manual_col: st.session_state.manual_priors,
    })

    if mode == _prior_opts[0]:  # Model mode
        styled_df = comparison_data.style.format({
            _prior_model_col: "{:.2f}",
            _prior_manual_col: "{:.2f}",
        })
        st.dataframe(
            styled_df,
            width="stretch",
            height=_glide_full_height_px(_st_table_row_count(styled_df)),
            hide_index=True,
        )
        st.session_state.final_priors = st.session_state.dynamic_priors
    else:  # Manual mode
        st.markdown(_ui_text("#### Manual Probability Configuration", "#### Configuración Manual de Probabilidades"))
        st.info(
            _ui_text(
                "💡 **Closure Rule**: Enter values for the first 3 groups. The **FARC** value will be automatically adjusted so the sum is 100%. All values must be strictly greater than 0.",
                "💡 **Regla de Cierre**: Ingresa los valores para los primeros 3 grupos. El valor de **FARC** se ajustará automáticamente para que la suma sea 100%. Todos los valores deben ser estrictamente mayores a 0.",
            )
        )
        c1, c2, c3 = st.columns(3)
        p_dc = c1.number_input("DC (%)", min_value=0.01, max_value=99.97, value=float(st.session_state.manual_priors[0]), step=1.0, key="manual_p_dc")
        p_par = c2.number_input("PAR (%)", min_value=0.01, max_value=99.97, value=float(st.session_state.manual_priors[1]), step=1.0, key="manual_p_par")
        p_eln = c3.number_input("ELN (%)", min_value=0.01, max_value=99.97, value=float(st.session_state.manual_priors[2]), step=1.0, key="manual_p_eln")

        total_3 = p_dc + p_par + p_eln
        p_farc = 100.0 - total_3

        if p_farc <= 0:
            st.error(
                _ui_text(
                    f"❌ Error: The sum of DC, PAR and ELN ({total_3:.2f}%) is already 100% or more. FARC would be {p_farc:.2f}%, which is not valid. Reduce the values.",
                    f"❌ Error: La suma de DC, PAR y ELN ({total_3:.2f}%) ya es el 100% o más. FARC quedaría en {p_farc:.2f}%, lo cual no es válido. Reduce los valores.",
                )
            )
            st.session_state.final_priors = st.session_state.dynamic_priors
        else:
            st.session_state.manual_priors = [p_dc, p_par, p_eln, p_farc]
            st.session_state.final_priors = st.session_state.manual_priors
            comparison_data[_prior_manual_col] = [round(p, 2) for p in st.session_state.manual_priors]
            styled_df = comparison_data.style.format({
                _prior_model_col: "{:.2f}",
                _prior_manual_col: "{:.2f}",
            })
            st.dataframe(
                styled_df,
                width="stretch",
                height=_glide_full_height_px(_st_table_row_count(styled_df)),
                hide_index=True,
            )
            st.success(
                _ui_text(
                    f"✅ Manual Configuration Active. FARC has been adjusted to: **{p_farc:.2f}%**",
                    f"✅ Configuración Manual Activa. FARC se ha ajustado a: **{p_farc:.2f}%**",
                )
            )

    st.markdown("---")
    st.markdown(_ui_text("### Regional Map by Municipality", "### Mapa regional por municipio"))
    st.markdown(
        _ui_text(
            "Each polygon is a **municipality**; the color shows the model **region**. "
            "Municipal totals and year ranges come from the same case database used by the application "
            "(**Data_CMH.csv**, aligned with public **CNMH** data).",
            "Cada polígono es un **municipio**; el color muestra la **región** del modelo. "
            "Los totales por municipio y el rango de años provienen de la misma base de casos que usa la aplicación "
            "(archivo **Data_CMH.csv**, alineado con la información pública del **CNMH**).",
        )
    )

    _load_map_now = st.button(
        _ui_text("Load municipal map", "Cargar mapa municipal"),
        key="btn_load_municipio_map",
        use_container_width=True,
        help=_ui_text(
            "Loads the municipal GeoJSON only when needed to view the map. Does not affect Table 5.2 or tab 6.",
            "Carga el GeoJSON municipal solo cuando se necesita ver el mapa. No afecta Tabla 5.2 ni pestaña 6.",
        ),
    )
    if _load_map_now:
        st.session_state["municipio_map_requested"] = True

    if bool(st.session_state.get("municipio_map_requested", False)):
        municipio_geojson, df_municipios = load_municipio_geojson_v4()
    else:
        municipio_geojson, df_municipios = None, pd.DataFrame()
        st.info(
            _ui_text(
                "Municipal map not loaded at startup to speed up the app. Use the button to view it.",
                "Mapa municipal no cargado en el inicio para acelerar la app. Use el botón para visualizarlo.",
            )
        )

    if municipio_geojson and not df_municipios.empty:
        _lbl_total = _ui_text("Total Kidnappings (municipality)", "Total de secuestros (municipio)")
        _lbl_first = _ui_text("First year with records (municipality)", "Primer año con registro (municipio)")
        _lbl_last = _ui_text("Last year with records (municipality)", "Último año con registro (municipio)")
        _lbl_dept = _ui_text("Department", "Departamento")
        _lbl_muni = _ui_text("Municipality", "Municipio")
        _lbl_reg = _ui_text("Region", "Región")

        # Region name translations for map category labels
        _REGION_EN = {
            "Metropolitana": "Metropolitan",
            "Andina": "Andean",
            "Caribe": "Caribbean",
            "Pacífica / Zona Roja": "Pacific / Red Zone",
            "Oriente / Selva": "East / Jungle",
            "Sin región": "No region",
        }
        _is_en = st.session_state.get("app_language", "English") == "English"
        _df_map = df_municipios.copy()
        if _is_en and "Region" in _df_map.columns:
            _df_map["Region"] = _df_map["Region"].map(lambda r: _REGION_EN.get(str(r), str(r)) if pd.notna(r) else r)
        _region_order = (
            [_REGION_EN.get(r, r) for r in REGIONES] + ["No region"]
            if _is_en
            else REGIONES + ["Sin región"]
        )
        _region_colors = (
            {_REGION_EN.get(k, k): v for k, v in REGION_COLORS.items()}
            if _is_en
            else REGION_COLORS
        )

        fig_map = px.choropleth(
            _df_map,
            geojson=municipio_geojson,
            locations="feature_id",
            featureidkey="properties.MPIO_CCNCT",
            color="Region",
            category_orders={"Region": _region_order},
            color_discrete_map=_region_colors,
            hover_name="Municipio",
            hover_data={
                "Departamento": True,
                "Region": True,
                "Total Secuestros": True,
                "Primer Año": True,
                "Último Año": True,
                "feature_id": False,
            },
            labels={
                "Municipio": _lbl_muni,
                "Departamento": _lbl_dept,
                "Region": _lbl_reg,
                "Total Secuestros": _lbl_total,
                "Primer Año": _lbl_first,
                "Último Año": _lbl_last,
            },
        )
        fig_map.update_traces(marker_line_width=0.25, marker_line_color="rgba(255,255,255,0.65)")
        fig_map.update_geos(fitbounds="locations", visible=False, projection_type="mercator")
        fig_map.update_layout(
            height=650, margin={"r": 0, "t": 20, "l": 0, "b": 0}, paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(title_text=_lbl_reg, orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
        )
        st.plotly_chart(fig_map, use_container_width=True)

        _ts = df_municipios["Total Secuestros"].sum()
        total_secuestros = int(_ts) if pd.notna(_ts) else 0
        df_con_casos = df_municipios[df_municipios["Total Secuestros"] > 0]
        año_primero_sec = None
        año_ultimo_sec = None
        if not df_con_casos.empty:
            _ymin = df_con_casos["Primer Año"].min()
            _ymax = df_con_casos["Último Año"].max()
            if pd.notna(_ymin) and pd.notna(_ymax):
                try:
                    año_primero_sec = int(_ymin)
                    año_ultimo_sec = int(_ymax)
                except (ValueError, TypeError):
                    año_primero_sec = None
                    año_ultimo_sec = None

        st.markdown(
            _ui_text(
                f"**Total accumulated kidnappings:** {total_secuestros:,} (sum of all municipalities with data).",
                f"**Total de secuestros acumulados:** {total_secuestros:,} (suma de todos los municipios con datos).",
            )
        )
        if año_primero_sec is not None and año_ultimo_sec is not None:
            st.markdown(
                _ui_text(
                    f"**Temporal coverage of the municipal panel (CMH):** **{año_primero_sec}**–**{año_ultimo_sec}**.",
                    f"**Cobertura temporal del panel municipal (CMH):** **{año_primero_sec}**–**{año_ultimo_sec}**.",
                )
            )
        elif not df_con_casos.empty:
            st.info(
                _ui_text(
                    "There are municipalities with at least one case, but the year range is not available "
                    "(check the **Year** column in **Data_CMH.csv**).",
                    "Hay municipios con al menos un caso, pero el rango de años no está disponible "
                    "(revisa la columna **Año** en **Data_CMH.csv**).",
                )
            )
        st.markdown(_cita_biblio_cmh())
    else:
        st.warning(
            _ui_text(
                "Could not load the municipal map or there are no data to plot. "
                "Check that **co_2018_MGN_MPIO_POLITICO.geojson** is in the same folder as **app.py** "
                "and that **Data_CMH.csv** has records; then restart the application.",
                "No se pudo cargar el mapa municipal o no hay datos para graficar. "
                "Comprueba que el archivo **co_2018_MGN_MPIO_POLITICO.geojson** esté en la misma carpeta que **app.py** "
                "y que **Data_CMH.csv** tenga registros; luego reinicia la aplicación.",
            )
        )

with tab_cal:
    _ensure_focus_cov_store_in_session()
    st.markdown(
        r"""
        <style>
        /* Estilo para eliminar gaps entre tablas e «Editar valores» (popovers) en toda la pestaña */
        div[data-testid="stVerticalBlock"]:has(iframe):has([data-testid="stPopover"]) {
            gap: 0 !important;
            row-gap: 0 !important;
        }
        /* Elimina márgenes de los contenedores de elementos consecutivos */
        div.stElementContainer:has(iframe) + div.stElementContainer:has([data-testid="stPopover"]) {
            margin-top: -5.3rem !important; /* Compensa el gap (~85px) identificado en pruebas */
        }
        /* Ajuste específico para el iframe cuando va seguido de un popover */
        div.stElementContainer:has(iframe) {
            margin-bottom: -0.5rem !important;
        }
        /* Ajuste para el popover cuando sigue a un iframe */
        div.stElementContainer:has([data-testid="stPopover"]) {
            margin-top: 0 !important;
            padding-top: 0 !important;
        }
        /* Elimina padding interno de los popovers para que el botón esté más arriba */
        [data-testid="stPopover"] {
            margin-top: -0.1rem !important;
        }
        /* Títulos de tablas: menos margen inferior */
        div[data-testid="stVerticalBlock"] h4 {
            margin-bottom: 0.15rem !important;
        }
        /* Popovers «Editar valores Prior» (portal fuera del tab): una línea por fila de coeficiente */
        div[data-testid="stPopoverBody"] div[data-testid="stVerticalBlock"] {
            gap: 0.2rem !important;
        }
        div[data-testid="stPopoverBody"] div[data-testid="stHorizontalBlock"] {
            flex-wrap: nowrap !important;
            align-items: center !important;
            row-gap: 0 !important;
        }
        div[data-testid="stPopoverBody"] div[data-testid="stHorizontalBlock"] > div {
            display: flex !important;
            flex-direction: column !important;
            justify-content: center !important;
            align-items: flex-start !important;
            min-height: 2rem !important;
        }
        div[data-testid="stPopoverBody"] div[data-testid="element-container"] {
            margin-bottom: 0 !important;
            padding-bottom: 0 !important;
        }
        div[data-testid="stPopoverBody"] .stMarkdown {
            margin-bottom: 0 !important;
            margin-top: 0 !important;
        }
        div[data-testid="stPopoverBody"] .stMarkdown p {
            margin: 0 !important;
            line-height: 1.2 !important;
        }
        div[data-testid="stPopoverBody"] [data-testid="stLatex"] {
            margin-bottom: 0 !important;
            margin-top: 0 !important;
        }
        div[data-testid="stPopoverBody"] .katex {
            font-size: 0.88em !important;
        }
        div[data-testid="stPopoverBody"] .katex-display {
            margin: 0 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    _M_cal = maturation_filter(int(st.session_state.cal_t_hazard), float(st.session_state.cal_rho_mat))
    _modelo_cal = ModeloSecuestro(
        betas=copy.deepcopy(st.session_state.cal_betas_dict),
        lambdas_0=copy.deepcopy(st.session_state.cal_lambdas_dict),
    )
    _mu_vec = [float(st.session_state.final_priors[i]) / 100.0 for i in range(len(TIPOS_SECUESTRADOR))]
    _iota_t = max(_mu_vec) if _mu_vec else 0.0

    # Configuración del incidente (θ_K, Desenlace, Z) ahora sincronizada con el panel global
    _th = tipo_real
    _sel_out = st.session_state.cal_desenlace
    _j_mech = _FOCUS_LABEL_TO_J[_sel_out]
    _frow = _cal_focus_row(_j_mech)
    _bkey_f, _hkey_f, _kappa_f = _frow[2], _frow[3], _frow[4]


    _h_now = _modelo_cal.calcular_hazards(
        int(st.session_state.cal_t_hazard),
        _th,
        float(st.session_state.cal_presion_S),
        maturity_mult=_M_cal,
        z_region=str(st.session_state.z_region),
        v_victim=str(st.session_state.v_victim),
        alpha=float(st.session_state.get("cal_alpha_star", 0.0)),
        gamma=float(st.session_state.get("cal_presion_S", 0.0)),
    )
    st.divider()
    st.markdown(
        r"### 1. Ecuación $\lambda_j$ completa (Mechanism.tex) y estimaciones"
    )
    _pk_cov = _focus_cov_profile_key(
        _j_mech,
        _th,
        str(st.session_state.z_region),
        str(st.session_state.v_victim),
        str(f_capa),
        str(s_tipo),
    )
    _zp_tab1 = _focus_cmh_endogenous_tentatives(_th)
    _df_cov0 = _build_focus_covariate_table(
        j_mech=_j_mech,
        theta_k=_th,
        z_region=st.session_state.z_region,
        v_victim=st.session_state.v_victim,
        f_capa=f_capa,
        s_tipo=s_tipo,
        lambdas_dict=st.session_state.cal_lambdas_dict,
        betas_dict=st.session_state.cal_betas_dict,
        M_t=_M_cal,
        presion_S=float(st.session_state.cal_presion_S),
        h_j_numeric=float(_h_now[_hkey_f]),
        tipo_incidente_p1=str(tipo_real),
        zeta_phi=_zp_tab1,
    )
    _df_cov = _apply_focus_cov_saved_values(
        _df_cov0, _pk_cov, st.session_state.focus_cov_store, _th
    )

    _df_edit_prior = _tab1_popover_rows_solo_calibrados(_df_cov)
    _fce_key = "fce_" + hashlib.md5(_pk_cov.encode("utf-8")).hexdigest()[:16]

    _ecLa, _ecRa = st.columns((0.46, 0.54), gap="medium")
    with _ecLa:
        st.markdown(f"**Causa** $j = {_j_mech}$.")
        # KaTeX: más compacta (~dos líneas visuales junto a Tabla 1).
        st.latex(r"\footnotesize " + _MECH_LATEX[_j_mech])
        _is_en_blurb = st.session_state.get("app_language", "English") == "English"
        _blurb_dict = _MECH_EQUATION_BLURB_EN if _is_en_blurb else _MECH_EQUATION_BLURB_ES
        _blurb = _blurb_dict.get(_j_mech, _blurb_dict[1])
        _blurb_lang = "en" if _is_en_blurb else "es"
        st.markdown(
            f'<div class="mech-eq-blurb" lang="{_blurb_lang}">\n\n{_blurb}\n\n</div>',
            unsafe_allow_html=True,
        )
    with _ecRa:
        _t1_head_txt = _ui_text("Table 1 \\cdot \\text{ Focus Covariates }", "Tabla 1 \\cdot \\text{ Covariables foco }")
        _t1_katex_expr = _t1_head_txt + "\\lambda_j"
        _t1_katex_json = json.dumps(_t1_katex_expr)
        _t1_fallback_json = json.dumps(
            _ui_text("Table 1 · Focus Covariates λ_j", "Tabla 1 · Covariables foco λ_j")
        )
        components.html(
            '<!DOCTYPE html><html><head><meta charset="utf-8"/>'
            f'<link rel="stylesheet" href="{_KATEX_BASE}/katex.min.css" crossorigin="anonymous"/>'
            f'<script src="{_KATEX_BASE}/katex.min.js" crossorigin="anonymous"></script>'
            '<style>'
            'html,body{margin:0;padding:0;overflow:hidden;background:transparent;}'
            '#t1h{font-size:1.1rem;font-weight:600;margin:4px 0 2px 0;padding:0;}'
            '.katex{font-size:1em;}'
            '</style>'
            '</head><body>'
            '<div id="t1h"></div>'
            f'<script>'
            f'try{{katex.render({_t1_katex_json},document.getElementById("t1h"),'
            '{displayMode:false,throwOnError:false,strict:false});}'
            f'catch(e){{document.getElementById("t1h").textContent={_t1_fallback_json};}}'
            '</script>'
            '</body></html>',
            height=38,
        )
        _render_focus_covariate_katex_table(
            _df_cov,
            show_origen=False,
            font_boost_pt=2.0,
            compact_iframe_bottom=True,
            iframe_slack_px=0,
            collapse_gap_below=True,
        )
        with st.popover(
            _ui_text("Edit Prior Values (Table 1)", "Editar valores Prior (Tabla 1)"),
            width="stretch",
        ):
            st.markdown(
                """
                <style>
                /* Texto del panel ~2 pt menor que el tamaño base de Streamlit */
                div[data-testid="stPopoverBody"] {
                    font-size: 0.86rem !important;
                }
                div[data-testid="stPopoverBody"] .katex {
                    font-size: 0.94em !important;
                }
                /* Una línea por fila Prior: texto + math + input */
                div[data-testid="stPopoverBody"] p {
                    margin: 0.08rem 0 !important;
                    white-space: nowrap !important;
                    overflow: hidden !important;
                    text-overflow: ellipsis !important;
                    line-height: 1.3 !important;
                }
                div[data-testid="stPopoverBody"] div[data-testid="stHorizontalBlock"] {
                    flex-wrap: nowrap !important;
                    align-items: center !important;
                }
                /* Botón primario «Guardar» solo dentro del popover: verde (evita el rojo por defecto). */
                div[data-testid="stPopoverBody"] button[data-testid="stBaseButton-primary"],
                div[data-testid="stPopover"] button[data-testid="stBaseButton-primary"] {
                    background-color: #198754 !important;
                    border: 1px solid #146c32 !important;
                    color: #ffffff !important;
                }
                div[data-testid="stPopoverBody"] button[data-testid="stBaseButton-primary"]:hover,
                div[data-testid="stPopover"] button[data-testid="stBaseButton-primary"]:hover {
                    background-color: #157347 !important;
                    border-color: #125a3a !important;
                    color: #ffffff !important;
                }
                </style>
                """,
                unsafe_allow_html=True,
            )
            _vals_current = _render_focus_prior_valor_inputs(
                _df_edit_prior, widget_stem=_fce_key, profile_key=_pk_cov
            )
            _fce_save_lbl = _ui_text(
                "Save {} · {}".format(
                    {1: "Payment", 2: "Death", 3: "Ransom"}.get(_j_mech, str(_j_mech)), _th
                ),
                "Guardar DC {} · {}".format(
                    _FOCUS_OUTCOME_BTN_SHORT.get(_j_mech, str(_j_mech)), _th
                ),
            )
            _btn_save, _btn_reset = st.columns(2)
            with _btn_save:
                if st.button(
                    _fce_save_lbl,
                    key=f"fce_btn_{_fce_key}",
                    type="primary",
                    help=_ui_text("Saves the edited Prior values to `user_focus_covariates.json` for this profile.", "Guarda los Prior editados en `user_focus_covariates.json` para este perfil."),
                ):
                    if st.session_state.get(f"fce_bad_{_pk_cov}"):
                        st.error(
                            _ui_text(
                                "Check the **Value** fields: use **.** for thousands and **,** for decimals "
                                "(e.g. **1,234.56**).",
                                "Revisa los **Valor**: miles con **.** y decimales con **,** "
                                "(ejemplo: **1.234,56**).",
                            )
                        )
                    else:
                        _vals = dict(_vals_current)
                        st.session_state.focus_cov_store[_pk_cov] = {"valores_por_termino": _vals}
                        _save_focus_cov_store(st.session_state.focus_cov_store)
                        st.success(
                            _ui_text(
                                f"Saved to **`user_focus_covariates.json`** "
                                f"(profile `{_pk_cov[:48]}{'…' if len(_pk_cov) > 48 else ''}`: "
                                f"{len(_vals)} **Prior** terms).",
                                f"Caso guardado en **`user_focus_covariates.json`** "
                                f"(perfil `{_pk_cov[:48]}{'…' if len(_pk_cov) > 48 else ''}`: "
                                f"{len(_vals)} términos **Prior**).",
                            )
                        )
                        st.rerun()
            with _btn_reset:
                if st.button(
                    _ui_text("Reset Prior (this profile)", "Restablecer Prior (este perfil)"),
                    key=f"fce_rst_{_fce_key}",
                    help=_ui_text("Removes saved entries for this profile; Prior values revert to model defaults.", "Elimina entradas guardadas para este perfil; los Prior vuelven al modelo."),
                ):
                    st.session_state.focus_cov_store.pop(_pk_cov, None)
                    _save_focus_cov_store(st.session_state.focus_cov_store)
                    st.session_state[f"fce_bump_{_pk_cov}"] = int(
                        st.session_state.get(f"fce_bump_{_pk_cov}", 0)
                    ) + 1
                    st.info(_ui_text("**Prior** overrides for this profile removed; model default values reloaded.", "Overrides **Prior** de este perfil eliminados; se recargan valores del modelo."))
                    st.rerun()

    st.divider()
    st.markdown(
        _ui_text("### 2. Effective Intensities (eq. 32–35, *Mechanism.tex*)", "### 2. Intensidades efectivas (ec. 32–35, *Mechanism.tex*)")
    )
    _simLa, _simRa = st.columns((0.46, 0.54), gap="medium")
    with _simLa:
        st.caption(_ui_text(f"Focal outcome in Table 1: **j = {_j_mech}** ({_sel_out}).", f"Desenlace focal en Tabla 1: **j = {_j_mech}** ({_sel_out})."))
        st.latex(
            r"\tilde{\lambda}_j(t)=M(t)\,\lambda_j(t)\ (j=1,2,3),\quad "
            r"\tilde{\lambda}_4(t)=\lambda_4,\quad \Delta t=1."
        )
        st.latex(r"M(t)=\min\!\bigl\{1,\,(t/T_{\mathrm{mad}})^2\bigr\}.")
        st.latex(
            r"p_{\mathrm{Cont},t}=\exp\!\Bigl(-\sum_{j=1}^{4}\tilde{\lambda}_j(t)\,\Delta t\Bigr)."
        )
        st.latex(r"q(t)=1-p_{\mathrm{Cont},t}.")
    with _simRa:
        rb_katex_title(
            r"#### Tabla 2 · $T_{\mathrm{mad}}$ y $\lambda_4$ (Prior)"
        )
        _Tmv = float(st.session_state.cal_T_mad)
        _L4v = float(st.session_state.cal_lambda_4)
        _df_eff = pd.DataFrame(
            [
                {
                    "#": 1,
                    "Término": _ui_text("Maturation threshold", "Umbral de maduración (texto)"),
                    "Coeficiente": r"T_{\mathrm{mad}}",
                    "Valor": _Tmv,
                    "Origen del valor": "Prior",
                },
                {
                    "#": 2,
                    "Término": _ui_text("Exogenous channel (baseline)", "Canal exógeno (basal)"),
                    "Coeficiente": r"\lambda_4",
                    "Valor": _L4v,
                    "Origen del valor": "Prior",
                },
            ]
        )
        _df_eff["Clase_tab7"] = "Prior"
        _render_focus_covariate_katex_table(
            _df_eff,
            show_origen=False,
            font_boost_pt=2.0,
            compact_iframe_bottom=True,
            iframe_slack_px=0,
            collapse_gap_below=True,
        )
        with st.popover(
            _ui_text("Edit Prior Values · Table 2", "Editar valores Prior · Tabla 2"),
            width="stretch",
        ):
            st.markdown(
                """
                <style>
                div[data-testid="stPopoverBody"] {
                    font-size: 0.86rem !important;
                }
                div[data-testid="stPopoverBody"] .katex {
                    font-size: 0.94em !important;
                }
                div[data-testid="stPopoverBody"] p {
                    margin: 0.08rem 0 !important;
                    white-space: nowrap !important;
                    overflow: hidden !important;
                    text-overflow: ellipsis !important;
                    line-height: 1.3 !important;
                }
                div[data-testid="stPopoverBody"] div[data-testid="stHorizontalBlock"] {
                    flex-wrap: nowrap !important;
                    align-items: center !important;
                }
                /* Botón primario «Guardar» solo dentro del popover: verde (evita el rojo por defecto). */
                div[data-testid="stPopoverBody"] button[data-testid="stBaseButton-primary"],
                div[data-testid="stPopover"] button[data-testid="stBaseButton-primary"] {
                    background-color: #198754 !important;
                    border: 1px solid #146c32 !important;
                    color: #ffffff !important;
                }
                div[data-testid="stPopoverBody"] button[data-testid="stBaseButton-primary"]:hover,
                div[data-testid="stPopover"] button[data-testid="stBaseButton-primary"]:hover {
                    background-color: #157347 !important;
                    border-color: #125a3a !important;
                    color: #ffffff !important;
                }
                </style>
                """,
                unsafe_allow_html=True,
            )
            _c2l1, _c2r1 = st.columns((0.68, 0.32), gap="small")
            with _c2l1:
                rb_katex_title(r"1. $T_{\mathrm{mad}}$")
            with _c2r1:
                st.number_input(
                    "Valor",
                    min_value=1.0,
                    max_value=1e6,
                    value=float(st.session_state.cal_T_mad),
                    step=1.0,
                    format="%.0f",
                    key="cal_T_mad",
                    label_visibility="collapsed",
                    help=_ui_text(r"Threshold in $M(t)=\min\{1,(t/T_{\mathrm{mad}})^2\}$ (days).", "Umbral en $M(t)=\min\{1,(t/T_{\mathrm{mad}})^2\}$ (días)."),
                )
            _c2l2, _c2r2 = st.columns((0.68, 0.32), gap="small")
            with _c2l2:
                rb_katex_title(r"2. $\lambda_4$")
            with _c2r2:
                st.number_input(
                    "Valor",
                    min_value=1e-12,
                    value=float(st.session_state.cal_lambda_4),
                    step=0.0001,
                    format="%.6f",
                    key="cal_lambda_4",
                    label_visibility="collapsed",
                    help=_ui_text("Baseline intensity of the fourth channel (exogenous), Mechanism.tex.", "Intensidad basal del cuarto canal (exógeno), Mechanism.tex."),
                )
            st.divider()
            _t2_btn_save, _t2_btn_reset = st.columns(2)
            with _t2_btn_save:
                if st.button(
                    _ui_text("Save", "Guardar"),
                    key="tab2_save_prior",
                    type="primary",
                    help=_ui_text("Confirms the edited values for this session.", "Confirma los valores editados para esta sesión."),
                ):
                    st.success(_ui_text("Table 2 **Prior** values saved to session.", "Valores **Prior** de Tabla 2 guardados en la sesión."))
                    st.rerun()
            with _t2_btn_reset:
                if st.button(
                    _ui_text("Reset", "Restablecer"),
                    key="tab2_reset_prior",
                    help=_ui_text("Reverts to base values (T_mad=5, λ4=0.0005).", "Vuelve a los valores base (T_mad=5, λ4=0,0005)."),
                ):
                    st.session_state.cal_T_mad = 5.0
                    st.session_state.cal_lambda_4 = 0.0005
                    st.session_state["cal_T_mad_default_version"] = "tmad_5"
                    st.session_state["cal_lambda_4_default_version"] = "lambda4_0p0005"
                    st.info(_ui_text("Table 2 **Prior** values reset.", "Valores **Prior** de Tabla 2 restablecidos."))
                    st.rerun()

    st.divider()
    st.markdown(r"### 3. $p_{\mathrm{det},t}$")
    _pdLa, _pdRa = st.columns((0.46, 0.54), gap="medium")
    with _pdLa:
        st.latex(
            r"p_{\mathrm{det},t}(\theta_K)=\Lambda\!\left(\eta_0(\theta_K)+\eta_1\alpha_t^\ast+\eta_2\gamma_t^\ast\right),\quad "
            r"\Lambda(u)=\frac{1}{1+e^{-u}}."
        )
        st.latex(
            r"\mathbb{P}(d_t\mid \theta_K,\alpha_t^\ast,\gamma_t^\ast)"
            r"=p_{\mathrm{det},t}(\theta_K)^{\,d_t}\bigl(1-p_{\mathrm{det},t}(\theta_K)\bigr)^{\,1-d_t}."
        )
        st.caption(
            _ui_text(
                r"**Type-specific** detectability: $\eta_0(\theta_K)$ varies across organisations; "
                r"$\eta_1,\eta_2$ capture the effect of instruments $(\alpha_t^\ast,\gamma_t^\ast)$; "
                r"Bernoulli signal $d_t\in\{0,1\}$.",
                r"Detectabilidad **tipo-específica**: $\eta_0(\theta_K)$ varía entre organizaciones; "
                r"$\eta_1,\eta_2$ capturan el efecto de instrumentos $(\alpha_t^\ast,\gamma_t^\ast)$; "
                r"señal Bernoulli $d_t\in\{0,1\}$.",
            )
        )
    with _pdRa:
        rb_katex_title(
            _ui_text(
                r"#### Table 3 · $\eta_0,\eta_1,\eta_2$ in $p_{\mathrm{det},t}$ (Prior)",
                r"#### Tabla 3 · $\eta_0,\eta_1,\eta_2$ en $p_{\mathrm{det},t}$ (Prior)",
            )
        )
        _e1 = float(st.session_state.cal_eta1_pdet)
        _e2 = float(st.session_state.cal_eta2_pdet)
        _t3_intercept_lbl = _ui_text("Intercept logit (p_det)", "Intercepto logit (p_det)")
        _t3_w_alpha_lbl   = _ui_text("Weight of α* in p_det",  "Peso de α* en p_det")
        _t3_w_gamma_lbl   = _ui_text("Weight of γ* in p_det",  "Peso de γ* en p_det")
        _df_pdet = pd.DataFrame(
            [
                {
                    "#": _idx_eta,
                    "Término": f"{_t3_intercept_lbl} · {_th_eta}",
                    "Coeficiente": rf"\eta_0(\mathrm{{{_th_eta}}})",
                    "Valor": _pdet_eta0_for_theta(_th_eta),
                    "Origen del valor": "Prior",
                }
                for _idx_eta, _th_eta in enumerate(("DC", "PAR", "ELN", "FARC"), start=1)
            ]
            + [
                {
                    "#": 5,
                    "Término": _t3_w_alpha_lbl,
                    "Coeficiente": r"\eta_1",
                    "Valor": _e1,
                    "Origen del valor": "Prior",
                },
                {
                    "#": 6,
                    "Término": _t3_w_gamma_lbl,
                    "Coeficiente": r"\eta_2",
                    "Valor": _e2,
                    "Origen del valor": "Prior",
                },
            ]
        )
        _df_pdet["Clase_tab7"] = "Prior"
        _render_focus_covariate_katex_table(
            _df_pdet,
            show_origen=False,
            font_boost_pt=2.0,
            compact_iframe_bottom=True,
            iframe_slack_px=0,
            collapse_gap_below=True,
        )
        with st.popover(
            _ui_text("Edit Prior values · Table 3", "Editar valores Prior · Tabla 3"),
            width="stretch",
        ):
            st.markdown(
                """
                <style>
                div[data-testid="stPopoverBody"] {
                    font-size: 0.86rem !important;
                }
                div[data-testid="stPopoverBody"] .katex {
                    font-size: 0.94em !important;
                }
                div[data-testid="stPopoverBody"] p {
                    margin: 0.08rem 0 !important;
                    white-space: nowrap !important;
                    overflow: hidden !important;
                    text-overflow: ellipsis !important;
                    line-height: 1.3 !important;
                }
                div[data-testid="stPopoverBody"] div[data-testid="stHorizontalBlock"] {
                    flex-wrap: nowrap !important;
                    align-items: center !important;
                }
                /* Botón primario «Guardar» solo dentro del popover: verde (evita el rojo por defecto). */
                div[data-testid="stPopoverBody"] button[data-testid="stBaseButton-primary"],
                div[data-testid="stPopover"] button[data-testid="stBaseButton-primary"] {
                    background-color: #198754 !important;
                    border: 1px solid #146c32 !important;
                    color: #ffffff !important;
                }
                div[data-testid="stPopoverBody"] button[data-testid="stBaseButton-primary"]:hover,
                div[data-testid="stPopover"] button[data-testid="stBaseButton-primary"]:hover {
                    background-color: #157347 !important;
                    border-color: #125a3a !important;
                    color: #ffffff !important;
                }
                </style>
                """,
                unsafe_allow_html=True,
            )
            rb_katex_title(
                _ui_text(
                    r"1. $\eta_0(\theta_K)$ — intercept by type",
                    r"1. $\eta_0(\theta_K)$ — intercepto por tipo",
                )
            )
            for _th_ui in ("DC", "PAR", "ELN", "FARC"):
                _c3l1, _c3r1 = st.columns((0.68, 0.32), gap="small")
                with _c3l1:
                    rb_katex_title(rf"$\eta_0(\mathrm{{{_th_ui}}})$")
                with _c3r1:
                    st.number_input(
                        _ui_text("Value", "Valor"),
                        value=float(st.session_state.get(f"cal_eta0_pdet_{_th_ui}", _ETA0_PDET_DEFAULTS.get(_th_ui, -2.0))),
                        step=0.05,
                        format="%.2f",
                        key=f"cal_eta0_pdet_{_th_ui}",
                        label_visibility="collapsed",
                        help=_ui_text(
                            rf"Baseline detectability of $\theta_K=\mathrm{{{_th_ui}}}$: $\eta_0(\mathrm{{{_th_ui}}})+\eta_1\alpha^\ast+\eta_2\gamma^\ast$.",
                            rf"Detectabilidad basal de $\theta_K=\mathrm{{{_th_ui}}}$: $\eta_0(\mathrm{{{_th_ui}}})+\eta_1\alpha^\ast+\eta_2\gamma^\ast$.",
                        ),
                    )
            _c3l2, _c3r2 = st.columns((0.68, 0.32), gap="small")
            with _c3l2:
                rb_katex_title(r"2. $\eta_1$")
            with _c3r2:
                st.number_input(
                    _ui_text("Value", "Valor"),
                    value=float(st.session_state.cal_eta1_pdet),
                    step=0.05,
                    format="%.2f",
                    key="cal_eta1_pdet",
                    label_visibility="collapsed",
                    help=_ui_text(
                        r"Weight of $\alpha_t^\ast$ (blockade / related instrument).",
                        r"Peso de $\alpha_t^\ast$ (bloqueo / instrumento relacionado).",
                    ),
                )
            _c3l3, _c3r3 = st.columns((0.68, 0.32), gap="small")
            with _c3l3:
                rb_katex_title(r"3. $\eta_2$")
            with _c3r3:
                st.number_input(
                    _ui_text("Value", "Valor"),
                    value=float(st.session_state.cal_eta2_pdet),
                    step=0.05,
                    format="%.2f",
                    key="cal_eta2_pdet",
                    label_visibility="collapsed",
                    help=_ui_text(
                        r"Weight of $\gamma_t^\ast$ (operational pressure).",
                        r"Peso de $\gamma_t^\ast$ (presión operativa).",
                    ),
                )
            st.divider()
            _t3_btn_save, _t3_btn_reset = st.columns(2)
            with _t3_btn_save:
                if st.button(
                    _ui_text("Save", "Guardar"),
                    key="tab3_save_prior",
                    type="primary",
                    help=_ui_text(
                        "Confirms the edited values for this session.",
                        "Confirma los valores editados para esta sesión.",
                    ),
                ):
                    st.success(_ui_text(
                        "**Prior** values for Table 3 saved in session.",
                        "Valores **Prior** de Tabla 3 guardados en la sesión.",
                    ))
                    st.rerun()
            with _t3_btn_reset:
                if st.button(
                    _ui_text("Reset", "Restablecer"),
                    key="tab3_reset_prior",
                    help=_ui_text(
                        "Returns to base values (η₀(DC)=-1.5, η₀(PAR)=-2.0, η₀(ELN)=-2.5, η₀(FARC)=-2.8, η₁=1, η₂=1).",
                        "Vuelve a los valores base (η₀(DC)=-1.5, η₀(PAR)=-2.0, η₀(ELN)=-2.5, η₀(FARC)=-2.8, η₁=1, η₂=1).",
                    ),
                ):
                    for _th_rst, _v_rst in _ETA0_PDET_DEFAULTS.items():
                        st.session_state[f"cal_eta0_pdet_{_th_rst}"] = _v_rst
                    st.session_state.cal_eta1_pdet = 1.0
                    st.session_state.cal_eta2_pdet = 1.0
                    st.info(_ui_text(
                        "**Prior** values for Table 3 reset.",
                        "Valores **Prior** de Tabla 3 restablecidos.",
                    ))
                    st.rerun()

    # ── Sorteo d₀ ~ Bernoulli(p_det,0) — Mechanism.tex ────────────────────────
    _theta_true_s3 = str(st.session_state.get("tipo_real_cal", "DC"))
    _eta0_s3 = float(st.session_state.get(f"cal_eta0_pdet_{_theta_true_s3}", _ETA0_PDET_DEFAULTS.get(_theta_true_s3, -2.0)))
    _eta1_s3 = float(st.session_state.get("cal_eta1_pdet", 1.0))
    _eta2_s3 = float(st.session_state.get("cal_eta2_pdet", 1.0))
    _a0_s3   = float(st.session_state.get("h0_alpha", 0.20))
    _g0_s3   = float(st.session_state.get("h0_gamma", 0.90))
    _idx_s3  = float(_eta0_s3 + _eta1_s3 * _a0_s3 + _eta2_s3 * _g0_s3)
    _pdet_s3 = float(1.0 / (1.0 + np.exp(-_idx_s3)))
    if "h0_d" not in st.session_state or str(st.session_state.h0_d) not in ("0", "1"):
        st.session_state.h0_d = "0"
    _d0_cur_s3 = str(st.session_state.h0_d)
    _sc1, _sc2, _sc3 = st.columns((0.38, 0.24, 0.38))
    with _sc1:
        st.metric(
            label=r"$p_{\mathrm{det},0}=\Lambda(\eta_0+\eta_1\alpha_0+\eta_2\gamma_0)$",
            value=f"{_pdet_s3:.4f}",
            help=(
                f"η₀={_eta0_s3:.2f}, η₁={_eta1_s3:.2f}, η₂={_eta2_s3:.2f}  |  "
                f"α₀={_a0_s3:.2f}, γ₀={_g0_s3:.2f}  |  "
                f"index={_idx_s3:.4f}"
            ),
        )
    with _sc2:
        st.metric(
            label=_ui_text("$d_0$ current", "$d_0$ actual"),
            value=_d0_cur_s3,
            help=_ui_text("Result of last Bernoulli(p_det,0) draw.", "Resultado del último sorteo Bernoulli(p_det,0)."),
        )
    with _sc3:
        if st.button(
            _ui_text("Generate d₀", "Generar d₀"),
            key="btn_gen_d0",
            use_container_width=True,
            help=_ui_text(r"Draws d₀ ~ Bernoulli(p_{det,0}) and updates everything depending on d₀.", r"Sortea d₀ ~ Bernoulli(p_{det,0}) y actualiza todo lo que depende de d₀."),
        ):
            _u_s3 = float(np.random.uniform(0.0, 1.0))
            _d0_gen = 1 if _u_s3 < _pdet_s3 else 0
            st.session_state.h0_d = str(_d0_gen)
            st.session_state["h0_d_u_draw"] = float(_u_s3)
            st.session_state["h0_d_pdet_used"] = float(_pdet_s3)
            st.rerun()
    _u_last_s3 = st.session_state.get("h0_d_u_draw")
    if _u_last_s3 is not None:
        _pdet_used_s3 = float(st.session_state.get("h0_d_pdet_used", _pdet_s3))
        st.caption(
            _ui_text(
                f"Last draw: u = {float(_u_last_s3):.4f} "
                f"{'<' if float(_u_last_s3) < _pdet_used_s3 else '≥'} "
                f"p_det,0 = {_pdet_used_s3:.4f} "
                f"→ **d₀ = {_d0_cur_s3}**",
                f"Último sorteo: u = {float(_u_last_s3):.4f} "
                f"{'<' if float(_u_last_s3) < _pdet_used_s3 else '≥'} "
                f"p_det,0 = {_pdet_used_s3:.4f} "
                f"→ **d₀ = {_d0_cur_s3}**",
            )
        )

    st.divider()
    st.markdown(_ui_text("### 4. Survival Probability", "### 4. Probabilidad de supervivencia"))
    _sv2La, _sv2Ra = st.columns((0.46, 0.54), gap="medium")
    with _sv2La:
        st.caption(
            _ui_text(r"**Eq. 37:** Modal precision · **Eq. 38:** Survival (adjusted logit).", r"**Ec. 37:** Precisión modal · **Ec. 38:** Supervivencia (logit ajustado).")
        )
        st.latex(
            r"\iota_t := \max_{\theta\in\Theta_K}\mu_t(\theta),\quad "
            r"\hat\theta_t = \operatorname*{arg\,max}_{\theta\in\Theta_K} \mu_t(\theta)."
        )
        st.latex(
            r"\mathbb{P}_{\mathrm{E}}(\mathrm{surv}\mid\iota_t,\hat\theta_t,\theta_K)="
            r"\Lambda\Bigl(\alpha_{\mathrm{leth}}(\theta_K)+\beta_R\,\iota_t\,\mathbf{1}\{\hat\theta_t=\theta_K\}\Bigr),\quad "
            r"\Lambda(u)=\frac{1}{1+e^{-u}}."
        )
        st.caption(
            _ui_text(
                r"$\iota_t$ and $\hat\theta_t$ are computed from **μ**; "
                r"$\alpha_{\mathrm{leth}}<0$ and high $\beta_R$ imply that, when $\iota_t$ is not close to 1, "
                r"modal survival falls below 0.5.",
                r"$\iota_t$ y $\hat\theta_t$ se calculan desde **μ**; "
                r"$\alpha_{\mathrm{leth}}<0$ y $\beta_R$ alto hacen que, si $\iota_t$ no está cerca de 1, "
                r"la supervivencia modal quede por debajo de 0.5.",
            )
        )
    with _sv2Ra:
        rb_katex_title(r"#### Tabla 4 · $\alpha_0$, $\beta_R$, $\iota_t$")
        st.caption(
            _ui_text(r"**Prior** for the survival logit; **Computed** from μ₀ and focal type **θ_K** above.", r"**Prior** para la logit de supervivencia; **Calculado** desde μ₀ y el tipo focal **θ_K** arriba.")
        )
        _a0s = float(st.session_state.cal_surv_alpha0[_th])
        _bRs = float(st.session_state.cal_surv_beta_R)
        _df_surv = pd.DataFrame(
            [
                {
                    "#": 1,
                    "Término": _ui_text("α_leth lethality (Prior · eq. 38)", "α_leth letalidad (Prior · ec. 38)"),
                    "Coeficiente": r"\alpha_{\mathrm{leth}}(\theta_K)",
                    "Valor": _a0s,
                    "Origen del valor": "Prior",
                },
                {
                    "#": 2,
                    "Término": _ui_text("β_R productivity (Prior · eq. 38)", "β_R productividad (Prior · ec. 38)"),
                    "Coeficiente": r"\beta_R",
                    "Valor": _bRs,
                    "Origen del valor": "Prior",
                },
                {
                    "#": 3,
                    "Término": _ui_text("Modal precision (eq. 37)", "Precisión modal (ec. 37)"),
                    "Coeficiente": r"\iota_t",
                    "Valor": float(_iota_t),
                    "Origen del valor": "Calculado",
                },
                {
                    "#": 4,
                    "Término": _ui_text("Match indicator (eq. 38)", "Indicadora coincidencia (ec. 38)"),
                    "Coeficiente": r"\mathbf{1}\{\hat\theta_t = \theta_K\}",
                    "Valor": 1.0 if TIPOS_SECUESTRADOR[_mu_vec.index(_iota_t)] == _th else 0.0,
                    "Origen del valor": "Calculado",
                },
            ]
        )
        _df_surv["Clase_tab7"] = _df_surv["Origen del valor"].apply(
            lambda x: "Prior" if x == "Prior" else "Endogenous"
        )
        _render_focus_covariate_katex_table(
            _df_surv,
            show_origen=False,
            font_boost_pt=2.0,
            compact_iframe_bottom=True,
            iframe_slack_px=0,
            collapse_gap_below=True,
        )
        with st.popover(
            _ui_text("Edit Prior Values · Table 4", "Editar valores Prior · Tabla 4"),
            width="stretch",
        ):
            st.markdown(
                """
                <style>
                div[data-testid="stPopoverBody"] {
                    font-size: 0.86rem !important;
                }
                div[data-testid="stPopoverBody"] .katex {
                    font-size: 0.94em !important;
                }
                div[data-testid="stPopoverBody"] p {
                    margin: 0.08rem 0 !important;
                    white-space: nowrap !important;
                    overflow: hidden !important;
                    text-overflow: ellipsis !important;
                    line-height: 1.3 !important;
                }
                div[data-testid="stPopoverBody"] div[data-testid="stHorizontalBlock"] {
                    flex-wrap: nowrap !important;
                    align-items: center !important;
                }
                /* Botón primario «Guardar» solo dentro del popover: verde (evita el rojo por defecto). */
                div[data-testid="stPopoverBody"] button[data-testid="stBaseButton-primary"],
                div[data-testid="stPopover"] button[data-testid="stBaseButton-primary"] {
                    background-color: #198754 !important;
                    border: 1px solid #146c32 !important;
                    color: #ffffff !important;
                }
                div[data-testid="stPopoverBody"] button[data-testid="stBaseButton-primary"]:hover,
                div[data-testid="stPopover"] button[data-testid="stBaseButton-primary"]:hover {
                    background-color: #157347 !important;
                    border-color: #125a3a !important;
                    color: #ffffff !important;
                }
                </style>
                """,
                unsafe_allow_html=True,
            )
            _s4l1, _s4r1 = st.columns((0.68, 0.32), gap="small")
            with _s4l1:
                rb_katex_title(r"1. $\alpha_{\mathrm{leth}}(\theta_K)$")
                st.caption(_ui_text("Intrinsic lethality (Prior · eq. 38)", "Letalidad intrínseca (Prior · ec. 38)"))
            with _s4r1:
                for _tk in TIPOS_SECUESTRADOR:
                    st.number_input(
                        f"α_leth ({_tk})",
                        value=float(st.session_state.cal_surv_alpha0[_tk]),
                        step=0.05,
                        format="%.2f",
                        key=f"cal_surv_alpha0_{_tk}",
                        help=_ui_text(f"Intrinsic lethality for {_tk}.", f"Letalidad intrínseca para {_tk}."),
                    )
                    # Sincronizar con el diccionario
                    st.session_state.cal_surv_alpha0[_tk] = st.session_state[f"cal_surv_alpha0_{_tk}"]
            _s4l2, _s4r2 = st.columns((0.68, 0.32), gap="small")
            with _s4l2:
                rb_katex_title(r"2. $\beta_R$")
            with _s4r2:
                st.number_input(
                    "Valor",
                    value=float(st.session_state.cal_surv_beta_R),
                    step=0.05,
                    format="%.2f",
                    key="cal_surv_beta_R",
                    label_visibility="collapsed",
                    help=_ui_text(r"Weight of $\iota_t$ when $\hat\theta_t=\theta_K$ (eq. 38).", r"Peso de $\iota_t$ cuando $\hat\theta_t=\theta_K$ (ec. 38)."),
                )
            st.divider()
            _t4_btn_save, _t4_btn_reset = st.columns(2)
            with _t4_btn_save:
                if st.button(
                    _ui_text("Save", "Guardar"),
                    key="tab4_save_prior",
                    type="primary",
                    help=_ui_text("Confirms the edited values for this session.", "Confirma los valores editados para esta sesión."),
                ):
                    st.success(_ui_text("Table 4 **Prior** values saved to session.", "Valores **Prior** de Tabla 4 guardados en la sesión."))
                    st.rerun()
            with _t4_btn_reset:
                if st.button(
                    _ui_text("Reset", "Restablecer"),
                    key="tab4_reset_prior",
                    help=_ui_text("Reverts to μ-sensitive values: negative α_leth and high β_R.", "Vuelve a los valores sensibles a μ: α_leth negativos y β_R alto."),
                ):
                    st.session_state.cal_surv_alpha0 = dict(_SURV_ALPHA0_SENSITIVE_DEFAULTS)
                    st.session_state.cal_surv_beta_R = float(_SURV_BETA_R_SENSITIVE_DEFAULT)
                    st.session_state.cal_surv_sensitivity_version = "mu_sensitive_v2"
                    st.info(_ui_text("Table 4 **Prior** values reset.", "Valores **Prior** de Tabla 4 restablecidos."))
                    st.rerun()

    st.divider()
    st.markdown(_ui_text("### 5. Voice Measurement", "### 5. Medición de voz"))
    _vp = st.session_state.cal_voz_params[_th]
    _sync_cal_voz_from_session_widgets(_th, _vp)
    _vzL, _vzR = st.columns((1.05, 1), gap="small")
    with _vzL:
        st.latex(
            r"x_t^{obs}=\bar{x}(\theta_K)+\varepsilon_L+\varepsilon_S,\qquad "
            r"\varepsilon_L\sim\mathcal{N}(0,\Sigma_L),\quad "
            r"\varepsilon_S\sim\mathcal{N}(0,\Sigma_S)."
        )
        st.latex(
            r"\mathcal{L}_{C,t}(\theta_K \mid V_t)=\begin{cases}\bigl[\mathcal{L}_{\mathrm{voz},t}(\theta_K)\pi_{\mathrm{call}}(\theta_K)\bigr]^{\omega_{\mathrm{voz}}}, & V_t=1,\\[0.6em]\bigl[1-\pi_{\mathrm{call}}(\theta_K)\bigr]^{\omega_{\mathrm{voz}}}, & V_t=0.\end{cases}"
        )
        st.markdown(_ui_text("**Oscilloscope (voice viewer)** · illustration with first trait ($i=1$).", "**Osciloscopio (visor de voz)** · ilustración con el primer rasgo ($i=1$)."))
        if st.button(
            _ui_text("📡 Capture 10s", "📡 Capturar 10s"),
            key=f"cal_voz_cap_{_th}",
            use_container_width=True,
        ):
            st.session_state.cal_voz_osc_bundle = _cal_sample_voice_bundle(_th)
            st.session_state.cal_voz_osc_th = _th
            st.rerun()
        _osc_n1, _osc_n2 = st.columns(2, gap="small")
        with _osc_n1:
            _incl_L = st.checkbox(
                _ui_text("Include ε_L (long-term)", "Incluir ε_L (largo plazo)"),
                value=True,
                key=f"cal_voz_incl_L_{_th}",
            )
        with _osc_n2:
            _incl_S = st.checkbox(
                _ui_text("Include ε_S (short-term)", "Incluir ε_S (corto plazo)"),
                value=True,
                key=f"cal_voz_incl_S_{_th}",
            )
        _m_k0 = float(_vp["x"][0])
        _t_audio = np.linspace(0, 10, 500)
        _pure = np.sin(2 * np.pi * _m_k0 / 20 * _t_audio) * 0.5
        if (
            st.session_state.cal_voz_osc_bundle is not None
            and st.session_state.cal_voz_osc_th == _th
        ):
            _bnd = st.session_state.cal_voz_osc_bundle
            _xb0 = float(_bnd["xb"][0])
            _eL0 = float(_bnd["eL"][0])
            _eS0 = float(_bnd["eS"][0])
            _eff0 = _xb0 + (_eL0 if _incl_L else 0.0) + (_eS0 if _incl_S else 0.0)
            _seed_v = int(
                (_eff0 * 1e6 + _xb0 * 1e3 + _eL0 * 1e2 + _eS0 * 10.0) % (2**31)
            )
            _rng_v = np.random.default_rng(_seed_v)
            _hf = _rng_v.normal(0, 0.05, 500) if _incl_S else np.zeros(500)
            if not _incl_L and not _incl_S:
                _dist = _pure
            else:
                _dist = np.sin(2 * np.pi * _eff0 / 20 * _t_audio) * 0.4 + _hf
            _fig_w = go.Figure()
            _fig_w.add_trace(
                go.Scatter(
                    x=_t_audio,
                    y=_pure,
                    name=_ui_text("Pure", "Pura"),
                    line=dict(color="#00CC96", width=1.5, dash="dash"),
                )
            )
            _fig_w.add_trace(
                go.Scatter(
                    x=_t_audio,
                    y=_dist,
                    name="Obs",
                    line=dict(color="#EF553B", width=2),
                )
            )
            _fig_w.update_layout(
                title=_ui_text(f"Oscilloscope: {_th}", f"Osciloscopio: {_th}"),
                height=300,
                template="plotly_dark",
                margin=dict(t=36, b=8, l=8, r=8),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1,
                ),
            )
            st.plotly_chart(_fig_w, use_container_width=True)
        else:
            st.caption(_ui_text("Press **Capture** to generate a sample signal.", "Pulse **Capturar** para generar una señal de ejemplo."))
            st.plotly_chart(
                go.Figure().update_layout(
                    height=220,
                    title=_ui_text("Oscilloscope (Inactive)", "Osciloscopio (Inactivo)"),
                    template="plotly_dark",
                    margin=dict(t=32, b=8, l=8, r=8),
                ),
                use_container_width=True,
            )
    with _vzR:
        st.markdown(
            r"""
            <style>
            /* Tabla 5: sin márgenes negativos (misma razón que Tabla 1). */
            div[role="tabpanel"]:not([hidden]) [data-testid="stElementContainer"]:has(iframe),
            div[role="tabpanel"]:not([hidden]) [data-testid="element-container"]:has(iframe) {
                margin-bottom: 0 !important;
                padding-bottom: 0.2rem !important;
            }
            div[role="tabpanel"]:not([hidden]) [data-testid="stElementContainer"]:has([data-testid="stPopover"]),
            div[role="tabpanel"]:not([hidden]) [data-testid="element-container"]:has([data-testid="stPopover"]) {
                margin-top: 0.1rem !important;
                padding-top: 0 !important;
            }
            div[data-testid="stPopoverBody"] {
                font-size: 0.86rem !important;
            }
            div[data-testid="stPopoverBody"] div[data-testid="stHorizontalBlock"] {
                flex-wrap: wrap !important;
                align-items: flex-end !important;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        st.caption(
            _ui_text(
                rf"**Observed** for focal type **θ_K = {_th}** · $k=4$ traits · "
                r"**edit** with **Edit values · Table 5**.",
                rf"**Observado** por tipo focal **θ_K = {_th}** · $k=4$ rasgos · "
                r"**edita** con **Editar valores · Tabla 5**.",
            )
        )
        rb_katex_title(_ui_text(r"**Table 5. Voice Measurement Parameters**", r"**Tabla 5. Parámetros Medición de Voz**"))
        _render_focus_covariate_katex_table(
            _build_cal_voz_tabla5_df(_vp),
            show_origen=False,
            font_boost_pt=2.0,
            term_font_boost_pt=-1.0,
            term_line_height=1.35,
            col_width_css_override="""
.cov-katex-table-root th.num,.cov-katex-table-root td.num{width:5%;}
.cov-katex-table-root th.term,.cov-katex-table-root td.term{width:48%;}
.cov-katex-table-root th.coef,.cov-katex-table-root td.coef{width:18%;}
.cov-katex-table-root th.val,.cov-katex-table-root td.val{width:12%;}
.cov-katex-table-root th.prior-flag,.cov-katex-table-root td.prior-flag{width:17%;}
""",
            compact_iframe_bottom=True,
            iframe_slack_px=0,
            collapse_gap_below=True,
        )
        with st.popover(
            _ui_text(f"Edit values · Table 5 · {_th}", f"Editar valores · Tabla 5 · {_th}"),
            width="stretch",
        ):
            st.markdown(
                r"""
                <style>
                /* Tabla 5 (popover): evitar solapes por márgenes/line-height de Markdown/KaTeX */
                div[data-testid="stPopoverBody"] .stMarkdown,
                div[data-testid="stPopoverBody"] [data-testid="stMarkdownContainer"] {
                    margin: 0 !important;
                    padding: 0 !important;
                }
                div[data-testid="stPopoverBody"] .stMarkdown p,
                div[data-testid="stPopoverBody"] [data-testid="stMarkdownContainer"] p {
                    margin: 0 !important;
                    padding: 0 !important;
                    line-height: 1.15 !important;
                }
                div[data-testid="stPopoverBody"] [data-testid="stLatex"] {
                    margin: 0 !important;
                    padding: 0 !important;
                }
                div[data-testid="stPopoverBody"] .katex-display {
                    margin: 0.05rem 0 0.15rem 0 !important;
                }
                div[data-testid="stPopoverBody"] div[data-testid="stHorizontalBlock"] {
                    align-items: flex-start !important;
                }
                div[data-testid="stPopoverBody"] .t5-row-title {
                    margin-top: 0.35rem !important;
                    margin-bottom: 0.15rem !important;
                    line-height: 1.2 !important;
                }
                </style>
                """,
                unsafe_allow_html=True,
            )
            _t5h1, _t5h2, _t5h3 = st.columns(3, gap="small")
            with _t5h1:
                rb_katex_title(r"$\bar{x}_i$")
            with _t5h2:
                rb_katex_title(r"$\sigma_{L,i}$")
            with _t5h3:
                rb_katex_title(r"$\sigma_{S,i}$")
            for _vi in range(4):
                st.markdown(
                    f"<div class='t5-row-title'><b>{_vi + 1}. {_VOZ_RASGO_LABELS[_vi]}</b></div>",
                    unsafe_allow_html=True,
                )
                _vx, _vsl, _vss = st.columns(3)
                with _vx:
                    _vp["x"][_vi] = float(
                        st.number_input(
                            "xbar",
                            value=float(_vp["x"][_vi]),
                            format="%.2f",
                            key=f"voz_x_{_th}_{_vi}",
                            label_visibility="collapsed",
                        )
                    )
                with _vsl:
                    _vp["sigma_L"][_vi] = float(
                        st.number_input(
                            "sigma_L",
                            min_value=1e-9,
                            value=float(_vp["sigma_L"][_vi]),
                            format="%.2f",
                            step=0.001,
                            key=f"voz_sL_{_th}_{_vi}",
                            label_visibility="collapsed",
                        )
                    )
                with _vss:
                    _vp["sigma_S"][_vi] = float(
                        st.number_input(
                            "sigma_S",
                            min_value=1e-9,
                            value=float(_vp["sigma_S"][_vi]),
                            format="%.2f",
                            step=0.001,
                            key=f"voz_sS_{_th}_{_vi}",
                            label_visibility="collapsed",
                        )
                    )
            st.divider()
            _t5_btn_save, _t5_btn_reset = st.columns(2)
            with _t5_btn_save:
                if st.button(
                    _ui_text("Save", "Guardar"),
                    key=f"tab5_save_{_th}",
                    type="primary",
                    help=_ui_text("Confirms the edited values for this session.", "Confirma los valores editados para esta sesión."),
                ):
                    st.success(_ui_text("Table 5 **Observed** values saved to session.", "Valores **Observado** de Tabla 5 guardados en la sesión."))
                    st.rerun()
            with _t5_btn_reset:
                if st.button(
                    _ui_text("Reset", "Restablecer"),
                    key=f"tab5_reset_{_th}",
                    help=_ui_text("Reverts to the base prior values for this type θ_K.", "Vuelve a los valores base del prior para este tipo θ_K."),
                ):
                    _defs = _default_cal_voz_params().get(_th, None)
                    if _defs is not None:
                        st.session_state.cal_voz_params[_th] = copy.deepcopy(_defs)
                        for _i in range(4):
                            st.session_state[f"voz_x_{_th}_{_i}"] = float(_defs["x"][_i])
                            st.session_state[f"voz_sL_{_th}_{_i}"] = float(_defs["sigma_L"][_i])
                            st.session_state[f"voz_sS_{_th}_{_i}"] = float(_defs["sigma_S"][_i])
                    st.info(_ui_text("Table 5 **Observed** values reset.", "Valores **Observado** de Tabla 5 restablecidos."))
                    st.rerun()

        # Nueva sub-sección: Frecuencia de contacto
        _sync_cal_voz_extra_from_session_widgets(_th)
        _pi_val = float(st.session_state.cal_voz_pi_call[_th])
        _omega_val = float(st.session_state.cal_voz_omega)
        
        st.divider()
        rb_katex_title(_ui_text(r"**Communication urgency** · $\pi_{\mathrm{call}}$, $\omega_{\mathrm{voz}}$", r"**Urgencia de comunicación** · $\pi_{\mathrm{call}}$, $\omega_{\mathrm{voz}}$"))
        _render_focus_covariate_katex_table(
            _build_cal_voz_extra_df(_pi_val, _omega_val),
            show_origen=False,
            font_boost_pt=2.0,
            term_font_boost_pt=-1.0,
            term_line_height=1.35,
            col_width_css_override="""
.cov-katex-table-root th.num,.cov-katex-table-root td.num{width:5%;}
.cov-katex-table-root th.term,.cov-katex-table-root td.term{width:48%;}
.cov-katex-table-root th.coef,.cov-katex-table-root td.coef{width:18%;}
.cov-katex-table-root th.val,.cov-katex-table-root td.val{width:12%;}
.cov-katex-table-root th.prior-flag,.cov-katex-table-root td.prior-flag{width:17%;}
""",
            compact_iframe_bottom=True,
            iframe_slack_px=0,
            collapse_gap_below=True,
        )
        with st.popover(_ui_text(f"Edit urgency and weight · {_th}", f"Editar urgencia y peso · {_th}"), width="stretch"):
            st.markdown(_ui_text("**Voice frequency mode by type**", "**Modo de frecuencia de voz por tipo**"))
            _mode_col_hi, _mode_col_lo = st.columns(2)
            with _mode_col_hi:
                if st.button(
                    _ui_text("High frequency", "Frecuencia alta"),
                    key=f"voz_freq_high_{_th}",
                    use_container_width=True,
                    help=_ui_text("Sets DC=0.65, PAR=0.55, ELN=0.45, FARC=0.35.", "Aplica DC=0.65, PAR=0.55, ELN=0.45, FARC=0.35."),
                ):
                    st.session_state.cal_voz_pi_call = dict(_VOICE_PI_FREQ_HIGH)
                    for _th_pi in TIPOS_SECUESTRADOR:
                        st.session_state[f"voz_pi_{_th_pi}"] = float(_VOICE_PI_FREQ_HIGH[_th_pi])
                    _clear_dynamic_cycles_only()
                    st.session_state["voice_frequency_mode"] = "alta"
                    st.success(_ui_text("High frequency applied. Use Advance Cycles to recalculate τ≥1.", "Frecuencia alta aplicada. Use Avanzar ciclos para recalcular τ≥1."))
                    st.rerun()
            with _mode_col_lo:
                if st.button(
                    _ui_text("Low frequency", "Frecuencia baja"),
                    key=f"voz_freq_low_{_th}",
                    use_container_width=True,
                    help=_ui_text("Sets DC=0.13, PAR=0.11, ELN=0.09, FARC=0.07.", "Aplica DC=0.13, PAR=0.11, ELN=0.09, FARC=0.07."),
                ):
                    st.session_state.cal_voz_pi_call = dict(_VOICE_PI_FREQ_LOW)
                    for _th_pi in TIPOS_SECUESTRADOR:
                        st.session_state[f"voz_pi_{_th_pi}"] = float(_VOICE_PI_FREQ_LOW[_th_pi])
                    _clear_dynamic_cycles_only()
                    st.session_state["voice_frequency_mode"] = "baja"
                    st.success(_ui_text("Low frequency applied. Use Advance Cycles to recalculate τ≥1.", "Frecuencia baja aplicada. Use Avanzar ciclos para recalcular τ≥1."))
                    st.rerun()
            st.caption(
                _ui_text(
                    "Scale applied in both modes: DC > PAR > ELN > FARC. "
                    "Low frequency is 20% of high.",
                    "Escala aplicada en ambos modos: DC > PAR > ELN > FARC. "
                    "La frecuencia baja es el 20% de la alta.",
                )
            )
            st.divider()
            _pi_col, _om_col = st.columns(2)
            with _pi_col:
                st.slider(
                    _ui_text(r"$\pi_{\mathrm{call}}$ (Urgency)", r"$\pi_{\mathrm{call}}$ (Urgencia)"),
                    0.01, 0.99, _pi_val, 0.01,
                    key=f"voz_pi_{_th}",
                    help=_ui_text("Daily voice emission probability.", "Probabilidad diaria de emisión de voz.")
                )
            with _om_col:
                st.slider(
                    _ui_text(r"$\omega_{\mathrm{voz}}$ (Weight)", r"$\omega_{\mathrm{voz}}$ (Peso)"),
                    0.0, 1.0, _omega_val, 0.05,
                    key="voz_omega",
                    help=_ui_text("Moderator of voice learning (eq. 1138).", "Moderador del aprendizaje por voz (ec. 1138).")
                )
            if st.button(_ui_text("Save urgency", "Guardar urgencia"), key=f"btn_save_pi_{_th}", type="primary"):
                st.success(_ui_text("Urgency parameters saved.", "Parámetros de urgencia guardados."))
                st.rerun()

    st.divider()
    st.markdown(_ui_text("### 6. Capture Probability", "### 6. Probabilidad de captura"))
    _pcap_L, _pcap_R = st.columns((1.05, 1), gap="small")
    with _pcap_L:
        st.latex(
            r"p_{\mathrm{cap},t}(\theta_K) = \Lambda\Bigl(\delta_a + c_0(\theta_K) + "
            r"c_\alpha(\theta_K)\alpha_t^\ast + c_\gamma(\theta_K)\gamma_t^\ast + c_S(\theta_S)\Bigr)"
        )
        st.caption(
            _ui_text(
                "Measures the technical capture probability given the environment and applied policies. "
                "It is fundamental for the captive's survival likelihood.",
                "Mide la probabilidad técnica de captura dado el entorno y las políticas aplicadas. "
                "Es fundamental para la verosimilitud de supervivencia del captor.",
            )
        )
    with _pcap_R:
        _prow = st.session_state.cal_pcap_params[_th]
        _delta_a = float(st.session_state.cal_pcap_delta_a)
        _c_S = float(st.session_state.cal_pcap_c_S)
        _sync_cal_pcap_from_session_widgets(_th, _prow)
        
        rb_katex_title(_ui_text(r"**Table 6. Technical Capture Parameters**", r"**Tabla 6. Parámetros de Captura Técnica**"))
        _render_focus_covariate_katex_table(
            _build_cal_pcap_tabla6_df(_prow, _delta_a, _c_S),
            show_origen=False,
            font_boost_pt=2.0,
            term_font_boost_pt=-1.0,
            term_line_height=1.35,
            col_width_css_override="""
.cov-katex-table-root th.num,.cov-katex-table-root td.num{width:5%;}
.cov-katex-table-root th.term,.cov-katex-table-root td.term{width:48%;}
.cov-katex-table-root th.coef,.cov-katex-table-root td.coef{width:18%;}
.cov-katex-table-root th.val,.cov-katex-table-root td.val{width:12%;}
.cov-katex-table-root th.prior-flag,.cov-katex-table-root td.prior-flag{width:17%;}
""",
            compact_iframe_bottom=True,
            iframe_slack_px=0,
            collapse_gap_below=True,
        )
        with st.popover(_ui_text(f"Edit capture · Table 6 · {_th}", f"Editar captura · Tabla 6 · {_th}"), width="stretch"):
            _pc1, _pc2 = st.columns(2)
            with _pc1:
                rb_katex_title(r"1. $\delta_a$ — " + _ui_text("action impact", "impacto acción"))
                st.number_input("delta_a", value=_delta_a, step=0.1, key="pcap_delta_a_widget", label_visibility="collapsed")
                st.session_state.cal_pcap_delta_a = st.session_state.pcap_delta_a_widget
                rb_katex_title(r"2. $c_0(\theta_K)$ — " + _ui_text("baseline heterogeneity", "heterogeneidad basal"))
                st.number_input("c0", value=float(_prow["c0"]), step=0.1, key=f"pcap_c0_{_th}", label_visibility="collapsed")
            with _pc2:
                rb_katex_title(r"3. $c_S(\theta_S)$ — " + _ui_text("institutional capacity", "capacidad institucional"))
                st.number_input("c_S", value=_c_S, step=0.1, key="pcap_c_S_widget", label_visibility="collapsed")
                st.session_state.cal_pcap_c_S = st.session_state.pcap_c_S_widget
                rb_katex_title(r"4. $c_\alpha(\theta_K)$ — " + _ui_text("blockade sensitivity", "sensibilidad bloqueo"))
                st.number_input("c_alpha", value=float(_prow["c_alpha"]), step=0.1, key=f"pcap_c_alpha_{_th}", label_visibility="collapsed")
            rb_katex_title(r"5. $c_\gamma(\theta_K)$ — " + _ui_text("pressure sensitivity", "sensibilidad presión"))
            st.number_input("c_gamma", value=float(_prow["c_gamma"]), step=0.1, key=f"pcap_c_gamma_{_th}", label_visibility="collapsed")
            st.divider()
            _btn_pc_save, _btn_pc_reset = st.columns(2)
            with _btn_pc_save:
                if st.button(_ui_text("Save capture", "Guardar captura"), key=f"btn_save_pcap_{_th}", type="primary"):
                    st.success(_ui_text("Capture parameters saved.", "Parámetros de captura guardados."))
                    st.rerun()
            with _btn_pc_reset:
                if st.button(_ui_text("Reset capture", "Restablecer captura"), key=f"btn_reset_pcap_{_th}"):
                    _defs = _default_cal_pcap_params().get(_th)
                    st.session_state.cal_pcap_params[_th] = _defs
                    st.session_state.cal_pcap_delta_a = 0.0
                    st.session_state.cal_pcap_c_S = 0.0
                    st.rerun()


modelo = ModeloSecuestro(
    betas=copy.deepcopy(st.session_state.cal_betas_dict),
    lambdas_0=copy.deepcopy(st.session_state.cal_lambdas_dict),
)
modelo.T_mad = float(st.session_state.get("cal_T_mad", 30.0))
presion_S = float(st.session_state.cal_presion_S)
alpha_star = float(st.session_state.get("cal_alpha_star", 0.0))

with tab_mdg:
    st.markdown(
        r"""
        <style>
        /*
         * Tabla 7 (MDG): título + KaTeX + «Editar valores» en columna derecha (más ancho); ecuaciones a la izquierda.
         * Cero hueco entre bloques; el botón «Editar» va justo debajo del iframe de la tabla.
         */
        [data-testid="column"]:has(h4.mdg-tabla7-title),
        [data-testid="stColumn"]:has(h4.mdg-tabla7-title) {
            margin-right: 0 !important;
            padding-right: 0 !important;
        }
        /* Toda la columna de Tabla 7: sin separación vertical entre título, iframe y popover. */
        [data-testid="column"]:has(h4.mdg-tabla7-title) > div[data-testid="stVerticalBlock"],
        [data-testid="stColumn"]:has(h4.mdg-tabla7-title) > div[data-testid="stVerticalBlock"] {
            gap: 0 !important;
            row-gap: 0 !important;
        }
        [data-testid="column"]:has(h4.mdg-tabla7-title),
        [data-testid="stColumn"]:has(h4.mdg-tabla7-title),
        div[data-testid="stVerticalBlock"]:has(h4.mdg-tabla7-title) {
            gap: 0 !important;
            row-gap: 0 !important;
        }
        [data-testid="column"]:has(h4.mdg-tabla7-title) div[data-testid="stVerticalBlock"],
        [data-testid="stColumn"]:has(h4.mdg-tabla7-title) div[data-testid="stVerticalBlock"],
        div[data-testid="stVerticalBlock"]:has(h4.mdg-tabla7-title) div[data-testid="stVerticalBlock"] {
            gap: 0 !important;
            row-gap: 0 !important;
        }
        h4.mdg-tabla7-title {
            margin-top: 0 !important;
            margin-bottom: 0.2rem !important;
            padding: 0 !important;
            line-height: 1.2 !important;
            font-size: 0.98rem;
            font-weight: 600;
        }
        [data-testid="column"]:has(h4.mdg-tabla7-title) [data-testid="stMarkdownContainer"],
        [data-testid="stColumn"]:has(h4.mdg-tabla7-title) [data-testid="stMarkdownContainer"],
        div[data-testid="stVerticalBlock"]:has(h4.mdg-tabla7-title) [data-testid="stMarkdownContainer"] {
            margin-bottom: 0 !important;
            padding-bottom: 0 !important;
        }
        [data-testid="column"]:has(h4.mdg-tabla7-title) [data-testid="stMarkdownContainer"] p,
        [data-testid="stColumn"]:has(h4.mdg-tabla7-title) [data-testid="stMarkdownContainer"] p {
            margin: 0 !important;
        }
        [data-testid="column"]:has(h4.mdg-tabla7-title) [data-testid="element-container"]:has(iframe),
        [data-testid="stColumn"]:has(h4.mdg-tabla7-title) [data-testid="element-container"]:has(iframe),
        div[data-testid="stVerticalBlock"]:has(h4.mdg-tabla7-title) [data-testid="element-container"]:has(iframe) {
            margin-top: 0 !important;
            margin-bottom: 0 !important;
            padding-top: 0 !important;
            padding-bottom: 0 !important;
            line-height: 0 !important;
            overflow: visible !important;
        }
        [data-testid="column"]:has(h4.mdg-tabla7-title) [data-testid="element-container"]:has(iframe) > div,
        [data-testid="stColumn"]:has(h4.mdg-tabla7-title) [data-testid="element-container"]:has(iframe) > div {
            margin: 0 !important;
            padding: 0 !important;
            line-height: 0 !important;
        }
        [data-testid="column"]:has(h4.mdg-tabla7-title) [data-testid="element-container"]:has(iframe) iframe,
        [data-testid="stColumn"]:has(h4.mdg-tabla7-title) [data-testid="element-container"]:has(iframe) iframe {
            display: block !important;
            margin: 0 !important;
            padding: 0 !important;
            border: 0 !important;
            vertical-align: bottom !important;
        }
        /* «Editar valores» inmediatamente debajo del iframe (hermano siguiente en la columna). */
        [data-testid="column"]:has(h4.mdg-tabla7-title) [data-testid="element-container"]:has(iframe) + [data-testid="element-container"],
        [data-testid="stColumn"]:has(h4.mdg-tabla7-title) [data-testid="element-container"]:has(iframe) + [data-testid="element-container"],
        div[data-testid="stVerticalBlock"]:has(h4.mdg-tabla7-title) [data-testid="element-container"]:has(iframe) + [data-testid="element-container"] {
            margin-top: 2px !important;
            padding-top: 0 !important;
        }
        [data-testid="column"]:has(h4.mdg-tabla7-title) [data-testid="element-container"]:has([data-testid="stPopover"]),
        [data-testid="stColumn"]:has(h4.mdg-tabla7-title) [data-testid="element-container"]:has([data-testid="stPopover"]),
        div[data-testid="stVerticalBlock"]:has(h4.mdg-tabla7-title) [data-testid="element-container"]:has([data-testid="stPopover"]),
        div[data-testid="stVerticalBlock"]:has(h4.mdg-tabla7-title) div[data-testid="stVerticalBlock"] [data-testid="element-container"]:has([data-testid="stPopover"]) {
            margin-top: 2px !important;
            padding-top: 0 !important;
        }
        [data-testid="column"]:has(h4.mdg-tabla7-title) [data-testid="stPopover"],
        [data-testid="stColumn"]:has(h4.mdg-tabla7-title) [data-testid="stPopover"],
        div[data-testid="stVerticalBlock"]:has(h4.mdg-tabla7-title) [data-testid="stPopover"] {
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("## The Mano de Dios–Guadalupe (MDG)")

    _mu_mdg_tab = {
        t: float(st.session_state.final_priors[i]) / 100.0
        for i, t in enumerate(TIPOS_SECUESTRADOR)
    }
    _R_mdg = 45.0 if f_capa == "Alta Riqueza" else 18.0
    _al_mdg, _ga_mdg = (0.38, 0.42) if s_tipo == "Duro" else (0.20, 0.28)
    _cmh_alive_mdg, _cmh_kill_mdg = cmh_alive_and_kill_shares()
    _rb = lambda k, d: float(st.session_state.get(k, d))
    _p_cap_mdg = _rb("rb_pcap", 0.12)
    _beta_k_mdg = _rb("rb_betak", 0.92)
    _V_L_mdg = _rb("rb_vl", 100.0)
    _F_col_mdg = _rb("rb_fcol", 40.0)
    _phi_mdg, _kap_mdg, _nu_mdg = _rb_family_phi_kappa_nu(f_capa)
    _pd0_mdg = _rb("rb_pdet0", 0.08)
    _pda_mdg = _rb("rb_pdeta", 0.35)
    _omk_mdg = _rb("rb_omk", 350.0)
    _omp_mdg = _rb("rb_omp", 15.0)
    _omg_mdg = _rb("rb_omg", 1.2)
    _ops_mdg = (
        _rb("rb_ops0", 2.0), _rb("rb_ops1", 0.6), _rb("rb_ops2", 0.9),
        _rb("rb_ops3", 0.30), _rb("rb_ops4", 0.40), _rb("rb_ops5", 0.20),
    )
    _mt_mdg = (
        _rb("rb_mt0", 1.5), _rb("rb_mt1", 0.45), _rb("rb_mt2", 0.75),
        _rb("rb_mt3", 0.25), _rb("rb_mt4", 0.35), _rb("rb_mt5", 0.15),
    )
    _cinst_mdg = (_rb("rb_calp", 0.8), _rb("rb_cgam", 0.5), _rb("rb_ccross", 0.2))

    _H_mu_mdg = shannon_entropy(_mu_mdg_tab)
    _H0_ref_mdg = float(np.log(len(TIPOS_SECUESTRADOR)))
    _agentes_mdg = [
        ("1 · Secuestrador (K)", "K"),
        ("2 · Familia (F)", "F"),
        ("3 · Estado (S)", "S"),
    ]
    _agent_mdg_label = st.selectbox(
        "Agente a analizar (MDG):",
        options=[l for l, c in _agentes_mdg],
        key="mdg_agent_selector"
    )
    _mdg_focal_code = next(c for l, c in _agentes_mdg if l == _agent_mdg_label)

    # Renderizado del análisis para el agente seleccionado
    for _agent_label, _mdg_focal_code in _agentes_mdg:
        if _agent_label == _agent_mdg_label:
            _actions_mdg: list[str] = []
            _astar_mdg = ""
            try:
                if _mdg_focal_code == "K":
                    _, _df_uk, _ = compute_kidnapper_by_type_tables(
                        modelo, _mu_mdg_tab, presion_S, _al_mdg, _ga_mdg, _R_mdg,
                        _p_cap_mdg, s_tipo == "Duro", tipo_real, _beta_k_mdg,
                    )
                    _rowk = _df_uk[_df_uk["theta_K"] == tipo_real].iloc[0]
                    _astar_mdg = str(_rowk["rama_optima"])
                    _actions_mdg = ["Liberar (a_rel)", "Matar (a_kill)", "Continuar (a_cont)"]
                elif _mdg_focal_code == "F":
                    _df_fm, _ = compute_family_table(
                        modelo, _mu_mdg_tab, presion_S, _V_L_mdg, _R_mdg, _ga_mdg,
                        _phi_mdg, _kap_mdg, _nu_mdg, _F_col_mdg, _pd0_mdg, _pda_mdg, _al_mdg,
                        _cmh_alive_mdg,
                    )
                    _actions_mdg = [str(x) for x in _df_fm["Rama"].tolist()]
                    _imax = int(_df_fm["EU ilustrativa"].astype(float).values.argmax())
                    _astar_mdg = _actions_mdg[_imax]
                else:
                    _df_sm, _ = compute_state_table(
                        _mu_mdg_tab, modelo, presion_S, precision_iota, _omk_mdg,
                        _omp_mdg, _omg_mdg, _al_mdg, _ga_mdg, _R_mdg, _ops_mdg,
                        _mt_mdg, _cinst_mdg, _cmh_kill_mdg, _cmh_alive_mdg,
                    )
                    _vr = float(_df_sm.loc[_df_sm["Rama"] == "Rescate", "Pérdida"].iloc[0])
                    _vn = float(_df_sm.loc[_df_sm["Rama"] == "Negociar", "Pérdida"].iloc[0])
                    _astar_mdg = "Rescate (a_res)" if _vr <= _vn else "Negociar (a_neg)"
                    _actions_mdg = ["Rescate (a_res)", "Negociar (a_neg)"]
            except (IndexError, KeyError, ValueError, TypeError):
                if _mdg_focal_code == "K":
                    _actions_mdg = ["Liberar (a_rel)", "Matar (a_kill)", "Continuar (a_cont)"]
                elif _mdg_focal_code == "F":
                    _actions_mdg = ["Cooperar (a_coop)", "Colusión (a_col)"]
                else:
                    _actions_mdg = ["Rescate (a_res)", "Negociar (a_neg)"]
                _astar_mdg = _actions_mdg[0]

            _key_T0 = f"mdg_T0_{_mdg_focal_code}"
            _key_cbar = f"mdg_cbar_{_mdg_focal_code}"
            if _key_T0 not in st.session_state: st.session_state[_key_T0] = 1.0
            if _key_cbar not in st.session_state: st.session_state[_key_cbar] = 0.05

            _T0_w = float(st.session_state[_key_T0])
            _cbar_w = float(st.session_state[_key_cbar])
            _t_int_mdg = int(st.session_state.get("mdg_tday", 1))
            _eta_w = float(st.session_state.cal_mdg_eta_cal_by_i[_mdg_focal_code])
            _T_t_mdg = float(hybrid_temperature(
                _H_mu_mdg, _T0_w, H0=_H0_ref_mdg, eta_cal=_eta_w, t=_t_int_mdg, c_bar=_cbar_w
            ))
            
            _p_map_theoretical = _mdg_implementation_logit_probs(_actions_mdg, _astar_mdg, _T_t_mdg)
            _p_map = dict(_p_map_theoretical)
            for _i_ed, _ac_ed in enumerate(_actions_mdg):
                _pk = f"mdg7_p_{_mdg_focal_code}_{_i_ed}"
                if _pk in st.session_state:
                    _p_map[_ac_ed] = float(st.session_state[_pk])
                else:
                    st.session_state[_pk] = float(_p_map_theoretical[_ac_ed])

            _coef_pi_mdg = {
                "K": r"\mathbb{P}_{\mathrm{I},K}(\tilde{a}=a \mid a^{K,\ast},X_t)",
                "S": r"\mathbb{P}_{\mathrm{I},S}(\tilde{a}=a \mid a^{S,\ast},X_t)",
                "F": r"\mathbb{P}_{\mathrm{I},F}(\tilde{a}=a \mid a^{F,\ast},X_t)",
            }
            _coef_focal_pi = _coef_pi_mdg.get(_mdg_focal_code, _coef_pi_mdg["S"])
            _rows_mdg_katex = []
            _n_u = 0
            for _a in _actions_mdg:
                _n_u += 1
                _lbl_a = f"{_a} ★" if str(_a) == str(_astar_mdg) else str(_a)
                _pv = float(_p_map[_a])
                _rows_mdg_katex.append({"#": _n_u, "Término": f"Implementación MDG · {_lbl_a}", "Coeficiente": _coef_focal_pi, "Valor": _pv, "Valor_KaTeX": rf"\text{{{_fmt_es_num(_pv, 2)}}}", "Clase_tab7": "No prior"})
            
            _prior_const = float(_mu_mdg_tab.get(tipo_real, 0.25))
            _n_u += 1
            _rows_mdg_katex.append({"#": _n_u, "Término": _ui_text("Prior marginal μ(θ_K) incident", "Prior marginal μ(θ_K) incidente"), "Coeficiente": r"\mu(\theta_K)", "Valor": _prior_const, "Valor_KaTeX": rf"\text{{{_fmt_es_num(_prior_const, 2)}}}", "Clase_tab7": "Prior"})

            _tt_specs = [
                (_ui_text("Base temperature T₀", "Temperatura base T₀"), "T_0", _T0_w),
                (_ui_text("Reference entropy H(μ₀)", "Entropía de referencia H(μ₀)"), r"H(\mu_0)", float(_H0_ref_mdg)),
                (_ui_text("Floor c̲ (lower)", "Piso c̲ (inferior)"), r"\underline{c}", _cbar_w),
            ]
            for _term_tt, _coef_tt, _val_tt in _tt_specs:
                _n_u += 1
                _rows_mdg_katex.append({"#": _n_u, "Término": _term_tt, "Coeficiente": _coef_tt, "Valor": _val_tt, "Valor_KaTeX": rf"\text{{{_fmt_es_num(_val_tt, 2)}}}", "Clase_tab7": "No prior"})
            
            _n_u += 1
            _rows_mdg_katex.append({"#": _n_u, "Término": _TABLA7_TERM_ETA_CAL, "Coeficiente": r"\eta_{\mathrm{cal}}", "Valor": _eta_w, "Valor_KaTeX": rf"\text{{{_fmt_es_num(_eta_w, 4)}}}", "Clase_tab7": "No prior"})
            
            df_tab7 = pd.DataFrame(_rows_mdg_katex)
            
            _c_eq, _c_tbl = st.columns([0.45, 0.55], gap="large")
            with _c_eq:
                st.markdown(_ui_text("#### Implementation Law", "#### Ley de Implementación"))
                st.latex(rf"\mathbb{{P}}_{{\mathrm{{I}},{_mdg_focal_code}}}(\tilde a=a \mid a^\ast, X_t) = \frac{{\exp(\mathbf{{1}}\{{a=a^\ast\}}/T_t)}}{{\sum_{{a'\in\mathcal{{A}}^{{{_mdg_focal_code}}}}}\exp(\mathbf{{1}}\{{a'=a^\ast\}}/T_t)}}")
                st.markdown(_ui_text("#### Temperature Dynamics", "#### Dinámica de Temperatura"))
                st.latex(r"T_t = T_0 \max\!\left\{\frac{H(\mu_t)}{H(\mu_0)}e^{-\eta_{\mathrm{cal}}t},\,\underline{c}\right\}")
                st.latex(rf"H(\mu_t) = -\sum_{{\theta \in \Theta_K}} \mu_t(\theta) \ln \mu_t(\theta) = \text{{{_fmt_es_num(_H_mu_mdg, 2)}}}")

            with _c_tbl:
                _t7_title = _ui_text(f"Table 7 · Implementation ({_mdg_focal_code})", f"Tabla 7 · Implementación ({_mdg_focal_code})")
                st.markdown(f'<h4 class="mdg-tabla7-title">{_t7_title}</h4>', unsafe_allow_html=True)
                _render_focus_covariate_katex_table(
                    df_tab7,
                    show_origen=False,
                    font_boost_pt=2.0,
                    compact_iframe_bottom=True,
                    iframe_slack_px=36,
                    collapse_gap_below=True,
                )
                with st.popover(_ui_text(f"Edit values · Table 7 ({_mdg_focal_code})", f"Editar valores · Tabla 7 ({_mdg_focal_code})"), use_container_width=True):
                    for _i_ed, _ac_ed in enumerate(_actions_mdg):
                        _pl, _pr = st.columns((0.58, 0.42), gap="small", vertical_alignment="center")
                        with _pl: st.markdown(f"**#{_i_ed+1}** · {_translate_text_to_english(_ac_ed)}")
                        with _pr: st.number_input(" ", min_value=0.0, max_value=1.0, value=float(_p_map[_ac_ed]), step=0.0001, format="%.2f", key=f"mdg7_p_{_mdg_focal_code}_{_i_ed}", label_visibility="collapsed")

                    _pl4, _pr4 = st.columns((0.58, 0.42), gap="small", vertical_alignment="center")
                    with _pl4: st.markdown(_ui_text(f"**#4** · $\\underline{{c}}$ (noise floor)", f"**#4** · $\\underline{{c}}$ (piso de ruido)"))
                    with _pr4: st.number_input(" ", min_value=0.0, max_value=1.0, value=_cbar_w, step=0.01, format="%.2f", key=_key_cbar, label_visibility="collapsed")

                    _pl5, _pr5 = st.columns((0.58, 0.42), gap="small", vertical_alignment="center")
                    with _pl5: st.markdown(_ui_text("**#5** · $T_0$ (base temperature)", "**#5** · $T_0$ (temperatura base)"))
                    with _pr5: st.number_input(" ", min_value=1e-12, value=_T0_w, step=0.01, format="%.2f", key=_key_T0, label_visibility="collapsed")

                    _pl6, _pr6 = st.columns((0.58, 0.42), gap="small", vertical_alignment="center")
                    with _pl6: st.markdown(_ui_text(f"**#6** · $\\eta_{{\\mathrm{{cal}}}}$ · decay", f"**#6** · $\\eta_{{\\mathrm{{cal}}}}$ · decaimiento"))
                    with _pr6:
                        _eta_ed_k = f"mdg_t7_eta_ed_{_mdg_focal_code}"
                        st.number_input(" ", min_value=0.0, value=_eta_w, step=0.0001, format="%.4f", key=_eta_ed_k, label_visibility="collapsed")
                        st.session_state.cal_mdg_eta_cal_by_i[_mdg_focal_code] = float(st.session_state[_eta_ed_k])

                    st.divider()
                    if st.button(_ui_text("Reset to theoretical model", "Resetear al modelo teórico"), key=f"mdg7_reset_{_mdg_focal_code}", use_container_width=True):
                        for _i_ed in range(len(_actions_mdg)):
                            _pk = f"mdg7_p_{_mdg_focal_code}_{_i_ed}"
                            if _pk in st.session_state: del st.session_state[_pk]
                        st.rerun()

    st.markdown("### Ecuaciones (28) y (29)")
    _psi8_mom = _cmh_outcome_moments_for_mdg()
    if (
        "cal_psi_params" not in st.session_state
        or int(st.session_state.get("cal_psi_params_cmh_version", -1)) < int(_PSI8_CMH_CALIB_VERSION)
    ):
        st.session_state.cal_psi_params = _default_cal_psi_params_from_cmh()
        st.session_state.cal_psi_params_cmh_version = int(_PSI8_CMH_CALIB_VERSION)
    _psi8_order = sorted(
        _psi8_mom["probs_terminal"].items(),
        key=lambda kv: float(kv[1]),
        reverse=True,
    )
    st.caption(
        "Diagnóstico Data_CMH.csv, por caso único: "
        + "; ".join(
            f"{_lbl}: {_psi8_mom['counts_terminal'][_lbl]} ({100.0 * _pr:.2f}%)"
            for _lbl, _pr in _psi8_order
        )
        + ". La ley activa de m_t ya no usa la logística Ψ_j; usa los hazards competitivos "
        + "de Mechanism.tex con α*, γ*, M(t), p_det y λ4."
    )
    
    _j8_sel = st.selectbox(
        "Desenlace físico a calibrar (j en Eq. 28-29)",
        options=[1, 2, 3, 4, 5],
        format_func=lambda x: f"j={x} · {_MDG_OUTCOME_LABELS[x]}",
        key="mdg_j8_sel"
    )
    _p8 = st.session_state.cal_psi_params[_j8_sel]

    _rows_psi = [
        {"#": 1, "Término": "Intercepto delta_j", "Coeficiente": r"\delta_j", "Valor": _p8["delta"], "Clase_tab7": "Prior"},
        {"#": 2, "Término": "Peso Acción K (gamma_j,K)", "Coeficiente": r"\gamma_{j,K}", "Valor": _p8["gamma_K"], "Clase_tab7": "No prior"},
        {"#": 3, "Término": "Peso Acción S (gamma_j,S)", "Coeficiente": r"\gamma_{j,S}", "Valor": _p8["gamma_S"], "Clase_tab7": "No prior"},
        {"#": 4, "Término": "Peso Acción F (gamma_j,F)", "Coeficiente": r"\gamma_{j,F}", "Valor": _p8["gamma_F"], "Clase_tab7": "No prior"},
        {"#": 5, "Término": "Capacidad operativa (phi_j,gamma)", "Coeficiente": r"\phi_{j,\gamma}", "Valor": _p8["phi_gamma"], "Clase_tab7": "Prior"},
        {"#": 6, "Término": f"Peso: {_THETA_K_LABELS[0]} (phi_j,1)", "Coeficiente": r"\phi_{j,1}", "Valor": _p8["phi_theta"][0], "Clase_tab7": "Prior"},
        {"#": 7, "Término": f"Peso: {_THETA_K_LABELS[1]} (phi_j,2)", "Coeficiente": r"\phi_{j,2}", "Valor": _p8["phi_theta"][1], "Clase_tab7": "Prior"},
        {"#": 8, "Término": f"Peso: {_THETA_K_LABELS[2]} (phi_j,3)", "Coeficiente": r"\phi_{j,3}", "Valor": _p8["phi_theta"][2], "Clase_tab7": "Prior"},
        {"#": 9, "Término": f"Peso: {_THETA_K_LABELS[3]} (phi_j,4)", "Coeficiente": r"\phi_{j,4}", "Valor": _p8["phi_theta"][3], "Clase_tab7": "Prior"},
        {"#": 10, "Término": "Precisión kappa_j", "Coeficiente": r"\kappa_j", "Valor": _p8["kappa"], "Clase_tab7": "Endógena"},
    ]
    for _r in _rows_psi:
        _r["Valor_KaTeX"] = rf"\text{{{_fmt_es_num(_r['Valor'], 2)}}}"

    _df_psi = pd.DataFrame(_rows_psi)

    _ec8L, _ec8R = st.columns((0.4, 0.6), gap="small")
    with _ec8L:
        st.markdown(
            r"**Desenlace físico $m_t$ (hazards competitivos).** "
            r"Condicionado a la tripleta ejecutada $\tilde A_t$ y al estado $\mathcal C_t$, "
            r"la probabilidad de continuar y las masas terminales salen de las lambdas tilda."
        )
        st.latex(
            r"p_{\mathrm{Cont},t\mid\theta_K}"
            r"=\exp\!\left[-\sum_{j=1}^{4}\tilde{\lambda}_j(t\mid\theta_K,\mathcal C_t)\Delta t\right]"
        )
        st.latex(
            r"h_j(t\mid\theta_K,\mathcal C_t)"
            r"=\left(1-p_{\mathrm{Cont},t\mid\theta_K}\right)"
            r"\frac{\tilde{\lambda}_j(t\mid\theta_K,\mathcal C_t)}{\sum_{\ell=1}^{4}\tilde{\lambda}_\ell(t\mid\theta_K,\mathcal C_t)}"
            r",\qquad j=1,\ldots,4"
        )
        st.latex(
            r"\mathbb P_E(m_t=\mathrm{Cont}\mid\theta_K,\mathcal C_t)=p_{\mathrm{Cont},t\mid\theta_K},"
            r"\qquad \mathbb P_E(m_t=j\mid\theta_K,\mathcal C_t)=h_j(t\mid\theta_K,\mathcal C_t)"
        )
    with _ec8R:
        st.markdown(f"#### Tabla 8 · Propensión $\Psi_{{j={_j8_sel}}}$ (referencia archivada)")
        _render_focus_covariate_katex_table(
            _df_psi,
            show_origen=False,
            font_boost_pt=2.0,
            compact_iframe_bottom=True,
            iframe_slack_px=0,
            collapse_gap_below=True,
        )
        with st.popover(
            f"Editar valores · Tabla 8 (j={_j8_sel})",
            width="stretch",
        ):
            # Reutilizamos el estilo de la Tabla 7 popover
            st.markdown(
                """
                <style>
                div[data-testid="stPopoverBody"] { font-size: 0.86rem !important; overflow: visible !important; }
                div[data-testid="stPopoverBody"] .katex { font-size: 0.94em !important; }
                </style>
                """,
                unsafe_allow_html=True,
            )
            
            _fields = [
                ("delta", r"\delta_j", "Intercepto"),
                ("gamma_K", r"\gamma_{j,K}", "Acción K"),
                ("gamma_S", r"\gamma_{j,S}", "Acción S"),
                ("gamma_F", r"\gamma_{j,F}", "Acción F"),
                ("phi_gamma", r"\phi_{j,\gamma}", "Capacidad operativa"),
                ("kappa", r"\kappa_j", "Precisión"),
            ]
            st.markdown("**Pesos del vector de tecnología** $\\vec{\\phi}_{j,\\theta}$:")
            _v_th = THETA_K_MAP.get(tipo_real, [0.0]*4)
            for _idx, _lbl in enumerate(_THETA_K_LABELS):
                _plw, _prw = st.columns((0.58, 0.42), gap="small", vertical_alignment="center")
                with _plw: st.markdown(f"**#T{_idx+1}** · {_lbl} ({_v_th[_idx]})")
                with _prw:
                    _val_phi_v = st.number_input(
                        f" ",
                        value=float(_p8["phi_theta"][_idx]),
                        step=0.01,
                        format="%.2f",
                        key=f"mdg8_{_j8_sel}_phi_v{_idx}",
                        label_visibility="collapsed"
                    )
                    st.session_state.cal_psi_params[_j8_sel]["phi_theta"][_idx] = float(_val_phi_v)
            for _i_f, (_f_key, _f_tex, _f_name) in enumerate(_fields):
                _pl, _pr = st.columns((0.58, 0.42), gap="small", vertical_alignment="center")
                with _pl:
                    st.markdown(f"**#{_i_f + 1}** · ${_f_tex}$ ({_f_name})")
                with _pr:
                    _new_v = st.number_input(
                        " ",
                        value=float(_p8[_f_key]),
                        step=0.01,
                        format="%.2f",
                        key=f"mdg8_{_j8_sel}_{_f_key}",
                        label_visibility="collapsed",
                    )
                    st.session_state.cal_psi_params[_j8_sel][_f_key] = float(_new_v)
    st.divider()
    st.markdown("### Materialización (Transformada Inversa)")
    st.caption("Arquitectura estocástica del **DGP** (Mechanism.tex, sección 6.2.2). Transforma la **intención estratégica** en **realizaciones observables**.")

    # 1. Recuperar Intenciones (a*) de la Pestaña 4 si existen, o usar defaults
    _a_k_star = "Continuar"
    _a_s_star = "No Rescatar"
    _a_f_star = "Cooperar"

    # 2. Sorteo DGP (Transformada Inversa) - Simulación de 3 Casos
    if "dgp_seed" not in st.session_state:
        st.session_state.dgp_seed = 42

    if st.button("🎰 Girar ruleta del DGP (Generar trayectorias)", type="primary", use_container_width=True, key="btn_dgp_tab3"):
        st.session_state.dgp_seed = np.random.randint(0, 100000)
        st.rerun()

    np.random.seed(st.session_state.dgp_seed)
    
    # --- PRESENTACIÓN POR JUGADOR (Fase 1) ---
    # --- PRESENTACIÓN POR JUGADOR (Fase 1) ---
    st.markdown("#### Fase 1: Implementación de la Intención ($\mathcal{H}_t^i$)")
    st.latex(r"\tilde{a}_t^i = a_k \iff \sum_{j=1}^{k-1} \mathbb{P}^I_i(a_j \mid a_t^{i\ast}, X_t) \le u_{t,i} < \sum_{j=1}^{k} \mathbb{P}^I_i(a_j \mid a_t^{i\ast}, X_t)")
    
    # Precision iota (calculada de la mu actual en el tab) y vector theta focal
    _iota_val = float(max(_mu_mdg_tab.values()))
    _v_th = THETA_K_MAP.get(tipo_real, [0.0]*4)
    
    # Los sorteos u y v provienen de una distribución Uniforme U(0, 1).
    # Se usan como entrada para el método de la Transformada Inversa.
    _u_vals = np.random.random(3)
    _v_val = np.random.random()

    _c1, _c2, _c3 = st.columns(3)
    _players = [
        ("Secuestrador (K)", "K", _a_k_star, _u_vals[0], r"a_{t}^{K,\ast}", r"\tilde{a}_t^K"),
        ("Estado (S)", "S", _a_s_star, _u_vals[1], r"a_{t}^{S,\ast}", r"\tilde{a}_t^S"),
        ("Familia (F)", "F", _a_f_star, _u_vals[2], r"a_{t}^{F,\ast}", r"\tilde{a}_t^F"),
    ]

    _exec_actions = []
    _tab3_action_probs = {}
    for _i, (_name, _key, _intent, _u, _tex_ast, _tex_tilde) in enumerate(_players):
        with [_c1, _c2, _c3][_i]:
            st.markdown(f"**{_name}**")
            st.latex(_tex_ast + r" = \text{" + _intent + "}")
            st.latex(r"u_{t," + _key + r"} = " + f"{_u:.4f}")
            
            # Definir probabilidades dinámicas basadas en precisión iota (Fase 1)
            # p_intent = 0.85 + 0.14 * iota. El resto se reparte.
            _p_int = 0.85 + 0.14 * _iota_val
            _p_others = (1.0 - _p_int) / 2.0
            
            _ints = []
            if _key == "K":
                # K tiene 3 acciones: Continuar, Liberar, Matar
                _acts = ["Continuar", "Liberar", "Matar"]
                _probs = [_p_others] * 3
                _idx_int = _acts.index(_intent)
                _probs[_idx_int] = _p_int
                # Normalizar por si acaso
                _probs = [x / sum(_probs) for x in _probs]
                
                _curr = 0.0
                for _a, _p in zip(_acts, _probs):
                    _ints.append((f"[{_curr:.2f}, {_curr+_p:.2f})", _a))
                    _curr += _p
            elif _key == "S":
                # S tiene 2 acciones: No Rescatar, Rescatar
                _p_s_int = 0.90 + 0.09 * _iota_val
                _acts = ["No Rescatar", "Rescatar"]
                _probs = [1.0 - _p_s_int, 1.0 - _p_s_int]
                _idx_int = _acts.index(_intent)
                _probs[_idx_int] = _p_s_int
                _probs = [x / sum(_probs) for x in _probs]
                
                _curr = 0.0
                for _a, _p in zip(_acts, _probs):
                    _ints.append((f"[{_curr:.2f}, {_curr+_p:.2f})", _a))
                    _curr += _p
            else: # F
                # F tiene 2 acciones: Cooperar, Coludir
                _p_f_int = 0.92 + 0.07 * _iota_val
                _acts = ["Cooperar", "Coludir"]
                _probs = [1.0 - _p_f_int, 1.0 - _p_f_int]
                _idx_int = _acts.index(_intent)
                _probs[_idx_int] = _p_f_int
                _probs = [x / sum(_probs) for x in _probs]
                
                _curr = 0.0
                for _a, _p in zip(_acts, _probs):
                    _ints.append((f"[{_curr:.2f}, {_curr+_p:.2f})", _a))
                    _curr += _p

            _tab3_action_probs[_key] = {str(_a): float(_p) for _a, _p in zip(_acts, _probs)}
            
            _exec = ""
            for _range, _act in _ints:
                _is_hit = False
                _low = float(_range.split(",")[0][1:])
                _high = float(_range.split(",")[1][:-1].replace(")", "").replace("]", ""))
                # Manejo de borde superior
                if _low <= _u < _high or (_u >= 0.99 and _high >= 0.99):
                    _is_hit = True
                    _exec = _act
                
                _prefix = "🎯 " if _is_hit else "▫️ "
                if _is_hit:
                    st.markdown(f"{_prefix}**{_range} → {_act}**")
                else:
                    st.caption(f"{_prefix}{_range} → {_act}")

            _exec_actions.append(_exec)
            st.latex(_tex_tilde + r" \to \text{" + _exec + "}")

    st.session_state.tab3_materialization_action_probs = _tab3_action_probs

    st.divider()

    # --- CÁLCULO DE PROBABILIDADES FASE 2 (Consistencia con Eq. 28-29) ---
    # Psi_j = delta_j + sum(gamma_ji * 1{a_tilde == obj_j}) + phi_gamma*gamma_t + phi_theta*theta_K + kappa*iota
    _eta0_tab3 = float(st.session_state.get(f"cal_eta0_pdet_{str(tipo_real)}", _ETA0_PDET_DEFAULTS.get(str(tipo_real), -2.0)))
    _eta1_tab3 = float(st.session_state.get("cal_eta1_pdet", 1.0))
    _eta2_tab3 = float(st.session_state.get("cal_eta2_pdet", 1.0))
    _pdet_tab3 = float(1.0 / (1.0 + np.exp(-(_eta0_tab3 + _eta1_tab3 * float(alpha_star) + _eta2_tab3 * float(presion_S)))))
    _probs_m_dict, _mfac_tab3 = _mechanism_m_probs_for_actions(
        str(tipo_real),
        1,
        float(alpha_star),
        float(presion_S),
        float(_pdet_tab3),
        str(_exec_actions[0]),
        str(_exec_actions[1]),
        str(_exec_actions[2]),
        z_region=str(st.session_state.z_region),
        v_victim=str(st.session_state.v_victim),
        f_capa=str(f_capa),
        s_tipo=str(s_tipo),
    )
    _probs_m = [
        float(_probs_m_dict["Liberación"]),
        float(_probs_m_dict["Rescate"]),
        float(_probs_m_dict["Pago"]),
        float(_probs_m_dict["Muerte"]),
        float(_probs_m_dict["Continuar"]),
    ]
    st.session_state.tab3_materialization_outcome_probs = {
        "lib": float(_probs_m[0]),
        "res": float(_probs_m[1]),
        "pay": float(_probs_m[2]),
        "kill": float(_probs_m[3]),
        "cont": float(_probs_m[4]),
        "mechanism_factors": dict(_mfac_tab3),
    }
    
    # --- PRESENTACIÓN DESENLACE (Fase 2) ---
    st.markdown("#### Fase 2: Materialización del Desenlace ($\mathcal{G}_t$)")
    st.latex(r"m_t = r \iff \sum_{\ell < r} \mathbb{P}^E(m_t = \ell \mid \theta_K,\mathcal C_t) \le v_t < \sum_{\ell \le r} \mathbb{P}^E(m_t = \ell \mid \theta_K,\mathcal C_t)")
    
    _cc1, _cc2 = st.columns((0.6, 0.4))
    with _cc1:
        st.latex(r"\tilde{A}_t = (\text{" + _exec_actions[0] + r"}, \text{" + _exec_actions[1] + r"}, \text{" + _exec_actions[2] + r"})")
        st.latex(r"v_t = " + f"{_v_val:.4f}")
        st.caption(
            rf"$M(1)={float(_mfac_tab3.get('M_t', 0.0)):.6f}$, "
            rf"$\alpha^\ast={float(alpha_star):.4f}$, $\gamma^\ast={float(presion_S):.4f}$, "
            rf"$p_{{det}}={float(_pdet_tab3):.4f}$; "
            rf"$p_{{Cont}}={float(_mfac_tab3.get('p_cont', 0.0)):.6f}$."
        )
        
        # Intervalos de desenlace construidos dinámicamente
        _acum = 0.0
        _ints_m = []
        for _idx in range(5):
            _label = _MDG_OUTCOME_LABELS[_idx + 1]
            _prob = _probs_m[_idx]
            _low = _acum
            _high = _acum + _prob
            _range_text = f"[{_low:.2f}, {_high:.2f})" if _idx < 4 else f"[{_low:.2f}, 1.00]"
            _ints_m.append((_range_text, _label, _low, _high, _prob))
            _acum = _high
        
        _m_res_text = ""
        for _range, _act, _l, _h, _p in _ints_m:
            _is_hit = False
            if _l <= _v_val < _h or (_act == "Continuar" and _v_val >= _l):
                _is_hit = True
                _m_res_text = _act
            
            _prefix = "🎯 " if _is_hit else "▫️ "
            _prob_text = f"($p={_p:.2f}$)"
            if _is_hit:
                st.markdown(f"{_prefix}**{_range} → {_act}** {_prob_text}")
            else:
                st.caption(f"{_prefix}{_range} → {_act} {_prob_text}")

        st.session_state.tab3_materialization_exec_actions = {
            "K": _exec_actions[0],
            "S": _exec_actions[1],
            "F": _exec_actions[2],
        }
        st.session_state.tab3_materialization_outcome = _m_res_text

    with _cc2:
        st.markdown(f"**Resultado materializado:**")
        st.title(_m_res_text)

    st.divider()
    st.caption("Nota: los intervalos se construyen con la ley física activa de Mechanism.tex: lambdas tilda, p_cont y h_j. Los corchetes en negrita indican la caída del sorteo.")

    # ── Ciclo τ=1 · Implementación MDG ──────────────────────────────────────
    _c1_diag_mdg = st.session_state.get("first_cycle_diag52") or {}
    _c1_tau1_mdg = st.session_state.get("first_cycle_tau1_52") or {}
    _has_c1_mdg = bool(
        isinstance(_c1_diag_mdg, dict)
        and _c1_diag_mdg.get("pk_probs")
        and isinstance(_c1_tau1_mdg, dict)
    )
    if _has_c1_mdg:
        with st.expander("Ciclo τ = 1 · Implementación MDG (resultado de Avanzar ciclos)", expanded=True):
            _mu1_mdg   = {th: float(_c1_tau1_mdg.get(f"μ({th})", 0.0)) for th in TIPOS_SECUESTRADOR}
            _iota1_mdg = float(_c1_diag_mdg.get("iota_prior", 0.0))
            _thetah1   = str(_c1_diag_mdg.get("theta_prior", "—"))
            _V1_mdg    = str(_c1_tau1_mdg.get("V (voz)", "—"))
            _d1_mdg    = str(_c1_tau1_mdg.get("d (det.)", "—"))
            _alpha1_mdg = str(_c1_tau1_mdg.get("α* Estado", "—"))
            _gamma1_mdg = str(_c1_tau1_mdg.get("γ* Estado", "—"))
            _pk1 = dict(_c1_diag_mdg.get("pk_probs", {}))
            _ps1 = dict(_c1_diag_mdg.get("ps_probs", {}))
            _pf1 = dict(_c1_diag_mdg.get("pf_probs", {}))
            _atk1 = str(_c1_diag_mdg.get("atk", "—"))
            _ats1 = str(_c1_diag_mdg.get("ats", "—"))
            _atf1 = str(_c1_diag_mdg.get("atf", "—"))
            _m1_mdg = str(_c1_tau1_mdg.get("m", "—")).split("(")[0].strip()

            # Contexto del ciclo
            _mu1_str = " · ".join(
                f"μ({th})={float(_mu1_mdg.get(th, 0.0)):.3f}" for th in TIPOS_SECUESTRADOR
            )
            st.caption(
                f"**Señales:** V₁={_V1_mdg}, d₁={_d1_mdg}  ·  "
                f"**Creencia prior μ₁:** {_mu1_str}  ·  "
                f"ι₁={_iota1_mdg:.4f}, θ̂₁={_thetah1}  ·  "
                f"α₁*={_alpha1_mdg}, γ₁*={_gamma1_mdg}"
            )

            # Tabla MDG τ=1
            _mdg1_rows = []
            for _ag1, _pdict1, _exec1 in [("K", _pk1, _atk1), ("S", _ps1, _ats1), ("F", _pf1, _atf1)]:
                for _act1, _pval1 in sorted(_pdict1.items(), key=lambda kv: -float(kv[1])):
                    _marker = "★" if str(_act1) == str(_exec1) else ""
                    _mdg1_rows.append({
                        "Agente": _ag1,
                        "Acción ã": f"{_marker}{_act1}",
                        "P_I(ã|a*)": f"{float(_pval1):.4f}",
                        "Sorteada": "✓" if str(_act1) == str(_exec1) else "",
                    })
            if _mdg1_rows:
                st.dataframe(
                    pd.DataFrame(_mdg1_rows),
                    use_container_width=True,
                    hide_index=True,
                    height=min(38 * len(_mdg1_rows) + 38, 320),
                )
            st.caption(
                f"Desenlace m₁ sorteado: **{_m1_mdg}**  "
                f"(★ = acción sorteada por MDG; P_I calculada con μ₁ e ι₁={_iota1_mdg:.4f})"
            )

    st.divider()

with tab_rb:
    # Unicode para etiquetas de widgets / columnas (Streamlit no aplica LaTeX ahí).
    _U = {
        "th": "\u03b8",
        "mu": "\u03bc",
        "al": "\u03b1",
        "ga": "\u03b3",
        "io": "\u03b9",
        "be": "\u03b2",
        "ka": "\u03ba",
        "et": "\u03b7",
        "nu": "\u03bd",
        "ph": "\u03c6",
        # Igual que \mathcal{U} en LaTeX; útil en títulos donde KaTeX en markdown puede variar.
        "calU": "\U0001d4b0",
    }
    st.markdown('<span class="rb-tab4-title-scale"></span>', unsafe_allow_html=True)
    st.markdown("## Comportamiento racional (IR / IC)")

    # --- NUEVA SECCIÓN: CONJUNTOS DE INFORMACIÓN Y ESPACIOS ---
    st.markdown("### 4.0 · Conjuntos de Información y Espacios del Modelo")
    
    with st.expander("Ver espacios y estructuras del mecanismo", expanded=True):
        st.markdown("#### Espacios de Acción y Resultados")
        st.latex(r"\mathcal{A}^K = \{a_{rel}, a_{kill}, a_{cont}\}, \quad \mathcal{A}^F = \{a_{coop}, a_{col}\}, \quad \mathcal{A}^S = \{a_{res}, a_{neg}\} \times \mathcal{A}^\alpha \times \mathcal{A}^\gamma")
        st.latex(r"\mathcal{M} = \{\text{cont, surv, rel, kill, pay}\}, \quad \Theta_K = \{DC, PAR, ELN, FARC\}")
        st.markdown("#### Estructura del mecanismo")
        st.latex(r"\mathcal{D} = \langle \mathcal{J}, \mathbb{T}, \Theta, \mathcal{R}, (\mathcal{X}_t)_{t \in \mathbb{T}}, (H_t)_{t \in \mathbb{T}}, (\mu_t)_{t \in \mathbb{T}}, \mathbb{P}_E \rangle")

    st.markdown(
        """
        <style>
        div[data-testid="stVerticalBlock"]:has(.rb-tab4-title-scale) h2,
        div[data-testid="stVerticalBlock"]:has(.rb-tab4-title-scale) h3,
        div[data-testid="stVerticalBlock"]:has(.rb-tab4-title-scale) h4 {
            font-size: calc(1em + 4pt) !important;
            line-height: 1.18 !important;
        }
        .rb-t0-title {
            margin: 0.15rem 0 0.2rem 0;
            line-height: 1.22;
            font-size: calc(1.12rem + 4pt);
            font-weight: 650;
        }
        .rb-t0-subtitle {
            margin: 0 0 0.55rem 0;
            color: rgba(49, 51, 63, 0.72);
            font-size: 0.86rem;
            line-height: 1.35;
        }
        div[data-testid="stPopover"] button {
            white-space: normal;
            min-height: 2.45rem;
            line-height: 1.15;
        }
        div[data-testid="stPopoverBody"] label,
        div[data-testid="stPopoverBody"] p {
            line-height: 1.25 !important;
        }
        div[data-testid="stPopoverBody"] [data-testid="stWidgetLabel"] {
            min-height: 1.25rem;
        }
        div[data-testid="stPopoverBody"] [data-testid="stSlider"] {
            padding-top: 0.28rem;
        }
        .rb-t0-table-label {
            margin: 0.25rem 0 0.08rem 0;
            line-height: 1.15;
            font-weight: 650;
            font-size: calc(0.88rem + 4pt);
        }
        .rb-t0-soft-gap {
            height: 0.28rem;
        }
        div[data-testid="stVerticalBlock"]:has(.rb-t10-tabs-marker) div[data-testid="stTabs"] [data-testid="stVerticalBlock"] {
            gap: 0 !important;
        }
        div[data-testid="stVerticalBlock"]:has(.rb-t10-tabs-marker) div[data-testid="stTabs"] div[role="tabpanel"] {
            padding-top: 0 !important;
            margin-top: 0 !important;
        }
        div[data-testid="stVerticalBlock"]:has(.rb-t10-tabs-marker) div[data-testid="stTabs"] div[role="tabpanel"] [data-testid="stElementContainer"]:first-child {
            margin-top: 0 !important;
            padding-top: 0 !important;
        }
        div[data-testid="stVerticalBlock"]:has(.rb-t10-tabs-marker) div[role="tabpanel"] [data-testid="stElementContainer"]:has(iframe) {
            margin-bottom: 0 !important;
            padding-bottom: 0 !important;
        }
        div[data-testid="stVerticalBlock"]:has(.rb-t10-tabs-marker) div[role="tabpanel"] [data-testid="stElementContainer"]:has(iframe) + [data-testid="stElementContainer"]:has(iframe) {
            margin-top: -0.55rem !important;
            padding-top: 0 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    with st.expander(
        "Historia pública y conjuntos de información (t₀)",
        expanded=False,
    ):
        _h0_title_col, _h0_action_col = st.columns([0.68, 0.32], gap="large", vertical_alignment="center")
        with _h0_title_col:
            st.markdown(
                '<div class="rb-t0-subtitle">Edita la fila inicial t = 0 y comprueba el resultado en la tabla pública.</div>',
                unsafe_allow_html=True,
            )

        # Preparar datos de información t=0
        z_val = st.session_state.get("z_region", "Andina")
        v_val = st.session_state.get("v_victim", "Privado")
        f_val = f_capa # Definido justo antes
        s_val = s_tipo # Definido justo antes
        theta_k_true = tipo_real # Tipo del incidente global
        priors_v = [f"{p:.2f}%" for p in st.session_state.final_priors]

        # Inicialización de acciones t=0 en session_state si no existen
        if "h0_alpha" not in st.session_state: st.session_state.h0_alpha = 0.20
        if "h0_gamma" not in st.session_state: st.session_state.h0_gamma = 0.90
        _tab3_exec_h0 = st.session_state.get("tab3_materialization_exec_actions", {})
        _tab3_outcome_h0 = st.session_state.get("tab3_materialization_outcome", "—")
        _base_m_tau0_h0 = st.session_state.get("base_cycle_m_tau0", {})
        _h0_Atilde_K_val = str(_tab3_exec_h0.get("K", "—")) if isinstance(_tab3_exec_h0, dict) else "—"
        _h0_Atilde_S_val = str(_tab3_exec_h0.get("S", "—")) if isinstance(_tab3_exec_h0, dict) else "—"
        _h0_Atilde_F_val = str(_tab3_exec_h0.get("F", "—")) if isinstance(_tab3_exec_h0, dict) else "—"
        _h0_m_val = (
            str(_base_m_tau0_h0.get("m", "—"))
            if isinstance(_base_m_tau0_h0, dict) and str(_base_m_tau0_h0.get("m", ""))
            else str(_tab3_outcome_h0 or "—")
        )
        if "h0_d" not in st.session_state or str(st.session_state.get("h0_d")) not in ("0", "1"):
            st.session_state.h0_d = "0"
        st.session_state.h0_Atilde_K = _h0_Atilde_K_val
        st.session_state.h0_Atilde_S = _h0_Atilde_S_val
        st.session_state.h0_Atilde_F = _h0_Atilde_F_val
        st.session_state.h0_m = _h0_m_val

        # --- 1. HISTORIA PÚBLICA (h_t) ---
        with _h0_action_col:
            with st.popover("Acciones iniciales t=0", use_container_width=True):
                _render_compact_katex_expr(
                    r"\text{Instrumentos públicos: }\alpha_0,\ \gamma_0",
                    height=28,
                )
                _ia, _ib = st.columns(2, gap="large")
                with _ia:
                    _render_widget_katex_label(r"\alpha_0\ \cdot\ \text{bloqueo financiero}")
                    st.info(
                        f"α₀ = **{float(st.session_state.get('h0_alpha', 0.20)):.2f}** · Ajustable en panel superior",
                        icon="🔒",
                    )
                with _ib:
                    _render_widget_katex_label(r"\gamma_0\ \cdot\ \text{presión operativa}")
                    st.info(
                        f"γ₀ = **{float(st.session_state.get('h0_gamma', 0.90)):.2f}** · Ajustable en panel superior",
                        icon="🔒",
                    )
                _render_h0_source_note(height=28)
                _render_widget_katex_label(r"d_0\ \cdot\ \text{detección}")
                _d0_display_val = str(st.session_state.get("h0_d", "0"))
                st.info(
                    f"d₀ = **{_d0_display_val}** · Generado en Pestaña 2 → Sección 3",
                    icon="🔒",
                )

        st.markdown('<div class="rb-t0-table-label">Tabla 9a · Historia pública inicial (h₀)</div>', unsafe_allow_html=True)
        _render_compact_katex_expr(
            r"h_0 = \bigl(t,\alpha_0,\gamma_0,\tilde{a}_0^S,\tilde{a}_0^F,\tilde{a}_0^K,m_0,d_0,\mathrm{evento}\bigr)",
            height=30,
        )
        df_history = pd.DataFrame([{
            "t": 0,
            "a": st.session_state.h0_alpha,
            "g": st.session_state.h0_gamma,
            "As": _h0_Atilde_S_val,
            "Af": _h0_Atilde_F_val,
            "Ak": _h0_Atilde_K_val,
            "m": _h0_m_val,
            "d": st.session_state.h0_d,
            "e": "Inicio del Cautiverio"
        }])
        h_headers = [
            "t", r"\alpha_t", r"\gamma_t",
            r"\tilde{a}_t^S", r"\tilde{a}_t^F", r"\tilde{a}_t^K",
            r"m_t", r"d_t", "Evento"
        ]
        render_generic_katex_table(df_history, h_headers, height=82, compact=True, header_font_boost_pt=2.0)

        st.markdown('<div class="rb-t0-soft-gap"></div>', unsafe_allow_html=True)

        # --- 2. CONJUNTOS DE INFORMACIÓN PRIVADA ---
        # Tabla Estado (S)
        df_info_S = pd.DataFrame([{
            "t": 0,
            "z": z_val,
            "f": f_val,
            "v": v_val,
            "s": s_val,
            "m1": priors_v[0],
            "m2": priors_v[1],
            "m3": priors_v[2],
            "m4": priors_v[3]
        }])
        s_headers = ["t", "z", r"\theta_F", r"\theta_V", r"\theta_S", r"\mu_t(DC)", r"\mu_t(PAR)", r"\mu_t(ELN)", r"\mu_t(FARC)"]

        # Tabla Familia (F)
        df_info_F = pd.DataFrame([{
            "t": 0,
            "z": z_val,
            "f": f_val,
            "v": v_val,
            "s": s_val
        }])
        f_headers = ["t", "z", r"\theta_F", r"\theta_V", r"\theta_S"]

        # Tabla Secuestrador (K)
        df_info_K = pd.DataFrame([{
            "t": 0,
            "z": z_val,
            "tk": theta_k_true,
            "f": f_val,
            "v": v_val,
            "s": s_val
        }])
        k_headers = ["t", "z", r"\theta_K (\text{Verdadero})", r"\theta_F", r"\theta_V", r"\theta_S"]

        c_info1, c_info2 = st.columns(2, gap="medium")
        with c_info1:
            st.markdown('<div class="rb-t0-table-label">Familia (F)</div>', unsafe_allow_html=True)
            _render_compact_katex_expr(
                r"\mathcal{I}_0^F = \bigl(h_0,\, z,\, \theta_F,\, \theta_V,\, \theta_S\bigr)"
            )
            render_generic_katex_table(df_info_F, f_headers, height=82, compact=True, header_font_boost_pt=2.0)
        with c_info2:
            st.markdown('<div class="rb-t0-table-label">Secuestrador (K)</div>', unsafe_allow_html=True)
            _render_compact_katex_expr(
                r"\mathcal{I}_0^K = \bigl(h_0,\, z,\, \theta_K,\, \theta_F,\, \theta_V,\, \theta_S\bigr)"
            )
            render_generic_katex_table(df_info_K, k_headers, height=82, compact=True, header_font_boost_pt=2.0)

        st.markdown('<div class="rb-t0-table-label">Estado (S)</div>', unsafe_allow_html=True)
        _render_compact_katex_expr(
            r"\mathcal{I}_0^S = \bigl(h_0,\, z,\, \theta_F,\, \theta_V,\, \theta_S,\, \mu_0\bigr)"
        )
        render_generic_katex_table(df_info_S, s_headers, height=92, compact=True, header_font_boost_pt=2.0)

    mu_tab = {
        t: float(st.session_state.final_priors[i]) / 100.0
        for i, t in enumerate(TIPOS_SECUESTRADOR)
    }
    st.markdown(
        '<div class="rb-t0-table-label" style="margin:0.1rem 0 0.14rem 0;line-height:1.18;">'
        "Tabla 10 · Trayectoria de riesgos e incidencias desde t=0"
        "</div>"
        '<div style="display:flex;flex-wrap:nowrap;align-items:baseline;gap:0.55rem 1rem;'
        "overflow-x:auto;padding:0.14rem 0 0.26rem 0;font-size:0.76rem;line-height:1.28;"
        'color:rgba(49,51,63,0.74);scrollbar-width:thin;-webkit-overflow-scrolling:touch;">'
        '<span style="flex:0 0 auto;white-space:nowrap;"><b>①</b> Intensidades y cuotas</span>'
        '<span style="flex:0 0 auto;opacity:0.4;padding:0 0.05rem;">|</span>'
        '<span style="flex:0 0 auto;white-space:nowrap;"><b>②</b> Supervivencia, detección, MDG K, S, F y expectativas</span>'
        '<span style="flex:0 0 auto;opacity:0.4;padding:0 0.05rem;">|</span>'
        '<span style="flex:0 0 auto;white-space:nowrap;"><b>③</b> 10b: captura + MDG / m₀</span>'
        '<span style="flex:0 0 auto;opacity:0.4;padding:0 0.05rem;">|</span>'
        '<span style="flex:0 0 auto;white-space:nowrap;"><b>④</b> 10c: efectivas (cap, surv, rel | pay, kill, det)</span>'
        "</div>",
        unsafe_allow_html=True,
    )
    _t0_gamma_eff = float(
        st.session_state.get("base_h0_gamma")
        if st.session_state.get("mechanism_started", False) and "base_h0_gamma" in st.session_state
        else st.session_state.get("h0_gamma", st.session_state.get("cal_presion_S", 0.0))
    )
    _t0_alpha_eff = float(
        st.session_state.get("base_h0_alpha")
        if st.session_state.get("mechanism_started", False) and "base_h0_alpha" in st.session_state
        else st.session_state.get("h0_alpha", st.session_state.get("cal_alpha_star", 0.0))
    )
    _Tmad_t0 = float(st.session_state.get("cal_T_mad", 30.0))
    _lambda4_t0 = float(st.session_state.get("cal_lambda_4", 0.0005))
    _iota_t0 = float(precision_iota)

    _t0_headers = [
        "t",
        r"M(t)",
        r"\tilde{\lambda}_1(t)\ \text{(Pago)}",
        r"\tilde{\lambda}_2(t)\ \text{(Muerte)}",
        r"\tilde{\lambda}_3(t)\ \text{(Rescate)}",
        r"\tilde{\lambda}_4(t)\ \text{(Exog.)}",
        r"p_{\mathrm{Cont},t}",
        r"q(t)",
        r"\xi_1(t)",
        r"\xi_2(t)",
        r"\xi_3(t)",
        r"\xi_4(t)",
        r"\mathbb{P}_{\mathrm{E}}(s_t{=}1\mid\iota_t,\hat{\theta}_t,\theta_K)",
        r"p_{\mathrm{det},t}",
        r"\mathbb{P}_{\mathrm{I},K}(\tilde{a}_{\mathrm{rel}})",
        r"\mathbb{P}_{\mathrm{I},K}(\tilde{a}_{\mathrm{kill}})",
        r"\mathbb{P}_{\mathrm{I},K}(\tilde{a}_{\mathrm{cont}})",
        r"S(t)",
        r"F_1(t)",
        r"F_2(t)",
        r"F_3(t)",
        r"F_4(t)",
        r"\mathbb{E}_{\tilde{A}_t\mid\mathcal{Q}_t^{\mathrm{Coop}}}\!\bigl[\mathbb{P}_{\mathrm{E}}(s_t{=}1\mid\cdot)\bigr]",
        r"\mathbb{E}_{\tilde{A}_t\mid\mathcal{Q}_t^{\mathrm{Col}}}\!\bigl[\mathbb{P}_{\mathrm{E}}(m_t{=}\mathrm{rel}\mid\cdot)\bigr]",
        r"\mathbb{P}_{\mathrm{E}}(d_t{=}1\mid\alpha_t,\tilde{a}_t^F,\mathcal{I}_t^F)",
    ]
    _t0_header_tooltips = [
        "Índice temporal de la historia pública. En esta tabla solo t=0 se calcula como estado inicial.",
        "M(t)=min{1,(t/T_mad)^2}. Para t=0, M(0)=0.",
        "λ̃1(t)=M(t)·λ_{10}(t)·exp(E1), causa Pago (j=1). E1 usa los valores de Tabla 1: β_F, −β_V, β_{S,1}, β_{K,1}, β_{z,1}, ζ_{α,1}, ζ_{γ,1}, ζ_{d,1}, φ_{F,1} y φ_{K,1}, evaluados con las acciones ejecutadas ã_t.",
        "λ̃2(t)=M(t)·λ_{20}(t)·exp(E2), causa Muerte (j=2). E2 usa Tabla 1: β_{S,2}, β_{K,2}, β_{z,2}, ζ_{α,2}, ζ_{γ,2}, ζ_{d,2}, φ_{F,2}, φ_{K,2}^{kill} y φ_{K,2}^{cont}.",
        "λ̃3(t)=M(t)·λ_{30}(t)·exp(E3), causa Rescate (j=3). E3 usa Tabla 1: −β_{S,3}, β_{K,3}, β_{z,3}, ζ_{α,3}, ζ_{γ,3}, ζ_{d,3}, ζ_R, φ_{F,3} y φ_{K,3}.",
        "λ̃4(t)=λ4, canal exógeno basal de Tabla 2.",
        "p_{Cont,t}=exp[-Σ_j λ̃j(t)·Δt], con Δt=1.",
        "q(t)=1-p_{Cont,t}.",
        "ξ1(t)=λ̃1(t)/Σ_l λ̃l(t), cuota condicional de Pago.",
        "ξ2(t)=λ̃2(t)/Σ_l λ̃l(t), cuota condicional de Muerte.",
        "ξ3(t)=λ̃3(t)/Σ_l λ̃l(t), cuota condicional de Rescate.",
        "ξ4(t)=λ̃4(t)/Σ_l λ̃l(t), cuota condicional del canal exógeno.",
        "Ley primitiva de supervivencia focal (Mechanism.tex, ec. p-surv-rescue-logit-ajustado): ℙ_E(s_t=1|ι_t,θ̂_t,θ_K)=Λ[α_leth(θ_K)+β_R·ι_t·1{θ̂_t=θ_K}]. Sin tilde: no promedia MDG. El objeto \\tilde{p}_{surv} (esperanza sobre ã^S) está en la subtabla siguiente con encabezado \\tilde{p}_{surv,0}.",
        "p_{det,t}=Λ(η_0+η_1·α_t*+η_2·γ_t*). Detectabilidad marginal de colusión (Tabla 3); instrumentos α_t*, γ_t* de Pestaña 4. La versión \\tilde{p}_{det} con peso por colusión ejecutada está en la subtabla de probabilidades efectivas.",
        "ℙ_{I,K}(ã_t=a_rel|a_t^{K*},X_t): prob. de que K ejecute liberar (MDG). Logit sobre utilidades con temperatura híbrida T_t.",
        "ℙ_{I,K}(ã_t=a_kill|a_t^{K*},X_t): prob. de que K ejecute matar (MDG). Logit sobre utilidades con temperatura híbrida T_t.",
        "ℙ_{I,K}(ã_t=a_cont|a_t^{K*},X_t): prob. de que K ejecute continuar (MDG). Logit sobre utilidades con temperatura híbrida T_t.",
        "S(t)=S(t-1)·p_{Cont,t}. Probabilidad de que el proceso permanezca en m=cont hasta t. Los 4 riesgos competitivos son Pago, Muerte, Rescate y Exógeno; Liberación se modela vía ℙ_{I,K}(ã=a_rel) (MDG de K), no como incidencia acumulada F_j.",
        "F1(t)=F1(t-1)+S(t-1)·q(t)·ξ1(t), incidencia acumulada Pago. La cuota ξ1 depende de α y γ vía ζ_α, ζ_γ, ζ_d (ver λ̃1).",
        "F2(t)=F2(t-1)+S(t-1)·q(t)·ξ2(t), incidencia acumulada Muerte. La cuota ξ2 depende de α y γ vía ζ_α, ζ_γ, ζ_d (ver λ̃2).",
        "F3(t)=F3(t-1)+S(t-1)·q(t)·ξ3(t), incidencia acumulada Rescate. La cuota ξ3 depende de α, γ, p_det y la acción de rescate del Estado (ver λ̃3).",
        "F4(t)=F4(t-1)+S(t-1)·q(t)·ξ4(t), incidencia acumulada exógena. Canal basal constante λ4, sin instrumentos de política.",
        "E_{Ã|Q^Coop}[P_E(s_t=1)]: esperanza sobre (ã^K,ã^S,ã^F) con ley de implementación de F centrada en Cooperar (Q^Coop); P_E(s=1|·) es el logit de supervivencia focal según ã^S (Mechanism.tex, p-surv-rescue-logit-ajustado). Coherente con la rama cooperación de la familia.",
        "E_{Ã|Q^Col}[P_E(m_t=rel|·)]: misma convolución MDG de K y S que en la trayectoria, con F bajo ley centrada en Coludir; P_E(m=lib) se calcula con hazards competitivos (m_rel≡liberación). Igual a p̃_rel,0 en la subtabla de probabilidades efectivas.",
        "P_E(d_t=1|α_t,ã_t^F,I_t^F): probabilidad de detección con instrumentos públicos α_t,γ_t de h₀; en la calibración actual coincide con p_{det,t}=Λ(η_0+η_1 α+η_2 γ) de la misma fila (sin término explícito adicional en η que dependa de ã^F).",
    ]
    _t0_header_tooltip_latex = [
        r"t\in\{0,1,2,\ldots\}",
        r"M(t)=\min\{1,(t/T_{\mathrm{mad}})^2\}",
        r"\tilde{\lambda}_1(t)=M(t)\lambda_{10}(t)\exp\!\left(\beta_F-\beta_V+\beta_{S,1}+\beta_{K,1}+\beta_{z,1}-\zeta_{\alpha,1}\alpha_t^\ast-\zeta_{\gamma,1}\gamma_t^\ast-\zeta_{d,1}p_{\mathrm{det},t}+\varphi_{F,1}\mathbf{1}\{\tilde a_t^F=\mathrm{Pagar}\}+\varphi_{K,1}\mathbf{1}\{\tilde a_t^K=\mathrm{cont}\}\right)",
        r"\tilde{\lambda}_2(t)=M(t)\lambda_{20}(t)\exp\!\left(\beta_{S,2}+\beta_{K,2}+\beta_{z,2}+\zeta_{\alpha,2}\alpha_t^\ast+\zeta_{\gamma,2}\gamma_t^\ast-\zeta_{d,2}p_{\mathrm{det},t}-\varphi_{F,2}\mathbf{1}\{\tilde a_t^F=\mathrm{Pagar}\}+\varphi_{K,2}^{kill}\mathbf{1}\{\tilde a_t^K=\mathrm{kill}\}+\varphi_{K,2}^{cont}\mathbf{1}\{\tilde a_t^K=\mathrm{cont}\}\right)",
        r"\tilde{\lambda}_3(t)=M(t)\lambda_{30}(t)\exp\!\left(-\beta_{S,3}+\beta_{K,3}+\beta_{z,3}+\zeta_{\alpha,3}\alpha_t^\ast+\zeta_{\gamma,3}\gamma_t^\ast+\zeta_{d,3}p_{\mathrm{det},t}+\zeta_R\mathbf{1}\{\tilde a_t^S=\mathrm{Rescate}\}-\varphi_{F,3}\mathbf{1}\{\tilde a_t^F=\mathrm{Pagar}\}+\varphi_{K,3}\mathbf{1}\{\tilde a_t^K=\mathrm{cont}\}\right)",
        r"\tilde{\lambda}_4(t)=\lambda_4",
        r"p_{\mathrm{Cont},t}=\exp\!\left(-\sum_{j=1}^{4}\tilde{\lambda}_j(t)\Delta t\right)",
        r"q(t)=1-p_{\mathrm{Cont},t}",
        r"\xi_1(t)=\frac{\tilde{\lambda}_1(t)}{\sum_{\ell=1}^{4}\tilde{\lambda}_{\ell}(t)}",
        r"\xi_2(t)=\frac{\tilde{\lambda}_2(t)}{\sum_{\ell=1}^{4}\tilde{\lambda}_{\ell}(t)}",
        r"\xi_3(t)=\frac{\tilde{\lambda}_3(t)}{\sum_{\ell=1}^{4}\tilde{\lambda}_{\ell}(t)}",
        r"\xi_4(t)=\frac{\tilde{\lambda}_4(t)}{\sum_{\ell=1}^{4}\tilde{\lambda}_{\ell}(t)}",
        r"\mathbb{P}_{\mathrm{E}}(s_t=1\mid\iota_t,\hat\theta_t,\theta_K)=\Lambda\!\left(\alpha_{\mathrm{leth}}(\theta_K)+\beta_R\iota_t\mathbf{1}\{\hat\theta_t=\theta_K\}\right)",
        r"p_{\mathrm{det},t}=\Lambda(\eta_0+\eta_1\alpha_t^\ast+\eta_2\gamma_t^\ast)",
        r"\mathbb{P}_{\mathrm{I},K}(\tilde a_t^K=a\mid a_t^{K\ast},X_t)=\frac{\exp(\mathbf{1}\{a=a_t^{K\ast}\}/T_t)}{\sum_{a'\in\mathcal A^K}\exp(\mathbf{1}\{a'=a_t^{K\ast}\}/T_t)}",
        r"\mathbb{P}_{\mathrm{I},K}(\tilde a_t^K=a\mid a_t^{K\ast},X_t)=\frac{\exp(\mathbf{1}\{a=a_t^{K\ast}\}/T_t)}{\sum_{a'\in\mathcal A^K}\exp(\mathbf{1}\{a'=a_t^{K\ast}\}/T_t)}",
        r"\mathbb{P}_{\mathrm{I},K}(\tilde a_t^K=a\mid a_t^{K\ast},X_t)=\frac{\exp(\mathbf{1}\{a=a_t^{K\ast}\}/T_t)}{\sum_{a'\in\mathcal A^K}\exp(\mathbf{1}\{a'=a_t^{K\ast}\}/T_t)}",
        r"S(t)=S(t-1)p_{\mathrm{Cont},t}",
        r"F_1(t)=F_1(t-1)+S(t-1)q(t)\xi_1(t)",
        r"F_2(t)=F_2(t-1)+S(t-1)q(t)\xi_2(t)",
        r"F_3(t)=F_3(t-1)+S(t-1)q(t)\xi_3(t)",
        r"F_4(t)=F_4(t-1)+S(t-1)q(t)\xi_4(t)",
        r"\mathbb{E}_{\tilde{A}_t\mid\mathcal{Q}_t^{\mathrm{Coop}}}\!\left[\mathbb{P}_{\mathrm{E}}(s_t{=}1\mid\gamma_t,\tilde{A}_t,\theta_K)\right]",
        r"\mathbb{E}_{\tilde{A}_t\mid\mathcal{Q}_t^{\mathrm{Col}}}\!\left[\mathbb{P}_{\mathrm{E}}(m_t{=}\mathrm{rel}\mid\tilde{A}_t,R,\theta_K)\right]",
        r"\mathbb{P}_{\mathrm{E}}(d_t{=}1\mid\alpha_t,\tilde{a}_t^F,\mathcal{I}_t^F)",
    ]

    def _render_t0_theta_table(_theta_sel: str) -> None:
        _mu_theta = {t: (1.0 if t == _theta_sel else 0.0) for t in TIPOS_SECUESTRADOR}
        _ransom_t0 = float(st.session_state.get("tab3_R_override", 100.0 if f_capa == "Alta Riqueza" else 50.0))
        _df_t0_traj = _build_t0_longitudinal_mechanism_table(
            modelo=modelo,
            mu=_mu_theta,
            presion_S=_t0_gamma_eff,
            t_mad=_Tmad_t0,
            lambda4=_lambda4_t0,
            precision_iota=_iota_t0,
            alpha0=_t0_alpha_eff,
            gamma0=_t0_gamma_eff,
            ransom_scale=_ransom_t0,
            estado_duro=(s_tipo == "Duro"),
            beta_k=float(st.session_state.get("rb_betak", 0.92)),
            t_max=0,
            z_region=str(st.session_state.z_region),
            v_victim=str(st.session_state.v_victim),
            theta_true=_theta_sel,
            f_capa=str(f_capa),
            s_tipo=str(s_tipo),
            atilde_F=str(st.session_state.get("h0_Atilde_F", "Cooperar")),
            atilde_K=str(st.session_state.get("h0_Atilde_K", "Continuar")),
            atilde_S=str(st.session_state.get("h0_Atilde_S", "No Rescatar")),
        )
        _agent_probs_t0 = _build_t0_family_state_mdg_probs(
            modelo,
            mu_tab,
            _t0_gamma_eff,
            _iota_t0,
            _t0_alpha_eff,
            _t0_gamma_eff,
            _ransom_t0,
            f_capa,
        )
        _row0 = _df_t0_traj.iloc[0]
        _pk_eq = _t10_traj_mdg_K_weights(_row0)
        _ps_eq = _t10b_mdg_S_weights(_agent_probs_t0)
        _pf_coop = _mdg_indicator_probs("F", ["Cooperar", "Coludir"], "Cooperar")
        _pf_col_act = _mdg_indicator_probs("F", ["Cooperar", "Coludir"], "Coludir")
        _theta_hat_t0 = max(mu_tab, key=lambda k: float(mu_tab.get(k, 0.0)))
        _iota_c = float(max(0.0, min(1.0, float(_iota_t0))))
        _p_qcoop_s1 = _mechanism_E_tildeA_Qcoop_PE_s1(
            _theta_sel,
            _iota_c,
            str(_theta_hat_t0),
            _pk_eq,
            _ps_eq,
            _pf_coop,
        )
        _p_det_struct = float(max(0.0, min(1.0, float(_row0.get("Pdet", 0.0)))))
        _q_col_t0 = _expected_outcomes_over_tilde_A_hazards(
            _theta_sel,
            0,
            _t0_alpha_eff,
            float(_t0_gamma_eff),
            _p_det_struct,
            _pk_eq,
            _ps_eq,
            _pf_col_act,
            z_region=str(st.session_state.z_region),
            v_victim=str(st.session_state.v_victim),
            f_capa=str(f_capa),
            s_tipo=str(s_tipo),
        )
        _df_t0_traj_x = _df_t0_traj.copy()
        _df_t0_traj_x["EQcoop_s1"] = _p_qcoop_s1
        _df_t0_traj_x["EQcol_mrel"] = float(max(0.0, min(1.0, float(_q_col_t0.get("lib", 0.0)))))
        _df_t0_traj_x["PE_d1_aF"] = _p_det_struct
        _c1_post_t10 = st.session_state.get("first_cycle_post54")
        _c1_diag_t10 = st.session_state.get("first_cycle_diag52") or {}
        _has_c1_t10 = isinstance(_c1_post_t10, dict) and isinstance(_c1_post_t10.get("mu_prior"), dict)
        _df_c1_traj_x = pd.DataFrame()
        _agent_probs_c1_t10: Optional[dict[str, float]] = None
        _alpha_c1_t10 = float(_c1_diag_t10.get("alpha_usado", _t0_alpha_eff))
        _gamma_c1_t10 = float(_c1_diag_t10.get("gamma_usado", _t0_gamma_eff))
        _iota_c1_t10 = _iota_t0
        if _has_c1_t10:
            try:
                _mu_prior_c1_t10 = {
                    str(_th): float((_c1_post_t10.get("mu_prior") or {}).get(_th, 0.0))
                    for _th in TIPOS_SECUESTRADOR
                }
                _s_mu_c1_t10 = float(sum(max(0.0, v) for v in _mu_prior_c1_t10.values()))
                if _s_mu_c1_t10 > 1e-12:
                    _mu_prior_c1_t10 = {k: max(0.0, v) / _s_mu_c1_t10 for k, v in _mu_prior_c1_t10.items()}
                else:
                    _mu_prior_c1_t10 = dict(mu_tab)
                _iota_c1_t10 = float(max(_mu_prior_c1_t10.values())) if _mu_prior_c1_t10 else _iota_t0
                _df_c1_traj_full = _build_t0_longitudinal_mechanism_table(
                    modelo=modelo,
                    mu=_mu_theta,
                    presion_S=_gamma_c1_t10,
                    t_mad=_Tmad_t0,
                    lambda4=_lambda4_t0,
                    precision_iota=_iota_c1_t10,
                    alpha0=_alpha_c1_t10,
                    gamma0=_gamma_c1_t10,
                    ransom_scale=_ransom_t0,
                    estado_duro=(s_tipo == "Duro"),
                    beta_k=float(st.session_state.get("rb_betak", 0.92)),
                    t_max=1,
                    z_region=str(st.session_state.z_region),
                    v_victim=str(st.session_state.v_victim),
                    theta_true=_theta_sel,
                    f_capa=str(f_capa),
                    s_tipo=str(s_tipo),
                    atilde_F=str(_c1_diag_t10.get("atf", st.session_state.get("h0_Atilde_F", "Cooperar"))),
                    atilde_K=str(_c1_diag_t10.get("atk", st.session_state.get("h0_Atilde_K", "Continuar"))),
                    atilde_S=str(_c1_diag_t10.get("ats", st.session_state.get("h0_Atilde_S", "No Rescatar"))),
                )
                _df_c1_traj = _df_c1_traj_full[_df_c1_traj_full["t"].astype(int) == 1].copy()
                _agent_probs_c1_t10 = _build_t0_family_state_mdg_probs(
                    modelo,
                    _mu_prior_c1_t10,
                    _gamma_c1_t10,
                    _iota_c1_t10,
                    _alpha_c1_t10,
                    _gamma_c1_t10,
                    _ransom_t0,
                    f_capa,
                )
                if not _df_c1_traj.empty:
                    _row_c1 = _df_c1_traj.iloc[0]
                    _pk_c1 = _t10_traj_mdg_K_weights(_row_c1)
                    _ps_c1 = _t10b_mdg_S_weights(_agent_probs_c1_t10)
                    _pf_coop_c1 = _mdg_indicator_probs("F", ["Cooperar", "Coludir"], "Cooperar")
                    _pf_col_c1 = _mdg_indicator_probs("F", ["Cooperar", "Coludir"], "Coludir")
                    _theta_hat_c1 = max(_mu_prior_c1_t10, key=lambda k: float(_mu_prior_c1_t10.get(k, 0.0)))
                    _qcoop_c1 = _mechanism_E_tildeA_Qcoop_PE_s1(
                        _theta_sel,
                        _iota_c1_t10,
                        str(_theta_hat_c1),
                        _pk_c1,
                        _ps_c1,
                        _pf_coop_c1,
                    )
                    _pdet_c1_col = float(max(0.0, min(1.0, float(_row_c1.get("Pdet", 0.0)))))
                    _q_col_c1 = _expected_outcomes_over_tilde_A_hazards(
                        _theta_sel,
                        1,
                        _alpha_c1_t10,
                        float(_gamma_c1_t10),
                        _pdet_c1_col,
                        _pk_c1,
                        _ps_c1,
                        _pf_col_c1,
                        z_region=str(st.session_state.z_region),
                        v_victim=str(st.session_state.v_victim),
                        f_capa=str(f_capa),
                        s_tipo=str(s_tipo),
                    )
                    _df_c1_traj_x = _df_c1_traj.copy()
                    _df_c1_traj_x["EQcoop_s1"] = _qcoop_c1
                    _df_c1_traj_x["EQcol_mrel"] = float(max(0.0, min(1.0, float(_q_col_c1.get("lib", 0.0)))))
                    _df_c1_traj_x["PE_d1_aF"] = float(max(0.0, min(1.0, float(_row_c1.get("Pdet", 0.0)))))
                    _df_t0_traj_x = pd.concat([_df_t0_traj_x, _df_c1_traj_x], ignore_index=True)
            except Exception:
                _df_c1_traj_x = pd.DataFrame()
                _agent_probs_c1_t10 = None
        if not _df_c1_traj_x.empty:
            st.caption(
                f"Tabla 10 actualizada para ciclo 1: se usa α={_alpha_c1_t10:.4f}, "
                f"γ={_gamma_c1_t10:.4f}; con esos valores se actualiza p_det, M(1), "
                "λ̃_j(1) y P_cap."
            )
        _df_t0_show = _df_t0_traj_x.copy()
        for _col in _df_t0_show.columns:
            if _col != "t":
                _df_t0_show[_col] = _df_t0_show[_col].map(
                    lambda x: "" if pd.isna(x) else f"{float(x):.4f}"
                )
        _n_t10_traj_a = 12
        _cols_t10a = list(_df_t0_show.columns[:_n_t10_traj_a])
        _cols_t10b_traj = ["t"] + list(_df_t0_show.columns[_n_t10_traj_a:])
        _hdr_t10a = _t0_headers[:_n_t10_traj_a]
        _hdr_t10b_traj = [_t0_headers[0]] + _t0_headers[_n_t10_traj_a:]
        _tip_t10a = _t0_header_tooltips[:_n_t10_traj_a]
        _tip_t10b_traj = [_t0_header_tooltips[0]] + _t0_header_tooltips[_n_t10_traj_a:]
        _ltip_t10a = _t0_header_tooltip_latex[:_n_t10_traj_a]
        _ltip_t10b_traj = [_t0_header_tooltip_latex[0]] + _t0_header_tooltip_latex[_n_t10_traj_a:]
        def _t10_block_label(text: str) -> None:
            st.markdown(
                f'<div style="margin:0.35rem 0 0.08rem 0;font-weight:700;font-size:0.86rem;">{html.escape(text)}</div>',
                unsafe_allow_html=True,
            )

        _t10_tab_a = st.expander("Grupo 10a · Riesgos e incidencias", expanded=True)
        _t10_tab_b = st.expander("Grupo 10b · Implementación y desenlace m", expanded=True)
        _t10_tab_c = st.expander("Grupo 10c · Probabilidades efectivas", expanded=True)
        with _t10_tab_a:
            _t10_block_label("10a · Intensidades, supervivencia y cuotas")
            render_generic_katex_table(
                _df_t0_show[_cols_t10a],
                _hdr_t10a,
                height=136,
                compact=True,
                relaxed_compact=True,
                header_nowrap=True,
                header_tooltips=_tip_t10a,
                header_tooltip_latex=_ltip_t10a,
                header_font_boost_pt=2.0,
                header_boost_by_index={2: 3.0, 3: 3.0, 4: 3.0, 5: 3.0},
                header_tooltips_open_up=True,
                header_tooltip_top_space_px=0,
                tight_spacing=False,
            )
            st.markdown('<div style="height:0.25rem" aria-hidden="true"></div>', unsafe_allow_html=True)
            _t10_block_label("10a · Supervivencia, detección, MDG K e incidencias")
            render_generic_katex_table(
                _df_t0_show[_cols_t10b_traj],
                _hdr_t10b_traj,
                height=146,
                compact=True,
                relaxed_compact=True,
                header_nowrap=True,
                header_tooltips=_tip_t10b_traj,
                header_tooltip_latex=_ltip_t10b_traj,
                header_font_boost_pt=2.0,
                header_tooltips_open_up=True,
                header_tooltip_font_delta_pt=-1.5,
                header_tooltip_top_space_px=0,
                tight_spacing=False,
            )
        st.markdown(
            '<div style="height:0.55rem" aria-hidden="true"></div>',
            unsafe_allow_html=True,
        )
        _atk_t10 = str(st.session_state.get("h0_Atilde_K", "Continuar"))
        _ats_t10 = str(st.session_state.get("h0_Atilde_S", "No Rescatar"))
        _atf_t10 = str(st.session_state.get("h0_Atilde_F", "Cooperar"))
        _mprobs_t0, _ = _mechanism_m_probs_for_actions(
            _theta_sel, 0, _t0_alpha_eff, _t0_gamma_eff, _p_det_struct,
            _atk_t10, _ats_t10, _atf_t10,
            z_region=str(st.session_state.z_region),
            v_victim=str(st.session_state.v_victim),
            f_capa=str(f_capa),
            s_tipo=str(s_tipo),
        )
        _outcome_probs_t0 = {
            "PE_m_lib":  float(_mprobs_t0.get("Liberación", 0.0)),
            "PE_m_res":  float(_mprobs_t0.get("Rescate",    0.0)),
            "PE_m_pay":  float(_mprobs_t0.get("Pago",       0.0)),
            "PE_m_kill": float(_mprobs_t0.get("Muerte",     0.0)),
            "PE_m_cont": float(_mprobs_t0.get("Continuar",  0.0)),
        }
        _t10b_probs = {**_agent_probs_t0, **_outcome_probs_t0}
        _df_t10b = _build_t0_capture_mdg_report(
            _theta_sel,
            _df_t0_traj.iloc[0],
            _t0_alpha_eff,
            _t0_gamma_eff,
            _t10b_probs,
        )
        if not _df_c1_traj_x.empty and isinstance(_agent_probs_c1_t10, dict):
            _pdet_c1_t10 = float(_df_c1_traj_x.iloc[0].get("Pdet", _p_det_struct))
            _mprobs_c1, _ = _mechanism_m_probs_for_actions(
                _theta_sel, 1, _alpha_c1_t10, _gamma_c1_t10, _pdet_c1_t10,
                _atk_t10, _ats_t10, _atf_t10,
                z_region=str(st.session_state.z_region),
                v_victim=str(st.session_state.v_victim),
                f_capa=str(f_capa),
                s_tipo=str(s_tipo),
            )
            _outcome_probs_c1 = {
                "PE_m_lib":  float(_mprobs_c1.get("Liberación", 0.0)),
                "PE_m_res":  float(_mprobs_c1.get("Rescate",    0.0)),
                "PE_m_pay":  float(_mprobs_c1.get("Pago",       0.0)),
                "PE_m_kill": float(_mprobs_c1.get("Muerte",     0.0)),
                "PE_m_cont": float(_mprobs_c1.get("Continuar",  0.0)),
            }
            _t10b_probs_c1 = {**_agent_probs_c1_t10, **_outcome_probs_c1}
            _df_t10b_c1 = _build_t0_capture_mdg_report(
                _theta_sel,
                _df_c1_traj_x.iloc[0],
                _alpha_c1_t10,
                _gamma_c1_t10,
                _t10b_probs_c1,
            )
            _df_t10b = pd.concat([_df_t10b, _df_t10b_c1], ignore_index=True)
        _df_t10b_show = _df_t10b.copy()
        for _col in _df_t10b_show.columns:
            _df_t10b_show[_col] = _df_t10b_show[_col].map(
                lambda x: f"{int(x)}" if _col == "t" else f"{float(x):.4f}"
            )
        _hdr10b_a = [
            "t",
            r"p_{\mathrm{cap},0}",
            r"P_{I,F}(\tilde{a}_{\mathrm{coop}})",
            r"P_{I,F}(\tilde{a}_{\mathrm{col}})",
            r"P_{I,S}(\tilde{a}_{\mathrm{res}})",
            r"P_{I,S}(\tilde{a}_{\mathrm{neg}})",
        ]
        _hdr10b_b = [
            "t",
            r"P_E(m_0=\mathrm{lib})",
            r"P_E(m_0=\mathrm{res})",
            r"P_E(m_0=\mathrm{pay})",
            r"P_E(m_0=\mathrm{kill})",
            r"P_E(m_0=\mathrm{cont})",
        ]
        _tip10b_a = [
            "Periodo inicial.",
            "Probabilidad técnica de captura en t=0, Tabla 6.",
            "Probabilidad MDG de F para cooperar.",
            "Probabilidad MDG de F para coludir.",
            "Probabilidad MDG de S para rescatar.",
            "Probabilidad MDG de S para negociar.",
        ]
        _tip10b_b = [
            "Periodo inicial.",
            "Probabilidad física del desenlace m_0=lib.",
            "Probabilidad física del desenlace m_0=res.",
            "Probabilidad física del desenlace m_0=pay.",
            "Probabilidad física del desenlace m_0=kill.",
            "Probabilidad física del desenlace m_0=cont.",
        ]
        _ltip10b_a = [
            r"t=0",
            r"p_{\mathrm{cap},0}=\Lambda(\delta_a+c_0(\theta_K)+c_\alpha(\theta_K)\alpha_0+c_\gamma(\theta_K)\gamma_0+c_S)",
            r"P_{I,F}(\tilde a=a\mid a^\ast,X_t)=\frac{\exp(\mathbf{1}\{a=a^\ast\}/T_t)}{\sum_{a'\in\mathcal A^F}\exp(\mathbf{1}\{a'=a^\ast\}/T_t)}",
            r"P_{I,F}(\tilde a=a\mid a^\ast,X_t)=\frac{\exp(\mathbf{1}\{a=a^\ast\}/T_t)}{\sum_{a'\in\mathcal A^F}\exp(\mathbf{1}\{a'=a^\ast\}/T_t)}",
            r"P_{I,S}(\tilde a=a\mid a^\ast,X_t)=\frac{\exp(\mathbf{1}\{a=a^\ast\}/T_t)}{\sum_{a'\in\mathcal A^S}\exp(\mathbf{1}\{a'=a^\ast\}/T_t)}",
            r"P_{I,S}(\tilde a=a\mid a^\ast,X_t)=\frac{\exp(\mathbf{1}\{a=a^\ast\}/T_t)}{\sum_{a'\in\mathcal A^S}\exp(\mathbf{1}\{a'=a^\ast\}/T_t)}",
        ]
        _ltip10b_b = [
            r"t=0",
            r"h_j=\bigl(1-p_{\mathrm{Cont}}\bigr)\frac{\tilde\lambda_j}{\sum_\ell\tilde\lambda_\ell},\quad p_{\mathrm{Cont}}=\exp\!\left[-\sum_j\tilde\lambda_j\Delta t\right]",
            r"h_j=\bigl(1-p_{\mathrm{Cont}}\bigr)\frac{\tilde\lambda_j}{\sum_\ell\tilde\lambda_\ell},\quad p_{\mathrm{Cont}}=\exp\!\left[-\sum_j\tilde\lambda_j\Delta t\right]",
            r"h_j=\bigl(1-p_{\mathrm{Cont}}\bigr)\frac{\tilde\lambda_j}{\sum_\ell\tilde\lambda_\ell},\quad p_{\mathrm{Cont}}=\exp\!\left[-\sum_j\tilde\lambda_j\Delta t\right]",
            r"h_j=\bigl(1-p_{\mathrm{Cont}}\bigr)\frac{\tilde\lambda_j}{\sum_\ell\tilde\lambda_\ell},\quad p_{\mathrm{Cont}}=\exp\!\left[-\sum_j\tilde\lambda_j\Delta t\right]",
            r"P_E(m=\mathrm{Cont})=p_{\mathrm{Cont}}=\exp\!\left[-\sum_j\tilde\lambda_j\Delta t\right]",
        ]
        with _t10_tab_b:
            _t10_block_label("10b · Captura técnica y probabilidades MDG")
            render_generic_katex_table(
                _df_t10b_show[["t", "p_cap", "PIF_coop", "PIF_col", "PIS_res", "PIS_neg"]],
                _hdr10b_a,
                height=132,
                compact=True,
                relaxed_compact=True,
                header_nowrap=True,
                header_tooltips=_tip10b_a,
                header_tooltip_latex=_ltip10b_a,
                header_font_boost_pt=2.0,
                header_tooltips_open_up=True,
                header_tooltip_font_delta_pt=-1.5,
                header_tooltip_top_space_px=0,
                tight_spacing=False,
            )
            st.markdown('<div style="height:0.25rem" aria-hidden="true"></div>', unsafe_allow_html=True)
            _t10_block_label("10b · Desenlaces físicos m")
            render_generic_katex_table(
                _df_t10b_show[
                    ["t", "PE_m_lib", "PE_m_res", "PE_m_pay", "PE_m_kill", "PE_m_cont"]
                ],
                _hdr10b_b,
                height=126,
                compact=True,
                relaxed_compact=True,
                header_nowrap=True,
                header_tooltips=_tip10b_b,
                header_tooltip_latex=_ltip10b_b,
                header_font_boost_pt=2.0,
                header_tooltips_open_up=True,
                header_tooltip_top_space_px=0,
                tight_spacing=False,
            )
        st.markdown(
            '<div style="height:0.55rem" aria-hidden="true"></div>',
            unsafe_allow_html=True,
        )
        _df_t10c = _build_t0_tilde_prob_report(
            modelo=modelo,
            mu=mu_tab,
            theta=_theta_sel,
            presion_S=_t0_gamma_eff,
            alpha0=_t0_alpha_eff,
            gamma0=_t0_gamma_eff,
            ransom_scale=_ransom_t0,
            estado_duro=(s_tipo == "Duro"),
            beta_k=float(st.session_state.get("rb_betak", 0.92)),
            precision_iota=_iota_t0,
            agent_mdg_probs=_agent_probs_t0,
            traj_row_t0=_df_t0_traj.iloc[0],
        )
        if not _df_c1_traj_x.empty and isinstance(_agent_probs_c1_t10, dict):
            _df_t10c_c1 = _build_t0_tilde_prob_report(
                modelo=modelo,
                mu=_mu_prior_c1_t10,
                theta=_theta_sel,
                presion_S=_gamma_c1_t10,
                alpha0=_alpha_c1_t10,
                gamma0=_gamma_c1_t10,
                ransom_scale=_ransom_t0,
                estado_duro=(s_tipo == "Duro"),
                beta_k=float(st.session_state.get("rb_betak", 0.92)),
                precision_iota=_iota_c1_t10,
                agent_mdg_probs=_agent_probs_c1_t10,
                traj_row_t0=_df_c1_traj_x.iloc[0],
            )
            _df_t10c = pd.concat([_df_t10c, _df_t10c_c1], ignore_index=True)
        _df_t10c_show = _df_t10c.copy()
        for _col in _df_t10c_show.columns:
            _df_t10c_show[_col] = _df_t10c_show[_col].map(
                lambda x: f"{int(x)}" if _col == "t" else f"{float(x):.4f}"
            )
        _hdr10c_a = [
            "t",
            r"\tilde{p}_{\mathrm{cap},0}",
            r"\tilde{p}_{\mathrm{surv},0}",
            r"\tilde{p}_{\mathrm{rel},0}",
        ]
        _hdr10c_b = [
            "t",
            r"\tilde{p}_{\mathrm{pay},0}",
            r"\tilde{p}_{\mathrm{kill},0}",
            r"\tilde{p}_{\mathrm{det},0}",
        ]
        _tip10c_a = [
            "Periodo inicial.",
            "Esperanza de captura técnica sobre la acción ejecutada del Estado.",
            "Esperanza de supervivencia focal sobre la acción ejecutada del Estado.",
            "Esperanza de h_lib (hazard exógeno, Eqs. 28-29) bajo Q^Col: F fija en Coludir, K y S con pesos MDG.",
        ]
        _tip10c_b = [
            "Periodo inicial.",
            "Esperanza de h_pay (hazard de pago, Eqs. 28-29) bajo Q^Cont: K fijo en Continuar, S y F con pesos MDG.",
            "Esperanza de h_kill (hazard de muerte, Eqs. 28-29) bajo Q^Neg: S fija en No Rescatar, K y F con pesos MDG.",
            "Esperanza de detección sobre la acción ejecutada de la familia.",
        ]
        _hz_tip = (
            r"h_j=\bigl(1-p_{\mathrm{Cont}}\bigr)\dfrac{\tilde\lambda_j}{\sum_\ell\tilde\lambda_\ell},"
            r"\quad p_{\mathrm{Cont}}=\exp\!\left[-\sum_j\tilde\lambda_j\,\Delta t\right]"
        )
        _ltip10c_a = [
            r"t=0",
            r"\tilde p_{\mathrm{cap},0}=\mathbb E_{\tilde a_0^S\mid\mathcal Q_0}\!\left[\mathbb P_E(d_0=1\mid\alpha_0,\gamma_0,\tilde a_0^S,\theta_K)\right]",
            r"\tilde p_{\mathrm{surv},0}=\mathbb E_{\tilde a_0^S\mid\mathcal Q_0^{\mathrm{Res}}}\!\left[\mathbb P_E(\mathrm{surv}\mid\iota_0,\hat\theta_0,\theta_K,\tilde a_0^S)\right]",
            (
                r"\tilde p_{\mathrm{rel},0}"
                r"=\mathbb E_{\tilde A_0\mid\mathcal Q_0^{\mathrm{Col}}}\!\left["
                r"\mathbb P_E(m_0=\mathrm{lib}\mid\tilde A_0,X'_0,\theta_K)\right]"
                r"\quad(\mathcal Q_0^{\mathrm{Col}}:\tilde a_0^F=\mathrm{Coludir})"
                r"\newline " + _hz_tip
            ),
        ]
        _ltip10c_b = [
            r"t=0",
            (
                r"\tilde p_{\mathrm{pay},0}"
                r"=\mathbb E_{\tilde A_0\mid\mathcal Q_0^{\mathrm{Cont}}}\!\left["
                r"\mathbb P_E(m_0=\mathrm{pay}\mid\tilde A_0,X'_0,\theta_K)\right]"
                r"\quad(\mathcal Q_0^{\mathrm{Cont}}:\tilde a_0^K=\mathrm{Continuar})"
                r"\newline " + _hz_tip
            ),
            (
                r"\tilde p_{\mathrm{kill},0}"
                r"=\mathbb E_{\tilde A_0\mid\mathcal Q_0^{\mathrm{Neg}}}\!\left["
                r"\mathbb P_E(m_0=\mathrm{kill}\mid\tilde A_0,X'_0,\theta_K)\right]"
                r"\quad(\mathcal Q_0^{\mathrm{Neg}}:\tilde a_0^S=\mathrm{NoRescatar})"
                r"\newline " + _hz_tip
            ),
            r"\tilde p_{\mathrm{det},0}=\mathbb E_{\tilde a_0^F\mid\mathcal Q_0^F}\!\left[\mathbb P_E(d_0=1\mid\alpha_0,\tilde a_0^F,\mathcal I_0^F)\right]",
        ]
        with _t10_tab_c:
            _t10_block_label("10c · Probabilidades efectivas: captura, supervivencia y liberación")
            render_generic_katex_table(
                _df_t10c_show[["t", "ptilde_cap", "ptilde_surv", "ptilde_rel"]],
                _hdr10c_a,
                height=122,
                compact=True,
                relaxed_compact=True,
                header_nowrap=True,
                header_tooltips=_tip10c_a,
                header_tooltip_latex=_ltip10c_a,
                header_font_boost_pt=2.0,
                header_tooltips_open_up=True,
                header_tooltip_top_space_px=0,
                tight_spacing=False,
            )
            st.markdown('<div style="height:0.25rem" aria-hidden="true"></div>', unsafe_allow_html=True)
            _t10_block_label("10c · Probabilidades efectivas: pago, muerte y detección")
            render_generic_katex_table(
                _df_t10c_show[["t", "ptilde_pay", "ptilde_kill", "ptilde_det"]],
                _hdr10c_b,
                height=122,
                compact=True,
                relaxed_compact=True,
                header_nowrap=True,
                header_tooltips=_tip10c_b,
                header_tooltip_latex=_ltip10c_b,
                header_font_boost_pt=2.0,
                header_tooltips_open_up=True,
                header_tooltip_top_space_px=0,
                tight_spacing=False,
            )

    st.markdown('<span class="rb-t10-tabs-marker"></span>', unsafe_allow_html=True)
    _theta_tabs = st.tabs([f"θK = {t}" for t in TIPOS_SECUESTRADOR])
    for _tab_theta, _theta_sel in zip(_theta_tabs, TIPOS_SECUESTRADOR):
        with _tab_theta:
            _render_t0_theta_table(_theta_sel)

    st.divider()

    _R_default = 20_000_000.0
    if st.session_state.get("tab15_r_base_version") != int(_TAB15_CALIB_VERSION):
        st.session_state.tab3_R_override = float(_R_default)
        st.session_state["tab15_r_base_version"] = int(_TAB15_CALIB_VERSION)
        st.session_state.pop("tab15_ransom_sig", None)
        st.session_state.pop("tab15_last_validation", None)
        _run_kidnapper_backward_induction_cached.clear()
    if "tab3_R_override" not in st.session_state:
        st.session_state.tab3_R_override = float(_R_default)
    R_escala = float(st.session_state.get("tab3_R_override", _R_default))
    # Inicializar overrides de costos por tipo desde coeficientes calibrados
    for _th_ci in TIPOS_SECUESTRADOR:
        _c0i = _TAB15_FIXED_COST_COEFFS.get(_th_ci, {"phi": 1.0, "kappa_c": 1.0, "nu": 0.0})
        if f"tab3_phi_{_th_ci}" not in st.session_state:
            st.session_state[f"tab3_phi_{_th_ci}"] = float(_c0i["phi"])
        if f"tab3_kc_{_th_ci}" not in st.session_state:
            st.session_state[f"tab3_kc_{_th_ci}"] = float(_c0i["kappa_c"])
        if f"tab3_nu_{_th_ci}" not in st.session_state:
            st.session_state[f"tab3_nu_{_th_ci}"] = float(_c0i["nu"])
    if s_tipo == "Duro":
        alpha_ill, gamma_ill = 0.38, 0.42
    else:
        alpha_ill, gamma_ill = 0.20, 0.28

    cmh_alive, cmh_kill = cmh_alive_and_kill_shares()
    cmh_meta = load_cmh_outcome_moments()

    # =====================================================================
    # SECCIÓN: Problema de los 3 jugadores (IR / IC)
    # Insertada después de Tabla 10 y antes del análisis individual por agente.
    # =====================================================================
    st.markdown("---")
    st.markdown("## Problema de los 3 jugadores — Optimización formal y valores calibrados")

    # ── Leer parámetros de sesión con defaults ──────────────────────────────
    _p3_beta_k  = float(st.session_state.get("rb_betak",  0.92))
    _p3_pcap    = 0.12
    _p3_vl      = float(st.session_state.get("rb_vl",    100.0))
    _p3_fcol    = float(st.session_state.get("rb_fcol",   40.0))
    _p3_phi_f, _p3_kappa_f, _p3_nu_f = _rb_family_phi_kappa_nu(f_capa)
    _p3_pd0     = float(st.session_state.get("rb_pdet0",  0.08))
    _p3_pda     = float(st.session_state.get("rb_pdeta",  0.35))
    _p3_omk     = 200000.0
    _p3_omp     = float(st.session_state.get("rb_omp",   15.0))
    _p3_omg     = float(st.session_state.get("rb_omg",    1.2))
    _p3_ops     = (
        float(st.session_state.get("rb_ops0", 2.0)),
        float(st.session_state.get("rb_ops1", 0.6)),
        float(st.session_state.get("rb_ops2", 0.9)),
        float(st.session_state.get("rb_ops3", 0.30)),
        float(st.session_state.get("rb_ops4", 0.40)),
        float(st.session_state.get("rb_ops5", 0.20)),
    )
    _p3_mt      = (
        float(st.session_state.get("rb_mt0",  1.5)),
        float(st.session_state.get("rb_mt1", 0.45)),
        float(st.session_state.get("rb_mt2", 0.75)),
        float(st.session_state.get("rb_mt3", 0.25)),
        float(st.session_state.get("rb_mt4", 0.35)),
        float(st.session_state.get("rb_mt5", 0.15)),
    )
    _p3_cinst   = (
        float(st.session_state.get("rb_calp",   0.8)),
        float(st.session_state.get("rb_cgam",   0.5)),
        float(st.session_state.get("rb_ccross", 0.2)),
    )
    def _calibrate_state_costs_interior(
        ops: tuple[float, float, float, float, float, float],
        mt: tuple[float, float, float, float, float, float],
        cinst: tuple[float, float, float],
        omega_g: float,
        omega_p: float,
        ransom: float,
        target_r: tuple[float, float] = (0.90, 0.95),
        target_n: tuple[float, float] = (0.65, 0.35),
    ) -> tuple[
        tuple[float, float, float, float, float, float],
        tuple[float, float, float, float, float, float],
        tuple[float, float, float],
    ]:
        """Recalibra pendientes: V_R óptimo cuando p_surv y presión son altos; V_N óptimo interior."""
        c_alpha = max(abs(float(cinst[0])), 0.8)
        c_gamma = max(abs(float(cinst[1])), 0.5)
        c_cross = min(abs(float(cinst[2])), 0.45 * (c_alpha * c_gamma) ** 0.5)
        cinst_i = (c_alpha, c_gamma, c_cross)

        ops_i = [float(x) for x in ops]
        mt_i = [float(x) for x in mt]
        ops_i[0] = max(abs(ops_i[0]), 230000.0)
        ops_i[2] = max(abs(ops_i[2]), 500000.0)
        ops_i[4] = max(abs(ops_i[4]), 0.4)
        ops_i[5] = min(abs(ops_i[5]), 0.45 * (ops_i[2] * ops_i[4]) ** 0.5)
        mt_i[2] = max(abs(mt_i[2]), 0.75)
        mt_i[4] = max(abs(mt_i[4]), 0.35)
        mt_i[5] = min(abs(mt_i[5]), 0.45 * (mt_i[2] * mt_i[4]) ** 0.5)

        alpha_r, gamma_r = target_r
        alpha_n, gamma_n = target_n
        qg_r = ops_i[2] + 2.0 * float(omega_g) * c_gamma
        qa_r = ops_i[4] + 2.0 * float(omega_g) * c_alpha
        qga_r = ops_i[5] + float(omega_g) * c_cross
        ops_i[1] = -(qg_r * gamma_r + qga_r * alpha_r)
        ops_i[3] = -(qa_r * alpha_r + qga_r * gamma_r)

        qg_n = mt_i[2] + 2.0 * float(omega_g) * c_gamma
        qa_n = mt_i[4] + 2.0 * float(omega_g) * c_alpha
        qga_n = mt_i[5] + float(omega_g) * c_cross
        mt_i[1] = -(qg_n * gamma_n + qga_n * alpha_n)
        mt_i[3] = float(omega_p) * float(ransom) - (qa_n * alpha_n + qga_n * gamma_n)
        return tuple(ops_i), tuple(mt_i), cinst_i

    _p3_ops, _p3_mt, _p3_cinst = _calibrate_state_costs_interior(
        _p3_ops, _p3_mt, _p3_cinst, _p3_omg, _p3_omp, R_escala
    )

    # ── Computar utilidades K ───────────────────────────────────────────────
    # Usar _t0_alpha_eff / _t0_gamma_eff para coherencia con lo que muestra Tabla 12
    # (rows α₀, γ₀, C₀). alpha_ill/gamma_ill son valores ilustrativos del panel K que
    # pueden diferir del incidente concreto reflejado en h0_alpha / h0_gamma.
    _snap_k_p3 = st.session_state.get("rb_k_params_snapshot")
    if _snap_k_p3 is not None and not _snap_k_p3.empty:
        _df_p3_k_params = refresh_kidnapper_endogenous_columns(
            _force_common_tab12_r(_snap_k_p3.copy(), R_escala), modelo, _t0_gamma_eff, _t0_gamma_eff, alpha=_t0_alpha_eff
        )
        _df_p3_util_k = kidnapper_util_df_from_param_df(
            _df_p3_k_params, modelo, _t0_gamma_eff, _t0_alpha_eff, _t0_gamma_eff,
            R_escala, tipo_real, _p3_beta_k,
        )
    else:
        _df_p3_k_params = build_kidnapper_params_df(
            modelo,
            R_base=float(R_escala),
            gamma_oper=float(_t0_gamma_eff),
            p_cap_base=float(_p3_pcap),
            estado_duro=(s_tipo == "Duro"),
        )
        _df_p3_k_params = _force_common_tab12_r(_df_p3_k_params, R_escala)
        _df_p3_util_k = kidnapper_util_df_from_param_df(
            _df_p3_k_params, modelo, _t0_gamma_eff, _t0_alpha_eff, _t0_gamma_eff,
            R_escala, tipo_real, _p3_beta_k,
        )
        # Inicializar p̃_cap con el modelo logístico calibrado (igual que Tabla 10)
        _pcap_all_init = st.session_state.get("cal_pcap_params", _default_cal_pcap_params())
        _delta_a_init = float(st.session_state.get("cal_pcap_delta_a", 0.0))
        _c_S_init = float(st.session_state.get("cal_pcap_c_S", 0.0))
        for _ii_pc in _df_p3_k_params.index:
            _tipo_pc = str(_df_p3_k_params.at[_ii_pc, "theta_K"])
            _pc_row = _pcap_all_init.get(_tipo_pc, _default_cal_pcap_params()[_tipo_pc])
            _logit_pc = (
                _delta_a_init
                + float(_pc_row["c0"])
                + float(_pc_row["c_alpha"]) * _t0_alpha_eff
                + float(_pc_row["c_gamma"]) * _t0_gamma_eff
                + _c_S_init
            )
            _df_p3_k_params.at[_ii_pc, "p_cap_tilde"] = round(float(1.0 / (1.0 + np.exp(-_logit_pc))), 4)
    _last_r_calib = st.session_state.get("tab3_last_R_calib")
    if "R_escala" not in _df_p3_k_params.columns:
        _T_calib0 = int(max(1, int(st.session_state.get("tab15_T_horizon", _TAB14_TRAJ_TMAX))))
        _b_items, _l_items = _betas_lambdas_cache_items(
            st.session_state.cal_betas_dict, st.session_state.cal_lambdas_dict
        )
        _df_p3_k_params = _run_kidnapper_scale_calibration_cached(
            _df_to_cache_records(_df_p3_k_params),
            tuple(_df_p3_k_params.columns),
            _b_items,
            _l_items,
            R_base=float(R_escala),
            gamma_oper=float(_t0_gamma_eff),
            p_cap_base=float(_p3_pcap),
            estado_duro=(s_tipo == "Duro"),
            presion_S=float(_t0_gamma_eff),
            alpha=float(_t0_alpha_eff),
            beta_k=float(_p3_beta_k),
            gamma_lo=float(_t0_gamma_eff),
            gamma_hi=float(min(0.95, float(_t0_gamma_eff) + 0.38)),
            T_horizon=int(_T_calib0),
            finalize=False,
        )
        _df_p3_k_params = _force_common_tab12_r(_df_p3_k_params, R_escala)
        st.session_state["tab3_last_R_calib"] = float(R_escala)
        st.session_state["rb_k_params_snapshot"] = _df_p3_k_params.copy()
        st.session_state["tab3_finalize_pending"] = True
    elif (
        _last_r_calib is not None
        and float(_last_r_calib) != float(R_escala)
    ):
        _df_p3_k_params = _force_common_tab12_r(_df_p3_k_params, R_escala)
        st.session_state["tab3_last_R_calib"] = float(R_escala)
        st.session_state["rb_k_params_snapshot"] = _df_p3_k_params.copy()
        st.session_state.pop("tab15_ransom_sig", None)
        _run_kidnapper_backward_induction_cached.clear()
    elif _last_r_calib is None:
        st.session_state["tab3_last_R_calib"] = float(R_escala)
    # Detectar cambios en costos por tipo y limpiar caché si ocurrieron
    _cost_snap_cur = {
        f"{_p}_{_th}": float(st.session_state.get(f"tab3_{_p}_{_th}", 0.0))
        for _th in TIPOS_SECUESTRADOR for _p in ["phi", "kc", "nu"]
    }
    if _cost_snap_cur != st.session_state.get("tab3_cost_snap_prev", {}):
        st.session_state["tab3_cost_snap_prev"] = _cost_snap_cur
        st.session_state.pop("tab15_ransom_sig", None)
        st.session_state.pop("tab15_last_validation", None)
        _run_kidnapper_backward_induction_cached.clear()
    _df_p3_k_params, _tab12_floor_changed = _enforce_tab12_continuar_defaults(
        _df_p3_k_params,
        modelo,
        R_base=float(R_escala),
        gamma_oper=float(_t0_gamma_eff),
        p_cap_base=float(_p3_pcap),
        estado_duro=(s_tipo == "Duro"),
    )
    _df_p3_k_params = _force_common_tab12_r(_df_p3_k_params, R_escala)
    _df_p3_k_params = _apply_fixed_tab15_cost_params(
        _df_p3_k_params,
        modelo,
        R_base=float(R_escala),
        gamma_oper=float(_t0_gamma_eff),
        p_cap_base=float(_p3_pcap),
        estado_duro=(s_tipo == "Duro"),
    )
    _df_p3_k_params = refresh_kidnapper_endogenous_columns(
        _df_p3_k_params, modelo, _t0_gamma_eff, _t0_gamma_eff, alpha=_t0_alpha_eff
    )
    if _tab12_floor_changed:
        st.session_state["rb_k_params_snapshot"] = _df_p3_k_params.copy()
        st.session_state.pop("tab15_ransom_sig", None)
        st.session_state.pop("tab15_last_validation", None)
        _run_kidnapper_backward_induction_cached.clear()
    elif st.session_state.get("tab15_cost_policy_version") != int(_TAB15_CALIB_VERSION):
        # Nueva versión calibrada: resetear overrides de costos a los nuevos defaults
        for _th_rv in TIPOS_SECUESTRADOR:
            _c_rv = _TAB15_FIXED_COST_COEFFS.get(_th_rv, {"phi": 1.0, "kappa_c": 1.0, "nu": 0.0})
            st.session_state[f"tab3_phi_{_th_rv}"] = float(_c_rv["phi"])
            st.session_state[f"tab3_kc_{_th_rv}"]  = float(_c_rv["kappa_c"])
            st.session_state[f"tab3_nu_{_th_rv}"]  = float(_c_rv["nu"])
        st.session_state["rb_k_params_snapshot"] = _df_p3_k_params.copy()
        st.session_state["tab15_cost_policy_version"] = int(_TAB15_CALIB_VERSION)
        st.session_state.pop("tab15_ransom_sig", None)
        st.session_state.pop("tab15_last_validation", None)
        _run_kidnapper_backward_induction_cached.clear()
    # ── p̃_cap,0(θ_K) y p̃_pay,0(θ_K) según Mechanism.tex ─────────────────────
    # p̃_cap = E_{\tilde a^S|Q^Cap}[p_cap(\tilde a^S,θ,α0,γ0)].
    # p̃_pay = E_{\tilde A|Q^Cont}[P_E(m=pay|\tilde A,X',θ)] con hazards competitivos.
    # Ambas usan la misma ley MDG de Tabla 10b.
    _p3_mdg_agent = _build_t0_family_state_mdg_probs(
        modelo, mu_tab, _t0_gamma_eff, _iota_t0, _t0_alpha_eff, _t0_gamma_eff, R_escala, f_capa
    )
    _ps_k12 = _t10b_mdg_S_weights(_p3_mdg_agent)
    _pf_k12 = _t10b_mdg_F_weights(_p3_mdg_agent)
    _kc_k12 = _mdg_indicator_probs("K", ["Liberar", "Matar", "Continuar"], "Continuar")
    _pdet_k12 = _pdet_logit_prob(str(tipo_real), float(_t0_alpha_eff), float(_t0_gamma_eff))
    _df_p3_k_params = _apply_tab12_mechanism_probabilities(
        _df_p3_k_params,
        _p3_mdg_agent,
        alpha0=float(_t0_alpha_eff),
        gamma0=float(_t0_gamma_eff),
        p_det=float(_pdet_k12),
        f_capa=str(f_capa),
        s_tipo=str(s_tipo),
        t_eval=0,
    )
    # Recomputar utilidades con p̃_cap/p̃_pay corregidos (V_cont y U_kill).
    _df_p3_util_k = kidnapper_util_df_from_param_df(
        _df_p3_k_params, modelo, _t0_gamma_eff, _t0_alpha_eff, _t0_gamma_eff,
        R_escala, tipo_real, _p3_beta_k,
    )
    _row_p3_k  = _df_p3_util_k[_df_p3_util_k["theta_K"] == tipo_real]
    _u_rel_p3  = float(_row_p3_k["U_rel"].iloc[0])  if not _row_p3_k.empty else float("nan")
    _u_kill_p3 = float(_row_p3_k["U_kill"].iloc[0]) if not _row_p3_k.empty else float("nan")
    _v_cont_p3 = float(_row_p3_k["V_cont"].iloc[0]) if not _row_p3_k.empty else float("nan")
    _rama_p3_k = str(_row_p3_k["rama_optima"].iloc[0]) if not _row_p3_k.empty else "—"
    _best_p3_k = (
        max(_u_rel_p3, _u_kill_p3, _v_cont_p3)
        if not any(np.isnan([_u_rel_p3, _u_kill_p3, _v_cont_p3]))
        else float("nan")
    )

    # ── μ para Tabla 11 / triple F: creencia bayesiana completa de Familia ───
    # Mechanism.tex usa E_{θ_K | I_t^F}; por tanto Tabla 11 promedia sobre todos
    # los tipos con la creencia vigente del tablero, no con masa unitaria en θ*.
    _s_mu_tab11 = float(sum(max(0.0, float(mu_tab.get(k, 0.0))) for k in TIPOS_SECUESTRADOR))
    _mu_tab11 = {
        k: (
            max(0.0, float(mu_tab.get(k, 0.0))) / _s_mu_tab11
            if _s_mu_tab11 > 1e-12
            else 1.0 / len(TIPOS_SECUESTRADOR)
        )
        for k in TIPOS_SECUESTRADOR
    }

    # ── Esperanzas anidadas (Mechanism.tex) coherentes con Tabla 10 / 10c ───
    _p3_nested = _family_nested_expectations_tab10(
        modelo=modelo,
        mu=_mu_tab11,
        presion_S=_t0_gamma_eff,
        precision_iota=_iota_t0,
        alpha0=_t0_alpha_eff,
        gamma0=_t0_gamma_eff,
        ransom_scale=R_escala,
        estado_duro=(s_tipo == "Duro"),
        beta_k=_p3_beta_k,
        f_capa=f_capa,
        z_region=str(st.session_state.z_region),
        v_victim=str(st.session_state.v_victim),
        t_mad=_Tmad_t0,
        lambda4=_lambda4_t0,
        mu_mdg_for_agent=mu_tab,
        p_det_theta_focal=None,
    )

    # ── Computar utilidades F (α₀, γ₀ alineados a Tabla 10 / historia h₀) ───
    _df_p3_f, _ = compute_family_table(
        modelo,
        _mu_tab11,
        _t0_gamma_eff,
        _p3_vl,
        R_escala,
        _t0_gamma_eff,
        _p3_phi_f,
        _p3_kappa_f,
        _p3_nu_f,
        _p3_fcol,
        _p3_pd0,
        _p3_pda,
        _t0_alpha_eff,
        cmh_alive,
        f1_nested_triple=_p3_nested,
    )
    _p3_coop_rows = _df_p3_f.loc[_df_p3_f["Rama"].str.startswith("Cooperar"), "EU ilustrativa"]
    _p3_col_rows  = _df_p3_f.loc[_df_p3_f["Rama"].str.startswith("Colusión"), "EU ilustrativa"]
    _u_coop_p3 = float(_p3_coop_rows.iloc[0]) if not _p3_coop_rows.empty else 0.0
    _u_col_p3  = float(_p3_col_rows.iloc[0])  if not _p3_col_rows.empty  else 0.0
    _df_p3_f_cal, _ = family_calibrated_vs_endogenous(
        modelo,
        _mu_tab11,
        _t0_gamma_eff,
        _p3_vl,
        R_escala,
        _t0_gamma_eff,
        _p3_phi_f,
        _p3_kappa_f,
        _p3_nu_f,
        _p3_fcol,
        _p3_pd0,
        _p3_pda,
        _t0_alpha_eff,
        cmh_alive,
        f1_nested_triple=_p3_nested,
    )

    # ── Computar pérdidas S ─────────────────────────────────────────────────
    _vr_p3, _vn_p3, _psurv_p3, _pkill_p3 = compute_state_VR_VN(
        mu_tab, modelo, presion_S, precision_iota,
        _p3_omk, _p3_omp, _p3_omg,
        alpha_ill, gamma_ill, R_escala,
        _p3_ops, _p3_mt, _p3_cinst,
        cmh_kill, cmh_alive,
    )
    _cops_p3 = (
        _p3_ops[0] + _p3_ops[1] * gamma_ill + 0.5 * _p3_ops[2] * gamma_ill**2
        + _p3_ops[3] * alpha_ill + 0.5 * _p3_ops[4] * alpha_ill**2
        + _p3_ops[5] * gamma_ill * alpha_ill
    )
    _cmaint_p3 = (
        _p3_mt[0] + _p3_mt[1] * gamma_ill + 0.5 * _p3_mt[2] * gamma_ill**2
        + _p3_mt[3] * alpha_ill + 0.5 * _p3_mt[4] * alpha_ill**2
        + _p3_mt[5] * gamma_ill * alpha_ill
    )
    _g_p3      = (
        _p3_cinst[0] * alpha_ill**2
        + _p3_cinst[1] * gamma_ill**2
        + _p3_cinst[2] * alpha_ill * gamma_ill
    )

    # =========================================================
    st.session_state["rb_p3_bundle"] = {
        "alpha_ill": float(alpha_ill),
        "gamma_ill": float(gamma_ill),
        "R_escala": float(R_escala),
        "p3_omk": float(_p3_omk),
        "p3_omp": float(_p3_omp),
        "p3_omg": float(_p3_omg),
        "p3_ops": tuple(float(x) for x in _p3_ops),
        "p3_mt": tuple(float(x) for x in _p3_mt),
        "p3_cinst": tuple(float(x) for x in _p3_cinst),
        "p3_beta_k": float(_p3_beta_k),
        "p3_vl": float(_p3_vl),
        "p3_fcol": float(_p3_fcol),
        "p3_pd0": float(_p3_pd0),
        "p3_pda": float(_p3_pda),
        "vr_p3": float(_vr_p3),
        "vn_p3": float(_vn_p3),
        "psurv_p3": float(_psurv_p3),
        "pkill_p3": float(_pkill_p3),
        "cops_p3": float(_cops_p3),
        "cmaint_p3": float(_cmaint_p3),
        "g_p3": float(_g_p3),
        "u_coop_p3": float(_u_coop_p3),
        "u_col_p3": float(_u_col_p3),
        "t0_alpha_eff": float(_t0_alpha_eff),
        "t0_gamma_eff": float(_t0_gamma_eff),
        "iota_t0": float(_iota_t0),
        "mu_tab": {k: float(v) for k, v in mu_tab.items()},
        "tipo_real": str(tipo_real),
        "s_tipo": str(s_tipo),
        "f_capa": str(f_capa),
        "df_p3_util_k": _df_p3_util_k.copy(),
        "df_p3_k_params": _df_p3_k_params.copy(),
    }

    # =========================================================
    # JUGADOR F — Maximización
    # =========================================================
    st.markdown("### 1 · Familia ($F$) — Maximización")
    _pf1, _pf2 = st.columns([2, 3], gap="large")
    with _pf1:
        st.markdown("**Programa formal** · Mechanism.tex, ec. `family-argmax-bayes`")
        st.latex(
            r"a_t^{F\ast} \in \arg\max_{a^F \in \{a_{\mathrm{coop}},\,a_{\mathrm{col}}\}}"
            r"\;\mathbb{E}_{\theta_K \mid \mathcal{I}_t^F}\!\left["
            r"\mathcal{U}^F_t(a^F \mid \theta_K,\theta_F)\right]"
        )
        st.markdown("**Rama cooperación** · ec. `family-utility-coop`")
        st.latex(
            r"\mathcal{U}^F_t(a_{\mathrm{coop}}\mid \theta_K,\theta_F) ="
            r"\mathbb{E}_{\theta_K \mid \mathcal{I}_t^F}\!\left["
            r"\mathbb{E}_{\tilde{A}_t \mid \mathcal{Q}_t^{\mathrm{Coop}}}\!\left["
            r"\mathbb{P}_{\mathrm{E}}(s_t{=}1\mid \gamma_t,\tilde{A}_t,\theta_K)\right]\right] V_L"
            r"- e_t(\gamma_t,\theta_F)"
        )
        st.markdown("**Costo vía institucional** · ec. `family-institutional-cost`")
        st.latex(
            r"e_t(\gamma_t,\theta_F)=\phi_F(\theta_F)\,\exp\!\bigl(\kappa_F(\theta_F)\,\gamma_t\bigr)"
            r"+\nu_F(\theta_F)"
        )
        st.markdown("**Rama colusión** · ec. `family-utility-col`")
        st.latex(
            r"\mathcal{U}^F_t(a_{\mathrm{col}}) ="
            r"\mathbb{E}_{\theta_K \mid \mathcal{I}_t^F}\!\left["
            r"\mathbb{E}_{\tilde{A}_t \mid \mathcal{Q}_t^{\mathrm{Col}}}\!\left["
            r"\mathbb{P}_{\mathrm{E}}(m_t{=}\mathrm{rel}\mid \tilde{A}_t,R,\theta_K)\right]\right] V_L"
            r"- R"
            r"- \mathbb{E}_{\tilde{a}_t^F \mid \mathcal{Q}_{t}^{F}}\!\left["
            r"\mathbb{P}_{\mathrm{E}}(d_t{=}1\mid \alpha_t,\tilde{a}_t^F,\mathcal{I}_t^F)\right]"
            r"\,F_{\mathrm{col}}"
        )
        st.markdown("**IR^F** (cooperación preferida) · ec. `ir-family`")
        st.latex(
            r"\mathcal{U}^F_t(a_{\mathrm{coop}}) \;\geq\; \mathcal{U}^F_t(a_{\mathrm{col}})"
        )

    with _pf2:
        st.markdown(
            "**1.1 · Tabla 11 · Parámetros calibrados e ilustrativos (F)** "
            "(misma fuente que `compute_family_table`; **KaTeX**)"
        )
        render_tabla11_family_calibrated_katex(_df_p3_f_cal)
        st.markdown("**1.2 · Utilidades esperadas por rama (ilustrativo)**")
        _df_f_cols = ["Rama", "EU ilustrativa"] + (["Ref."] if "Ref." in _df_p3_f.columns else [])
        _col_cfg_f = {
            "Rama": st.column_config.TextColumn("Rama", width="medium"),
            "EU ilustrativa": st.column_config.NumberColumn(
                "EU ilustrativa",
                format="%.2f",
                help=(
                    "Esperanza con probabilidades anidadas de Tabla 11: coop ≈ P̂(s=1)·V_L − e(γ); "
                    "col ≈ P̂(rel)·V_L − R − P̂(det)·F_col."
                ),
            ),
        }
        if "Ref." in _df_f_cols:
            _col_cfg_f["Ref."] = st.column_config.TextColumn("Ref.", width="small")
        st.dataframe(
            _df_p3_f[_df_f_cols],
            hide_index=True,
            use_container_width=True,
            height=_glide_full_height_px(_st_table_row_count(_df_p3_f)),
            column_config=_col_cfg_f,
        )
        _ir_f_p3 = _u_coop_p3 >= _u_col_p3
        _ir_f_icon = "✅" if _ir_f_p3 else "❌"
        _a_f_opt_p3 = "Cooperar" if _ir_f_p3 else "Coludir"
        st.markdown(
            f"**IR^F**: "
            f"$\\mathcal{{U}}_{{\\mathrm{{coop}}}}$ = **{_u_coop_p3:.2f}** "
            f"{'≥' if _ir_f_p3 else '<'} "
            f"$\\mathcal{{U}}_{{\\mathrm{{col}}}}$ = **{_u_col_p3:.2f}** {_ir_f_icon} "
            f"→ **decisión óptima: {_a_f_opt_p3}**"
        )
        if not _ir_f_p3:
            st.warning(
                "Familia prefiere coludir: IR^F no se satisface con los parámetros actuales. "
                "Aumentar $V_L$ o reducir R / $F_{\\mathrm{col}}$ puede restablecer la condición."
            )

    # =========================================================
    # JUGADOR K — Maximización
    # =========================================================
    st.markdown("### 2 · Secuestrador ($K$) — Maximización")
    _pk1, _pk2 = st.columns([2, 3], gap="large")
    with _pk1:
        st.markdown("**Programa formal** · Mechanism.tex, ec. `kidnapper-argmax`")
        st.latex(
            r"a_t^{K\ast} = \arg\max_{a^K \in \mathcal{A}^K}"
            r"\;\mathcal{U}_t^K\!\left(a^K \mid h_t, \theta_K, \theta_S\right)"
        )
        st.latex(r"\mathcal{A}^K = \{a_{\mathrm{rel}},\;a_{\mathrm{kill}},\;a_{\mathrm{cont}}\}")
        st.markdown("**Liberación** ($a_{\mathrm{rel}}$)")
        st.latex(r"U^K_{\mathrm{rel}}(\theta)=-\kappa_{\mathrm{rel}}(\theta)")
        st.markdown("**Muerte** ($a_{\mathrm{kill}}$) · ec. `kidnapper-kill`")
        st.latex(
            r"U^K_{\mathrm{kill}}(\theta,\theta_S)"
            r"=\bigl(1-\tilde{p}_{\mathrm{cap},t}(\theta)\bigr)\,\eta(\theta)"
            r"-\tilde{p}_{\mathrm{cap},t}(\theta)\,F_{\mathrm{cap}}(\theta,\theta_S)"
        )
        st.markdown("**Continuación** ($a_{\mathrm{cont}}$) · ec. `kidnapper-cont`")
        st.latex(
            r"\begin{aligned}"
            r"V^K_{\mathrm{cont},t}={}&\tilde{p}_{\mathrm{pay},t}(\theta)\,R\,(1-\alpha_t)"
            r"-\Bigl[\phi(\theta)\,\exp\!\bigl(\kappa_c(\theta)\,\gamma_t\bigr)+\nu(\theta)\Bigr]\\"
            r"&-\tilde{p}_{\mathrm{cap},t}(\theta)\,F_{\mathrm{cap}}(\theta,\theta_S)\\"
            r"&+\beta(\theta)\,\bigl(1-\tilde{p}_{\mathrm{cap},t}(\theta)\bigr)\,"
            r"\mathbb{E}\!\left[V^K_{t+1}\!\left(\mu_{t+1}(a_{\mathrm{cont}})\right)\right]"
            r"\end{aligned}"
        )

    with _pk2:
        st.markdown(
            f"**2.1 · Tabla 12 · Parámetros calibrados del secuestrador (K)** "
            f"($\\theta_K={tipo_real}$; $\\alpha_0={_t0_alpha_eff:.2f}$, "
            f"$\\gamma_0={_t0_gamma_eff:.2f}$; Tabla 9 + Tabla 9a, $t=0$)"
        )
        _k12_panel_slot = st.container()
        # ── Editor Tabla 12 ────────────────────────────────────────────────
        with st.expander(
            "Editar parámetros · Tabla 12",
            expanded=False,
        ):
            st.caption(
                "**p̃_cap**, **p̃_pay** y **C(γ,θ)** se recalculan automáticamente con Mechanism.tex y Tabla 10."
            )
            _tipo_sel_k12 = str(tipo_real)
            _r_input_k12 = float(R_escala)
            # ── R común a todos los tipos ─────────────────────────────────
            _render_widget_katex_label(r"R", "Escala de rescate — común a todos los tipos (DC, PAR, ELN, FARC)")
            _new_r_val = st.number_input(
                "R",
                step=10.0,
                format="%.1f",
                help=(
                    "Rescate esperado R — valor común para DC, PAR, ELN y FARC. "
                    "Cambiarlo recalibra automáticamente Tabla 12 y Tabla 15."
                ),
                key="tab3_R_override",
                label_visibility="collapsed",
            )
            if float(_new_r_val) != float(R_escala):
                R_escala = float(_new_r_val)
            # ── Parámetros de utilidad del tipo del incidente ─────────────
            _row_k12_sel = _df_p3_k_params[
                _df_p3_k_params["theta_K"].astype(str) == str(_tipo_sel_k12)
            ]
            _ed_krel, _ed_eta, _ed_fcap, _ed_pcap = None, None, None, None
            if _row_k12_sel.empty:
                st.warning(f"No hay fila de parámetros para θ_K = {_tipo_sel_k12}.")
            else:
                _r0k = _row_k12_sel.iloc[0]
                st.markdown(f"**Utilidades · θ_K = {tipo_real}** (U_kill, U_rel)")
                _ka, _kb, _kc_col = st.columns(3)
                with _ka:
                    _render_widget_katex_label(RB_LATEX_K12_EDITOR["kappa_rel"], RB_LATEX_K12_EDITOR_CAPTION["kappa_rel"])
                    _ed_krel = st.number_input(
                        "kappa_rel",
                        value=float(_r0k["kappa_rel"]),
                        min_value=0.0, step=0.5, format="%.3f",
                        help="U_rel = −κ_rel",
                        key=f"k12_{_tipo_sel_k12}_krel",
                        label_visibility="collapsed",
                    )
                with _kb:
                    _render_widget_katex_label(RB_LATEX_K12_EDITOR["eta"], RB_LATEX_K12_EDITOR_CAPTION["eta"])
                    _ed_eta = st.number_input(
                        "eta",
                        value=float(_r0k["eta"]),
                        step=0.5, format="%.3f",
                        help="U_kill = (1−p̃_cap)·η − p̃_cap·F_cap",
                        key=f"k12_{_tipo_sel_k12}_eta",
                        label_visibility="collapsed",
                    )
                with _kc_col:
                    _render_widget_katex_label(RB_LATEX_K12_EDITOR["F_cap"], RB_LATEX_K12_EDITOR_CAPTION["F_cap"])
                    _ed_fcap = st.number_input(
                        "F_cap",
                        value=float(_r0k["F_cap"]),
                        min_value=0.0, step=1.0, format="%.3f",
                        help="Penalidad al ser capturado (entra en U_kill y V_cont).",
                        key=f"k12_{_tipo_sel_k12}_fcap",
                        label_visibility="collapsed",
                    )
                _render_widget_katex_label(RB_LATEX_K12_EDITOR["p_cap"], RB_LATEX_K12_EDITOR_CAPTION["p_cap"])
                _ed_pcap = st.number_input(
                    "p_cap_tilde",
                    value=float(_r0k["p_cap_tilde"]),
                    min_value=0.0, max_value=1.0, step=0.005, format="%.4f",
                    help="Valor calculado: E_{ã^S|Q^Cap}[p_cap(ã^S,θ,α0,γ0)] con pesos de Tabla 10b.",
                    key=f"k12_{_tipo_sel_k12}_pcap",
                    label_visibility="collapsed",
                    disabled=True,
                )
            # ── C(γ,θ) y β — pestañas por tipo ───────────────────────────
            st.markdown(
                r"$C(\gamma,\theta_K)=\phi\,\exp(\kappa_c\,\gamma)+\nu$ **y** $\beta$ **— por tipo**"
            )
            _cost_tabs_k12 = st.tabs(TIPOS_SECUESTRADOR)
            for _tab_k12, _th_k12 in zip(_cost_tabs_k12, TIPOS_SECUESTRADOR):
                with _tab_k12:
                    if f"tab3_beta_{_th_k12}" not in st.session_state:
                        _row_bi = _df_p3_k_params[
                            _df_p3_k_params["theta_K"].astype(str) == _th_k12
                        ]
                        st.session_state[f"tab3_beta_{_th_k12}"] = float(
                            _row_bi.iloc[0]["beta_k"]
                            if not _row_bi.empty and "beta_k" in _row_bi.columns
                            else _p3_beta_k
                        )
                    _col_phi, _col_beta = st.columns(2)
                    with _col_phi:
                        st.number_input(
                            "ϕ (escala costo)",
                            step=0.5,
                            format="%.2f",
                            help=f"Coeficiente base de costo para {_th_k12}: C=ϕ·exp(κ_c·γ)+ν.",
                            key=f"tab3_phi_{_th_k12}",
                        )
                        st.number_input(
                            "κ_c (sensibilidad γ)",
                            step=0.01,
                            format="%.4f",
                            help=f"Exponente de presión para {_th_k12}.",
                            key=f"tab3_kc_{_th_k12}",
                        )
                    with _col_beta:
                        st.number_input(
                            "ν (costo fijo)",
                            step=0.05,
                            format="%.3f",
                            help=f"Costo fijo de cautiverio para {_th_k12}.",
                            key=f"tab3_nu_{_th_k12}",
                        )
                        st.number_input(
                            "β (descuento)",
                            min_value=0.0,
                            max_value=0.999,
                            step=0.005,
                            format="%.4f",
                            help=f"Factor de descuento del tipo {_th_k12} en la rama continuar.",
                            key=f"tab3_beta_{_th_k12}",
                        )
            # ── Actualizar df con todos los parámetros ────────────────────
            _df_k12_new = _df_p3_k_params.copy()
            for _th_up in TIPOS_SECUESTRADOR:
                _idx_th_up = _df_k12_new[
                    _df_k12_new["theta_K"].astype(str) == _th_up
                ].index
                if len(_idx_th_up) > 0:
                    _ii_up = _idx_th_up[0]
                    _c0_up = _TAB15_FIXED_COST_COEFFS.get(_th_up, {"phi": 1.0, "kappa_c": 1.0, "nu": 0.0})
                    _df_k12_new.at[_ii_up, "phi"]    = float(st.session_state.get(f"tab3_phi_{_th_up}", _c0_up["phi"]))
                    _df_k12_new.at[_ii_up, "kappa_c"] = float(st.session_state.get(f"tab3_kc_{_th_up}",  _c0_up["kappa_c"]))
                    _df_k12_new.at[_ii_up, "nu"]     = float(st.session_state.get(f"tab3_nu_{_th_up}",  _c0_up["nu"]))
                    _df_k12_new.at[_ii_up, "beta_k"] = float(st.session_state.get(f"tab3_beta_{_th_up}", _p3_beta_k))
                    _df_k12_new.at[_ii_up, "R_escala"] = round(float(R_escala), 2)
            _idx_sel_k12 = _df_k12_new[
                _df_k12_new["theta_K"].astype(str) == str(_tipo_sel_k12)
            ].index
            if len(_idx_sel_k12) > 0 and _ed_krel is not None:
                _ii = _idx_sel_k12[0]
                _df_k12_new.at[_ii, "kappa_rel"]   = float(_ed_krel)
                _df_k12_new.at[_ii, "eta"]         = float(_ed_eta)
                _df_k12_new.at[_ii, "F_cap"]       = float(_ed_fcap)
            _df_k12_new = _force_common_tab12_r(_df_k12_new, R_escala)
            _df_k12_refreshed = refresh_kidnapper_endogenous_columns(
                _df_k12_new, modelo, _t0_gamma_eff, _t0_gamma_eff, alpha=_t0_alpha_eff
            )
            _df_k12_refreshed = _apply_tab12_mechanism_probabilities(
                _df_k12_refreshed,
                _p3_mdg_agent,
                alpha0=float(_t0_alpha_eff),
                gamma0=float(_t0_gamma_eff),
                p_det=float(_pdet_k12),
                f_capa=str(f_capa),
                s_tipo=str(s_tipo),
                t_eval=0,
            )
            st.session_state["rb_k_params_snapshot"] = _df_k12_refreshed
            st.session_state.pop("tab15_ransom_sig", None)
            _run_kidnapper_backward_induction_cached.clear()
            # Derivados informativos — evaluar en γ₀ = _t0_gamma_eff
            if not _row_k12_sel.empty:
                _ed_phi_cap = float(st.session_state.get(f"tab3_phi_{_tipo_sel_k12}", float(_r0k["phi"])))
                _ed_kc_cap  = float(st.session_state.get(f"tab3_kc_{_tipo_sel_k12}",  float(_r0k["kappa_c"])))
                _ed_nu_cap  = float(st.session_state.get(f"tab3_nu_{_tipo_sel_k12}",  float(_r0k["nu"])))
                _C_der = kidnapper_cost_c(float(_t0_gamma_eff), _ed_phi_cap, _ed_kc_cap, _ed_nu_cap)
                _qc_der = _expected_outcomes_over_tilde_A_hazards(
                    str(_tipo_sel_k12),
                    0,
                    float(_t0_alpha_eff),
                    float(_t0_gamma_eff),
                    float(_pdet_k12),
                    _kc_k12,
                    _ps_k12,
                    _pf_k12,
                    z_region=str(st.session_state.get("z_region", "Andina")),
                    v_victim=str(st.session_state.get("v_victim", "Privado")),
                    f_capa=str(f_capa),
                    s_tipo=str(s_tipo),
                )
                _pcap_der = _mechanism_tilde_p_cap_from_t10b_S(
                    str(_tipo_sel_k12), float(_t0_alpha_eff), float(_t0_gamma_eff), _p3_mdg_agent
                )
                _ppay_der = float(_qc_der.get("pay", float("nan")))
                st.caption(
                    f"**{_tipo_sel_k12}** — Derivados con γ₀={_t0_gamma_eff:.3f}: "
                    f"p̃_cap (Mec.) = **{_pcap_der:.4f}** · "
                    f"p̃_pay (Mec.) = **{_ppay_der:.4f}** · "
                    f"C(γ₀,θ) = ϕ·exp(κ_c·γ₀)+ν = **{_C_der:.4f}** · "
                    f"R = **{R_escala:.1f}**"
                )

    # Re-sincronizar _df_p3_k_params con el snapshot que acaba de escribir el editor.
    st.session_state.pop("tab3_force_scale_calib", None)
    st.session_state.pop("tab3_force_farc_r", None)
    _snap_k12_now = st.session_state.get("rb_k_params_snapshot")
    if _snap_k12_now is not None and not _snap_k12_now.empty:
        _df_p3_k_params = refresh_kidnapper_endogenous_columns(
            _force_common_tab12_r(_snap_k12_now.copy(), R_escala), modelo, _t0_gamma_eff, _t0_gamma_eff, alpha=_t0_alpha_eff
        )
        _df_p3_k_params = _force_common_tab12_r(_df_p3_k_params, R_escala)
        _df_p3_k_params = _apply_fixed_tab15_cost_params(
            _df_p3_k_params,
            modelo,
            R_base=float(R_escala),
            gamma_oper=float(_t0_gamma_eff),
            p_cap_base=float(_p3_pcap),
            estado_duro=(s_tipo == "Duro"),
        )
        _df_p3_k_params = refresh_kidnapper_endogenous_columns(
            _df_p3_k_params, modelo, _t0_gamma_eff, _t0_gamma_eff, alpha=_t0_alpha_eff
        )
    # Re-aplicar probabilidades Mechanism.tex después del editor/snapshot.
    _df_p3_k_params = _apply_tab12_mechanism_probabilities(
        _df_p3_k_params,
        _p3_mdg_agent,
        alpha0=float(_t0_alpha_eff),
        gamma0=float(_t0_gamma_eff),
        p_det=float(_pdet_k12),
        f_capa=str(f_capa),
        s_tipo=str(s_tipo),
        t_eval=0,
    )
    st.session_state["rb_k_params_snapshot"] = _df_p3_k_params.copy()
    # Detectar cambios en beta_k por tipo (no cubiertos por la detección de costos).
    # Se ejecuta sobre _df_p3_k_params ya re-sincronizado para capturar el valor actualizado.
    _beta_snap_cur = {
        str(_df_p3_k_params.at[_ii_b, "theta_K"]): round(
            float(_df_p3_k_params.at[_ii_b, "beta_k"])
            if "beta_k" in _df_p3_k_params.columns
            else float(_p3_beta_k), 6
        )
        for _ii_b in _df_p3_k_params.index
    }
    if _beta_snap_cur != st.session_state.get("tab3_beta_snap_prev", {}):
        st.session_state["tab3_beta_snap_prev"] = _beta_snap_cur
        st.session_state.pop("tab15_ransom_sig", None)
        st.session_state.pop("tab15_last_validation", None)
        st.session_state.pop("tab15_k_params_calibrated", None)
        _run_kidnapper_backward_induction_cached.clear()

    # Recalcular y renderizar Tabla 12 visible después de sincronizar el editor.
    _df_p3_util_k = kidnapper_util_df_from_param_df(
        _df_p3_k_params, modelo, _t0_gamma_eff, _t0_alpha_eff, _t0_gamma_eff,
        R_escala, tipo_real, _p3_beta_k,
    )
    _row_p3_k = _df_p3_util_k[_df_p3_util_k["theta_K"] == tipo_real]
    _u_rel_p3 = float(_row_p3_k["U_rel"].iloc[0]) if not _row_p3_k.empty else float("nan")
    _u_kill_p3 = float(_row_p3_k["U_kill"].iloc[0]) if not _row_p3_k.empty else float("nan")
    _v_cont_p3 = float(_row_p3_k["V_cont"].iloc[0]) if not _row_p3_k.empty else float("nan")
    _rama_p3_k = str(_row_p3_k["rama_optima"].iloc[0]) if not _row_p3_k.empty else "—"
    _best_p3_k = (
        max(_u_rel_p3, _u_kill_p3, _v_cont_p3)
        if not any(np.isnan([_u_rel_p3, _u_kill_p3, _v_cont_p3]))
        else float("nan")
    )
    _row_k_panel = _df_p3_k_params[_df_p3_k_params["theta_K"] == tipo_real]
    if "_k12_panel_slot" in locals():
        with _k12_panel_slot:
            if _row_k_panel.empty:
                st.warning(f"No hay fila de parámetros para θ_K = {tipo_real}.")
            else:
                _df_k_cal = _build_kidnapper_panel_calibrated_util_df(
                    _row_k_panel.iloc[0],
                    alpha=_t0_alpha_eff,
                    gamma=_t0_gamma_eff,
                    R=float(R_escala),
                    beta=float(_row_k_panel.iloc[0].get("beta_k", _p3_beta_k)),
                )
                render_kidnapper_calibrated_params_katex(_df_k_cal)
                if not _row_p3_k.empty:
                    st.caption(
                        f"Utilidades evaluadas: "
                        f"$U^K_{{\\mathrm{{rel}}}}$ = **{_u_rel_p3:.2f}**, "
                        f"$U^K_{{\\mathrm{{kill}}}}$ = **{_u_kill_p3:.2f}**, "
                        f"$V^K_{{\\mathrm{{cont}},t}}$ = **{_v_cont_p3:.2f}** "
                        f"→ rama **{_rama_p3_k}**."
                    )

    _mu0_traj = {
        t: float(st.session_state.final_priors[i]) / 100.0
        for i, t in enumerate(TIPOS_SECUESTRADOR)
    }
    _h0_d_raw = str(st.session_state.get("h0_d", "0"))
    _d0_traj = 0 if _h0_d_raw in ("—", "", "0") else 1
    _exec_h0_traj = st.session_state.get("tab3_materialization_exec_actions", {})
    _a_s_h0 = (
        str(_exec_h0_traj.get("S", st.session_state.get("h0_Atilde_S", "No Rescatar")))
        if isinstance(_exec_h0_traj, dict)
        else "No Rescatar"
    )
    _a_f_h0 = (
        str(_exec_h0_traj.get("F", st.session_state.get("h0_Atilde_F", "Cooperar")))
        if isinstance(_exec_h0_traj, dict)
        else "Cooperar"
    )
    _a_k_h0 = (
        str(_exec_h0_traj.get("K", st.session_state.get("h0_Atilde_K", "Continuar")))
        if isinstance(_exec_h0_traj, dict)
        else "Continuar"
    )
    _estado_rescate_traj = str(_a_s_h0) in ("Rescatar", "Rescate")
    _zp_traj = _focus_cmh_endogenous_tentatives(str(tipo_real))
    _pdet_traj = _pdet_logit_prob(str(tipo_real), float(_t0_alpha_eff), float(_t0_gamma_eff))
    _m_traj = str(st.session_state.get("h0_m", "Continuar"))
    if _m_traj in ("—", ""):
        _m_traj = "Continuar"
    _li_impl_traj = _build_t0_implementation_likelihood_by_theta(
        _mu0_traj,
        _t0_gamma_eff,
        _iota_t0,
        _t0_alpha_eff,
        _t0_gamma_eff,
        R_escala,
        f_capa,
        (s_tipo == "Duro"),
        float(_p3_beta_k),
        _a_f_h0,
        _a_k_h0,
        _a_s_h0,
    )
    _omega_voz_t14, _pi_call_t14, _voz_params_t14 = _resolve_voice_tab2_params()
    _tab2_bundles_traj = _tab2_bundles_all_types(
        z_region=str(st.session_state.z_region),
        v_victim=str(st.session_state.v_victim),
        f_capa=str(f_capa),
        s_tipo=str(s_tipo),
    )
    if (
        "tab15_T_horizon" not in st.session_state
        or st.session_state.get("tab15_T_horizon_version") != int(_TAB15_CALIB_VERSION)
    ):
        st.session_state["tab15_T_horizon"] = int(_TAB14_TRAJ_TMAX)
        st.session_state["tab15_T_horizon_version"] = int(_TAB15_CALIB_VERSION)
    _T_traj = int(max(1, int(st.session_state.get("tab15_T_horizon", _TAB14_TRAJ_TMAX))))
    _snapshot_mu_tab14 = pd.DataFrame()
    with st.expander(
        f"Trayectoria $\\mu_t$ ($t=0,\\ldots,{_T_traj})$ · verosimilitud Mechanism.tex",
        expanded=False,
    ):
        _render_mechanism_lh_physical_equations()
        if resolve_observed_desenlace(_m_traj) == "Continuar":
            _render_belief_update_continuar_table(
                modelo,
                _mu0_traj,
                t_eval=0,
                table_title="Tabla 13 · Verosimilitud física y actualización de creencias (μ)",
                d_obs=_d0_traj,
                presion_S=_t0_gamma_eff,
                z_region=str(st.session_state.z_region),
                v_victim=str(st.session_state.v_victim),
                alpha=_t0_alpha_eff,
                gamma=_t0_gamma_eff,
                p_det=_pdet_traj,
                zeta_alpha=float(_zp_traj.get("zeta_alpha", 0.1)),
                zeta_gamma=float(_zp_traj.get("zeta_gamma", 0.1)),
                zeta_d=float(_zp_traj.get("zeta_d", 0.1)),
                zeta_R=float(_zp_traj.get("zeta_R", 0.1)),
                estado_rescata=_estado_rescate_traj,
                t_mad=float(st.session_state.get("cal_T_mad", 30.0)),
                lambda4=float(st.session_state.get("cal_lambda_4", 0.0005)),
                omega_voz=_omega_voz_t14,
                pi_call_by_theta=_pi_call_t14,
                voz_params_by_theta=_voz_params_t14,
                V_t=None,
                atilde_F=_a_f_h0,
                atilde_K=_a_k_h0,
                atilde_S=_a_s_h0,
                implementation_likelihood_by_theta=_li_impl_traj,
                tab2_bundle_by_theta=_tab2_bundles_traj,
                aggregate_unknown_theta=False,
                aggregate_lc_unknown_theta=True,
            )
        _render_mechanism_bayes_likelihood_rest(_omega_voz_t14)
        if resolve_observed_desenlace(_m_traj) == "Continuar":
            _, _mu1_t0, _ = build_t0_bayesian_posterior_report(
                modelo,
                _mu0_traj,
                "Continuar",
                int(_d0_traj),
                presion_S=_t0_gamma_eff,
                z_region=str(st.session_state.z_region),
                v_victim=str(st.session_state.v_victim),
                alpha=_t0_alpha_eff,
                gamma=_t0_gamma_eff,
                p_det=_pdet_traj,
                zeta_alpha=float(_zp_traj.get("zeta_alpha", 0.1)),
                zeta_gamma=float(_zp_traj.get("zeta_gamma", 0.1)),
                zeta_d=float(_zp_traj.get("zeta_d", 0.1)),
                zeta_R=float(_zp_traj.get("zeta_R", 0.1)),
                estado_rescata=_estado_rescate_traj,
                t_mad=float(st.session_state.get("cal_T_mad", 30.0)),
                lambda4=float(st.session_state.get("cal_lambda_4", 0.0005)),
                t_eval=0,
                omega_voz=_omega_voz_t14,
                pi_call_by_theta=_pi_call_t14,
                voz_params_by_theta=_voz_params_t14,
                V_t=None,
                atilde_F=_a_f_h0,
                atilde_K=_a_k_h0,
                atilde_S=_a_s_h0,
                implementation_likelihood_by_theta=_li_impl_traj,
                tab2_bundle_by_theta=_tab2_bundles_traj,
            )
            with st.expander(
                "Actualización en $t=1$ (primer día con $M(1)>0$; misma señal Continuar)",
                expanded=False,
            ):
                _render_belief_update_continuar_table(
                    modelo,
                    _mu1_t0,
                    t_eval=1,
                    table_title="Tabla 13 · Verosimilitud física y actualización de creencias (μ)",
                    d_obs=_d0_traj,
                    presion_S=_t0_gamma_eff,
                    z_region=str(st.session_state.z_region),
                    v_victim=str(st.session_state.v_victim),
                    alpha=_t0_alpha_eff,
                    gamma=_t0_gamma_eff,
                    p_det=_pdet_traj,
                    zeta_alpha=float(_zp_traj.get("zeta_alpha", 0.1)),
                    zeta_gamma=float(_zp_traj.get("zeta_gamma", 0.1)),
                    zeta_d=float(_zp_traj.get("zeta_d", 0.1)),
                    zeta_R=float(_zp_traj.get("zeta_R", 0.1)),
                    estado_rescata=_estado_rescate_traj,
                    t_mad=float(st.session_state.get("cal_T_mad", 30.0)),
                    lambda4=float(st.session_state.get("cal_lambda_4", 0.0005)),
                    omega_voz=_omega_voz_t14,
                    pi_call_by_theta=_pi_call_t14,
                    voz_params_by_theta=_voz_params_t14,
                    V_t=None,
                    atilde_F=_a_f_h0,
                    atilde_K=_a_k_h0,
                    atilde_S=_a_s_h0,
                    implementation_likelihood_by_theta=_li_impl_traj,
                    tab2_bundle_by_theta=_tab2_bundles_traj,
                )
        st.markdown("**Tabla 14 · Trayectoria de creencias $\\mu_t$**")
        st.number_input(
            "Horizonte temporal T (t = 0, …, T)",
            min_value=1,
            max_value=500,
            step=1,
            format="%d",
            key="tab15_T_horizon",
            help=(
                "Horizonte compartido con Tabla 15: filas de μ_t e inducción hacia atrás "
                "hasta τ = T. Máximo permitido: 500."
            ),
        )
        _T_traj = int(max(1, int(st.session_state["tab15_T_horizon"])))
        st.caption(
            r"**`Epi_pay_Qcont_mu`**: "
            r"$\tilde{\mathbb{E}}_{\tilde{A}_t\mid\mathcal{Q}_t^{\mathrm{Cont}}}[\mathbb{P}_{\mathbb E}(m_t=\mathrm{pay}"
            r"\mid\cdot,\theta_K=\theta^\ast)]$ (sin $\mu$). "
            r"**`Epi_pcap_Qcap`**: "
            r"$\mathbb{E}_{\tilde{a}^S_t\mid\mathcal{Q}_t^{\mathrm{Cap}}}[p_{\mathrm{cap}}(\tilde{a}^S_t,\theta_i,\theta_S,"
            r"\alpha_t^\ast,\gamma_t^\ast)]$ con $(\alpha_t,\gamma_t)$ de la fila y pesos $S$ de Tabla 10b."
        )
        _voice_path_tab14 = None
        _voice_emit_mu_tab14 = True
        _inc_path_ok = (
            st.session_state.get("incident_voice_path") is not None
            and str(st.session_state.get("incident_voice_theta", "")) == str(tipo_real)
        )
        if _inc_path_ok:
            _voice_path_tab14 = st.session_state.incident_voice_path
            _voice_emit_mu_tab14 = False
        if not _inc_path_ok and st.session_state.get("incident_voice_path") is None:
            st.info(
                "Genere la trayectoria de voz en **Simulación e Incidente** (panel superior) "
                "para fijar $(V_t,x^{obs})$ del captor θ*."
            )
        _alpha_by_t = None
        _gamma_by_t = None
        _pdet_by_t = None
        _res3j_mu = st.session_state.get("opt3j_result")
        if _res3j_mu is not None:
            _df3j_mu = _res3j_mu.get("trajectory")
            if isinstance(_df3j_mu, pd.DataFrame) and not _df3j_mu.empty:
                if "α_t*" in _df3j_mu.columns and "γ_t*" in _df3j_mu.columns:
                    _alpha_by_t = [
                        float(x) for x in _df3j_mu["α_t*"].iloc[:_T_traj].tolist()
                    ]
                    _gamma_by_t = [
                        float(x) for x in _df3j_mu["γ_t*"].iloc[:_T_traj].tolist()
                    ]
                    _pdet_by_t = [
                        float(
                            1.0
                            / (
                                1.0
                                + np.exp(
                                    -(
                                        _eta0_tr
                                        + _eta1_tr * a
                                        + _eta2_tr * g
                                    )
                                )
                            )
                        )
                        for a, g in zip(_alpha_by_t, _gamma_by_t)
                    ]
                    st.caption(
                        r"Instrumentos $\alpha_t^\ast,\gamma_t^\ast$ por periodo tomados de la "
                        r"trayectoria de equilibrio (pestaña 4, optimización 3 jugadores); "
                        rf"se rellena hasta $t={_T_traj}$ repitiendo el último valor disponible."
                    )
        if (
            _alpha_by_t is not None
            and _gamma_by_t is not None
            and _pdet_by_t is not None
        ):
            _Tfill = int(_T_traj)
            while len(_alpha_by_t) < _Tfill:
                _alpha_by_t.append(
                    float(_alpha_by_t[-1]) if _alpha_by_t else float(_t0_alpha_eff)
                )
            while len(_gamma_by_t) < _Tfill:
                _gamma_by_t.append(
                    float(_gamma_by_t[-1]) if _gamma_by_t else float(_t0_gamma_eff)
                )
            _alpha_by_t = _alpha_by_t[:_Tfill]
            _gamma_by_t = _gamma_by_t[:_Tfill]
            _pdet_by_t = [
                float(
                    1.0
                    / (1.0 + np.exp(-(_eta0_tr + _eta1_tr * a + _eta2_tr * g)))
                )
                for a, g in zip(_alpha_by_t, _gamma_by_t)
            ]
        _zeta_traj = (
            float(_zp_traj.get("zeta_alpha", 0.1)),
            float(_zp_traj.get("zeta_gamma", 0.1)),
            float(_zp_traj.get("zeta_d", 0.1)),
            float(_zp_traj.get("zeta_R", 0.1)),
        )
        _mu_traj_sig = _rb_mu_traj_signature(
            t_max=int(_T_traj),
            mu0=_mu0_traj,
            m_obs="Continuar",
            d_obs=int(_d0_traj),
            presion_S=float(_t0_gamma_eff),
            z_region=str(st.session_state.z_region),
            v_victim=str(st.session_state.v_victim),
            f_capa=str(f_capa),
            s_tipo=str(s_tipo),
            alpha=float(_t0_alpha_eff),
            gamma=float(_t0_gamma_eff),
            p_det=float(_pdet_traj),
            zeta=_zeta_traj,
            estado_rescata=bool(_estado_rescate_traj),
            t_mad=float(st.session_state.get("cal_T_mad", 30.0)),
            lambda4=float(st.session_state.get("cal_lambda_4", 0.0005)),
            omega_voz=float(_omega_voz_t14),
            voice_seed=int(st.session_state.get("global_semilla_rng", 123)),
            tipo_emit=str(tipo_real),
            voice_emit_from_mu=bool(_voice_emit_mu_tab14),
            voice_digest=_rb_voice_path_digest(_voice_path_tab14),
            alpha_by_t=_rb_hashable_float_seq(_alpha_by_t),
            gamma_by_t=_rb_hashable_float_seq(_gamma_by_t),
            epi_tag=(
                str(tipo_real),
                round(float(_t0_alpha_eff), 6),
                round(float(_t0_gamma_eff), 6),
                round(float(_iota_t0), 6),
                tuple(
                    (str(_rw["theta_K"]), round(float(R_escala), 2))
                    for _, _rw in _df_p3_k_params.iterrows()
                    if "R_escala" in _df_p3_k_params.columns
                ),
            ),
        )
        _mu_snap_cached = st.session_state.get("rb_mu_traj_snapshot")
        _has_mu_cache = (
            st.session_state.get("rb_mu_traj_sig") == _mu_traj_sig
            and isinstance(_mu_snap_cached, pd.DataFrame)
            and not _mu_snap_cached.empty
        )
        _force_mu_traj_from_start = bool(
            st.session_state.pop("force_tab14_compute_from_start", False)
        )
        _compute_mu_traj = st.button(
            "Recalcular Tabla 14",
            key="tab14_compute_mu_btn",
            help="Fuerza un nuevo cálculo de μ_t (por defecto se actualiza al cambiar θ* o los parámetros).",
        ) or _force_mu_traj_from_start
        _mechanism_started = bool(st.session_state.get("mechanism_started", False))
        if not _mechanism_started:
            st.info(
                "Tabla 14 se calculará cuando presione **Iniciar proceso** en "
                "**Simulación e Incidente**."
            )
            _df_mu_traj = pd.DataFrame()
        elif _has_mu_cache and not bool(_compute_mu_traj):
            _df_mu_traj = _mu_snap_cached.copy()
        else:
            with st.spinner(f"Calculando Tabla 14 (θ* = {tipo_real})…"):
                _df_mu_traj, _ = build_mechanism_mu_trajectory(
                    modelo,
                    _mu0_traj,
                    t_max=int(_T_traj),
                    m_obs="Continuar",
                    d_obs=_d0_traj,
                    presion_S=_t0_gamma_eff,
                    z_region=str(st.session_state.z_region),
                    v_victim=str(st.session_state.v_victim),
                    alpha=_t0_alpha_eff,
                    gamma=_t0_gamma_eff,
                    p_det=_pdet_traj,
                    zeta_alpha=_zeta_traj[0],
                    zeta_gamma=_zeta_traj[1],
                    zeta_d=_zeta_traj[2],
                    zeta_R=_zeta_traj[3],
                    estado_rescata=_estado_rescate_traj,
                    t_mad=float(st.session_state.get("cal_T_mad", 30.0)),
                    lambda4=float(st.session_state.get("cal_lambda_4", 0.0005)),
                    omega_voz=_omega_voz_t14,
                    continuation_path=True,
                    alpha_by_t=_alpha_by_t,
                    gamma_by_t=_gamma_by_t,
                    p_det_by_t=_pdet_by_t,
                    pi_call_by_theta=_pi_call_t14,
                    voz_params_by_theta=_voz_params_t14,
                    voice_seed=int(st.session_state.get("global_semilla_rng", 123)),
                    tipo_emit_voz=str(tipo_real),
                    voice_emit_from_mu=_voice_emit_mu_tab14,
                    voice_path=_voice_path_tab14,
                    voice_theta_focus=str(tipo_real) if _voice_path_tab14 is not None else None,
                    tab2_bundle_by_theta=_tab2_bundles_traj,
                    atilde_F=_a_f_h0,
                    atilde_K=_a_k_h0,
                    atilde_S=_a_s_h0,
                    implementation_likelihood_by_theta=_li_impl_traj,
                )
                _df_mu_traj = _rb_attach_mu_traj_epi_columns(
                    _df_mu_traj,
                    tipo_real=str(tipo_real),
                    t0_gamma=float(_t0_gamma_eff),
                    t0_alpha=float(_t0_alpha_eff),
                    iota_t0=float(_iota_t0),
                    kc_k12=_kc_k12,
                    ps_k12=_ps_k12,
                    pf_k12=_pf_k12,
                    p3_mdg_agent=_p3_mdg_agent,
                )
            st.session_state["rb_mu_traj_sig"] = _mu_traj_sig
            st.session_state["rb_mu_traj_snapshot"] = _df_mu_traj.copy()
        _snapshot_mu_tab14 = _df_mu_traj.copy()
        if _df_mu_traj.empty:
            _df_mu_show = pd.DataFrame()
        else:
            _df_mu_show = _df_mu_traj.rename(
            columns={
                "t": "t",
                "m_t": "m_t",
                "d_t": "d_t",
                "omega_voz": "ω_voz",
                "V_t": "V_t",
                "L_H_cont": "ℒ_H^{cont}",
                "q_comp": "q(t)",
                "M_t": "M(t)",
                "L_I": "ℒ_{I,t}",
                "L_H": "ℒ_H",
                "L_d": "ℒ_d",
                "L_voz": "ℒ_voz",
                "L_F": "ℒ_{F,t}",
                "L_C": "ℒ_{C,t}",
                "L_bayes": "ℒ_F·ℒ_C",
                "L_bayes_DC": "ℒ_t(DC)",
                "L_bayes_PAR": "ℒ_t(PAR)",
                "L_bayes_ELN": "ℒ_t(ELN)",
                "L_bayes_FARC": "ℒ_t(FARC)",
                "Z_t": "Z_t",
                "p_cont": "p_{Cont,t}",
                "mu_DC": "μ(DC)",
                "mu_PAR": "μ(PAR)",
                "mu_ELN": "μ(ELN)",
                "mu_FARC": "μ(FARC)",
                "alpha_t": "α_t",
                "gamma_t": "γ_t",
                "p_det_t": "p_det,t",
                "iota": "ι_t",
                "Epi_pay_Qcont_mu": (
                    f"Ê_Ã|Q^Cont[P_E(pay|θ*={tipo_real})]"
                ),
                "Epi_pcap_Qcap": (
                    f"Ê_ã^S|Q^Cap[p_cap(·,θ*={tipo_real})]"
                ),
            }
            )
        _cols14 = [
            c
            for c in [
                "t",
                "m_t",
                "d_t",
                "ω_voz",
                "V_t",
                "ℒ_H^{cont}",
                "q(t)",
                "M(t)",
                "ℒ_{I,t}",
                "ℒ_d",
                "ℒ_voz",
                "ℒ_{F,t}",
                "ℒ_{C,t}",
                "ℒ_F·ℒ_C",
                "ℒ_t(DC)",
                "ℒ_t(PAR)",
                "ℒ_t(ELN)",
                "ℒ_t(FARC)",
                "Z_t",
                "p_{Cont,t}",
                "μ(DC)",
                "μ(PAR)",
                "μ(ELN)",
                "μ(FARC)",
                "α_t",
                "γ_t",
                "p_det,t",
                "ι_t",
                f"Ê_ã^S|Q^Cap[p_cap(·,θ*={tipo_real})]",
                f"Ê_Ã|Q^Cont[P_E(pay|θ*={tipo_real})]",
            ]
            if c in _df_mu_show.columns
        ]
        if _cols14:
            _df_mu_show_render = _df_mu_show[_cols14].copy()
            for _cn in _df_mu_show_render.columns:
                if _cn != "m_t":
                    _df_mu_show_render[_cn] = pd.to_numeric(
                        _df_mu_show_render[_cn], errors="coerce"
                    )
            if "m_t" in _df_mu_show_render.columns:
                _df_mu_show_render["m_t"] = _df_mu_show_render["m_t"].astype(str)
            render_tab14_mu_katex_table(_df_mu_show_render, str(tipo_real))

    # Tabla 15: inducción hacia atrás (mismo horizonte T que Tabla 14)
    _T_tab15 = int(max(1, int(st.session_state.get("tab15_T_horizon", _TAB14_TRAJ_TMAX))))
    # tab15_view_theta: tipo activo en la vista de Tabla 15 (puede diferir del selectbox global).
    # Se resetea cuando el usuario cambia el selectbox global.
    if str(st.session_state.get("tab15_main_theta_last", "")) != str(tipo_real):
        st.session_state["tab15_view_theta"] = str(tipo_real)
        st.session_state["tab15_main_theta_last"] = str(tipo_real)
    if st.session_state.get("tab15_view_theta") not in TIPOS_SECUESTRADOR:
        st.session_state["tab15_view_theta"] = str(tipo_real)
    _view_theta = str(st.session_state["tab15_view_theta"])

    # ── Acción óptima del secuestrador en τ=0 (Col. 14, Tabla 15) ──────────────
    _tab15_calib_ready_pre = (
        isinstance(st.session_state.get("tab15_k_params_calibrated"), pd.DataFrame)
        and not st.session_state["tab15_k_params_calibrated"].empty
        and str(st.session_state.get("tab15_theta", "")) == str(tipo_real)
        and st.session_state.get("tab15_mu_sig") == st.session_state.get("rb_mu_traj_sig")
        and int(st.session_state.get("tab15_T_cached", -1)) == int(_T_tab15)
        and int(st.session_state.get("tab15_calib_version", -1)) == int(_TAB15_CALIB_VERSION)
    )
    _opt_t0_display = "—"
    if _tab15_calib_ready_pre and isinstance(_snapshot_mu_tab14, pd.DataFrame) and not _snapshot_mu_tab14.empty:
        _df_p3_kp_pre = st.session_state["tab15_k_params_calibrated"].copy()
        _b_items_pre, _l_items_pre = _betas_lambdas_cache_items(
            st.session_state.cal_betas_dict, st.session_state.cal_lambdas_dict
        )
        try:
            _df_ia_pre, _ = _run_kidnapper_backward_induction_cached(
                _df_to_cache_records(_snapshot_mu_tab14),
                tuple(_snapshot_mu_tab14.columns),
                _df_to_cache_records(_df_p3_kp_pre),
                tuple(_df_p3_kp_pre.columns),
                _b_items_pre,
                _l_items_pre,
                tipo_real=str(_view_theta),
                beta_k=float(_p3_beta_k),
                R=float(R_escala),
                t_mad=float(st.session_state.get("cal_T_mad", 30.0)),
                T=int(_T_tab15),
                alpha_fallback=float(_t0_alpha_eff),
                gamma_fallback=float(_t0_gamma_eff),
                alpha_tab12=float(_t0_alpha_eff),
                ransom_tab12=float(R_escala),
            )
            if not _df_ia_pre.empty:
                _r0_pre = _df_ia_pre.loc[_df_ia_pre["t"].astype(int) == 0]
                if not _r0_pre.empty:
                    _opt_t0_display = str(_r0_pre.iloc[0].get("opcion_BW", "—"))
        except Exception:
            pass
    if _opt_t0_display != "—":
        _opt_t0_label = _opt_t0_display.replace("(a_cont)", "").replace("(a_rel)", "").replace("(a_kill)", "").strip()
        st.metric(
            label=f"Acción óptima secuestrador τ=0 · θ = {tipo_real} · Tabla 15, col. 14",
            value=_opt_t0_label,
        )
    else:
        st.info(
            "Calcule la **Tabla 15** (abajo) para ver la acción óptima del secuestrador en τ=0."
        )

    with st.expander(
        f"Tabla 15 · Secuestrador (θ* = {_view_theta}): inducción hacia atrás, T = {_T_tab15}",
        expanded=False,
    ):
        st.caption("Horizonte **T** compartido con Tabla 14 (control arriba en la trayectoria $\\mu_t$).")
        _t_mad_v = float(st.session_state.get("cal_T_mad", 30.0))
        _force_tab15_from_start = bool(
            st.session_state.pop("force_tab15_compute_from_start", False)
        )
        _run_tab15_now = st.button(
            "Recalcular Tabla 15",
            key="tab15_compute_btn",
            help="Fuerza inducción hacia atrás y calibración (por defecto al cambiar θ* o Tabla 14).",
        ) or _force_tab15_from_start
        _tab15_cached_ready = (
            isinstance(st.session_state.get("tab15_mu_snapshot"), pd.DataFrame)
            and not st.session_state["tab15_mu_snapshot"].empty
            and isinstance(st.session_state.get("tab15_k_params_calibrated"), pd.DataFrame)
            and not st.session_state["tab15_k_params_calibrated"].empty
            and isinstance(st.session_state.get("tab15_last_validation"), dict)
            and bool(st.session_state["tab15_last_validation"])
            and str(st.session_state.get("tab15_theta", "")) == str(tipo_real)
            and st.session_state.get("tab15_mu_sig") == st.session_state.get("rb_mu_traj_sig")
            and int(st.session_state.get("tab15_T_cached", -1)) == int(_T_tab15)
            and int(st.session_state.get("tab15_calib_version", -1)) == int(_TAB15_CALIB_VERSION)
        )
        _mu_traj_ready = (
            isinstance(_snapshot_mu_tab14, pd.DataFrame)
            and not _snapshot_mu_tab14.empty
            and not _df_p3_k_params.empty
            and "theta_K" in _df_p3_k_params.columns
        )
        _need_tab15_run = bool(_run_tab15_now) or (bool(_mu_traj_ready) and not _tab15_cached_ready)
        if not _mu_traj_ready:
            st.info(
                "Calcule primero la **Tabla 14** (trayectoria $\\mu_t$) para habilitar la Tabla 15."
            )
        else:
            if _need_tab15_run:
                with st.spinner(f"Calculando Tabla 15 (θ* = {tipo_real})…"):
                    _df_p3_k_params, _val15_all = _tab15_calibrated_switch_summary(
                        _df_p3_k_params,
                        modelo,
                        _snapshot_mu_tab14,
                        beta_k=float(_p3_beta_k),
                        R_base=float(R_escala),
                        t_mad=float(_t_mad_v),
                        T=int(_T_tab15),
                        alpha=float(_t0_alpha_eff),
                        gamma=float(_t0_gamma_eff),
                    )
                    _df_p3_k_params = _force_common_tab12_r(_df_p3_k_params, R_escala)
                st.session_state["rb_k_params_snapshot"] = _df_p3_k_params.copy()
                st.session_state["tab15_k_params_calibrated"] = _df_p3_k_params.copy()
                st.session_state["tab15_mu_snapshot"] = _snapshot_mu_tab14.copy()
                st.session_state["tab15_last_validation"] = _val15_all
                st.session_state["tab15_T_cached"] = int(_T_tab15)
                st.session_state["tab15_theta"] = str(tipo_real)
                st.session_state["tab15_mu_sig"] = st.session_state.get("rb_mu_traj_sig")
                st.session_state["tab15_calib_version"] = int(_TAB15_CALIB_VERSION)
                st.session_state.pop("tab15_ransom_sig", None)
                _run_kidnapper_backward_induction_cached.clear()
            else:
                _df_p3_k_params = st.session_state["tab15_k_params_calibrated"].copy()
                _val15_all = dict(st.session_state["tab15_last_validation"])
                _T_tab15 = int(st.session_state.get("tab15_T_cached", _T_tab15))
            _snapshot_mu_tab15 = _snapshot_mu_tab14.copy()
            _b_items, _l_items = _betas_lambdas_cache_items(
                st.session_state.cal_betas_dict, st.session_state.cal_lambdas_dict
            )
            _ransom_t15 = float(R_escala)
            _last_r15 = st.session_state.get("tab15_ransom_sig")
            if _last_r15 is None or float(_last_r15) != float(_ransom_t15):
                _run_kidnapper_backward_induction_cached.clear()
            st.session_state["tab15_ransom_sig"] = float(_ransom_t15)
            _df_ia, _meta_ia = _run_kidnapper_backward_induction_cached(
                _df_to_cache_records(_snapshot_mu_tab15),
                tuple(_snapshot_mu_tab15.columns),
                _df_to_cache_records(_df_p3_k_params),
                tuple(_df_p3_k_params.columns),
                _b_items,
                _l_items,
                tipo_real=str(_view_theta),
                beta_k=float(_p3_beta_k),
                R=float(R_escala),
                t_mad=float(_t_mad_v),
                T=int(_T_tab15),
                alpha_fallback=float(_t0_alpha_eff),
                gamma_fallback=float(_t0_gamma_eff),
                alpha_tab12=float(_t0_alpha_eff),
                ransom_tab12=float(_ransom_t15),
            )
            _tau_switch_selected = (
                _val15_all.get(str(_view_theta), {}).get("primer_tau")
                if isinstance(_val15_all, dict)
                else _meta_ia.get("primer_tau_backward")
            )
            _r_col9 = float(_meta_ia.get("R_col9_flow_rev", _ransom_t15))
            _pp_t1 = float("nan")
            _alpha_t1 = float(_t0_alpha_eff)
            _flow_t1 = float("nan")
            _chk_t1 = float("nan")
            _opt_t0_full = "—"
            _opt_t1_full = "—"
            _opt_t2_full = "—"
            if not _df_ia.empty:
                _r1 = _df_ia.loc[_df_ia["t"].astype(int) == 1]
                if not _r1.empty:
                    _opt_t1_full = str(_r1.iloc[0].get("opcion_BW", "—"))
                _r2 = _df_ia.loc[_df_ia["t"].astype(int) == 2]
                if not _r2.empty:
                    _opt_t2_full = str(_r2.iloc[0].get("opcion_BW", "—"))
                    if "flow_rev" in _r1.columns:
                        _flow_t1 = float(_r1.iloc[0]["flow_rev"])
                    if "Epi_pay_Qcont_mu" in _snapshot_mu_tab15.columns:
                        _m1 = _snapshot_mu_tab15.loc[
                            _snapshot_mu_tab15["t"].astype(int) == 1
                        ]
                        if not _m1.empty:
                            _pp_t1 = float(_m1.iloc[0]["Epi_pay_Qcont_mu"])
                            _alpha_t1 = float(_m1.iloc[0].get("alpha_t", _t0_alpha_eff))
                _r0 = _df_ia.loc[_df_ia["t"].astype(int) == 0]
                _opt_t0_full = str(_r0.iloc[0].get("opcion_BW", "—")) if not _r0.empty else "—"
                st.session_state["tab15_opt_t0"] = str(_opt_t0_full)
                st.session_state["tab15_opt_t0_theta"] = str(_view_theta)
            if np.isfinite(_pp_t1):
                _chk_t1 = float(
                    kidnapper_tab15_flow_rev_col9(
                        _pp_t1, _r_col9, float(_alpha_t1)
                    )
                )
            if isinstance(_val15_all, dict) and str(_view_theta) in _val15_all:
                _prev_switches_view = [
                    int(_val15_all[_pth]["primer_tau"])
                    for _pth in ["DC", "PAR", "ELN", "FARC"][:["DC", "PAR", "ELN", "FARC"].index(str(_view_theta))]
                    if _pth in _val15_all and _val15_all[_pth].get("primer_tau") is not None
                ]
                _min_sw_view = max([3] + _prev_switches_view)
                _sw_full = _meta_ia.get("primer_tau_backward")
                _ok_sw_full = (
                    _sw_full is not None
                    and int(_sw_full) > max(2, int(_min_sw_view))
                    and int(_sw_full) < int(min(100, _T_tab15 + 1))
                )
                _val15_all[str(_view_theta)] = {
                    **dict(_val15_all[str(_view_theta)]),
                    "opcion_tau0": _opt_t0_full,
                    "ok_tau1": bool(_opt_t1_full == "Continuar (a_cont)"),
                    "ok_tau2": bool(_opt_t2_full == "Continuar (a_cont)"),
                    "ok_tau1_tau2": bool(
                        _opt_t1_full == "Continuar (a_cont)"
                        and _opt_t2_full == "Continuar (a_cont)"
                    ),
                    "ok_switch": bool(_ok_sw_full),
                    "ok_switch_before_100": bool(_ok_sw_full),
                    "primer_tau": int(_sw_full) if _sw_full is not None else None,
                    "opcion_tau1": _opt_t1_full,
                    "opcion_tau2": _opt_t2_full,
                    "verificado_con_tabla_completa": True,
                }
                st.session_state["tab15_last_validation"] = dict(_val15_all)
            st.caption(
                f"Col. 9 = $\\tilde p_{{pay}} \\cdot R_{{Tabla\\,12}}(\\theta^\\ast) "
                f"\\cdot (1-\\alpha_\\tau)$ con **R = {_r_col9:.2f}** "
                f"(R_base = {R_escala:.1f}). "
                f"τ=1: flow_rev = {_flow_t1:.4f}"
                + (
                    f", verificación pp×R×(1−α_τ) = {_chk_t1:.4f}"
                    if np.isfinite(_chk_t1)
                    else ""
                )
                + ". Columna **R_tab12_col9** repite el R usado en cada fila."
            )
            _c1_b, _c2_b, _c3_b = st.columns(3)
            with _c1_b:
                st.metric(
                    "Primera τ (inducción IA), acción ≠ Continuar",
                    str(_tau_switch_selected or "—"),
                )
            with _c2_b:
                st.metric(
                    "Primera τ con V_stationary ≤ max(U_rel, U_kill) (tras dominar)",
                    str(_meta_ia.get("primer_tau_stationary_below") or "—"),
                )
            with _c3_b:
                st.metric("Filas (τ = T … 1)", str(len(_df_ia)))
            if _opt_t1_full != "Continuar (a_cont)":
                st.error(
                    "Restricción incumplida en la Tabla 15 renderizada: "
                    f"en τ=1 la columna 14 es **{_opt_t1_full}**, debe ser **Continuar (a_cont)**. "
                    "Ajuste los parámetros de costos o β en Tabla 12."
                )
            st.caption(
                "Col. 14 ($a_K^*$): $\\arg\\max\\{\\text{cols. 7, 8, 13}\\}$ en cada $\\tau$. "
                "El momento del cambio de rama depende de $\\theta^\\ast$ vía costos fijos por tipo, captura y β; "
                "R es común para todos los tipos y Tabla 15 no recalibra costos durante el cálculo."
            )
            if _tau_switch_selected is not None and int(_tau_switch_selected) <= 1:
                st.error(
                    "El escenario base no logró garantizar primera τ de cambio > 1. "
                    "Revise costos y β de Tabla 12 para este θ."
                )
            if _val15_all:
                _val15_rows = []
                for _vth, _vd in _val15_all.items():
                    _val15_rows.append(
                        {
                            "θ_K": str(_vth),
                            "τ=1 col.14": "Continuar" if _vd.get("ok_tau1") else _vd.get("opcion_tau1", "—"),
                            "Cambio de Continuar en τ": int(_vd.get("primer_tau")) if _vd.get("primer_tau") is not None else None,
                            "Costos fijos": bool(_vd.get("costos_fijos_tabla12", False)),
                            "R común": _vd.get("R_escala", "—"),
                        }
                    )
                _all15_ok = all(
                    bool(r["τ=1 col.14"] == "Continuar")
                    and r["Cambio de Continuar en τ"] is not None
                    and int(r["Cambio de Continuar en τ"]) > 1
                    for r in _val15_rows
                )
                _sw_order_vals = [
                    _val15_all.get(_th, {}).get("primer_tau")
                    for _th in ["DC", "PAR", "ELN", "FARC"]
                    if isinstance(_val15_all, dict)
                ]
                _order15_ok = (
                    len(_sw_order_vals) == 4
                    and all(_v is not None for _v in _sw_order_vals)
                    and int(_sw_order_vals[0]) < int(_sw_order_vals[1]) < int(_sw_order_vals[2]) < int(_sw_order_vals[3])
                )
                if _all15_ok and _order15_ok:
                    st.success(
                        "Verificación Tabla 15: los 4 tipos tienen Continuar en τ=1 "
                        "y cumplen el orden DC < PAR < ELN < FARC."
                    )
                else:
                    st.warning(
                        "Verificación Tabla 15 pendiente: algún tipo no cumple Continuar en τ=1 "
                        "o el orden DC < PAR < ELN < FARC."
                    )
                if _view_theta != str(tipo_real):
                    st.info(
                        f"Vista: **θ = {_view_theta}** · "
                        f"Grupo secuestrador del incidente: **{tipo_real}** (sin cambios). "
                        "Haz clic en la fila del incidente para restaurar."
                    )
                st.caption(
                    "Haz clic en una fila para ver la Tabla 15 (col. 12 incluida) para ese θ_K. "
                    "El selector **Grupo secuestrador** no cambia."
                )
                _val15_sel_event = st.dataframe(
                    pd.DataFrame(_val15_rows),
                    use_container_width=True,
                    hide_index=True,
                    on_select="rerun",
                    selection_mode="single-row",
                    key="tab15_val_summary_sel",
                )
                _sel_rows_15 = (
                    _val15_sel_event.selection.rows
                    if hasattr(_val15_sel_event, "selection")
                    else []
                )
                if _sel_rows_15:
                    _sel_th_15 = str(_val15_rows[_sel_rows_15[0]]["θ_K"])
                    if _sel_th_15 != _view_theta:
                        st.session_state["tab15_view_theta"] = _sel_th_15
                        st.rerun()
            else:
                st.warning("No se pudo construir el resumen de cambio por tipo para Tabla 15.")
            if _df_ia.empty:
                st.warning("No se pudo construir la tabla de inducción hacia atrás.")
            else:
                _order_ia = [
                    "t",
                    "mu_star",
                    "mu_DC",
                    "mu_PAR",
                    "mu_ELN",
                    "mu_FARC",
                    "U_kill",
                    "U_rel",
                    "flow_rev",
                    "flow_cost",
                    "flow_cap",
                    "V_next",
                    "V_cont",
                    "opcion_BW",
                ]
                _col_ia = {
                    "t": "1. τ",
                    "mu_star": "2. μ(θ*)",
                    "mu_DC": "3. μ(DC)",
                    "mu_PAR": "4. μ(PAR)",
                    "mu_ELN": "5. μ(ELN)",
                    "mu_FARC": "6. μ(FARC)",
                    "U_kill": "7. U_kill",
                    "U_rel": "8. U_rel",
                    "flow_rev": "9. p̃_pay·R·(1−α_τ)",
                    "flow_cost": "10. −C_t",
                    "flow_cap": "11. −p̃_cap F_cap",
                    "V_next": "12. Σμ β(1−p̃_cap)V_cont(τ+1)",
                    "V_cont": "13. V̄_cont",
                    "opcion_BW": "14. a_K*",
                }
                _df_ia_disp = _df_ia[_order_ia].rename(columns=_col_ia)
                render_tab15_backward_katex_table(_df_ia_disp)

                st.markdown("---")
                st.markdown(
                    f"**Resumen · cambio de rama (Tabla 15)** · "
                    f"θ* = **{_view_theta}**"
                    + (f" (incidente: {tipo_real})" if _view_theta != str(tipo_real) else "")
                )
                _sw_sum = build_tab15_incident_switch_summary(
                    _df_ia,
                    _snapshot_mu_tab15,
                    str(_view_theta),
                    alpha_fallback=float(_t0_alpha_eff),
                    gamma_fallback=float(_t0_gamma_eff),
                )
                _opt_t0_label_sum = (
                    str(_opt_t0_full)
                    .replace("(a_cont)", "")
                    .replace("(a_rel)", "")
                    .replace("(a_kill)", "")
                    .strip()
                    or "—"
                )
                st.metric(
                    label="Elección óptima τ=0 · Tabla 15, col. 14",
                    value=_opt_t0_label_sum,
                    help="Acción óptima reportada en la columna 14 (a_K*) de la Tabla 15 para τ=0.",
                )
                render_tab15_switch_summary_katex(_sw_sum)
                if _sw_sum.get("tau_cambio") is None:
                    st.caption(
                        "En todo el horizonte **T** la inducción hacia atrás mantiene "
                        "**Continuar** como rama óptima (col. 14)."
                    )
                else:
                    st.caption(
                        f"En **τ = {_sw_sum['tau_cambio']}** la política pasa de "
                        f"**{_sw_sum['rama_anterior']}** a **{_sw_sum['rama_nueva']}**; "
                        f"μ({_sw_sum['theta_star']}) = {float(_sw_sum['mu_theta_star']):.4f}, "
                        f"α_t = {float(_sw_sum['alpha_t']):.4f}, "
                        f"γ_t = {float(_sw_sum['gamma_t']):.4f} "
                        f"(valores de la trayectoria Tabla 14 en ese τ)."
                    )

with tab_mech_sol:
    _U = {
        "th": "\u03b8",
        "mu": "\u03bc",
        "al": "\u03b1",
        "ga": "\u03b3",
        "io": "\u03b9",
        "be": "\u03b2",
        "ka": "\u03ba",
        "et": "\u03b7",
        "nu": "\u03bd",
        "ph": "\u03c6",
        "calU": "\U0001d4b0",
    }
    st.markdown('<span class="rb-tab4-title-scale"></span>', unsafe_allow_html=True)
    st.markdown("## Solución Mecanismo")
    st.caption(
        "Estado ($S$): minimización en $(\alpha,\gamma)$, verificación $\Gamma_t(\mu_t)$ "
        "y análisis interactivo. Los valores calibrados provienen del "
        "*Problema de los 3 jugadores* en la pestaña 4."
    )
    _p3b = st.session_state.get("rb_p3_bundle")
    if not _p3b:
        st.info(
            "Abra la pestaña **4 · Familia-Secuestrador** y desplácese hasta "
            "**Problema de los 3 jugadores** para generar los valores calibrados."
        )
    else:
        alpha_ill = float(_p3b["alpha_ill"])
        gamma_ill = float(_p3b["gamma_ill"])
        R_escala = float(_p3b["R_escala"])
        _p3_omk = 200000.0
        _p3_omp = float(_p3b["p3_omp"])
        _p3_omg = float(_p3b["p3_omg"])
        def _state_nonzero_value(value: Any, fallback: float = 0.0001) -> float:
            """Evita ceros exactos en Tabla 5.1 preservando el signo calibrado."""
            try:
                v = float(value)
            except (TypeError, ValueError):
                v = float(fallback)
            if abs(v) < 1e-12:
                return float(fallback)
            return float(v)

        def _pad_state_cost_tuple(
            vals: Any,
            defaults: tuple[float, float, float, float, float, float],
        ) -> tuple[float, float, float, float, float, float]:
            _raw = tuple(float(x) for x in vals)
            _padded = tuple((_raw + defaults)[:6])  # compatibilidad con sesiones antiguas c0,c1,c2
            return tuple(_state_nonzero_value(v, defaults[i]) for i, v in enumerate(_padded))

        _p3_ops = _pad_state_cost_tuple(_p3b["p3_ops"], (2.0, 0.6, 0.9, 0.30, 0.40, 0.20))
        _p3_mt = _pad_state_cost_tuple(_p3b["p3_mt"], (1.5, 0.45, 0.75, 0.25, 0.35, 0.15))
        _p3_cinst = tuple(
            _state_nonzero_value(v, (0.8, 0.5, 0.2)[i])
            for i, v in enumerate(tuple(_p3b["p3_cinst"]))
        )
        _p3_chi_alpha = float(max(0.0, abs(_p3_cinst[0])))
        _p3_chi_gamma = float(max(0.0, abs(_p3_cinst[1])))
        _p3_ops, _p3_mt, _p3_cinst = _calibrate_state_costs_interior(
            _p3_ops, _p3_mt, _p3_cinst, _p3_omg, _p3_omp, R_escala
        )
        _STATE_COST_SCALE = {"DC": 0.78, "PAR": 0.92, "ELN": 1.14, "FARC": 1.34}
        _STATE_TARGETS = {
            "DC": ((0.78, 0.80), (0.54, 0.30)),
            "PAR": ((0.84, 0.86), (0.60, 0.34)),
            "ELN": ((0.90, 0.92), (0.68, 0.40)),
            "FARC": ((0.94, 0.96), (0.74, 0.46)),
        }

        def _state_costs_by_type_from_base(
            ops_base: tuple[float, float, float, float, float, float],
            mt_base: tuple[float, float, float, float, float, float],
        ) -> tuple[
            dict[str, tuple[float, float, float, float, float, float]],
            dict[str, tuple[float, float, float, float, float, float]],
        ]:
            """Coeficientes C_ops y C_maint por tipo, sin costo fiscal G separado."""
            ops_by: dict[str, tuple[float, float, float, float, float, float]] = {}
            mt_by: dict[str, tuple[float, float, float, float, float, float]] = {}
            for th in TIPOS_SECUESTRADOR:
                sc = float(_STATE_COST_SCALE.get(str(th), 1.0))
                (alpha_r, gamma_r), (alpha_n, gamma_n) = _STATE_TARGETS.get(
                    str(th), ((0.88, 0.90), (0.65, 0.36))
                )
                ops_i = [float(x) for x in ops_base]
                mt_i = [float(x) for x in mt_base]
                ops_i[0] = max(abs(ops_i[0]), 230000.0) * sc
                ops_i[2] = max(abs(ops_i[2]), 500000.0) * sc
                ops_i[4] = max(abs(ops_i[4]), 0.40) * sc
                ops_i[5] = min(abs(ops_i[5]) * sc, 0.45 * (ops_i[2] * ops_i[4]) ** 0.5)
                ops_i[1] = -(ops_i[2] * gamma_r + ops_i[5] * alpha_r)
                ops_i[3] = -(ops_i[4] * alpha_r + ops_i[5] * gamma_r)

                mt_i[0] = max(abs(mt_i[0]), 1.5) * sc
                mt_i[2] = max(abs(mt_i[2]), 0.75) * sc
                mt_i[4] = max(abs(mt_i[4]), 0.35) * sc
                mt_i[5] = min(abs(mt_i[5]) * sc, 0.45 * (mt_i[2] * mt_i[4]) ** 0.5)
                mt_i[1] = -(mt_i[2] * gamma_n + mt_i[5] * alpha_n)
                mt_i[3] = float(_p3_omp) * float(R_escala) - (mt_i[4] * alpha_n + mt_i[5] * gamma_n)
                ops_by[str(th)] = tuple(float(x) for x in ops_i)
                mt_by[str(th)] = tuple(float(x) for x in mt_i)
            return ops_by, mt_by

        _p3_ops_by_type, _p3_mt_by_type = _state_costs_by_type_from_base(_p3_ops, _p3_mt)

        def _state_weighted_cost_tuple(
            cost_by_type: dict[str, tuple[float, float, float, float, float, float]],
            mu_v: dict[str, float],
        ) -> tuple[float, float, float, float, float, float]:
            vals = []
            for idx in range(6):
                vals.append(
                    float(
                        sum(
                            float(mu_v.get(th, 0.0)) * float(cost_by_type.get(th, cost_by_type[TIPOS_SECUESTRADOR[0]])[idx])
                            for th in TIPOS_SECUESTRADOR
                        )
                    )
                )
            return tuple(vals)  # type: ignore[return-value]

        def _state_minimize_quadratic_box(
            coeffs: tuple[float, float, float, float, float, float],
            *,
            b_alpha_extra: float = 0.0,
        ) -> tuple[float, float, float]:
            """Minimiza c0+b_g*g+.5*q_g*g²+(b_a+b_extra)*a+.5*q_a*a²+q_ga*g*a en [0,1]²."""
            c0, b_g, q_g, b_a, q_a, q_ga = (float(x) for x in coeffs)
            b_a = float(b_a + b_alpha_extra)
            candidates: set[tuple[float, float]] = {
                (0.0, 0.0),
                (0.0, 1.0),
                (1.0, 0.0),
                (1.0, 1.0),
            }
            det = float(q_g * q_a - q_ga * q_ga)
            if abs(det) > 1e-12:
                g_int = float((-b_g * q_a + q_ga * b_a) / det)
                a_int = float((q_ga * b_g - q_g * b_a) / det)
                if 0.0 <= g_int <= 1.0 and 0.0 <= a_int <= 1.0:
                    candidates.add((round(g_int, 10), round(a_int, 10)))
            for a_fix in (0.0, 1.0):
                if abs(q_g) > 1e-12:
                    g_b = float(-(b_g + q_ga * a_fix) / q_g)
                else:
                    g_b = 0.0 if b_g + q_ga * a_fix >= 0.0 else 1.0
                candidates.add((round(float(min(1.0, max(0.0, g_b))), 10), a_fix))
            for g_fix in (0.0, 1.0):
                if abs(q_a) > 1e-12:
                    a_b = float(-(b_a + q_ga * g_fix) / q_a)
                else:
                    a_b = 0.0 if b_a + q_ga * g_fix >= 0.0 else 1.0
                candidates.add((g_fix, round(float(min(1.0, max(0.0, a_b))), 10)))

            def _val(pair: tuple[float, float]) -> float:
                g_v, a_v = pair
                return float(c0 + b_g * g_v + 0.5 * q_g * g_v * g_v + b_a * a_v + 0.5 * q_a * a_v * a_v + q_ga * g_v * a_v)

            gamma_star, alpha_star = min(candidates, key=_val)
            return float(alpha_star), float(gamma_star), float(_val((gamma_star, alpha_star)))

        def _state_reference_centers(mu_v: dict[str, float]) -> dict[str, float]:
            raw = {th: max(0.0, float(dict(mu_v).get(th, 0.0))) for th in TIPOS_SECUESTRADOR}
            total = float(sum(raw.values()))
            mu_n = (
                {th: float(raw.get(th, 0.0)) / total for th in TIPOS_SECUESTRADOR}
                if total > 1e-12
                else {th: 1.0 / len(TIPOS_SECUESTRADOR) for th in TIPOS_SECUESTRADOR}
            )
            out = {
                "alpha_R_mu": 0.0,
                "gamma_R_mu": 0.0,
                "alpha_N_mu": 0.0,
                "gamma_N_mu": 0.0,
            }
            bench: dict[str, dict[str, float]] = {}
            for th in TIPOS_SECUESTRADOR:
                ar, gr, vr = _state_minimize_quadratic_box(_p3_ops_by_type[str(th)])
                an, gn, vn = _state_minimize_quadratic_box(
                    _p3_mt_by_type[str(th)],
                    b_alpha_extra=-float(_p3_omp) * float(R_escala),
                )
                w = float(mu_n.get(th, 0.0))
                out["alpha_R_mu"] += w * float(ar)
                out["gamma_R_mu"] += w * float(gr)
                out["alpha_N_mu"] += w * float(an)
                out["gamma_N_mu"] += w * float(gn)
                bench[str(th)] = {
                    "mu": w,
                    "alpha_R_bench": float(ar),
                    "gamma_R_bench": float(gr),
                    "V_R_bench": float(vr),
                    "alpha_N_bench": float(an),
                    "gamma_N_bench": float(gn),
                    "V_N_bench": float(vn),
                }
            out["bench"] = bench  # type: ignore[assignment]
            return out

        def _state_cost_stationary_point(
            coeffs: tuple[float, float, float, float, float, float],
        ) -> tuple[float, float, float, bool, bool]:
            _, c1, c2, c3, c4, c5 = (float(x) for x in coeffs)
            det = float(c2 * c4 - c5 * c5)
            if det <= 1e-12:
                return float("nan"), float("nan"), det, False, False
            gamma_min = float((-c1 * c4 + c5 * c3) / det)
            alpha_min = float((c5 * c1 - c2 * c3) / det)
            interior = bool(0.0 < alpha_min < 1.0 and 0.0 < gamma_min < 1.0)
            pd = bool(c2 > 0.0 and c4 > 0.0 and det > 0.0)
            return alpha_min, gamma_min, det, interior, pd

        def _enforce_k_continue_dominated_at_state_targets(
            df_params: pd.DataFrame,
        ) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
            """Aumenta ν si hace falta para que continuar no domine liberar/matar en el objetivo por tipo."""
            out = df_params.copy().reset_index(drop=True)
            audit: dict[str, dict[str, Any]] = {}
            for th in TIPOS_SECUESTRADOR:
                (_, _), (alpha_n, gamma_n) = _STATE_TARGETS.get(str(th), ((0.9, 0.9), (0.65, 0.35)))
                mask = out["theta_K"].astype(str) == str(th)
                if not bool(mask.any()):
                    continue
                nu_added = 0.0
                util_row = pd.Series(dtype=object)
                margin = float("-inf")
                for _ in range(14):
                    df_ref = refresh_kidnapper_endogenous_columns(
                        out.copy(), modelo, float(gamma_n), float(gamma_n), alpha=float(alpha_n)
                    )
                    util = kidnapper_util_df_from_param_df(
                        df_ref, modelo, float(gamma_n), float(alpha_n), float(gamma_n),
                        float(R_escala), str(tipo_real), float(_p3_beta_k)
                    )
                    row = util[util["theta_K"].astype(str) == str(th)]
                    if row.empty:
                        break
                    util_row = row.iloc[0]
                    outside = max(float(util_row["U_rel"]), float(util_row["U_kill"]))
                    margin = float(outside - float(util_row["V_cont"]))
                    if margin >= 0.0:
                        break
                    bump = float(abs(margin) + 10.0)
                    out.loc[mask, "nu"] = out.loc[mask, "nu"].astype(float) + bump
                    nu_added += bump
                audit[str(th)] = {
                    "alpha_target": float(alpha_n),
                    "gamma_target": float(gamma_n),
                    "U_rel": float(util_row.get("U_rel", np.nan)) if not util_row.empty else float("nan"),
                    "U_kill": float(util_row.get("U_kill", np.nan)) if not util_row.empty else float("nan"),
                    "V_cont": float(util_row.get("V_cont", np.nan)) if not util_row.empty else float("nan"),
                    "outside_best": max(
                        float(util_row.get("U_rel", np.nan)) if not util_row.empty else float("nan"),
                        float(util_row.get("U_kill", np.nan)) if not util_row.empty else float("nan"),
                    ),
                    "margin": float(margin),
                    "nu_added": float(nu_added),
                    "dominado": bool(margin >= 0.0),
                }
            return out, audit
        _p3_beta_k = float(_p3b["p3_beta_k"])
        _p3_vl = float(_p3b["p3_vl"])
        _p3_fcol = float(_p3b["p3_fcol"])
        _p3_pd0 = float(_p3b["p3_pd0"])
        _p3_pda = float(_p3b["p3_pda"])
        _vr_p3 = float(_p3b["vr_p3"])
        _vn_p3 = float(_p3b["vn_p3"])
        _psurv_p3 = float(_p3b["psurv_p3"])
        _pkill_p3 = float(_p3b["pkill_p3"])
        _cops_p3 = float(_p3b["cops_p3"])
        _cmaint_p3 = float(_p3b["cmaint_p3"])
        _g_p3 = float(_p3b["g_p3"])
        _u_coop_p3 = float(_p3b["u_coop_p3"])
        _u_col_p3 = float(_p3b["u_col_p3"])
        _t0_alpha_eff = float(_p3b["t0_alpha_eff"])
        _t0_gamma_eff = float(_p3b["t0_gamma_eff"])
        _iota_t0 = float(_p3b["iota_t0"])
        mu_tab = dict(_p3b["mu_tab"])
        tipo_real = str(_p3b["tipo_real"])
        s_tipo = str(_p3b["s_tipo"])
        f_capa = str(_p3b["f_capa"])
        _df_p3_util_k = _p3b["df_p3_util_k"].copy()
        _df_p3_k_params = _p3b["df_p3_k_params"].copy()
        _df_p3_k_params, _state_k_deterrence_audit = _enforce_k_continue_dominated_at_state_targets(_df_p3_k_params)
        _df_p3_k_params = refresh_kidnapper_endogenous_columns(
            _df_p3_k_params.copy(), modelo, float(_t0_gamma_eff), float(_t0_gamma_eff), alpha=float(_t0_alpha_eff)
        )
        _df_p3_util_k = kidnapper_util_df_from_param_df(
            _df_p3_k_params, modelo, float(_t0_gamma_eff), float(_t0_alpha_eff),
            float(_t0_gamma_eff), float(R_escala), str(tipo_real), float(_p3_beta_k)
        )
        _p3_phi_f, _p3_kappa_f, _p3_nu_f = _rb_family_phi_kappa_nu(f_capa)

        # =========================================================
        # JUGADOR S — Minimización
        # =========================================================
        st.markdown("### 3 · Estado ($S$) — Minimización, α, γ ∈ [0, 1]")
        _ps1, _ps2 = st.columns([2, 3], gap="large")
        with _ps1:
            st.markdown("**Bloque 1 · Información del Estado** · ec. `hat-theta-definition`")
            st.latex(
                r"\iota_t:=\max_{\theta\in\Theta_K}\mu_t(\theta),"
                r"\qquad"
                r"\hat{\theta}_t=\arg\max_{\theta\in\Theta_K}\mu_t(\theta)"
            )
            st.markdown("**Bloque 2 · Pérdida condicional por rama** · ec. `state-loss-conditional-branches`")
            st.latex(
                r"L_t(a_t^S,\alpha_t,\gamma_t,\theta_K,\iota_t)="
                r"\begin{cases}"
                r"V_t^R(\iota_t,\hat{\theta}_t,\theta_K,\alpha_t,\gamma_t),"
                r"& a_t^S=a_{\mathrm{res}},\\[3pt]"
                r"V_t^N(\theta_K,\alpha_t,\gamma_t),"
                r"& a_t^S=a_{\mathrm{neg}}."
                r"\end{cases}"
            )
            st.markdown("**Bloque 3A · Rama rescate: definición de \(V_t^R\)** · ec. `state-loss-rescue-branch`")
            st.latex(
                r"\begin{aligned}"
                r"V_t^R(\iota_t,\hat{\theta}_t,\theta_K,\alpha_t,\gamma_t)"
                r"&:=\omega_k\Bigl[1-\mathbb{P}_{\mathbb E}"
                r"(s_t=1\mid\iota_t,\hat{\theta}_t,\theta_K)\Bigr]\\"
                r"&\quad +C_{\mathrm{ops}}(\gamma_t,\alpha_t;\theta_K)."
                r"\end{aligned}"
            )
            st.latex(
                r"\begin{aligned}"
                r"C_{\mathrm{ops}}(\gamma_t,\alpha_t;\theta_K)"
                r"&=c_0(\theta_K)+c_1(\theta_K)\gamma_t"
                r"+\frac{c_2(\theta_K)}{2}\gamma_t^2\\"
                r"&\quad+c_3(\theta_K)\alpha_t"
                r"+\frac{c_4(\theta_K)}{2}\alpha_t^2"
                r"+c_5(\theta_K)\gamma_t\alpha_t."
                r"\end{aligned}"
            )
            st.markdown("**Bloque 3B · Rama negociación: definición de \(V_t^N\)** · ec. `negotiation-cost`")
            st.latex(
                r"\begin{aligned}"
                r"V_t^N(\theta_K,\alpha_t,\gamma_t)"
                r"&:=\omega_pR(1-\alpha_t)"
                r"+\omega_k h_2(t\mid\theta_K,\mathcal C_t)\\"
                r"&\quad+C_{\mathrm{maint}}(\gamma_t,\alpha_t;\theta_K)."
                r"\end{aligned}"
            )
            st.latex(
                r"\begin{aligned}"
                r"V_t^N(\mu_t,\alpha_t,\gamma_t)"
                r"&=\sum_{\theta\in\Theta_K}\mu_t(\theta)"
                r"\Bigl[\omega_pR(1-\alpha_t)+\omega_k h_2(t\mid\theta,\mathcal C_t)\\"
                r"&\qquad\qquad+C_{\mathrm{maint}}(\gamma_t,\alpha_t;\theta)\Bigr]."
                r"\end{aligned}"
            )
            st.latex(
                r"\begin{aligned}"
                r"C_{\mathrm{maint}}(\gamma_t,\alpha_t;\theta_K)"
                r"&=m_0(\theta_K)+m_1(\theta_K)\gamma_t"
                r"+\frac{m_2(\theta_K)}{2}\gamma_t^2\\"
                r"&\quad+m_3(\theta_K)\alpha_t"
                r"+\frac{m_4(\theta_K)}{2}\alpha_t^2"
                r"+m_5(\theta_K)\gamma_t\alpha_t."
                r"\end{aligned}"
            )
            st.markdown("**Bloque 4 · Centros bayesianos y penalización \(\Pi_{t,b}^S\)** · ec. `belief-weighted-policy`")
            st.latex(
                r"\begin{aligned}"
                r"\alpha_{t,b}^{\mu}"
                r"&:=\sum_{\theta\in\Theta_K}\mu_t(\theta)\alpha_{t,b}^{\mathrm{bench}}(\theta),\\"
                r"\gamma_{t,b}^{\mu}"
                r"&:=\sum_{\theta\in\Theta_K}\mu_t(\theta)\gamma_{t,b}^{\mathrm{bench}}(\theta),"
                r"\qquad b\in\{R,N\}."
                r"\end{aligned}"
            )
            st.latex(
                r"\begin{aligned}"
                r"\Pi_{t,b}^S(\alpha_t,\gamma_t;\mu_t)"
                r"&=\chi_\alpha(\alpha_t-\alpha_{t,b}^{\mu})^2"
                r"+\chi_\gamma(\gamma_t-\gamma_{t,b}^{\mu})^2,\\"
                r"&\qquad b\in\{R,N\},\quad \chi_\alpha,\chi_\gamma\ge0."
                r"\end{aligned}"
            )
            st.markdown("**Bloque 5 · Entropía y ganancia informacional** · ecs. `belief-entropy`, `expected-information-gain`")
            st.latex(
                r"H(\mu_t):=-\sum_{\theta\in\Theta_K}\mu_t(\theta)\log\mu_t(\theta),"
                r"\qquad 0\log 0:=0."
            )
            st.latex(
                r"\begin{aligned}"
                r"\mu_{t+1}^{m}(\theta)"
                r"&=\frac{\mu_t(\theta)\mathbb{P}_{\mathbb E}"
                r"(m_t=m\mid\theta,\alpha_t,\gamma_t,\mathcal C_t)}"
                r"{\sum_{\theta'\in\Theta_K}\mu_t(\theta')\mathbb{P}_{\mathbb E}"
                r"(m_t=m\mid\theta',\alpha_t,\gamma_t,\mathcal C_t)},\\[0.35em]"
                r"\Delta H_t(\alpha_t,\gamma_t)"
                r"&=H(\mu_t)-\sum_m\mathbb{P}_{\mathbb E}"
                r"(m_t=m\mid\mu_t,\alpha_t,\gamma_t,\mathcal C_t)"
                r"H(\mu_{t+1}^{m})."
                r"\end{aligned}"
            )
            st.markdown("**Bloque 6 · Pisos continuos con motivo informacional** · ecs. `rescue-cost`, `negotiation-piso`, `state-dual-control-objective`")
            st.latex(
                r"\begin{aligned}"
                r"\widetilde V_t^{R}(\iota_t,\hat{\theta}_t,\mu_t)"
                r"&=\min_{(\alpha_t,\gamma_t)\in\mathcal{A}_{\mathrm{grid}}}"
                r"\Biggl\{\sum_{\theta_K\in\Theta_K}\mu_t(\theta_K)"
                r"V_t^R(\iota_t,\hat{\theta}_t,\theta_K,\alpha_t,\gamma_t)\\"
                r"&\qquad\qquad+\Pi_{t,R}^S(\alpha_t,\gamma_t;\mu_t)"
                r"-\psi_H\Delta H_t(\alpha_t,\gamma_t)\Biggr\},\\[0.5em]"
                r"\widetilde V_t^{N}(\mu_t)"
                r"&=\min_{(\alpha_t,\gamma_t)\in\mathcal{A}_{\mathrm{grid}}}"
                r"\Bigl\{V_t^N(\mu_t,\alpha_t,\gamma_t)"
                r"+\Pi_{t,N}^S(\alpha_t,\gamma_t;\mu_t)"
                r"-\psi_H\Delta H_t(\alpha_t,\gamma_t)\Bigr\}."
                r"\end{aligned}"
            )
            st.markdown("**Bloque 7 · Programa global y regla discreta** · ecs. `state-expected-loss`, `state-discrete-rule`")
            st.latex(
                r"\widetilde{\mathcal L}^{S\ast}_t"
                r"=\min\left\{"
                r"\widetilde V_t^{R}(\iota_t,\hat{\theta}_t,\mu_t),"
                r"\widetilde V_t^{N}(\mu_t)"
                r"\right\}"
            )
            st.latex(
                r"\begin{aligned}"
                r"(a_t^{S\ast},\alpha_t^\ast,\gamma_t^\ast)"
                r"&=\begin{cases}"
                r"(a_{\mathrm{res}},\alpha_{\mathrm{res}}^\ast,\gamma_{\mathrm{res}}^\ast),"
                r"&\text{si }\widetilde V_t^{R}(\iota_t,\hat{\theta}_t,\mu_t)\le \widetilde V_t^{N}(\mu_t),\\[4pt]"
                r"(a_{\mathrm{neg}},\alpha_{\mathrm{neg}}^\ast,\gamma_{\mathrm{neg}}^\ast),"
                r"&\text{si }\widetilde V_t^{R}(\iota_t,\hat{\theta}_t,\mu_t)> \widetilde V_t^{N}(\mu_t)."
                r"\end{cases}"
                r"\end{aligned}"
            )
            st.markdown("**Bloque 8 · Programa restringido implementable** · def. `implementable-mechanism`, ec. `gamma-factible`")
            st.latex(
                r"\begin{aligned}"
                r"\chi_t^\ast(h_t)\in"
                r"\arg\min_{(a_t^S,\alpha_t,\gamma_t)\in\Gamma_t(\mu_t)}"
                r"\Biggl\{&\sum_{\theta_K\in\Theta_K}\mu_t(\theta_K)"
                r"L_t(a_t^S,\alpha_t,\gamma_t,\theta_K,\iota_t)\\"
                r"&+\Pi_{t,a^S}^S(\alpha_t,\gamma_t;\mu_t)"
                r"-\psi_H\Delta H_t(\alpha_t,\gamma_t)\Biggr\}."
                r"\end{aligned}"
            )
            st.markdown("---")
            st.markdown(
                "**Restricciones del programa** · def. `implementable-mechanism` · "
                "ec. `gamma-factible`"
            )
            st.latex(
                r"\Gamma_t(\mu_t):=\Bigl\{"
                r"(a_t^S,\alpha_t,\gamma_t)\in"
                r"\mathcal{A}^{S,\mathrm{disc}}\times[0,1]^2"
                r":IC^K,\,IR^K,\,IR^F"
                r"\text{ se satisfacen}\Bigr\}"
            )
            st.markdown("**IC^K** — no-mimetismo · def. `implementable-mechanism`")
            st.latex(
                r"\mathbb{E}_{\theta\sim\mu_t}"
                r"\left["
                r"V_t^K(a^\ast(\theta)\mid\theta,\alpha_t^\ast,\gamma_t^\ast)"
                r"-V_t^K(a^\ast(\theta_j)\mid\theta,\alpha_t^\ast,\gamma_t^\ast)"
                r"\right]\ge0,\quad\forall\,\theta_j\in\Theta_K"
            )
            st.markdown("**IR^K** — disuasión / salida pacífica · ec. `ir-K`")
            st.latex(
                r"\mathbb{E}_{\theta\sim\mu_t}"
                r"\left[U^K_{\mathrm{rel}}(\theta)-"
                r"\max\left\{"
                r"V^K_{\mathrm{cont},t}(\theta,\alpha_t^\ast,\gamma_t^\ast),"
                r"U^K_{\mathrm{kill}}(\theta,\theta_S)"
                r"\right\}"
                r"\right]\ge0"
            )
            st.markdown("**IR^F** — cooperación preferida · ec. `ir-family`")
            st.latex(
                r"\mathcal{U}_t^F(a_{\mathrm{coop}})\;\geq\;\mathcal{U}_t^F(a_{\mathrm{col}})"
            )
            st.markdown("**Actualización de información por ciclo**")
            st.markdown(
                "El **ciclo base** se registra en la columna $\\tau=0$ de la Tabla 5.2: "
                "$V_0$, $d_0$, $a^*$, $\\tilde a$ y $m_0$. Con esa información se obtiene "
                "$\\mu_1$ para alimentar la columna $\\tau=1$."
            )
            st.latex(r"\text{Cierre ciclo base en }\tau=0\Rightarrow (\mu_1,\iota_1,\alpha_0^\ast,\gamma_0^\ast,M_0)")
            st.latex(
                r"(\alpha_1^\ast,\gamma_1^\ast,\mu_1,\Delta H_1)\Rightarrow "
                r"P_{\mathrm{det},1},P_{\mathrm{cap},1},P_{\mathrm{surv},1},"
                r"P_{\mathrm{kill},1},P_I,\lambda_1"
            )
            st.latex(
                r"V_1\sim Bernoulli\!\left(\sum_{\theta}\mu_1(\theta)\,"
                r"P(V_1=1\mid\theta,\alpha_1^\ast,\gamma_1^\ast)\right),"
                r"\qquad d_1\sim Bernoulli(P_{\mathrm{det},1})"
            )
            st.latex(
                r"a_F^\ast=\arg\max\{U_1^F(a_{\mathrm{coop}}),"
                r"U_1^F(a_{\mathrm{col}})\},\qquad "
                r"a_K^\ast\leftarrow \text{Tabla 15, }\tau=0,\text{ columna 14}"
            )
            st.latex(
                r"a_S^\ast=\begin{cases}"
                r"\mathrm{Rescate},& \widetilde V_R(\mu_1,\alpha_1^\ast,\gamma_1^\ast)"
                r"\le \widetilde V_N(\mu_1,\alpha_1^\ast,\gamma_1^\ast),\\"
                r"\mathrm{Negociar},& \widetilde V_R>\widetilde V_N,"
                r"\end{cases}"
            )
            st.latex(
                r"\tilde a_F\sim P_I^F(\tilde a_F\mid a_F^\ast),\quad "
                r"\tilde a_K\sim P_I^K(\tilde a_K\mid a_K^\ast),\quad "
                r"\tilde a_S\sim P_I^S(\tilde a_S\mid a_S^\ast)"
            )
            st.latex(
                r"m_1\sim P^E\!\left(m_1=j\mid "
                r"\tilde a_F,\tilde a_K,\tilde a_S,\alpha_1^\ast,\gamma_1^\ast,\iota_1,\theta_K\right)"
            )
            st.latex(
                r"\mu_{t+1}(\theta)=\frac{\mu_t(\theta)\,"
                r"\mathcal{L}_F(\theta;d_t,m_{t+1},\tilde a_t,\alpha_t,\gamma_t)\,"
                r"\mathcal{L}_C(\theta;V_t,x_t)}"
                r"{\sum_{\theta'}\mu_t(\theta')\,"
                r"\mathcal{L}_F(\theta';d_t,m_{t+1},\tilde a_t,\alpha_t,\gamma_t)\,"
                r"\mathcal{L}_C(\theta';V_t,x_t)}"
            )
            st.caption(
                "En Tabla 5.2, τ=0 cierra el escenario base y produce μ1. Desde τ=1, "
                "la trayectoria dinámica repite la misma lógica reemplazando 1 por t."
            )

        with _ps2:
            st.markdown(
                "**Tabla 5.1 · Parámetros calibrados de las funciones del Estado por tipo** "
                "· `state-expected-loss`, `rescue-cost`, `negotiation-cost`"
            )
            _global_state_rows = [
                {
                    "Tipo": "Todos",
                    "Función": r"\mathcal{L}_t^S",
                    "Parámetro": "Peso costo humano",
                    "Símbolo": r"\omega_k",
                    "Valor calibrado": round(float(_p3_omk), 4),
                },
                {
                    "Tipo": "Todos",
                    "Función": r"\mathcal{L}_t^S",
                    "Parámetro": "Peso transferencias",
                    "Símbolo": r"\omega_p",
                    "Valor calibrado": round(float(_p3_omp), 4),
                },
                {
                    "Tipo": "Todos",
                    "Función": r"V_t^N",
                    "Parámetro": "Rescate / transferencia base",
                    "Símbolo": "R",
                    "Valor calibrado": round(float(R_escala), 4),
                },
                {
                    "Tipo": "Todos",
                    "Función": r"\Pi_{t,b}^S",
                    "Parámetro": "Peso desviación en bloqueo",
                    "Símbolo": r"\chi_\alpha",
                    "Valor calibrado": round(float(_p3_chi_alpha), 4),
                },
                {
                    "Tipo": "Todos",
                    "Función": r"\Pi_{t,b}^S",
                    "Parámetro": "Peso desviación en presión",
                    "Símbolo": r"\chi_\gamma",
                    "Valor calibrado": round(float(_p3_chi_gamma), 4),
                },
                {
                    "Tipo": "Todos",
                    "Función": r"\widetilde V_t^b",
                    "Parámetro": "Peso ganancia informacional",
                    "Símbolo": r"\psi_H",
                    "Valor calibrado": round(float(st.session_state.get("t52_entropy_info_weight", 25.0)), 4),
                },
                {
                    "Tipo": "Todos",
                    "Función": "Política actual",
                    "Parámetro": "Bloqueo financiero inicial",
                    "Símbolo": r"\alpha_0",
                    "Valor calibrado": round(float(_t0_alpha_eff), 4),
                },
                {
                    "Tipo": "Todos",
                    "Función": "Política actual",
                    "Parámetro": "Presión operativa inicial",
                    "Símbolo": r"\gamma_0",
                    "Valor calibrado": round(float(_t0_gamma_eff), 4),
                },
            ]
            _state_cost_rows = []
            _state_param_names = {
                "c0": "Intercepto operativo",
                "c1": "Pendiente operativa",
                "c2": "Curvatura operativa",
                "c3": "Pendiente bloqueo operativo",
                "c4": "Curvatura bloqueo operativo",
                "c5": "Interacción rescate",
                "m0": "Intercepto mantenimiento",
                "m1": "Pendiente mantenimiento",
                "m2": "Curvatura mantenimiento",
                "m3": "Pendiente bloqueo mantenimiento",
                "m4": "Curvatura bloqueo mantenimiento",
                "m5": "Interacción mantenimiento",
            }
            for _th_state_cost in TIPOS_SECUESTRADOR:
                _ops_th = _p3_ops_by_type[str(_th_state_cost)]
                _mt_th = _p3_mt_by_type[str(_th_state_cost)]
                _state_cost_rows.extend(
                    [
                        {
                            "Tipo": str(_th_state_cost),
                            "Función": r"C_{\mathrm{ops}}(\gamma,\alpha;\theta)",
                            "Parámetro": _state_param_names[f"c{idx}"],
                            "Símbolo": f"c{idx}",
                            "Valor calibrado": round(float(val), 6),
                        }
                        for idx, val in enumerate(_ops_th)
                    ]
                )
                _state_cost_rows.extend(
                    [
                        {
                            "Tipo": str(_th_state_cost),
                            "Función": r"C_{\mathrm{maint}}(\gamma,\alpha;\theta)",
                            "Parámetro": _state_param_names[f"m{idx}"],
                            "Símbolo": f"m{idx}",
                            "Valor calibrado": round(float(val), 6),
                        }
                        for idx, val in enumerate(_mt_th)
                    ]
                )
            _df_s_params_global = pd.DataFrame(_global_state_rows)
            _df_s_params_global["Valor calibrado"] = _df_s_params_global["Valor calibrado"].apply(
                lambda v: round(_state_nonzero_value(v), 6)
            )
            _df_s_params_global.insert(0, "#", range(1, len(_df_s_params_global) + 1))
            _df_state_cost_all = pd.DataFrame(_state_cost_rows)
            _df_state_cost_all["Valor calibrado"] = _df_state_cost_all["Valor calibrado"].apply(
                lambda v: round(_state_nonzero_value(v), 6)
            )

            def _fmt_state_51_value(value: Any) -> str:
                try:
                    v = float(value)
                except (TypeError, ValueError):
                    return str(value)
                if not np.isfinite(v):
                    return "—"
                if abs(v) > 1.0:
                    return _fmt_es_num(v, 0)
                return _fmt_es_num(v, 6).rstrip("0").rstrip(
                    "." if st.session_state.get("app_language", "English") == "English" else ","
                )

            def _state_param_math_cell(val: Any) -> str:
                s = str(val)
                if s == "Política actual":
                    return html.escape(s, quote=False)
                return f'<span class="math">{html.escape(s, quote=False)}</span>'

            _df_s_params_render = _df_s_params_global.copy()
            _df_s_params_render["Valor calibrado"] = _df_s_params_render["Valor calibrado"].map(_fmt_state_51_value)
            _df_s_params_render["Función"] = _df_s_params_render["Función"].map(_state_param_math_cell)
            _df_s_params_render["Símbolo"] = _df_s_params_render["Símbolo"].map(_state_param_math_cell)
            render_generic_katex_table(
                _df_s_params_render,
                [
                    r"\#",
                    r"\text{Función}",
                    r"\text{Parámetro}",
                    r"\text{Símbolo}",
                    r"\text{Valor calibrado}",
                ],
                height=_glide_full_height_px(_st_table_row_count(_df_s_params_render)) + 70,
                compact=True,
                relaxed_compact=True,
                header_nowrap=True,
            )
            _tab2_bundles_kh = _tab2_bundles_all_types(
                z_region=str(st.session_state.get("z_region", "Andina")),
                v_victim=str(st.session_state.get("v_victim", "Privado")),
                f_capa=str(f_capa),
                s_tipo=str(s_tipo),
            )
            _state_cost_tabs = st.tabs([str(th) for th in TIPOS_SECUESTRADOR])
            for _tab_state_cost, _th_state_cost in zip(_state_cost_tabs, TIPOS_SECUESTRADOR):
                with _tab_state_cost:
                    _df_state_cost_type = _df_state_cost_all[
                        _df_state_cost_all["Tipo"].astype(str) == str(_th_state_cost)
                    ].drop(columns=["Tipo"]).reset_index(drop=True)
                    # κh(θK, t): sensibilidad neta de presión sobre duración esperada
                    # Mechanism.tex eq. kappa-h: κh = ζγ2·λ̃2 + ζγ3·λ̃3 − ζγ1·λ̃1
                    try:
                        _bnd_kh = _tab2_bundles_kh.get(str(_th_state_cost), {})
                        _zbj_kh = _bnd_kh.get("zeta_by_j", {}) or {}
                        _h_kh = modelo.calcular_hazards(
                            1,
                            str(_th_state_cost),
                            float(st.session_state.get("cal_presion_S", 0.0)),
                            maturity_mult=1.0,
                            z_region=str(st.session_state.get("z_region", "Andina")),
                            v_victim=str(st.session_state.get("v_victim", "Privado")),
                            alpha=float(_t0_alpha_eff),
                            gamma=float(_t0_gamma_eff),
                            zeta_by_j=_zbj_kh,
                        )
                        _lam1_kh = float(_h_kh.get("Pago", 0.0))
                        _lam2_kh = float(_h_kh.get("Muerte", 0.0))
                        _lam3_kh = float(_h_kh.get("Rescate", 0.0))
                        _zg1_kh = float((_zbj_kh.get("Pago") or {}).get("gamma", 0.0))
                        _zg2_kh = float((_zbj_kh.get("Muerte") or {}).get("gamma", 0.0))
                        _zg3_kh = float((_zbj_kh.get("Rescate") or {}).get("gamma", 0.0))
                        _kh_val = _zg2_kh * _lam2_kh + _zg3_kh * _lam3_kh - _zg1_kh * _lam1_kh
                        _neg_sgn_kh = float(-1 if _kh_val > 1e-12 else (1 if _kh_val < -1e-12 else 0))
                    except Exception:
                        _kh_val = 0.0
                        _neg_sgn_kh = 0.0
                    _df_state_cost_type = pd.concat([
                        _df_state_cost_type,
                        pd.DataFrame([{
                            "Función":         r"\kappa_h(\theta_K,t)",
                            "Parámetro":       "Predicción duración vs γ* (↓ si −1)",
                            "Símbolo":         r"-\operatorname{sgn}(\kappa_h)",
                            "Valor calibrado": _neg_sgn_kh,
                        }]),
                    ], ignore_index=True)
                    _df_state_cost_type.insert(0, "#", range(1, len(_df_state_cost_type) + 1))
                    _df_state_cost_type_render = _df_state_cost_type.copy()
                    _df_state_cost_type_render["Valor calibrado"] = _df_state_cost_type_render["Valor calibrado"].map(_fmt_state_51_value)
                    _df_state_cost_type_render["Función"] = _df_state_cost_type_render["Función"].map(_state_param_math_cell)
                    _df_state_cost_type_render["Símbolo"] = _df_state_cost_type_render["Símbolo"].map(_state_param_math_cell)
                    render_generic_katex_table(
                        _df_state_cost_type_render,
                        [
                            r"\#",
                            r"\text{Función}",
                            r"\text{Parámetro}",
                            r"\text{Símbolo}",
                            r"\text{Valor calibrado}",
                        ],
                        height=_glide_full_height_px(_st_table_row_count(_df_state_cost_type_render)) + 70,
                        compact=True,
                        relaxed_compact=True,
                        header_nowrap=True,
                    )
            # ── Tabla 5.2 — τ por columna, variables por fila ────────────
            st.markdown(
                "**Tabla 5.2 · Equilibrio por τ: acciones óptimas, implementación MDG y desenlace** "
                "· pestaña 4 (a*) → pestaña 3 (ã, m)"
            )
            _val15_52   = st.session_state.get("tab15_last_validation") or {}
            _psi_52     = st.session_state.get("cal_psi_params") or {}
            _ir_f_52    = float(_u_coop_p3) >= float(_u_col_p3)
            _af52_star  = "Cooperar" if _ir_f_52 else "Coludir"
            _af52_full  = "Cooperar (a_coop)" if _ir_f_52 else "Colusión (a_col)"
            _mu0_52 = {
                _th: float(st.session_state.final_priors[_ii]) / 100.0
                for _ii, _th in enumerate(TIPOS_SECUESTRADOR)
            }
            _iota_52 = float(max(_mu0_52.values())) if _mu0_52 else 0.7
            _vr_lt_vn = float(_vr_p3) <= float(_vn_p3)
            _s_rule_p3 = "Rescate" if _vr_lt_vn else "Negociar"
            _s52_star = "Rescatar" if _vr_lt_vn else "No Rescatar"
            _s52_full = "Rescate (a_res)" if _vr_lt_vn else "Negociar (a_neg)"
            _agent52 = _build_t0_family_state_mdg_probs(
                modelo,
                _mu0_52,
                float(_t0_gamma_eff),
                float(_iota_52),
                float(_t0_alpha_eff),
                float(_t0_gamma_eff),
                float(R_escala),
                f_capa,
            )

            def _t52_clean_ak(s: str) -> str:
                base = str(s).split("(")[0].strip()
                return base  # "Continuar", "Liberar", "Matar"

            def _t52_impl_law_probs(
                actions: list[str],
                intent: str,
                player: str,
                mu0: Optional[dict[str, float]] = None,
                mu_tau: Optional[dict[str, float]] = None,
                tau: int = 1,
            ) -> dict[str, float]:
                """Ley de Implementación de Tabla 7 para el agente indicado.
                T_t = T0 * max((H(mu_t)/H(mu_0)) * exp(-eta_cal * t), c_bar)  (eq. temperatura-piso)
                """
                intent_i = str(intent) if str(intent) in actions else actions[0]

                def _entropy_mu(mu_in: Optional[dict[str, float]]) -> float:
                    vals = [max(0.0, float((mu_in or {}).get(th, 0.0))) for th in TIPOS_SECUESTRADOR]
                    total = float(sum(vals))
                    if total <= 1e-12:
                        vals = [1.0 / len(TIPOS_SECUESTRADOR)] * len(TIPOS_SECUESTRADOR)
                    else:
                        vals = [v / total for v in vals]
                    return float(-sum(v * np.log(v) for v in vals if v > 1e-12))

                h0_i = _entropy_mu(mu0)
                ht_i = _entropy_mu(mu_tau if mu_tau is not None else mu0)
                t_tau = float(max(0, int(tau)))
                t0_i = float(st.session_state.get(f"mdg_T0_{player}", 1.0))
                cbar_i = float(st.session_state.get(f"mdg_cbar_{player}", 0.05))
                eta_i = float(st.session_state.get("cal_mdg_eta_cal_by_i", {}).get(player, 0.0))
                temp_i = float(max(
                    hybrid_temperature(ht_i, t0_i, H0=h0_i, eta_cal=eta_i, t=int(t_tau), c_bar=cbar_i),
                    1e-12,
                ))
                return _mdg_implementation_logit_probs(actions, intent_i, temp_i)

            def _t52_p1(
                intent: str,
                player: str,
                iota: float,
                mu0: Optional[dict[str, float]] = None,
                mu_tau: Optional[dict[str, float]] = None,
                tau: int = 1,
            ) -> dict:
                """P_I(ã | a*) con Ley de Implementación de Tabla 7."""
                _ = iota
                if player == "K":
                    acts = ["Continuar", "Liberar", "Matar"]
                else:  # F
                    acts = ["Cooperar", "Coludir"]
                return _t52_impl_law_probs(acts, intent, player, mu0, mu_tau, tau=tau)

            def _t52_p1_s(
                agent_probs: dict[str, float],
                intent: Optional[str] = None,
                mu0: Optional[dict[str, float]] = None,
                mu_tau: Optional[dict[str, float]] = None,
                tau: int = 1,
            ) -> dict[str, float]:
                """P_I^S desde Ley de Implementación y coeficientes de Tabla 7 (pestaña 3)."""
                _ = agent_probs
                return _t52_impl_law_probs(["Rescatar", "No Rescatar"], str(intent), "S", mu0, mu_tau, tau=tau)

            def _t52_p2(ak: str, af: str, psi_p: dict,
                         presion: float, iota: float) -> dict:
                """P_E(m | ã_K, ã_F, ã_S=NoRescatar) — Phase 2 Tab 3 (sin condicionar en tipo)."""
                v_th = THETA_K_MAP.get(str(tipo_real), [0.0] * 4)
                psis = []
                for _j52 in [1, 2, 3, 4, 5]:
                    _p52 = psi_p.get(_j52, {})
                    psi = float(_p52.get("delta", 0.0))
                    if _j52 == 1 and ak == "Liberar":
                        psi += float(_p52.get("gamma_K", 0.0))
                    elif _j52 == 3 and af == "Coludir":
                        psi += float(_p52.get("gamma_F", 0.0))
                    elif _j52 == 4 and ak == "Matar":
                        psi += float(_p52.get("gamma_K", 0.0))
                    psi += float(_p52.get("phi_gamma", 0.0)) * presion
                    for _idx52 in range(4):
                        psi += float((_p52.get("phi_theta") or [0]*4)[_idx52]) * v_th[_idx52]
                    psi += float(_p52.get("kappa", 0.0)) * iota
                    psis.append(np.exp(psi))
                _s52 = sum(psis) or 1.0
                _labels52 = ["Liberación", "Rescate", "Pago", "Muerte", "Continuar"]
                return {lbl: float(e / _s52) for lbl, e in zip(_labels52, psis)}

            def _t52_realize(probs: dict, u: float) -> str:
                """Acción realizada: intervalo [lo,hi) donde cae u (transformada inversa)."""
                acts = list(probs.keys())
                curr = 0.0
                for _i, (_act, _p) in enumerate(probs.items()):
                    lo, hi = curr, curr + _p
                    is_last = (_i == len(acts) - 1)
                    if (lo <= u < hi) or (is_last and u >= lo):
                        return _act
                    curr += _p
                return acts[-1]

            def _t52_tip_html(
                player_label: str,
                intent: str,
                probs: dict,
                atilde_realized: str,
                atilde_argmax: str,
                iota: float,
                eu: float,
                u_draw: float,
                u_extras: str = "",
            ) -> str:
                """Tooltip HTML con KaTeX: P_I, intervalos, sorteo u, E[U]."""
                import html as _h
                if player_label.startswith("F"):
                    p_int = float(probs.get(intent, 0.0))
                    formula_kx = r"\mathbb{P}_I^F\ \text{por Ley de Implementación, Tabla 7}"
                    sub = "F"
                elif player_label.startswith("S"):
                    p_int = float(probs.get(intent, 0.0))
                    formula_kx = r"\mathbb{P}_I^S\ \text{por Ley de Implementación, Tabla 7}"
                    sub = "S"
                else:
                    p_int = float(probs.get(intent, 0.0))
                    formula_kx = r"\mathbb{P}_I^K\ \text{por Ley de Implementación, Tabla 7}"
                    sub = r"K"
                acts_list = list(probs.keys())
                # ── 1. P_I rows ───────────────────────────────────────────
                pi_rows = ""
                for _act, _p in probs.items():
                    star = '<span class="star">&#9733;</span>' if _act == intent else ""
                    pi_rows += (
                        f"<tr><td>{_h.escape(_act)}</td>"
                        f"<td class='num'>{_p:.4f}</td>"
                        f"<td>{star}</td></tr>"
                    )
                # ── 2. Interval rows ──────────────────────────────────────
                int_rows = ""
                curr = 0.0
                for _i, (_act, _p) in enumerate(probs.items()):
                    lo, hi = curr, curr + _p
                    is_last = (_i == len(acts_list) - 1)
                    hi_str = "1.0000" if is_last else f"{hi:.4f}"
                    in_iv = (lo <= u_draw < hi) or (is_last and u_draw >= lo)
                    s_mark = '<span class="star">&#9733;</span>' if _act == atilde_argmax else ""
                    d_mark = '<span class="dart">&#127919;</span>' if in_iv else ""
                    rc = ' class="hit"' if in_iv else ""
                    int_rows += (
                        f"<tr{rc}>"
                        f"<td class='iv'>[{lo:.4f},&nbsp;{hi_str})</td>"
                        f"<td>{_h.escape(_act)}</td>"
                        f"<td class='num'>{_p:.4f}</td>"
                        f"<td>{s_mark}{d_mark}</td></tr>"
                    )
                    curr += _p
                # ── 3. E[U] extras ───────────────────────────────────────
                eu_extra = ""
                if u_extras:
                    eu_extra = f"<div class='ux'>{_h.escape(u_extras)}</div>"
                return (
                    f'<div class="t52h">'
                    f'<div class="hdr">'
                    f'\\(\\iota = {iota:.4f}\\)&ensp;'
                    f'\\({formula_kx} = {p_int:.4f}\\)'
                    f'</div>'
                    f'<div class="sec">'
                    f'\\(\\mathbb{{P}}_I(\\tilde{{a}}_{{{sub}}} \\mid a^*_{{{sub}}} = \\text{{{_h.escape(intent)}}})\\)'
                    f'</div>'
                    f'<table>{pi_rows}</table>'
                    f'<div class="sec">Intervalos acumulados \\([l_o,\\,h_i)\\)</div>'
                    f'<table class="int">{int_rows}</table>'
                    f'<div class="draw">'
                    f'\\(u_{{{sub}}} = {u_draw:.4f}\\) &rarr; '
                    f'\\(\\tilde{{a}}_{{{sub}}} = \\text{{{_h.escape(atilde_realized)}}}\\) &#127919;'
                    f'</div>'
                    f'<div class="argm">argmax \\(\\mathbb{{P}}_I\\): '
                    f'<b>{_h.escape(atilde_argmax)}</b> &#9733;</div>'
                    f'{eu_extra}'
                    f'<div class="eu">\\(\\mathbb{{E}}[U_{{{sub}}}] = {eu:.4f}\\)</div>'
                    f'</div>'
                )

            # ── Columna τ=1: a_F*, a_S* se recalculan; a_K* queda fijo desde τ=0 ──
            _t52_vals: dict = {}
            _t52_tips: dict = {}
            _t52_tips0: dict = {}   # τ=0 tooltips (primer pase, μ₀)

            # Sorteos U(0,1) individuales por jugador, seeded con dgp_seed
            _t52_rng_seed = int(st.session_state.get("dgp_seed", 42)) % (2**31)
            _t52_rng = np.random.default_rng(_t52_rng_seed)
            _u52_f = float(_t52_rng.random())   # sorteo familia
            _u52_k = float(_t52_rng.random())   # sorteo secuestrador
            _u52_s = float(_t52_rng.random())   # sorteo Estado
            _base_m_tau0_reroll = int(st.session_state.get("base_m_tau0_reroll_counter", 0))
            _base_m_tau0_resample_version = 1
            _t52_m_rng_seed = int(
                _t52_rng_seed
                + 7919 * int(_PSI8_CMH_CALIB_VERSION)
                + 104729 * _base_m_tau0_reroll
                + 15485863 * _base_m_tau0_resample_version
            ) % (2**31)
            _t52_m_rng = np.random.default_rng(_t52_m_rng_seed)
            _v52_m0 = float(_t52_m_rng.random())   # sorteo desenlace Fase 2
            def _t52_screening_policy(
                tau_v: int,
                alpha_star_v: float,
                gamma_star_v: float,
            ) -> dict[str, float]:
                """Política usada: coincide con el óptimo del Estado en Mechanism.tex."""
                _ = tau_v
                alpha_star = float(min(1.0, max(0.0, float(alpha_star_v))))
                gamma_star = float(min(1.0, max(0.0, float(gamma_star_v))))
                return {
                    "alpha_star": alpha_star,
                    "gamma_star": gamma_star,
                }

            # ── Paso 1-3 Familia: a_F*(τ=1), P_I^F, intervalos, sorteo ──
            _t52_vals["a_F*"] = "—"

            _pf52         = _t52_p1(_af52_star, "F", _iota_52, _mu0_52, _mu0_52, tau=1)
            _atf52_argmax = max(_pf52, key=_pf52.get)
            _atf52        = _t52_realize(_pf52, _u52_f)          # ã_F: donde cae u_F
            _t52_vals["ã_F"] = f"{_atf52}  (u={_u52_f:.4f}, p={_pf52[_atf52]:.4f})"
            _eu_f52 = (
                _pf52.get("Cooperar", 0.0) * float(_u_coop_p3)
                + _pf52.get("Coludir", 0.0) * float(_u_col_p3)
            )
            _t52_tips["ã_F"] = _t52_tip_html(
                "F", _af52_star, _pf52,
                _atf52, _atf52_argmax,
                _iota_52, _eu_f52, _u52_f,
                f"(U_coop={float(_u_coop_p3):.4f}, U_col={float(_u_col_p3):.4f})",
            )
            _t52_tips0["ã_F"] = _t52_tips["ã_F"]

            # ── Paso 1-3 Secuestrador: a_K* fijo desde Tabla 15, τ=0, col. 14 ──
            _vd52_real = (_val15_52.get(str(tipo_real)) or {}) if isinstance(_val15_52, dict) else {}
            _ru52_real = _df_p3_util_k[_df_p3_util_k["theta_K"].astype(str) == str(tipo_real)]
            _ak52_session_t0 = (
                str(st.session_state.get("tab15_opt_t0", ""))
                if str(st.session_state.get("tab15_opt_t0_theta", "")) == str(tipo_real)
                else ""
            )
            _ak52_t0_pre_raw = str(
                _vd52_real.get("opcion_tau0")
                or _ak52_session_t0
                or (_ru52_real.iloc[0]["rama_optima"] if not _ru52_real.empty else "—")
            )
            _t52_vals[f"a_K* ({tipo_real})"] = "—"

            _ak52_intent  = _t52_clean_ak(_ak52_t0_pre_raw)
            _pk52_real    = _t52_p1(_ak52_intent, "K", _iota_52, _mu0_52, _mu0_52, tau=1)
            _atk52_argmax = max(_pk52_real, key=_pk52_real.get)
            _atk52_real   = _t52_realize(_pk52_real, _u52_k)     # ã_K: donde cae u_K
            _t52_vals[f"ã_K ({tipo_real})"] = f"{_atk52_real}  (u={_u52_k:.4f}, p={_pk52_real[_atk52_real]:.4f})"
            if not _ru52_real.empty:
                _ur52t = float(_ru52_real.iloc[0].get("U_rel",  float("nan")))
                _uk52t = float(_ru52_real.iloc[0].get("U_kill", float("nan")))
                _vc52t = float(_ru52_real.iloc[0].get("V_cont", float("nan")))
            else:
                _ur52t = _uk52t = _vc52t = float("nan")
            _eu_k52_real = (
                _pk52_real.get("Liberar",    0.0) * _ur52t
                + _pk52_real.get("Matar",    0.0) * _uk52t
                + _pk52_real.get("Continuar", 0.0) * _vc52t
            )
            _t52_tips[f"ã_K ({tipo_real})"] = _t52_tip_html(
                f"K({tipo_real})", _ak52_intent, _pk52_real,
                _atk52_real, _atk52_argmax,
                _iota_52, _eu_k52_real, _u52_k,
                f"(U_rel={_ur52t:.4f}, U_kill={_uk52t:.4f}, V_cont={_vc52t:.4f})",
            )
            _t52_tips0[f"ã_K ({tipo_real})"] = _t52_tips[f"ã_K ({tipo_real})"]

            # ── Paso 1-3 Estado: a_S*(τ=1), P_I^S desde Tabla 7, sorteo ──
            _t52_vals["a_S* óptima"] = _s52_full
            _ps52 = _t52_p1_s(_agent52, _s52_star, _mu0_52, _mu0_52, tau=1)
            _ats52_argmax = max(_ps52, key=_ps52.get)
            _ats52 = _t52_realize(_ps52, _u52_s)
            _t52_vals["ã_S"] = f"{_ats52}  (u={_u52_s:.4f}, p={_ps52[_ats52]:.4f})"
            _eu_s52 = (
                _ps52.get("Rescatar", 0.0) * (-float(_vr_p3))
                + _ps52.get("No Rescatar", 0.0) * (-float(_vn_p3))
            )
            _t52_tips["ã_S"] = _t52_tip_html(
                "S", _s52_star, _ps52,
                _ats52, _ats52_argmax,
                _iota_52, _eu_s52, _u52_s,
                f"(−V_R={-float(_vr_p3):.4f}, −V_N={-float(_vn_p3):.4f})",
            )
            _t52_tips0["ã_S"] = _t52_tips["ã_S"]

            # ── Probabilidades del Estado en τ=1: entran en L_t^S ───────
            _s_exec52 = "Rescatar" if str(_ats52).strip().lower().startswith("rescat") else "No Rescatar"
            _theta_hat52 = max(_mu0_52, key=_mu0_52.get) if _mu0_52 else str(tipo_real)
            _p_surv_r52_t1 = 0.0
            _p_kill_n52_t1 = 0.0
            for _th52 in TIPOS_SECUESTRADOR:
                _w52 = float(_mu0_52.get(_th52, 0.0))
                _p_surv_r52_t1 += _w52 * float(
                    _p_surv_rescue_logit_for_executed_S(
                        _th52, _iota_52, _theta_hat52, _s_exec52
                    )
                )
                _p_kill_n52_t1 += _w52 * float(
                    _outcome_probs_for_actions(
                        _th52,
                        float(_t0_gamma_eff),
                        float(_iota_52),
                        _atk52_real,
                        _s_exec52,
                        _atf52,
                    )["kill"]
                )
            # ── P(m) se calcula en Flow 2 con μ₁ y γ* de τ=1 ─────────────
            _ak52_t0_raw = str((_vd52_real or {}).get("opcion_tau0", "—"))
            if _ak52_t0_raw in ("", "—", "None", "nan"):
                if (
                    str(st.session_state.get("tab15_opt_t0_theta", "")) == str(tipo_real)
                    and str(st.session_state.get("tab15_opt_t0", "—")) not in ("", "—", "None", "nan")
                ):
                    _ak52_t0_raw = str(st.session_state.get("tab15_opt_t0", "—"))
            if _ak52_t0_raw in ("", "—", "None", "nan"):
                try:
                    _df_tab15_mu52 = st.session_state.get("tab15_mu_snapshot")
                    _df_tab15_k52 = st.session_state.get("tab15_k_params_calibrated")
                    if (
                        isinstance(_df_tab15_mu52, pd.DataFrame)
                        and not _df_tab15_mu52.empty
                        and isinstance(_df_tab15_k52, pd.DataFrame)
                        and not _df_tab15_k52.empty
                    ):
                        _b_items52, _l_items52 = _betas_lambdas_cache_items(
                            st.session_state.cal_betas_dict,
                            st.session_state.cal_lambdas_dict,
                        )
                        _df_ia52, _ = _run_kidnapper_backward_induction_cached(
                            _df_to_cache_records(_df_tab15_mu52),
                            tuple(_df_tab15_mu52.columns),
                            _df_to_cache_records(_df_tab15_k52),
                            tuple(_df_tab15_k52.columns),
                            _b_items52,
                            _l_items52,
                            tipo_real=str(tipo_real),
                            beta_k=float(_p3_beta_k),
                            R=float(R_escala),
                            t_mad=float(st.session_state.get("cal_T_mad", 30.0)),
                            T=int(st.session_state.get("tab15_T_cached", _TAB14_TRAJ_TMAX)),
                            alpha_fallback=float(_t0_alpha_eff),
                            gamma_fallback=float(_t0_gamma_eff),
                            alpha_tab12=float(_t0_alpha_eff),
                            ransom_tab12=float(R_escala),
                        )
                        _r0_ia52 = _df_ia52.loc[_df_ia52["t"].astype(int) == 0] if not _df_ia52.empty else pd.DataFrame()
                        if not _r0_ia52.empty:
                            _ak52_t0_raw = str(_r0_ia52.iloc[0].get("opcion_BW", "—"))
                            st.session_state["tab15_opt_t0"] = str(_ak52_t0_raw)
                            st.session_state["tab15_opt_t0_theta"] = str(tipo_real)
                            if isinstance(_val15_52, dict):
                                _vd_store52 = dict(_val15_52.get(str(tipo_real), {}) or {})
                                _vd_store52["opcion_tau0"] = str(_ak52_t0_raw)
                                _val15_52[str(tipo_real)] = _vd_store52
                                st.session_state["tab15_last_validation"] = dict(_val15_52)
                except Exception:
                    _ak52_t0_raw = "—"
            _af52_tau0_default = str(_af52_star) if str(_af52_star) in ("Cooperar", "Coludir") else "Cooperar"
            _ak52_tau0_default = _t52_clean_ak(str(_ak52_t0_raw))
            if _ak52_tau0_default not in ("Continuar", "Liberar", "Matar"):
                _ak52_tau0_default = "Continuar"
            _as52_tau0_default = "Rescatar" if str(_s52_star).startswith("Rescat") else "No Rescatar"
            with st.expander("Escenario base τ=0 · elegir a*", expanded=False):
                st.caption("Solo fija la columna τ=0 y alimenta Tabla 10.")
                _af52_tau0_star = st.selectbox(
                    "Familia · a_F*",
                    ["Cooperar", "Coludir"],
                    index=["Cooperar", "Coludir"].index(_af52_tau0_default),
                    key="t52_tau0_aF_star",
                )
                st.text_input(
                    f"Secuestrador · a_K* ({tipo_real})",
                    value=str(_ak52_tau0_default),
                    disabled=True,
                )
                _ak52_tau0_star = str(_ak52_tau0_default)
                _as52_tau0_star = st.selectbox(
                    "Estado · a_S*",
                    ["No Rescatar", "Rescatar"],
                    index=["No Rescatar", "Rescatar"].index(_as52_tau0_default),
                    key="t52_tau0_aS_star",
                )
                _V52_tau0 = st.selectbox(
                    "Voz · V₀",
                    ["0 · sin señal", "1 · hay señal"],
                    index=int(st.session_state.get("t52_tau0_V0", 0)),
                    key="t52_tau0_V0_label",
                )
                _d52_tau0 = st.selectbox(
                    "Detección · d₀",
                    ["0 · no detectado", "1 · detectado"],
                    index=int(str(st.session_state.get("h0_d", "0")) == "1"),
                    key="t52_tau0_d0_label",
                )
                st.slider(
                    "Sensibilidad ℒ a α*,γ*",
                    min_value=1.0,
                    max_value=10.0,
                    value=float(st.session_state.get("t52_likelihood_policy_sensitivity", 4.0)),
                    step=0.5,
                    format="%.1f",
                    key="t52_likelihood_policy_sensitivity",
                    help="Multiplica ζ_α y ζ_γ en las verosimilitudes físicas para que α* y γ* impacten más L_H, L_F y μ.",
                )
                st.slider(
                    "ψ_H · ganancia informacional",
                    min_value=0.0,
                    max_value=100.0,
                    value=float(st.session_state.get("t52_entropy_info_weight", 25.0)),
                    step=1.0,
                    format="%.1f",
                    key="t52_entropy_info_weight",
                    help="Peso del término -ψ_H·ΔH en el problema del Estado. En 0 se recupera el objetivo sin motivo de exploración.",
                )
                st.number_input(
                    "Top-K auditoría IR/IC",
                    min_value=5,
                    max_value=441,
                    value=int(st.session_state.get("t52_iric_top_k", 25)),
                    step=5,
                    key="t52_iric_top_k",
                    help=(
                        "Audita IR/IC solo en los K mejores candidatos por score continuo. "
                        "Suba este valor para mayor precisión; bájelo para acelerar Tabla 5.2."
                    ),
                )
                st.number_input(
                    "Top-M cálculo ΔH",
                    min_value=5,
                    max_value=441,
                    value=int(st.session_state.get("t52_delta_h_top_m", 40)),
                    step=5,
                    key="t52_delta_h_top_m",
                    help=(
                        "Calcula la ganancia informacional ΔH solo en los M mejores "
                        "candidatos por valor continuo antes de auditar IR/IC."
                    ),
                )
                st.checkbox(
                    "Calcular benchmark PI en todos los ciclos",
                    value=bool(st.session_state.get("t52_pi_benchmark_all_cycles", False)),
                    key="t52_pi_benchmark_all_cycles",
                    help=(
                        "Si está desactivado, la referencia de información perfecta se calcula "
                        "solo para τ=0 y τ=1, y el valor de τ=1 se registra en los ciclos siguientes. "
                        "Activarlo recalcula PI en cada ciclo, pero aumenta el tiempo."
                    ),
                )
            _V52_tau0_int = 1 if str(_V52_tau0).startswith("1") else 0
            _d52_tau0_int = 1 if str(_d52_tau0).startswith("1") else 0
            st.session_state["t52_tau0_V0"] = int(_V52_tau0_int)
            st.session_state["h0_d"] = str(int(_d52_tau0_int))
            _af52_tau0_full = "Cooperar (a_coop)" if _af52_tau0_star == "Cooperar" else "Colusión (a_col)"
            _ak52_t0_raw = str(_ak52_tau0_star)
            _ak52_intent = _t52_clean_ak(_ak52_t0_raw)
            _s52_tau0_full = "Rescate (a_res)" if _as52_tau0_star == "Rescatar" else "Negociar (a_neg)"
            _pf52_tau0 = _t52_p1(_af52_tau0_star, "F", _iota_52, _mu0_52, _mu0_52, tau=0)
            _pk52_tau0 = _t52_p1(_ak52_tau0_star, "K", _iota_52, _mu0_52, _mu0_52, tau=0)
            _ps52_tau0 = _t52_p1_s(_agent52, _as52_tau0_star, _mu0_52, _mu0_52, tau=0)
            _atf52_tau0 = _t52_realize(_pf52_tau0, _u52_f)
            _atk52_tau0 = _t52_realize(_pk52_tau0, _u52_k)
            _ats52_tau0 = _t52_realize(_ps52_tau0, _u52_s)
            st.session_state.tab3_materialization_action_probs = {
                "F": {str(k): float(v) for k, v in _pf52_tau0.items()},
                "K": {str(k): float(v) for k, v in _pk52_tau0.items()},
                "S": {str(k): float(v) for k, v in _ps52_tau0.items()},
            }
            st.session_state.tab3_materialization_exec_actions = {
                "F": str(_atf52_tau0),
                "K": str(_atk52_tau0),
                "S": str(_ats52_tau0),
            }
            _t52_vals0 = {
                "a_F*": str(_af52_tau0_star),
                "ã_F": f"{_atf52_tau0}  (u={_u52_f:.4f}, p={_pf52_tau0[_atf52_tau0]:.4f})",
                f"a_K* ({tipo_real})": str(_ak52_t0_raw),
                f"ã_K ({tipo_real})": f"{_atk52_tau0}  (u={_u52_k:.4f}, p={_pk52_tau0[_atk52_tau0]:.4f})",
                "a_S* óptima": str(_s52_tau0_full),
                "ã_S": f"{_ats52_tau0}  (u={_u52_s:.4f}, p={_ps52_tau0[_ats52_tau0]:.4f})",
                "m": "Continuar (base inicial)",
            }
            _t52_tips0["ã_F"] = _t52_tip_html(
                "F", _af52_tau0_star, _pf52_tau0,
                _atf52_tau0, max(_pf52_tau0, key=_pf52_tau0.get),
                _iota_52,
                _pf52_tau0.get("Cooperar", 0.0) * float(_u_coop_p3)
                + _pf52_tau0.get("Coludir", 0.0) * float(_u_col_p3),
                _u52_f,
                f"(U_coop={float(_u_coop_p3):.4f}, U_col={float(_u_col_p3):.4f})",
            )
            _t52_tips0[f"ã_K ({tipo_real})"] = _t52_tip_html(
                f"K({tipo_real})", _ak52_tau0_star, _pk52_tau0,
                _atk52_tau0, max(_pk52_tau0, key=_pk52_tau0.get),
                _iota_52,
                _pk52_tau0.get("Liberar", 0.0) * _ur52t
                + _pk52_tau0.get("Matar", 0.0) * _uk52t
                + _pk52_tau0.get("Continuar", 0.0) * _vc52t,
                _u52_k,
                f"(U_rel={_ur52t:.4f}, U_kill={_uk52t:.4f}, V_cont={_vc52t:.4f})",
            )
            _t52_tips0["ã_S"] = _t52_tip_html(
                "S", _as52_tau0_star, _ps52_tau0,
                _ats52_tau0, max(_ps52_tau0, key=_ps52_tau0.get),
                _iota_52,
                _ps52_tau0.get("Rescatar", 0.0) * (-float(_vr_p3))
                + _ps52_tau0.get("No Rescatar", 0.0) * (-float(_vn_p3)),
                _u52_s,
                f"(−V_R={-float(_vr_p3):.4f}, −V_N={-float(_vn_p3):.4f})",
            )
            _atf0_m52 = str(_atf52_tau0)
            _atk0_m52 = str(_atk52_tau0)
            _ats0_m52 = (
                "Rescatar"
                if str(_ats52_tau0).strip().lower().startswith("rescat")
                else "No Rescatar"
            )
            _policy0_52 = _t52_screening_policy(
                0,
                float(_t0_alpha_eff),
                float(_t0_gamma_eff),
            )
            _eta1_52 = float(st.session_state.get("cal_eta1_pdet", 1.0))
            _eta2_52 = float(st.session_state.get("cal_eta2_pdet", 1.0))
            _t52_entropy_weight = float(max(0.0, st.session_state.get("t52_entropy_info_weight", 25.0)))

            def _t52_norm_mu(mu_v: dict[str, float]) -> dict[str, float]:
                vals = {th: max(0.0, float(dict(mu_v).get(th, 0.0))) for th in TIPOS_SECUESTRADOR}
                total = float(sum(vals.values()))
                if total <= 1e-12:
                    return {th: 1.0 / len(TIPOS_SECUESTRADOR) for th in TIPOS_SECUESTRADOR}
                return {th: float(vals[th] / total) for th in TIPOS_SECUESTRADOR}

            def _t52_shannon_entropy(mu_v: dict[str, float]) -> float:
                mu_n = _t52_norm_mu(mu_v)
                return float(-sum(p * np.log(max(p, 1e-15)) for p in mu_n.values()))

            _t52_entropy_gain_cache: dict[tuple[Any, ...], dict[str, Any]] = {}

            def _t52_mu_signature(mu_v: dict[str, float]) -> tuple[tuple[str, float], ...]:
                mu_n = _t52_norm_mu(mu_v)
                return tuple((str(th), round(float(mu_n.get(str(th), 0.0)), 8)) for th in TIPOS_SECUESTRADOR)

            def _t52_expected_entropy_gain(
                mu_v: dict[str, float],
                tau_v: int,
                alpha_v: float,
                gamma_v: float,
                a_k_exec_v: str,
                a_s_exec_v: str,
                a_f_exec_v: str,
            ) -> dict[str, Any]:
                mu_n = _t52_norm_mu(mu_v)
                _cache_key = (
                    _t52_mu_signature(mu_n),
                    int(tau_v),
                    round(float(alpha_v), 4),
                    round(float(gamma_v), 4),
                    str(a_k_exec_v),
                    str(a_s_exec_v),
                    str(a_f_exec_v),
                    str(st.session_state.z_region),
                    str(st.session_state.v_victim),
                    str(f_capa),
                    str(s_tipo),
                    round(float(st.session_state.get("t52_likelihood_policy_sensitivity", 4.0)), 6),
                )
                if _cache_key in _t52_entropy_gain_cache:
                    return copy.deepcopy(_t52_entropy_gain_cache[_cache_key])
                h_now = float(_t52_shannon_entropy(mu_n))
                # η₀(θ_K) tipo-específico — detectabilidad basal varía entre organizaciones
                p_det_by_theta: dict[str, float] = {
                    str(th): float(1.0 / (1.0 + np.exp(-(
                        _pdet_eta0_for_theta(str(th))
                        + _eta1_52 * float(alpha_v)
                        + _eta2_52 * float(gamma_v)
                    ))))
                    for th in TIPOS_SECUESTRADOR
                }
                # p_det promedio ponderado por creencia (para referencia y display)
                p_det_v = float(sum(float(mu_n.get(str(th), 0.0)) * p_det_by_theta[str(th)] for th in TIPOS_SECUESTRADOR))
                probs_by_theta: dict[str, dict[str, float]] = {}
                mix = {lbl: 0.0 for lbl in ["Liberación", "Rescate", "Pago", "Muerte", "Continuar"]}
                for th in TIPOS_SECUESTRADOR:
                    pm_th, _ = _mechanism_m_probs_for_actions(
                        str(th),
                        int(tau_v),
                        float(alpha_v),
                        float(gamma_v),
                        float(p_det_by_theta[str(th)]),
                        str(a_k_exec_v),
                        str(a_s_exec_v),
                        str(a_f_exec_v),
                        z_region=str(st.session_state.z_region),
                        v_victim=str(st.session_state.v_victim),
                        f_capa=str(f_capa),
                        s_tipo=str(s_tipo),
                        allow_zero_time=(int(tau_v) == 0),
                        policy_sensitivity=float(st.session_state.get("t52_likelihood_policy_sensitivity", 4.0)),
                    )
                    probs_by_theta[str(th)] = {lbl: float(pm_th.get(lbl, 0.0)) for lbl in mix}
                    for lbl in mix:
                        mix[lbl] += float(mu_n.get(str(th), 0.0)) * float(probs_by_theta[str(th)].get(lbl, 0.0))
                # ΔH_t sobre señal conjunta (m,d) — Mechanism.tex eq:delta-H
                # P(m,d|μ) = Σ_θ μ(θ)·P(m|θ)·P(d|θ,α,γ) con P(d=1|θ)=p_det(θ)
                e_h_next = 0.0
                post_by_event: dict[str, dict[str, float]] = {}
                for lbl, p_m in mix.items():
                    for d_val in (0, 1):
                        p_d_mix = float(sum(
                            float(mu_n.get(str(th), 0.0))
                            * (p_det_by_theta[str(th)] if d_val == 1 else (1.0 - p_det_by_theta[str(th)]))
                            for th in TIPOS_SECUESTRADOR
                        ))
                        p_event = float(max(0.0, float(p_m) * p_d_mix))
                        event_key = f"{lbl}_d{d_val}"
                        if p_event <= 1e-15:
                            post = dict(mu_n)
                        else:
                            post = {
                                str(th): float(
                                    mu_n.get(str(th), 0.0)
                                    * probs_by_theta[str(th)].get(lbl, 0.0)
                                    * (p_det_by_theta[str(th)] if d_val == 1 else (1.0 - p_det_by_theta[str(th)]))
                                    / p_event
                                )
                                for th in TIPOS_SECUESTRADOR
                            }
                        post = _t52_norm_mu(post)
                        post_by_event[event_key] = post
                        e_h_next += p_event * float(_t52_shannon_entropy(post))
                delta_h = float(max(0.0, h_now - e_h_next))
                _out = {
                    "H": h_now,
                    "E_H_next": float(e_h_next),
                    "Delta_H": delta_h,
                    "p_event": {lbl: float(v) for lbl, v in mix.items()},
                    "post_by_event": post_by_event,
                    "p_det": p_det_v,
                }
                _t52_entropy_gain_cache[_cache_key] = copy.deepcopy(_out)
                return _out

            _pdet0_m52 = _pdet_logit_prob(
                str(tipo_real),
                float(_policy0_52["alpha_star"]),
                float(_policy0_52["gamma_star"]),
            )
            _pm0_52, _mfac0_52 = _mechanism_m_probs_for_actions(
                str(tipo_real),
                0,
                float(_policy0_52["alpha_star"]),
                float(_policy0_52["gamma_star"]),
                float(_pdet0_m52),
                str(_atk0_m52),
                str(_ats0_m52),
                str(_atf0_m52),
                z_region=str(st.session_state.z_region),
                v_victim=str(st.session_state.v_victim),
                f_capa=str(f_capa),
                s_tipo=str(s_tipo),
                allow_zero_time=True,
                policy_sensitivity=float(st.session_state.get("t52_likelihood_policy_sensitivity", 4.0)),
            )
            _m_mode_ignore_stop52 = str(st.session_state.get("t52_m_mode", "Sorteo")) == "Continuar"
            _m0_real52 = _t52_realize(_pm0_52, _v52_m0)
            _t52_vals0["m"] = f"{_m0_real52}  (v={_v52_m0:.4f}, p={_pm0_52[_m0_real52]:.4f})"
            st.session_state.tab3_materialization_outcome_probs = {
                "lib": float(_pm0_52.get("Liberación", 0.0)),
                "res": float(_pm0_52.get("Rescate", 0.0)),
                "pay": float(_pm0_52.get("Pago", 0.0)),
                "kill": float(_pm0_52.get("Muerte", 0.0)),
                "cont": float(_pm0_52.get("Continuar", 0.0)),
                "mechanism_factors": dict(_mfac0_52),
            }
            st.session_state.tab3_materialization_outcome = str(_m0_real52)
            st.session_state.h0_m = str(_m0_real52)
            st.session_state["base_cycle_m_tau0"] = {
                "tau": 0,
                "m": str(_m0_real52),
                "u": float(_v52_m0),
                "p": float(_pm0_52[_m0_real52]),
                "m_mode": "Continuar" if _m_mode_ignore_stop52 else "Sorteo",
                "m_ignore_stop": bool(_m_mode_ignore_stop52),
                "probs": {str(_k): float(_v) for _k, _v in _pm0_52.items()},
                "theta": str(tipo_real),
                "t_eval": 0,
                "alpha": float(_policy0_52["alpha_star"]),
                "gamma": float(_policy0_52["gamma_star"]),
                "p_det": float(_pdet0_m52),
                "iota": float(_iota_52),
                "V_t": int(_V52_tau0_int),
                "d": int(_d52_tau0_int),
                "atilde_F": str(_atf0_m52),
                "atilde_K": str(_atk0_m52),
                "atilde_S": str(_ats0_m52),
                "mechanism_factors": dict(_mfac0_52),
            }
            _base_m_by_theta52 = dict(st.session_state.get("base_cycle_m_tau0_by_theta", {}))
            _base_m_by_theta52[str(tipo_real)] = dict(st.session_state["base_cycle_m_tau0"])
            st.session_state["base_cycle_m_tau0_by_theta"] = _base_m_by_theta52
            _m0_int_rows52 = ""
            _m0_cur52 = 0.0
            for _i_m0_52, (_lbl_m0_52, _prob_m0_52) in enumerate(_pm0_52.items()):
                _lo_m0_52, _hi_m0_52 = _m0_cur52, _m0_cur52 + float(_prob_m0_52)
                _hi_m0_str52 = "1.0000" if _i_m0_52 == len(_pm0_52) - 1 else f"{_hi_m0_52:.4f}"
                _hit_m0_52 = (_lo_m0_52 <= _v52_m0 < _hi_m0_52) or (_i_m0_52 == len(_pm0_52) - 1 and _v52_m0 >= _lo_m0_52)
                _rc_m0_52 = ' class="hit"' if _hit_m0_52 else ""
                _mk_m0_52 = '<span class="dart">&#127919;</span>' if _hit_m0_52 else ""
                _m0_int_rows52 += (
                    f"<tr{_rc_m0_52}>"
                    f"<td class='iv'>[{_lo_m0_52:.4f},&nbsp;{_hi_m0_str52})</td>"
                    f"<td>{html.escape(_lbl_m0_52)}</td>"
                    f"<td class='num'>{float(_prob_m0_52):.4f}</td>"
                    f"<td>{_mk_m0_52}</td></tr>"
                )
                _m0_cur52 += float(_prob_m0_52)
            _m0_mode_note52 = (
                f'<div class="sec">Modo Continuar: parada desactivada</div>'
                f'<div class="ux">El selector <b>Modo m</b> está en <b>Continuar</b>: '
                f'\\(m_0\\) se sortea normalmente y entra en \\(\\mathcal{{L}}_H(m_0)\\), '
                f'pero la condición \\(m_0\\ne\\mathrm{{Continuar}}\\) no detiene la trayectoria. '
                f'Sorteo: \\(v_0={_v52_m0:.4f}\\), \\(m_0={html.escape(str(_m0_real52))}\\).</div>'
                f'<div class="sec">Transformada inversa</div>'
                f'<table class="int"><tr><td>Intervalo</td><td>m</td><td>P</td><td></td></tr>{_m0_int_rows52}</table>'
                if _m_mode_ignore_stop52
                else (
                    f'<div class="sec">Transformada inversa</div>'
                    f'<table class="int"><tr><td>Intervalo</td><td>m</td><td>P</td><td></td></tr>{_m0_int_rows52}</table>'
                    f'<div class="draw">\\(v_0={_v52_m0:.4f}\\) &rarr; <b>{html.escape(str(_m0_real52))}</b></div>'
                )
            )
            _t52_tips0["m"] = (
                f'<div class="t52h">'
                f'<div class="hdr">m en ciclo base · Fase 2</div>'
                f'<div class="sec">Tripleta ejecutada del ciclo base</div>'
                f'<div>\\(\\tilde A_0=(\\tilde a_K={html.escape(str(_atk0_m52))},'
                f'\\tilde a_S={html.escape(str(_ats0_m52))},'
                f'\\tilde a_F={html.escape(str(_atf0_m52))})\\)</div>'
                f'<div class="sec">Ecuaciones (28)-(29) · Mechanism.tex</div>'
                f'<div>\\(P^E(m_t=\\mathrm{{Cont}})=p_{{\\mathrm{{Cont}},t}}\\), '
                f'\\(P^E(m_t=j)=h_j(t\\mid\\theta_K,\\mathcal C_t)\\).</div>'
                f'<div class="sec">Política del Estado usada</div>'
                f'<div>Se usa directamente \\(\\alpha_0^*={float(_policy0_52["alpha_star"]):.4f}\\) '
                f'y \\(\\gamma_0^*={float(_policy0_52["gamma_star"]):.4f}\\), como instrumentos óptimos del Estado.</div>'
                f'{_t52_mechanism_m_tooltip_lines(_mfac0_52, 0, float(_policy0_52["alpha_star"]), float(_policy0_52["gamma_star"]), float(_pdet0_m52))}'
                f'{_m0_mode_note52}'
                f'</div>'
            )
            _mu1_52 = dict(_mu0_52)
            _df_post52 = pd.DataFrame()
            _meta_post52: dict[str, Any] = {}
            try:
                _zp52 = _focus_cmh_endogenous_tentatives(str(tipo_real))
                _eta1_52 = float(st.session_state.get("cal_eta1_pdet", 1.0))
                _eta2_52 = float(st.session_state.get("cal_eta2_pdet", 1.0))
                _pdet52 = _pdet_logit_prob(
                    str(tipo_real),
                    float(_policy0_52["alpha_star"]),
                    float(_policy0_52["gamma_star"]),
                )
                _h0_d_raw52 = str(st.session_state.get("h0_d", "0"))
                _d0_52 = 0 if _h0_d_raw52 in ("—", "", "0") else 1
                _omega_voz52, _pi_call52, _voz_params52 = _resolve_voice_tab2_params()
                _tab2_bundles52 = _tab2_bundles_all_types(
                    z_region=str(st.session_state.z_region),
                    v_victim=str(st.session_state.v_victim),
                    f_capa=str(f_capa),
                    s_tipo=str(s_tipo),
                )
                _lik_sens52 = float(st.session_state.get("t52_likelihood_policy_sensitivity", 4.0))
                _tab2_bundles52 = _scale_policy_zeta_bundles(_tab2_bundles52, _lik_sens52)
                _li0_52 = _build_t0_implementation_likelihood_by_theta(
                    _mu0_52,
                    presion_S=float(_policy0_52["gamma_star"]),
                    precision_iota=float(_iota_52),
                    alpha0=float(_policy0_52["alpha_star"]),
                    gamma0=float(_policy0_52["gamma_star"]),
                    ransom_scale=float(R_escala),
                    f_capa=str(f_capa),
                    estado_duro=(str(s_tipo) == "Duro"),
                    beta_k=float(_p3_beta_k),
                    atilde_F=str(_atf0_m52),
                    atilde_K=str(_atk0_m52),
                    atilde_S=str(_ats0_m52),
                )
                _df_post52, _mu1_52_calc, _meta_post52 = build_t0_bayesian_posterior_report(
                    modelo,
                    _mu0_52,
                    str(_m0_real52),
                    int(_d0_52),
                    presion_S=float(_policy0_52["gamma_star"]),
                    z_region=str(st.session_state.z_region),
                    v_victim=str(st.session_state.v_victim),
                    alpha=float(_policy0_52["alpha_star"]),
                    gamma=float(_policy0_52["gamma_star"]),
                    p_det=float(_pdet52),
                    zeta_alpha=float(_zp52.get("zeta_alpha", 0.1)),
                    zeta_gamma=float(_zp52.get("zeta_gamma", 0.1)),
                    zeta_d=float(_zp52.get("zeta_d", 0.1)),
                    zeta_R=float(_zp52.get("zeta_R", 0.1)),
                    estado_rescata=str(_ats0_m52).strip().lower().startswith("rescat"),
                    t_mad=float(st.session_state.get("cal_T_mad", 30.0)),
                    lambda4=float(st.session_state.get("cal_lambda_4", 0.0005)),
                    t_eval=0,
                    omega_voz=float(_omega_voz52),
                    pi_call_by_theta=_pi_call52,
                    voz_params_by_theta=_voz_params52,
                    V_t=int(_V52_tau0_int),
                    atilde_F=_atf0_m52,
                    atilde_K=_atk0_m52,
                    atilde_S=_ats0_m52,
                    implementation_likelihood_by_theta=_li0_52,
                    tab2_bundle_by_theta=_tab2_bundles52,
                    aggregate_unknown_theta=False,
                )
                _mu1_52 = {th: float(_mu1_52_calc.get(th, _mu0_52.get(th, 0.0))) for th in TIPOS_SECUESTRADOR}
            except Exception:
                _mu1_52 = dict(_mu0_52)
            _pf52 = _t52_p1(_af52_star, "F", _iota_52, _mu0_52, _mu1_52, tau=1)
            _atf52_argmax = max(_pf52, key=_pf52.get)
            _atf52 = _t52_realize(_pf52, _u52_f)
            _t52_vals["ã_F"] = f"{_atf52}  (u={_u52_f:.4f}, p={_pf52[_atf52]:.4f})"
            _eu_f52 = (
                _pf52.get("Cooperar", 0.0) * float(_u_coop_p3)
                + _pf52.get("Coludir", 0.0) * float(_u_col_p3)
            )
            _t52_tips["ã_F"] = _t52_tip_html(
                "F", _af52_star, _pf52,
                _atf52, _atf52_argmax,
                _iota_52, _eu_f52, _u52_f,
                f"(U_coop={float(_u_coop_p3):.4f}, U_col={float(_u_col_p3):.4f})",
            )
            _pk52_real = _t52_p1(_ak52_intent, "K", _iota_52, _mu0_52, _mu1_52, tau=1)
            _atk52_argmax = max(_pk52_real, key=_pk52_real.get)
            _atk52_real = _t52_realize(_pk52_real, _u52_k)
            _t52_vals[f"ã_K ({tipo_real})"] = f"{_atk52_real}  (u={_u52_k:.4f}, p={_pk52_real[_atk52_real]:.4f})"
            _eu_k52_real = (
                _pk52_real.get("Liberar", 0.0) * _ur52t
                + _pk52_real.get("Matar", 0.0) * _uk52t
                + _pk52_real.get("Continuar", 0.0) * _vc52t
            )
            _t52_tips[f"ã_K ({tipo_real})"] = _t52_tip_html(
                f"K({tipo_real})", _ak52_intent, _pk52_real,
                _atk52_real, _atk52_argmax,
                _iota_52, _eu_k52_real, _u52_k,
                f"(U_rel={_ur52t:.4f}, U_kill={_uk52t:.4f}, V_cont={_vc52t:.4f})",
            )

            def _t52_minimize_state_quadratic(
                const: float,
                b_gamma: float,
                q_gamma: float,
                b_alpha: float,
                q_alpha: float,
                q_gamma_alpha: float,
                info_bonus_func=None,
            ) -> tuple[float, float, float]:
                candidates: set[tuple[float, float]] = {
                    (0.0, 0.0),
                    (0.0, 1.0),
                    (1.0, 0.0),
                    (1.0, 1.0),
                }
                if float(_t52_entropy_weight) > 0.0 and info_bonus_func is not None:
                    for _g_grid in np.linspace(0.0, 1.0, 21):
                        for _a_grid in np.linspace(0.0, 1.0, 21):
                            candidates.add((round(float(_g_grid), 8), round(float(_a_grid), 8)))
                det = float(q_gamma * q_alpha - q_gamma_alpha * q_gamma_alpha)
                if abs(det) > 1e-12:
                    gamma_int = float((-b_gamma * q_alpha + q_gamma_alpha * b_alpha) / det)
                    alpha_int = float((q_gamma_alpha * b_gamma - q_gamma * b_alpha) / det)
                    if 0.0 <= gamma_int <= 1.0 and 0.0 <= alpha_int <= 1.0:
                        candidates.add((round(gamma_int, 8), round(alpha_int, 8)))
                for alpha_fix in (0.0, 1.0):
                    if abs(q_gamma) > 1e-12:
                        gamma_b = float(-(b_gamma + q_gamma_alpha * alpha_fix) / q_gamma)
                    else:
                        gamma_b = 0.0 if b_gamma + q_gamma_alpha * alpha_fix >= 0.0 else 1.0
                    candidates.add((round(float(min(1.0, max(0.0, gamma_b))), 8), alpha_fix))
                for gamma_fix in (0.0, 1.0):
                    if abs(q_alpha) > 1e-12:
                        alpha_b = float(-(b_alpha + q_gamma_alpha * gamma_fix) / q_alpha)
                    else:
                        alpha_b = 0.0 if b_alpha + q_gamma_alpha * gamma_fix >= 0.0 else 1.0
                    candidates.add((gamma_fix, round(float(min(1.0, max(0.0, alpha_b))), 8)))

                def _raw_val(gamma_v: float, alpha_v: float) -> float:
                    return float(
                        const
                        + b_gamma * gamma_v
                        + 0.5 * q_gamma * gamma_v * gamma_v
                        + b_alpha * alpha_v
                        + 0.5 * q_alpha * alpha_v * alpha_v
                        + q_gamma_alpha * gamma_v * alpha_v
                    )

                def _score(gamma_v: float, alpha_v: float) -> float:
                    bonus = 0.0
                    if float(_t52_entropy_weight) > 0.0 and info_bonus_func is not None:
                        try:
                            bonus = float(info_bonus_func(float(alpha_v), float(gamma_v)).get("Delta_H", 0.0))
                        except Exception:
                            bonus = 0.0
                    return float(_raw_val(gamma_v, alpha_v) - float(_t52_entropy_weight) * bonus)

                gamma_star, alpha_star = min(candidates, key=lambda x: _score(x[0], x[1]))
                return float(alpha_star), float(gamma_star), _score(gamma_star, alpha_star)

            def _t52_quad_value(
                gamma_v: float,
                alpha_v: float,
                const: float,
                b_gamma: float,
                q_gamma: float,
                b_alpha: float,
                q_alpha: float,
                q_gamma_alpha: float,
            ) -> float:
                return float(
                    const
                    + b_gamma * gamma_v
                    + 0.5 * q_gamma * gamma_v * gamma_v
                    + b_alpha * alpha_v
                    + 0.5 * q_alpha * alpha_v * alpha_v
                    + q_gamma_alpha * gamma_v * alpha_v
                )

            def _t52_iric_status(
                alpha_v: float,
                gamma_v: float,
                mu_tau_v: dict[str, float],
                V_R_v: float,
                V_N_v: float,
            ) -> dict[str, Any]:
                _ir_gap = float("nan")
                _ic_k_gap_min = float("nan")
                _ir_f_gap = float("nan")
                _ir_k_true_gap = float("nan")
                _ir_k_true_u_rel = float("nan")
                _ir_k_true_v_cont = float("nan")
                _ir_k_true_u_kill = float("nan")
                _ir_k_true = False
                try:
                    _df_k_cand = refresh_kidnapper_endogenous_columns(
                        _df_p3_k_params.copy(),
                        modelo,
                        float(gamma_v),
                        float(gamma_v),
                        alpha=float(alpha_v),
                    )
                    _df_util_cand = kidnapper_util_df_from_param_df(
                        _df_k_cand,
                        modelo,
                        float(gamma_v),
                        float(alpha_v),
                        float(gamma_v),
                        float(R_escala),
                        str(tipo_real),
                        float(_p3_beta_k),
                    )
                    _mu_ir_raw = {str(k): max(0.0, float(v)) for k, v in dict(mu_tau_v).items()}
                    _mu_ir_sum = float(sum(_mu_ir_raw.values()))
                    _mu_ir = (
                        {th: float(_mu_ir_raw.get(th, 0.0)) / _mu_ir_sum for th in TIPOS_SECUESTRADOR}
                        if _mu_ir_sum > 1e-12
                        else {th: 1.0 / len(TIPOS_SECUESTRADOR) for th in TIPOS_SECUESTRADOR}
                    )
                    _ir_gap = 0.0
                    for _, _ri_ir in _df_util_cand.iterrows():
                        _theta_ir = str(_ri_ir["theta_K"])
                        _outside_criminal = max(float(_ri_ir["V_cont"]), float(_ri_ir["U_kill"]))
                        _ir_gap += float(_mu_ir.get(_theta_ir, 0.0)) * (
                            float(_ri_ir["U_rel"]) - _outside_criminal
                        )
                    _ir_k = bool(not _df_util_cand.empty and _ir_gap >= -1e-9)
                    _df_true_ir = _df_util_cand[
                        _df_util_cand["theta_K"].astype(str) == str(tipo_real)
                    ]
                    if not _df_true_ir.empty:
                        _rt_ir = _df_true_ir.iloc[0]
                        _ir_k_true_u_rel = float(_rt_ir["U_rel"])
                        _ir_k_true_v_cont = float(_rt_ir["V_cont"])
                        _ir_k_true_u_kill = float(_rt_ir["U_kill"])
                        _ir_k_true_gap = float(
                            _ir_k_true_u_rel
                            - max(_ir_k_true_v_cont, _ir_k_true_u_kill)
                        )
                        _ir_k_true = bool(_ir_k_true_gap >= -1e-9)
                    _mu_ic_raw = {str(k): max(0.0, float(v)) for k, v in dict(mu_tau_v).items()}
                    _mu_ic_sum = float(sum(_mu_ic_raw.values()))
                    _mu_ic = (
                        {th: float(_mu_ic_raw.get(th, 0.0)) / _mu_ic_sum for th in TIPOS_SECUESTRADOR}
                        if _mu_ic_sum > 1e-12
                        else {th: 1.0 / len(TIPOS_SECUESTRADOR) for th in TIPOS_SECUESTRADOR}
                    )
                    _ic_k_expected: list[bool] = []
                    _ic_k_gaps: list[float] = []
                    _util_cols = {
                        "Liberar (a_rel)": "U_rel",
                        "Matar (a_kill)": "U_kill",
                        "Continuar (a_cont)": "V_cont",
                    }
                    for _, _rj in _df_util_cand.iterrows():
                        _gain_j = 0.0
                        for _, _ri in _df_util_cand.iterrows():
                            _theta_i = str(_ri["theta_K"])
                            _best_i = max(float(_ri["U_rel"]), float(_ri["U_kill"]), float(_ri["V_cont"]))
                            _col_j = _util_cols.get(str(_rj["rama_optima"]), "V_cont")
                            _gain_j += float(_mu_ic.get(_theta_i, 0.0)) * (_best_i - float(_ri[_col_j]))
                        _ic_k_gaps.append(float(_gain_j))
                        _ic_k_expected.append(_gain_j >= -1e-9)
                    _ic_k_gap_min = float(min(_ic_k_gaps)) if _ic_k_gaps else float("nan")
                    _ic_k = bool(all(_ic_k_expected) if _ic_k_expected else True)
                except Exception:
                    _ir_k = False
                    _ic_k = False
                    _ir_k_true = False

                try:
                    _mu_f_raw = {str(k): max(0.0, float(v)) for k, v in dict(mu_tau_v).items()}
                    _mu_f_sum = float(sum(_mu_f_raw.values()))
                    _mu_f_norm = (
                        {th: float(_mu_f_raw.get(th, 0.0)) / _mu_f_sum for th in TIPOS_SECUESTRADOR}
                        if _mu_f_sum > 1e-12
                        else {th: 1.0 / len(TIPOS_SECUESTRADOR) for th in TIPOS_SECUESTRADOR}
                    )
                    _df_f_cand, _ = compute_family_table(
                        modelo,
                        _mu_f_norm,
                        float(gamma_v),
                        float(_p3_vl),
                        float(R_escala),
                        float(gamma_v),
                        float(_p3_phi_f),
                        float(_p3_kappa_f),
                        float(_p3_nu_f),
                        float(_p3_fcol),
                        float(_p3_pd0),
                        float(_p3_pda),
                        float(alpha_v),
                        float(cmh_alive),
                    )
                    _ucoop = float(_df_f_cand.loc[_df_f_cand["Rama"].str.startswith("Cooperar"), "EU ilustrativa"].iloc[0])
                    _ucol = float(_df_f_cand.loc[_df_f_cand["Rama"].str.startswith("Colusión"), "EU ilustrativa"].iloc[0])
                    _ir_f_gap = float(_ucoop - _ucol)
                    _ir_f = bool(_ucoop >= _ucol)
                    _ic_f = bool(np.isfinite(_ucoop) and np.isfinite(_ucol))
                except Exception:
                    _ir_f = False
                    _ic_f = False

                _ic_s = bool(np.isfinite(float(V_R_v)) and np.isfinite(float(V_N_v)))
                _gamma_formal = bool(_ir_k and _ic_k and _ir_f)
                return {
                    "IR_K": _ir_k,
                    "IC_K": _ic_k,
                    "IR_F": _ir_f,
                    "IC_F": _ic_f,
                    "IC_S": _ic_s,
                    "IR_K_gap_E": float(_ir_gap),
                    "IC_K_gap_E_min": float(_ic_k_gap_min),
                    "IR_F_gap_E": float(_ir_f_gap),
                    "IR_K_true": bool(_ir_k_true),
                    "IR_K_true_gap": float(_ir_k_true_gap),
                    "IR_K_true_U_rel": float(_ir_k_true_u_rel),
                    "IR_K_true_V_cont": float(_ir_k_true_v_cont),
                    "IR_K_true_U_kill": float(_ir_k_true_u_kill),
                    "Gamma_formal": _gamma_formal,
                    "feasible": _gamma_formal,
                    "audit_feasible": bool(_gamma_formal and _ic_f and _ic_s),
                }

            def _t52_minimize_state_quadratic_iric(
                const: float,
                b_gamma: float,
                q_gamma: float,
                b_alpha: float,
                q_alpha: float,
                q_gamma_alpha: float,
                mu_tau_v: dict[str, float],
                other_value_func,
                info_bonus_func=None,
                extra_score_fn=None,
            ) -> tuple[float, float, float, dict[str, Any]]:
                _continuous_value_cache: dict[tuple[float, float], float] = {}

                def _continuous_value(_g_v: float, _a_v: float) -> float:
                    _key_v = (round(float(_g_v), 8), round(float(_a_v), 8))
                    if _key_v in _continuous_value_cache:
                        return float(_continuous_value_cache[_key_v])
                    _val_v = _t52_quad_value(
                        float(_g_v), float(_a_v),
                        const, b_gamma, q_gamma, b_alpha, q_alpha, q_gamma_alpha,
                    )
                    if extra_score_fn is not None:
                        try:
                            _val_v = float(_val_v) + float(extra_score_fn(float(_a_v), float(_g_v)))
                        except Exception:
                            pass
                    _continuous_value_cache[_key_v] = float(_val_v)
                    return float(_val_v)

                _structural_cands = {
                    (0.0, 0.0),
                    (0.0, 1.0),
                    (1.0, 0.0),
                    (1.0, 1.0),
                }
                _coarse_cands: set[tuple[float, float]] = set()
                _coarse_grid = np.linspace(0.0, 1.0, 11)
                for _g in _coarse_grid:
                    for _a in _coarse_grid:
                        _coarse_cands.add((round(float(_g), 8), round(float(_a), 8)))
                _coarse_cands.update(_structural_cands)

                _analytic_cands: set[tuple[float, float]] = set()
                det = float(q_gamma * q_alpha - q_gamma_alpha * q_gamma_alpha)
                if abs(det) > 1e-12:
                    _g_int = float((-b_gamma * q_alpha + q_gamma_alpha * b_alpha) / det)
                    _a_int = float((q_gamma_alpha * b_gamma - q_gamma * b_alpha) / det)
                    if 0.0 <= _g_int <= 1.0 and 0.0 <= _a_int <= 1.0:
                        _analytic_cands.add((round(_g_int, 8), round(_a_int, 8)))
                for _a_fix in (0.0, 1.0):
                    _g_b = (
                        float(-(b_gamma + q_gamma_alpha * _a_fix) / q_gamma)
                        if abs(q_gamma) > 1e-12
                        else (0.0 if b_gamma + q_gamma_alpha * _a_fix >= 0.0 else 1.0)
                    )
                    _analytic_cands.add((round(float(min(1.0, max(0.0, _g_b))), 8), round(_a_fix, 8)))
                for _g_fix in (0.0, 1.0):
                    _a_b = (
                        float(-(b_alpha + q_gamma_alpha * _g_fix) / q_alpha)
                        if abs(q_alpha) > 1e-12
                        else (0.0 if b_alpha + q_gamma_alpha * _g_fix >= 0.0 else 1.0)
                    )
                    _analytic_cands.add((round(_g_fix, 8), round(float(min(1.0, max(0.0, _a_b))), 8)))

                _coarse_cands.update(_analytic_cands)
                _refine_centers = sorted(
                    _coarse_cands,
                    key=lambda _ga: _continuous_value(float(_ga[0]), float(_ga[1])),
                )[:5]
                _cands: set[tuple[float, float]] = set(_coarse_cands)
                _refine_step = 0.05
                for _g0, _a0 in _refine_centers:
                    for _dg in (-_refine_step, 0.0, _refine_step):
                        for _da in (-_refine_step, 0.0, _refine_step):
                            _g_ref = float(min(1.0, max(0.0, float(_g0) + float(_dg))))
                            _a_ref = float(min(1.0, max(0.0, float(_a0) + float(_da))))
                            _cands.add((round(_g_ref, 8), round(_a_ref, 8)))

                _cheap_rank: list[tuple[float, float, float, float, float]] = []
                for _g_v, _a_v in _cands:
                    _val_v = _continuous_value(float(_g_v), float(_a_v))
                    _cheap_rank.append((float(_val_v), float(_g_v), float(_a_v), float(_val_v), 0.0))

                _cheap_rank.sort(key=lambda x: x[0])
                _delta_h_top_m = int(max(1, min(len(_cheap_rank), int(st.session_state.get("t52_delta_h_top_m", 40)))))
                _delta_h_cands: dict[tuple[float, float], tuple[float, float, float]] = {}
                for _score_v, _g_v, _a_v, _val_v, _bonus_v in _cheap_rank[:_delta_h_top_m]:
                    _delta_h_cands[(round(_g_v, 8), round(_a_v, 8))] = (_score_v, _val_v, _bonus_v)
                for _g_s, _a_s in _structural_cands:
                    for _score_v, _g_v, _a_v, _val_v, _bonus_v in _cheap_rank:
                        if abs(_g_v - _g_s) < 1e-9 and abs(_a_v - _a_s) < 1e-9:
                            _delta_h_cands[(round(_g_v, 8), round(_a_v, 8))] = (_score_v, _val_v, _bonus_v)
                            break

                _rank_with_info: list[tuple[float, float, float, float, float]] = []
                for _score_v, _g_v, _a_v, _val_v, _bonus_v in _cheap_rank:
                    _key_v = (round(_g_v, 8), round(_a_v, 8))
                    if _key_v in _delta_h_cands and float(_t52_entropy_weight) > 0.0 and info_bonus_func is not None:
                        try:
                            _bonus_v = float(info_bonus_func(float(_a_v), float(_g_v)).get("Delta_H", 0.0))
                        except Exception:
                            _bonus_v = 0.0
                        _score_v = float(_val_v - float(_t52_entropy_weight) * _bonus_v)
                    _rank_with_info.append((float(_score_v), float(_g_v), float(_a_v), float(_val_v), float(_bonus_v)))

                _rank_with_info.sort(key=lambda x: x[0])
                _top_k = int(max(1, min(len(_cheap_rank), int(st.session_state.get("t52_iric_top_k", 25)))))
                _audit_cands: dict[tuple[float, float], tuple[float, float]] = {}
                for _score_v, _g_v, _a_v, _val_v, _bonus_v in _rank_with_info[:_top_k]:
                    _audit_cands[(round(_g_v, 8), round(_a_v, 8))] = (_score_v, _val_v)
                for _g_s, _a_s in _structural_cands:
                    for _score_v, _g_v, _a_v, _val_v, _bonus_v in _rank_with_info:
                        if abs(_g_v - _g_s) < 1e-9 and abs(_a_v - _a_s) < 1e-9:
                            _audit_cands[(round(_g_v, 8), round(_a_v, 8))] = (_score_v, _val_v)
                            break

                _best_any: Optional[tuple[float, float, float, dict[str, Any]]] = None
                _best_feas: Optional[tuple[float, float, float, dict[str, Any]]] = None
                for (_g_v, _a_v), (_score_v, _val_v) in _audit_cands.items():
                    _other_v = float(other_value_func(_g_v, _a_v))
                    _st_v = _t52_iric_status(_a_v, _g_v, mu_tau_v, _other_v, _val_v)
                    _st_v["iric_audit_top_k"] = int(_top_k)
                    _st_v["iric_audited_candidates"] = int(len(_audit_cands))
                    _st_v["iric_total_candidates"] = int(len(_cheap_rank))
                    _st_v["grid_mode"] = "adaptive_11x11_top5_local3x3"
                    _st_v["grid_coarse_candidates"] = int(len(_coarse_cands))
                    _st_v["grid_refine_centers"] = int(len(_refine_centers))
                    _st_v["delta_h_top_m"] = int(_delta_h_top_m)
                    _st_v["delta_h_evaluated_candidates"] = int(len(_delta_h_cands))
                    _st_v["iric_audit_mode"] = "top_k"
                    _rec_v = (float(_a_v), float(_g_v), float(_score_v), _st_v)
                    if _best_any is None or _score_v < _best_any[2]:
                        _best_any = _rec_v
                    if bool(_st_v.get("feasible", False)) and (_best_feas is None or _score_v < _best_feas[2]):
                        _best_feas = _rec_v
                if _best_feas is None:
                    for _score_v, _g_v, _a_v, _val_v, _bonus_v in _rank_with_info[_top_k:]:
                        _key_v = (round(_g_v, 8), round(_a_v, 8))
                        if _key_v in _audit_cands:
                            continue
                        _other_v = float(other_value_func(_g_v, _a_v))
                        _st_v = _t52_iric_status(_a_v, _g_v, mu_tau_v, _other_v, _val_v)
                        _st_v["iric_audit_top_k"] = int(_top_k)
                        _st_v["iric_audited_candidates"] = int(len(_audit_cands) + 1)
                        _st_v["iric_total_candidates"] = int(len(_cheap_rank))
                        _st_v["grid_mode"] = "adaptive_11x11_top5_local3x3"
                        _st_v["grid_coarse_candidates"] = int(len(_coarse_cands))
                        _st_v["grid_refine_centers"] = int(len(_refine_centers))
                        _st_v["delta_h_top_m"] = int(_delta_h_top_m)
                        _st_v["delta_h_evaluated_candidates"] = int(len(_delta_h_cands))
                        _st_v["iric_audit_mode"] = "top_k_fallback"
                        _rec_v = (float(_a_v), float(_g_v), float(_score_v), _st_v)
                        if _best_any is None or _score_v < _best_any[2]:
                            _best_any = _rec_v
                        if bool(_st_v.get("feasible", False)):
                            _best_feas = _rec_v
                            break
                # Γ_t(μ_t) = ∅ si no existe punto con IR^K, IC^K e IR^F.
                # En ese caso se devuelve el mínimo irrestricto solo como diagnóstico,
                # marcado con feasible=False para no tratarlo como óptimo formal.
                return _best_feas if _best_feas is not None else _best_any  # type: ignore[return-value]

            def _t52_p_kill_exp_at(
                alpha: float,
                gamma: float,
                mu: dict[str, float],
                tau: int,
                a_k_exec: str,
                a_f_exec: str,
            ) -> float:
                """h_2(τ|θ,C_τ(α,γ)) esperado bajo μ via hazards competitivos (Mechanism.tex).
                Evalúa λ_2 en el candidato (α,γ) con p_det(α,γ)=Λ(η₀+η₁α+η₂γ).
                ã_S no entra en λ_2; se fija en 'No Rescatar' (rama VN)."""
                _result = 0.0
                _tau_eff = int(max(1, int(tau)))
                for _th in TIPOS_SECUESTRADOR:
                    _w = float(mu.get(_th, 0.0))
                    if _w < 1e-12:
                        continue
                    _p_det_cand = float(max(
                        0.0,
                        min(0.99, _pdet_logit_prob(str(_th), float(alpha), float(gamma))),
                    ))
                    try:
                        _probs, _ = _mechanism_m_probs_for_actions(
                            str(_th),
                            _tau_eff,
                            float(alpha),
                            float(gamma),
                            _p_det_cand,
                            str(a_k_exec),
                            "No Rescatar",
                            str(a_f_exec),
                            z_region=str(st.session_state.get("z_region", "")),
                            v_victim=str(st.session_state.get("v_victim", "")),
                            f_capa=str(st.session_state.get("f_capa", "Alta Riqueza")),
                            s_tipo=str("Duro" if st.session_state.get("tab2_estado_duro", True) else "Laxo"),
                        )
                        _result += _w * float(_probs.get("Muerte", 0.0))
                    except Exception:
                        pass
                return float(_result)

            def _t52_perfect_info_state_reference(
                theta_ref_v: str,
                tau_v: int,
                gamma_prev_v: float,
                a_k_exec_v: str,
                a_s_exec_v: str,
                a_f_exec_v: str,
            ) -> dict[str, Any]:
                theta_ref = str(theta_ref_v)
                mu_pi = {th: (1.0 if th == theta_ref else 0.0) for th in TIPOS_SECUESTRADOR}
                iota_pi = 1.0
                psurv_pi = float(_p_surv_precision_logit(theta_ref, iota_pi, theta_ref))
                ops_pi = _state_weighted_cost_tuple(_p3_ops_by_type, mu_pi)
                mt_pi = _state_weighted_cost_tuple(_p3_mt_by_type, mu_pi)
                ref_pi = _state_reference_centers(mu_pi)
                alpha_vr, gamma_vr, v_vr, _iric_vr_pi = _t52_minimize_state_quadratic_iric(
                    const=float(
                        _p3_omk * (1.0 - psurv_pi)
                        + ops_pi[0]
                        + _p3_chi_alpha * ref_pi["alpha_R_mu"] ** 2
                        + _p3_chi_gamma * ref_pi["gamma_R_mu"] ** 2
                    ),
                    b_gamma=float(ops_pi[1] - 2.0 * _p3_chi_gamma * ref_pi["gamma_R_mu"]),
                    q_gamma=float(ops_pi[2] + 2.0 * _p3_chi_gamma),
                    b_alpha=float(ops_pi[3] - 2.0 * _p3_chi_alpha * ref_pi["alpha_R_mu"]),
                    q_alpha=float(ops_pi[4] + 2.0 * _p3_chi_alpha),
                    q_gamma_alpha=float(ops_pi[5]),
                    mu_tau_v=mu_pi,
                    other_value_func=lambda _g, _a: 0.0,
                    info_bonus_func=None,
                )

                def _vr_pi_val(_g: float, _a: float) -> float:
                    return _t52_quad_value(
                        _g,
                        _a,
                        float(
                            _p3_omk * (1.0 - psurv_pi)
                            + ops_pi[0]
                            + _p3_chi_alpha * ref_pi["alpha_R_mu"] ** 2
                            + _p3_chi_gamma * ref_pi["gamma_R_mu"] ** 2
                        ),
                        float(ops_pi[1] - 2.0 * _p3_chi_gamma * ref_pi["gamma_R_mu"]),
                        float(ops_pi[2] + 2.0 * _p3_chi_gamma),
                        float(ops_pi[3] - 2.0 * _p3_chi_alpha * ref_pi["alpha_R_mu"]),
                        float(ops_pi[4] + 2.0 * _p3_chi_alpha),
                        float(ops_pi[5]),
                    )

                _tau_pi = int(max(1, int(tau_v)))
                alpha_vn, gamma_vn, v_vn, iric_vn = _t52_minimize_state_quadratic_iric(
                    const=float(
                        _p3_omp * R_escala
                        + mt_pi[0]
                        + _p3_chi_alpha * ref_pi["alpha_N_mu"] ** 2
                        + _p3_chi_gamma * ref_pi["gamma_N_mu"] ** 2
                    ),
                    b_gamma=float(mt_pi[1] - 2.0 * _p3_chi_gamma * ref_pi["gamma_N_mu"]),
                    q_gamma=float(mt_pi[2] + 2.0 * _p3_chi_gamma),
                    b_alpha=float(mt_pi[3] - _p3_omp * R_escala - 2.0 * _p3_chi_alpha * ref_pi["alpha_N_mu"]),
                    q_alpha=float(mt_pi[4] + 2.0 * _p3_chi_alpha),
                    q_gamma_alpha=float(mt_pi[5]),
                    mu_tau_v=mu_pi,
                    other_value_func=_vr_pi_val,
                    info_bonus_func=None,
                    extra_score_fn=lambda _av, _gv: _p3_omk * _t52_p_kill_exp_at(
                        _av, _gv, mu_pi, _tau_pi, str(a_k_exec_v), str(a_f_exec_v)
                    ),
                )
                rescue_pi = bool(float(v_vr) <= float(v_vn))
                _pkill_pi_rep = _t52_p_kill_exp_at(
                    float(alpha_vn if not rescue_pi else alpha_vr),
                    float(gamma_vn if not rescue_pi else gamma_vr),
                    mu_pi, _tau_pi, str(a_k_exec_v), str(a_f_exec_v),
                )
                return {
                    "theta": theta_ref,
                    "mu": mu_pi,
                    "branch": "VR" if rescue_pi else "VN",
                    "a_s_full": "Rescate (a_res)" if rescue_pi else "Negociar (a_neg)",
                    "alpha_star": float(alpha_vr if rescue_pi else alpha_vn),
                    "gamma_star": float(gamma_vr if rescue_pi else gamma_vn),
                    "alpha_vr": float(alpha_vr),
                    "gamma_vr": float(gamma_vr),
                    "V_R": float(v_vr),
                    "alpha_vn": float(alpha_vn),
                    "gamma_vn": float(gamma_vn),
                    "V_N": float(v_vn),
                    "p_surv": float(psurv_pi),
                    "p_kill": float(_pkill_pi_rep),
                    "IRIC_VR": dict(_iric_vr_pi or {}),
                    "IRIC_VN": dict(iric_vn or {}),
                }

            def _t52_blank_pi_reference(theta_ref_v: str, tau_v: int) -> dict[str, Any]:
                return {
                    "theta": str(theta_ref_v),
                    "tau": int(tau_v),
                    "skipped": True,
                    "branch": "—",
                    "a_s_full": "—",
                    "alpha_star": float("nan"),
                    "gamma_star": float("nan"),
                    "alpha_vr": float("nan"),
                    "gamma_vr": float("nan"),
                    "V_R": float("nan"),
                    "alpha_vn": float("nan"),
                    "gamma_vn": float("nan"),
                    "V_N": float("nan"),
                    "p_surv": float("nan"),
                    "p_kill": float("nan"),
                    "IRIC_VR": {},
                    "IRIC_VN": {},
                }

            def _t52_reuse_pi_reference(pi_ref_v: dict[str, Any], tau_v: int) -> dict[str, Any]:
                _ref = copy.deepcopy(dict(pi_ref_v))
                _ref["tau"] = int(tau_v)
                _ref["reused_from_tau"] = int(_ref.get("source_tau", _ref.get("tau_source", 1)) or 1)
                _ref["source_tau"] = int(_ref.get("reused_from_tau", 1) or 1)
                _ref["skipped"] = False
                return _ref

            def _t52_pi_display(pi_ref_v: dict[str, Any], key_v: str) -> str:
                try:
                    _val = float(dict(pi_ref_v).get(key_v, float("nan")))
                    return f"{_val:.4f}" if np.isfinite(_val) else "—"
                except Exception:
                    return "—"

            _theta_hat_link52 = max(_mu0_52, key=_mu0_52.get) if _mu0_52 else str(tipo_real)
            _iota_link52 = float(max(_mu0_52.values())) if _mu0_52 else float(_iota_52)
            _p_kill_by_type_link52: dict[str, float] = {}
            _p_surv_by_type_link52: dict[str, float] = {}
            for _th_link52 in TIPOS_SECUESTRADOR:
                _p_surv_by_type_link52[_th_link52] = _p_surv_precision_logit(
                    _th_link52, _iota_link52, _theta_hat_link52
                )
                _p_kill_by_type_link52[_th_link52] = float(
                    _outcome_probs_for_actions(
                        str(_th_link52),
                        float(_t0_gamma_eff),
                        float(_iota_link52),
                        _atk52_real,
                        _s_exec52,
                        _atf52,
                    )["kill"]
                )
            _p_kill_exp_link52 = float(
                sum(
                    float(_mu1_52.get(_th_link52, 0.0))
                    * float(_p_kill_by_type_link52.get(_th_link52, 0.0))
                    for _th_link52 in TIPOS_SECUESTRADOR
                )
            )
            _p_surv_exp_link52 = float(
                sum(
                    float(_mu1_52.get(_th_link52, 0.0))
                    * float(_p_surv_by_type_link52.get(_th_link52, 0.0))
                    for _th_link52 in TIPOS_SECUESTRADOR
                )
            )
            _ops_mu1_link52 = _state_weighted_cost_tuple(_p3_ops_by_type, _mu1_52)
            _mt_mu1_link52 = _state_weighted_cost_tuple(_p3_mt_by_type, _mu1_52)
            _ref_mu1_link52 = _state_reference_centers(_mu1_52)
            _ref_mu0_link52 = _state_reference_centers(_mu0_52)

            def _info_gain_vr_link52(alpha_v: float, gamma_v: float) -> dict[str, Any]:
                return _t52_expected_entropy_gain(
                    _mu1_52,
                    1,
                    float(alpha_v),
                    float(gamma_v),
                    _atk52_real,
                    "Rescatar",
                    _atf52,
                )

            def _info_gain_vn_link52(alpha_v: float, gamma_v: float) -> dict[str, Any]:
                return _t52_expected_entropy_gain(
                    _mu1_52,
                    1,
                    float(alpha_v),
                    float(gamma_v),
                    _atk52_real,
                    "No Rescatar",
                    _atf52,
                )

            _alpha_vr_link52, _gamma_vr_link52, _vstar_vr_link52, _iric_vr_link52 = _t52_minimize_state_quadratic_iric(
                const=float(
                    _p3_omk * (1.0 - _p_surv_exp_link52)
                    + _ops_mu1_link52[0]
                    + _p3_chi_alpha * _ref_mu1_link52["alpha_R_mu"] ** 2
                    + _p3_chi_gamma * _ref_mu1_link52["gamma_R_mu"] ** 2
                ),
                b_gamma=float(_ops_mu1_link52[1] - 2.0 * _p3_chi_gamma * _ref_mu1_link52["gamma_R_mu"]),
                q_gamma=float(_ops_mu1_link52[2] + 2.0 * _p3_chi_gamma),
                b_alpha=float(_ops_mu1_link52[3] - 2.0 * _p3_chi_alpha * _ref_mu1_link52["alpha_R_mu"]),
                q_alpha=float(_ops_mu1_link52[4] + 2.0 * _p3_chi_alpha),
                q_gamma_alpha=float(_ops_mu1_link52[5]),
                mu_tau_v=_mu1_52,
                other_value_func=lambda _g, _a: 0.0,
                info_bonus_func=_info_gain_vr_link52,
            )

            def _vr_val_link52(_g: float, _a: float) -> float:
                return _t52_quad_value(
                    _g,
                    _a,
                    float(
                        _p3_omk * (1.0 - _p_surv_exp_link52)
                        + _ops_mu1_link52[0]
                        + _p3_chi_alpha * _ref_mu1_link52["alpha_R_mu"] ** 2
                        + _p3_chi_gamma * _ref_mu1_link52["gamma_R_mu"] ** 2
                    ),
                    float(_ops_mu1_link52[1] - 2.0 * _p3_chi_gamma * _ref_mu1_link52["gamma_R_mu"]),
                    float(_ops_mu1_link52[2] + 2.0 * _p3_chi_gamma),
                    float(_ops_mu1_link52[3] - 2.0 * _p3_chi_alpha * _ref_mu1_link52["alpha_R_mu"]),
                    float(_ops_mu1_link52[4] + 2.0 * _p3_chi_alpha),
                    float(_ops_mu1_link52[5]),
                )

            _alpha_vn_link52, _gamma_vn_link52, _vstar_vn_link52, _iric_vn_link52 = _t52_minimize_state_quadratic_iric(
                const=float(
                    _p3_omp * R_escala
                    + _mt_mu1_link52[0]
                    + _p3_chi_alpha * _ref_mu1_link52["alpha_N_mu"] ** 2
                    + _p3_chi_gamma * _ref_mu1_link52["gamma_N_mu"] ** 2
                ),
                b_gamma=float(_mt_mu1_link52[1] - 2.0 * _p3_chi_gamma * _ref_mu1_link52["gamma_N_mu"]),
                q_gamma=float(_mt_mu1_link52[2] + 2.0 * _p3_chi_gamma),
                b_alpha=float(_mt_mu1_link52[3] - _p3_omp * R_escala - 2.0 * _p3_chi_alpha * _ref_mu1_link52["alpha_N_mu"]),
                q_alpha=float(_mt_mu1_link52[4] + 2.0 * _p3_chi_alpha),
                q_gamma_alpha=float(_mt_mu1_link52[5]),
                mu_tau_v=_mu1_52,
                other_value_func=_vr_val_link52,
                info_bonus_func=_info_gain_vn_link52,
                extra_score_fn=lambda _av, _gv: _p3_omk * _t52_p_kill_exp_at(
                    _av, _gv, _mu1_52, 1, _atk52_real, _atf52
                ),
            )
            _vr_formal_link52 = bool(dict(_iric_vr_link52 or {}).get("feasible", False))
            _vn_formal_link52 = bool(dict(_iric_vn_link52 or {}).get("feasible", False))
            if _vr_formal_link52 and _vn_formal_link52:
                _vr_lt_vn = bool(float(_vstar_vr_link52) <= float(_vstar_vn_link52))
            elif _vr_formal_link52:
                _vr_lt_vn = True
            elif _vn_formal_link52:
                _vr_lt_vn = False
            else:
                _vr_lt_vn = bool(float(_vstar_vr_link52) <= float(_vstar_vn_link52))
            _gamma_formal_link52 = bool(_vr_formal_link52 or _vn_formal_link52)
            _s_rule_p3 = (
                "Rescate"
                if _gamma_formal_link52 and _vr_lt_vn
                else ("Negociar" if _gamma_formal_link52 else "Γ vacío")
            )
            _s52_star = (
                "Rescatar"
                if _gamma_formal_link52 and _vr_lt_vn
                else ("No Rescatar" if _gamma_formal_link52 else "Γ vacío")
            )
            _s52_mdg_intent = "Rescatar" if _vr_lt_vn else "No Rescatar"
            _s52_full = (
                "Rescate (a_res)"
                if _gamma_formal_link52 and _vr_lt_vn
                else ("Negociar (a_neg)" if _gamma_formal_link52 else "Γ vacío (sin óptimo formal)")
            )
            _gamma_state_star52 = float(_gamma_vr_link52 if _vr_lt_vn else _gamma_vn_link52)
            _alpha_state_star52 = float(_alpha_vr_link52 if _vr_lt_vn else _alpha_vn_link52)
            _gamma_state_t0_52 = float(_t0_gamma_eff)
            _alpha_state_t0_52 = float(_t0_alpha_eff)
            _state_star_branch52 = "VR" if _vr_lt_vn else "VN"
            _info_vr_link52 = _info_gain_vr_link52(float(_alpha_vr_link52), float(_gamma_vr_link52))
            _info_vn_link52 = _info_gain_vn_link52(float(_alpha_vn_link52), float(_gamma_vn_link52))
            _info_sel_link52 = _info_vr_link52 if _vr_lt_vn else _info_vn_link52
            _pi_ref0_52 = _t52_perfect_info_state_reference(
                str(tipo_real),
                0,
                float(_t0_gamma_eff),
                str(_atk0_m52),
                str(_ats0_m52),
                str(_atf0_m52),
            )
            _pi_ref1_52 = _t52_perfect_info_state_reference(
                str(tipo_real),
                1,
                float(_t0_gamma_eff),
                str(_atk52_real),
                str(_s_exec52),
                str(_atf52),
            )
            _gamma_r_pi_label52 = rf"γ_R^{{{tipo_real},*}}"
            _alpha_r_pi_label52 = rf"α_R^{{{tipo_real},*}}"
            _gamma_n_pi_label52 = rf"γ_N^{{{tipo_real},*}}"
            _alpha_n_pi_label52 = rf"α_N^{{{tipo_real},*}}"
            _t52_vals["a_S* óptima"] = _s52_full
            _ps52 = _t52_p1_s(_agent52, _s52_mdg_intent, _mu0_52, _mu1_52, tau=1)
            _ats52_argmax = max(_ps52, key=_ps52.get)
            _ats52 = _t52_realize(_ps52, _u52_s)
            _s_exec52 = "Rescatar" if str(_ats52).strip().lower().startswith("rescat") else "No Rescatar"
            _t52_vals["ã_S"] = f"{_ats52}  (u={_u52_s:.4f}, p={_ps52[_ats52]:.4f})"
            _t52_vals["γ* Estado"] = f"{float(_gamma_state_star52):.4f}"
            _t52_vals["α* Estado"] = f"{float(_alpha_state_star52):.4f}"
            _t52_vals[_gamma_r_pi_label52] = f"{float(_pi_ref1_52['gamma_vr']):.4f}"
            _t52_vals[_alpha_r_pi_label52] = f"{float(_pi_ref1_52['alpha_vr']):.4f}"
            _t52_vals[_gamma_n_pi_label52] = f"{float(_pi_ref1_52['gamma_vn']):.4f}"
            _t52_vals[_alpha_n_pi_label52] = f"{float(_pi_ref1_52['alpha_vn']):.4f}"
            _t52_vals["H(μ)"] = f"{float(_info_sel_link52.get('H', _t52_shannon_entropy(_mu1_52))):.4f}"
            _t52_vals["ΔH Estado"] = f"{float(_info_sel_link52.get('Delta_H', 0.0)):.4f}"
            _t52_vals0["γ* Estado"] = f"{_gamma_state_t0_52:.4f}"
            _t52_vals0["α* Estado"] = f"{_alpha_state_t0_52:.4f}"
            _t52_vals0[_gamma_r_pi_label52] = f"{float(_pi_ref0_52['gamma_vr']):.4f}"
            _t52_vals0[_alpha_r_pi_label52] = f"{float(_pi_ref0_52['alpha_vr']):.4f}"
            _t52_vals0[_gamma_n_pi_label52] = f"{float(_pi_ref0_52['gamma_vn']):.4f}"
            _t52_vals0[_alpha_n_pi_label52] = f"{float(_pi_ref0_52['alpha_vn']):.4f}"
            _t52_vals0["H(μ)"] = f"{float(_t52_shannon_entropy(_mu0_52)):.4f}"
            _t52_vals0["ΔH Estado"] = "—"
            _policy1_52 = _t52_screening_policy(
                1,
                float(_alpha_state_star52),
                float(_gamma_state_star52),
            )
            _eu_s52 = (
                _ps52.get("Rescatar", 0.0) * (-float(_vstar_vr_link52))
                + _ps52.get("No Rescatar", 0.0) * (-float(_vstar_vn_link52))
            )
            _t52_tips["a_S* óptima"] = (
                r'<div>Resultado vinculado a Tabla 5.3: '
                r'\(a_S^*=\mathrm{Rescate}\) si \(V_R^*\le V_N^*\); '
                r'\(a_S^*=\mathrm{Negociar}\) si no.</div>'
                rf'<div>\(V_R^*={float(_vstar_vr_link52):.4f}\), '
                rf'\(V_N^*={float(_vstar_vn_link52):.4f}\).</div>'
            )
            _t52_gamma_disp = float(_gamma_state_star52)
            _t52_alpha_disp = float(_alpha_state_star52)
            _t52_t53_is_rescue = bool(_vr_lt_vn)
            _t52_t53_branch = "VR" if _t52_t53_is_rescue else "VN"
            _t52_t53_asstar = "Rescate" if _t52_t53_is_rescue else "Negociar"
            _t52_t53_vvr = float(_vstar_vr_link52)
            _t52_t53_vvn = float(_vstar_vn_link52)
            _t52_t53_gvr = float(_gamma_vr_link52)
            _t52_t53_avr = float(_alpha_vr_link52)
            _t52_t53_gvn = float(_gamma_vn_link52)
            _t52_t53_avn = float(_alpha_vn_link52)
            _t52_tips["γ* Estado"] = (
                rf'<div>τ=0: \(\gamma_0={_gamma_state_t0_52:.4f}\) (valor inicial de la sesión).</div>'
                rf'<div>τ=1: \(\gamma^*={_t52_gamma_disp:.4f}\) · Fuente: Tabla 5.3 τ=1.</div>'
                rf'<div>Regla: \(a_S^*=\mathrm{{{_t52_t53_asstar}}}\) porque '
                rf'\(V_R^*={_t52_t53_vvr:.4f}\) {"≤" if _t52_t53_is_rescue else ">"} '
                rf'\(V_N^*={_t52_t53_vvn:.4f}\).</div>'
                rf'<div>Rama {_t52_t53_branch}: '
                rf'\(\gamma_{{VR}}^*={_t52_t53_gvr:.4f},\ \gamma_{{VN}}^*={_t52_t53_gvn:.4f}\).</div>'
                rf'<div style="font-size:0.85em;color:#9ab;">argmin V_{{{_t52_t53_branch}}}(\gamma,\alpha) '
                rf'sobre \([0,1]^2\)'
                + (r' sin IR/IC.' if _t52_t53_is_rescue else r' s.t. IR/IC.')
                + rf' Objetivo: pérdida - \(\psi_H\Delta H\), \(\psi_H={_t52_entropy_weight:.1f}\), '
                + rf'\(\Delta H={float(_info_sel_link52.get("Delta_H", 0.0)):.4f}\).'
                + r'</div>'
            )
            _t52_tips["α* Estado"] = (
                rf'<div>τ=0: \(\alpha_0={_alpha_state_t0_52:.4f}\) (valor inicial de la sesión).</div>'
                rf'<div>τ=1: \(\alpha^*={_t52_alpha_disp:.4f}\) · Fuente: Tabla 5.3 τ=1.</div>'
                rf'<div>Regla: \(a_S^*=\mathrm{{{_t52_t53_asstar}}}\) porque '
                rf'\(V_R^*={_t52_t53_vvr:.4f}\) {"≤" if _t52_t53_is_rescue else ">"} '
                rf'\(V_N^*={_t52_t53_vvn:.4f}\).</div>'
                rf'<div>Rama {_t52_t53_branch}: '
                rf'\(\alpha_{{VR}}^*={_t52_t53_avr:.4f},\ \alpha_{{VN}}^*={_t52_t53_avn:.4f}\).</div>'
                rf'<div style="font-size:0.85em;color:#9ab;">argmin V_{{{_t52_t53_branch}}}(\gamma,\alpha) '
                rf'sobre \([0,1]^2\)'
                + (r' sin IR/IC.' if _t52_t53_is_rescue else r' s.t. IR/IC.')
                + rf' Objetivo: pérdida - \(\psi_H\Delta H\), \(\psi_H={_t52_entropy_weight:.1f}\), '
                + rf'\(\Delta H={float(_info_sel_link52.get("Delta_H", 0.0)):.4f}\).'
                + r'</div>'
            )
            _t52_pi_r_tip1 = (
                rf'<div>Benchmark de información perfecta para el tipo incidente '
                rf'\(\theta^\ast=\mathrm{{{html.escape(str(tipo_real))}}}\).</div>'
                rf'<div>\(\mu^{{PI}}(\theta^\ast)=1\), \(\mu^{{PI}}(\theta\ne\theta^\ast)=0\).</div>'
                rf'<div>Rama \(R\) rescate: \(V_R^{{PI}}={float(_pi_ref1_52["V_R"]):.4f}\), '
                rf'\((\alpha_R^{{\theta^\ast,*}},\gamma_R^{{\theta^\ast,*}})'
                rf'=({float(_pi_ref1_52["alpha_vr"]):.4f},{float(_pi_ref1_52["gamma_vr"]):.4f})\).</div>'
                r'<div>Referencia por rama; no alimenta Tabla 10 ni la posterior.</div>'
            )
            _t52_pi_n_tip1 = (
                rf'<div>Benchmark de información perfecta para el tipo incidente '
                rf'\(\theta^\ast=\mathrm{{{html.escape(str(tipo_real))}}}\).</div>'
                rf'<div>\(\mu^{{PI}}(\theta^\ast)=1\), \(\mu^{{PI}}(\theta\ne\theta^\ast)=0\).</div>'
                rf'<div>Rama \(N\) negociación: \(V_N^{{PI}}={float(_pi_ref1_52["V_N"]):.4f}\), '
                rf'\((\alpha_N^{{\theta^\ast,*}},\gamma_N^{{\theta^\ast,*}})'
                rf'=({float(_pi_ref1_52["alpha_vn"]):.4f},{float(_pi_ref1_52["gamma_vn"]):.4f})\).</div>'
                r'<div>Referencia por rama; no alimenta Tabla 10 ni la posterior.</div>'
            )
            _t52_pi_r_tip0 = (
                rf'<div>Benchmark τ=0 de información perfecta para '
                rf'\(\theta^\ast=\mathrm{{{html.escape(str(tipo_real))}}}\).</div>'
                rf'<div>Rama \(R\): \(V_R^{{PI}}={float(_pi_ref0_52["V_R"]):.4f}\), '
                rf'\((\alpha_R^{{\theta^\ast,*}},\gamma_R^{{\theta^\ast,*}})'
                rf'=({float(_pi_ref0_52["alpha_vr"]):.4f},{float(_pi_ref0_52["gamma_vr"]):.4f})\).</div>'
                r'<div>Referencia sin incertidumbre; no modifica probabilidades ni posterior.</div>'
            )
            _t52_pi_n_tip0 = (
                rf'<div>Benchmark τ=0 de información perfecta para '
                rf'\(\theta^\ast=\mathrm{{{html.escape(str(tipo_real))}}}\).</div>'
                rf'<div>Rama \(N\): \(V_N^{{PI}}={float(_pi_ref0_52["V_N"]):.4f}\), '
                rf'\((\alpha_N^{{\theta^\ast,*}},\gamma_N^{{\theta^\ast,*}})'
                rf'=({float(_pi_ref0_52["alpha_vn"]):.4f},{float(_pi_ref0_52["gamma_vn"]):.4f})\).</div>'
                r'<div>Referencia sin incertidumbre; no modifica probabilidades ni posterior.</div>'
            )
            _t52_tips[_gamma_r_pi_label52] = _t52_pi_r_tip1
            _t52_tips[_alpha_r_pi_label52] = _t52_pi_r_tip1
            _t52_tips[_gamma_n_pi_label52] = _t52_pi_n_tip1
            _t52_tips[_alpha_n_pi_label52] = _t52_pi_n_tip1
            _t52_tips0[_gamma_r_pi_label52] = _t52_pi_r_tip0
            _t52_tips0[_alpha_r_pi_label52] = _t52_pi_r_tip0
            _t52_tips0[_gamma_n_pi_label52] = _t52_pi_n_tip0
            _t52_tips0[_alpha_n_pi_label52] = _t52_pi_n_tip0
            _t52_tips["H(μ)"] = (
                r'<div>Entropía de Shannon de la creencia vigente:</div>'
                r'<div>\(H(\mu_t)=-\sum_{\theta}\mu_t(\theta)\ln\mu_t(\theta)\).</div>'
                rf'<div>τ=1: \(H(\mu_1)={float(_info_sel_link52.get("H", 0.0)):.4f}\).</div>'
            )
            _t52_tips["ΔH Estado"] = (
                r'<div>Ganancia esperada de información usada en el problema del Estado:</div>'
                r'<div>\(\Delta H=H(\mu_t)-\mathbb E_t[H(\mu_{t+1})]\).</div>'
                rf'<div>\(E[H(\mu_2)]={float(_info_sel_link52.get("E_H_next", 0.0)):.4f}\), '
                rf'\(\Delta H={float(_info_sel_link52.get("Delta_H", 0.0)):.4f}\), '
                rf'\(\psi_H={_t52_entropy_weight:.1f}\).</div>'
                r'<div>La celda óptima minimiza pérdida social menos \(\psi_H\Delta H\); usa directamente \(\alpha_t^*,\gamma_t^*\).</div>'
            )
            _t52_gamma_row_label = "Cumple Γ_t(μ_t) bajo EV"
            _t52_ir_true_row_label = f"Cumple IR^K({tipo_real}) tipo verdadero"

            def _t52_yes_no(v: Any) -> str:
                return "Sí" if bool(v) else "No"

            def _t52_gamma_tip(iric_v: dict[str, Any], tau_v: int) -> str:
                _formal_v = bool(iric_v.get("Gamma_formal", False))
                return (
                    f'<div class="t52h">'
                    f'<div class="hdr">Verificación formal bajo valor esperado · τ={int(tau_v)}</div>'
                    f'<div class="sec">Criterio</div>'
                    f'<div>\\(\\Gamma_t(\\mu_t)=1 \\Longleftrightarrow IR^K \\land IC^K \\land IR^F\\).</div>'
                    f'<div class="sec">Resultado</div>'
                    f'<div><b>{_t52_yes_no(_formal_v)}</b></div>'
                    f'<div class="sec">Gaps esperados</div>'
                    f'<div>\\(IR^K_E={float(iric_v.get("IR_K_gap_E", float("nan"))):.4f}\\), '
                    f'\\(IC^K_{{E,\\min}}={float(iric_v.get("IC_K_gap_E_min", float("nan"))):.4f}\\), '
                    f'\\(IR^F_E={float(iric_v.get("IR_F_gap_E", float("nan"))):.4f}\\).</div>'
                    f'<div class="ux">La fila audita el candidato elegido por el Estado; no reoptimiza ni cambia la trayectoria.</div>'
                    f'</div>'
                )

            def _t52_ir_true_tip(iric_v: dict[str, Any], tau_v: int, alpha_v: float, gamma_v: float) -> str:
                _ok_v = bool(iric_v.get("IR_K_true", False))
                return (
                    f'<div class="t52h">'
                    f'<div class="hdr">IR puntual del tipo verdadero · τ={int(tau_v)}</div>'
                    f'<div class="sec">Criterio</div>'
                    f'<div>\\(IR^K(\\theta^*)=1 \\Longleftrightarrow '
                    f'U^K_{{rel}}(\\theta^*)-\\max\\{{V^K_{{cont}}(\\theta^*),U^K_{{kill}}(\\theta^*)\\}}\\ge 0\\).</div>'
                    f'<div class="sec">Resultado</div>'
                    f'<div><b>{_t52_yes_no(_ok_v)}</b> para \\(\\theta^*=\\mathrm{{{html.escape(str(tipo_real))}}}\\).</div>'
                    f'<div class="sec">Instrumentos usados por el Estado</div>'
                    f'<div>\\(\\alpha_t={float(alpha_v):.4f}\\), \\(\\gamma_t={float(gamma_v):.4f}\\).</div>'
                    f'<div class="sec">Utilidades del tipo verdadero</div>'
                    f'<div>\\(U^K_{{rel}}={float(iric_v.get("IR_K_true_U_rel", float("nan"))):.4f}\\), '
                    f'\\(V^K_{{cont}}={float(iric_v.get("IR_K_true_V_cont", float("nan"))):.4f}\\), '
                    f'\\(U^K_{{kill}}={float(iric_v.get("IR_K_true_U_kill", float("nan"))):.4f}\\).</div>'
                    f'<div class="sec">Gap puntual</div>'
                    f'<div>\\(IR^K(\\theta^*)={float(iric_v.get("IR_K_true_gap", float("nan"))):.4f}\\).</div>'
                    f'<div class="ux">Auditoría puntual; no reemplaza la verificación bajo valor esperado.</div>'
                    f'</div>'
                )

            _t52_iric_tau0 = _t52_iric_status(
                float(_alpha_state_t0_52),
                float(_gamma_state_t0_52),
                dict(_mu0_52),
                float(_vr_p3),
                float(_vn_p3),
            )
            _t52_iric_tau1 = dict(_iric_vr_link52 if _vr_lt_vn else _iric_vn_link52)
            _t52_vals0[_t52_gamma_row_label] = _t52_yes_no(_t52_iric_tau0.get("Gamma_formal", False))
            _t52_vals[_t52_gamma_row_label] = _t52_yes_no(_t52_iric_tau1.get("Gamma_formal", False))
            _t52_tips0[_t52_gamma_row_label] = _t52_gamma_tip(_t52_iric_tau0, 0)
            _t52_tips[_t52_gamma_row_label] = _t52_gamma_tip(_t52_iric_tau1, 1)
            _t52_vals0[_t52_ir_true_row_label] = _t52_yes_no(_t52_iric_tau0.get("IR_K_true", False))
            _t52_vals[_t52_ir_true_row_label] = _t52_yes_no(_t52_iric_tau1.get("IR_K_true", False))
            # κh por tipo: τ=0 indeterminado (M(0)=0 → λ̃=0), τ=1 calculado con hazards del ciclo
            for _th52_kh in TIPOS_SECUESTRADOR:
                _kh_key52 = f"−sgn(κh) ({_th52_kh})"
                _t52_vals0[_kh_key52] = "—"
                _t52_tips0[_kh_key52] = (
                    f'<div class="t52h">'
                    f'<div class="hdr">−sgn(κ_h) · θ={html.escape(str(_th52_kh))} · τ=0</div>'
                    f'<div class="sec">Fórmula</div>'
                    f'<div>\\(\\kappa_h(\\theta,t)=\\zeta_\\gamma^{{(2)}}\\tilde{{\\lambda}}_2+'
                    f'\\zeta_\\gamma^{{(3)}}\\tilde{{\\lambda}}_3-\\zeta_\\gamma^{{(1)}}\\tilde{{\\lambda}}_1\\)</div>'
                    f'<div class="sec">Nota τ=0</div>'
                    f'<div>El filtro de maduración \\(M(0)=0\\) anula todas las intensidades efectivas '
                    f'\\(\\tilde{{\\lambda}}_j=0\\), por lo que \\(\\kappa_h\\) es indeterminado en el período inicial.</div>'
                    f'<div class="ux">El signo se reporta a partir de τ=1, cuando M(t)&gt;0.</div>'
                    f'</div>'
                )
                try:
                    _zbj_kh1 = (_tab2_bundles52.get(str(_th52_kh), {}) or {}).get("zeta_by_j", {}) or {}
                    _h_kh1 = modelo.calcular_hazards(
                        1,
                        str(_th52_kh),
                        float(_gamma_state_star52),
                        maturity_mult=float(_t52_M1),
                        z_region=str(st.session_state.get("z_region", "Andina")),
                        v_victim=str(st.session_state.get("v_victim", "Privado")),
                        alpha=float(_alpha_state_star52),
                        gamma=float(_gamma_state_star52),
                        zeta_by_j=_zbj_kh1,
                    )
                    _zg1_kh1 = float((_zbj_kh1.get("Pago")    or {}).get("gamma", 0.0))
                    _zg2_kh1 = float((_zbj_kh1.get("Muerte")  or {}).get("gamma", 0.0))
                    _zg3_kh1 = float((_zbj_kh1.get("Rescate") or {}).get("gamma", 0.0))
                    _l1_kh1  = float(_h_kh1.get("Pago",    0.0))
                    _l2_kh1  = float(_h_kh1.get("Muerte",  0.0))
                    _l3_kh1  = float(_h_kh1.get("Rescate", 0.0))
                    _kh1_val = _zg2_kh1 * _l2_kh1 + _zg3_kh1 * _l3_kh1 - _zg1_kh1 * _l1_kh1
                    _kh1_sgn = -1 if _kh1_val > 1e-12 else (1 if _kh1_val < -1e-12 else 0)
                    _t52_vals[_kh_key52] = str(_kh1_sgn)
                    if _kh1_sgn == -1:
                        _kh1_interp = (
                            "γ↑ empeora la posición del Estado: el efecto neto de la presión operativa "
                            "aumenta el peso relativo de causas de salida adversas (Muerte, Pago) "
                            "sobre la favorable (Rescate). El Estado paga un costo informacional "
                            "al subir γ sin lograr separación útil para este tipo."
                        )
                        _kh1_color = "#c0392b"
                    elif _kh1_sgn == 1:
                        _kh1_interp = (
                            "γ↑ mejora la posición del Estado: la presión operativa eleva "
                            "proporcionalmente más el hazard de Rescate que el de Pago o Muerte. "
                            "El Estado tiene incentivo a subir γ para acelerar la resolución favorable "
                            "y separar este tipo del resto."
                        )
                        _kh1_color = "#27ae60"
                    else:
                        _kh1_interp = (
                            "El efecto neto de γ sobre los hazards ponderados es nulo: las fuerzas "
                            "que aceleran Rescate y las que aceleran Pago/Muerte se cancelan exactamente. "
                            "El instrumento γ es informativamente neutro para este tipo en este período."
                        )
                        _kh1_color = "#7f8c8d"
                    _t52_tips[_kh_key52] = (
                        f'<div class="t52h">'
                        f'<div class="hdr">−sgn(κ_h) · θ={html.escape(str(_th52_kh))} · τ=1</div>'
                        f'<div class="sec">Fórmula</div>'
                        f'<div>\\(\\kappa_h(\\theta,t)='
                        f'\\underbrace{{\\zeta_\\gamma^{{(2)}}\\tilde{{\\lambda}}_2}}_{{\\text{{Muerte}}}}'
                        f'+\\underbrace{{\\zeta_\\gamma^{{(3)}}\\tilde{{\\lambda}}_3}}_{{\\text{{Rescate}}}}'
                        f'-\\underbrace{{\\zeta_\\gamma^{{(1)}}\\tilde{{\\lambda}}_1}}_{{\\text{{Pago}}}}\\)</div>'
                        f'<div class="sec">Componentes</div>'
                        f'<div>'
                        f'\\(\\zeta_\\gamma^{{(1)}}={_zg1_kh1:.4f}\\), '
                        f'\\(\\tilde{{\\lambda}}_1={_l1_kh1:.4f}\\) '
                        f'→ término Pago: \\({-_zg1_kh1 * _l1_kh1:+.4f}\\)<br/>'
                        f'\\(\\zeta_\\gamma^{{(2)}}={_zg2_kh1:.4f}\\), '
                        f'\\(\\tilde{{\\lambda}}_2={_l2_kh1:.4f}\\) '
                        f'→ término Muerte: \\({_zg2_kh1 * _l2_kh1:+.4f}\\)<br/>'
                        f'\\(\\zeta_\\gamma^{{(3)}}={_zg3_kh1:.4f}\\), '
                        f'\\(\\tilde{{\\lambda}}_3={_l3_kh1:.4f}\\) '
                        f'→ término Rescate: \\({_zg3_kh1 * _l3_kh1:+.4f}\\)'
                        f'</div>'
                        f'<div class="sec">Valor κ_h</div>'
                        f'<div>\\(\\kappa_h={_kh1_val:+.6f}\\)</div>'
                        f'<div class="sec">Signo resultante</div>'
                        f'<div style="font-size:1.1em;font-weight:bold;color:{_kh1_color};">'
                        f'−sgn(κ_h) = {_kh1_sgn:+d}</div>'
                        f'<div class="sec">Interpretación</div>'
                        f'<div>{html.escape(_kh1_interp)}</div>'
                        f'<div class="ux">Fuente: coeficientes ζ_γ del bundle de {html.escape(str(_th52_kh))} '
                        f'y hazards calculados con γ*={float(_gamma_state_star52):.4f}, '
                        f'α*={float(_alpha_state_star52):.4f}, M(1)={float(_t52_M1):.4f}.</div>'
                        f'</div>'
                    )
                except Exception:
                    _t52_vals[_kh_key52] = "—"
                    _t52_tips[_kh_key52] = (
                        f'<div class="t52h">'
                        f'<div class="hdr">−sgn(κ_h) · θ={html.escape(str(_th52_kh))} · τ=1</div>'
                        f'<div class="sec">Error</div>'
                        f'<div>No fue posible calcular κ_h para este tipo. '
                        f'Verifique que el bundle de {html.escape(str(_th52_kh))} '
                        f'contenga coeficientes ζ_γ válidos.</div>'
                        f'</div>'
                    )
            _t52_tips0[_t52_ir_true_row_label] = _t52_ir_true_tip(
                _t52_iric_tau0, 0, float(_alpha_state_t0_52), float(_gamma_state_t0_52)
            )
            _t52_tips[_t52_ir_true_row_label] = _t52_ir_true_tip(
                _t52_iric_tau1, 1, float(_alpha_state_star52), float(_gamma_state_star52)
            )
            _t52_tips["ã_S"] = _t52_tip_html(
                "S", _s52_mdg_intent, _ps52,
                _ats52, _ats52_argmax,
                _iota_52, _eu_s52, _u52_s,
                f"(−V_R*={-float(_vstar_vr_link52):.4f}, −V_N*={-float(_vstar_vn_link52):.4f})",
            )
            # ── P(m): θ* oculto genera el dato; Bayes evalúa todos los θ después ──
            _theta_mu1_52 = max(_mu1_52, key=_mu1_52.get) if _mu1_52 else str(tipo_real)
            _iota1_52 = float(max(_mu1_52.values())) if _mu1_52 else float(_iota_52)
            _pdet1_m52 = _pdet_logit_prob(
                str(tipo_real),
                float(_policy1_52["alpha_star"]),
                float(_policy1_52["gamma_star"]),
            )
            _pm52, _mfac1_52 = _mechanism_m_probs_for_actions(
                str(tipo_real),
                1,
                float(_policy1_52["alpha_star"]),
                float(_policy1_52["gamma_star"]),
                float(_pdet1_m52),
                _atk52_real,
                _ats52,
                _atf52,
                z_region=str(st.session_state.z_region),
                v_victim=str(st.session_state.v_victim),
                f_capa=str(f_capa),
                s_tipo=str(s_tipo),
                policy_sensitivity=float(st.session_state.get("t52_likelihood_policy_sensitivity", 4.0)),
            )
            _mstar52 = max(_pm52, key=_pm52.get)
            _lambda_diag52 = dict(_mfac1_52.get("lam", {}))
            _psi_rows52 = ""
            for _lbl52, _pv52 in {
                "Liberación": float(_lambda_diag52.get("Exógeno", 0.0)),
                "Rescate": float(_lambda_diag52.get("Rescate", 0.0)),
                "Pago": float(_lambda_diag52.get("Pago", 0.0)),
                "Muerte": float(_lambda_diag52.get("Muerte", 0.0)),
                "Continuar": float(_mfac1_52.get("p_cont", 0.0)),
            }.items():
                _is_win52 = (_lbl52 == _mstar52)
                _wst52 = " font-weight:700;background:#2a3a2a;" if _is_win52 else ""
                _wmk52 = " &#9733;" if _is_win52 else ""
                _psi_rows52 += (
                    f"<tr style='{_wst52}'>"
                    f"<td>{html.escape(_lbl52)}{_wmk52}</td>"
                    f"<td class='num'>{_pv52:.4f}</td>"
                    f"<td class='num'>{float(_mfac1_52.get('q', 0.0)):.4f}</td>"
                    f"<td class='num'>{_pm52[_lbl52]:.4f}</td>"
                    f"</tr>"
                )
            _int_rows52 = ""
            _cur52 = 0.0
            for _i52, (_lbl52, _prob52) in enumerate(_pm52.items()):
                _lo52, _hi52 = _cur52, _cur52 + _prob52
                _hi_str52 = "1.0000" if _i52 == len(_pm52) - 1 else f"{_hi52:.4f}"
                _is_win52 = (_lbl52 == _mstar52)
                _smk52 = '<span class="star">&#9733;</span>' if _is_win52 else ""
                _rc52 = ' class="hit"' if _is_win52 else ""
                _int_rows52 += (
                    f"<tr{_rc52}>"
                    f"<td class='iv'>[{_lo52:.4f},&nbsp;{_hi_str52})</td>"
                    f"<td>{html.escape(_lbl52)}</td>"
                    f"<td class='num'>{_prob52:.4f}</td>"
                    f"<td>{_smk52}</td></tr>"
                )
                _cur52 += _prob52
            _t52_tips["m"] = (
                f'<div class="t52h">'
                f'<div class="hdr">Desenlace \\(m\\) en \\(\\tau=1\\) — ley física</div>'
                f'<div class="sec">Condición para calcular</div>'
                f'<div class="ux">La celda queda vacía mientras no estén generadas/reportadas '
                f'\\(\\tilde{{a}}_K\\) y \\(\\tilde{{a}}_F\\) para \\(\\tau=1\\).</div>'
                f'<div class="sec">Cálculo cuando estén disponibles</div>'
                f'<div>\\(P^E(m_1=\\mathrm{{Cont}})=p_{{\\mathrm{{Cont}},1}}\\), '
                f'\\(P^E(m_1=j)=h_j(1\\mid\\theta_K,\\mathcal C_1)\\).</div>'
                f'<div class="ux">Se usa \\(\\theta_K=\\arg\\max_\\theta\\mu_1(\\theta)\\), '
                f'\\((\\alpha_1^*,\\gamma_1^*)\\) del Estado y \\(\\iota_1=\\max_\\theta\\mu_1(\\theta)\\).</div>'
                f'<div class="sec">Hazards competitivos</div>'
                f'<div class="sec">Política del Estado usada</div>'
                f'<div>Se usa directamente \\(\\alpha_1^*={float(_policy1_52["alpha_star"]):.4f}\\) '
                f'y \\(\\gamma_1^*={float(_policy1_52["gamma_star"]):.4f}\\), como instrumentos óptimos del Estado.</div>'
                f'{_t52_mechanism_m_tooltip_lines(_mfac1_52, 1, float(_policy1_52["alpha_star"]), float(_policy1_52["gamma_star"]), float(_pdet1_m52))}'
                f'<table><tr><td>m</td><td>λ/pCont</td><td>q</td><td>P</td></tr>{_psi_rows52}</table>'
                f'<div class="ux">Al generarse \\(\\tilde{{a}}_K\\) y \\(\\tilde{{a}}_F\\), '
                f'el globo despliega lambdas, probabilidades e intervalos acumulados.</div>'
                f'</div>'
            )
            _t52_vals["m"] = ""
            for _th_mu52 in TIPOS_SECUESTRADOR:
                _mu_label52 = f"μ({_th_mu52})"
                _t52_vals0[_mu_label52] = f"{float(_mu0_52.get(_th_mu52, 0.0)):.4f}"
                _t52_vals[_mu_label52] = f"{float(_mu1_52.get(_th_mu52, 0.0)):.4f}"
            _t52_vals0["α^μ (R)"] = f"{float(_ref_mu0_link52['alpha_R_mu']):.4f}"
            _t52_vals["α^μ (R)"]  = f"{float(_ref_mu1_link52['alpha_R_mu']):.4f}"
            _t52_vals0["γ^μ (R)"] = f"{float(_ref_mu0_link52['gamma_R_mu']):.4f}"
            _t52_vals["γ^μ (R)"]  = f"{float(_ref_mu1_link52['gamma_R_mu']):.4f}"
            _t52_vals0["α^μ (N)"] = f"{float(_ref_mu0_link52['alpha_N_mu']):.4f}"
            _t52_vals["α^μ (N)"]  = f"{float(_ref_mu1_link52['alpha_N_mu']):.4f}"
            _t52_vals0["γ^μ (N)"] = f"{float(_ref_mu0_link52['gamma_N_mu']):.4f}"
            _t52_vals["γ^μ (N)"]  = f"{float(_ref_mu1_link52['gamma_N_mu']):.4f}"

            # ── V, d, ι, M_t ─ fuente: Escenario base τ=0 y trayectoria τ=1 ──
            _t52_Tmad52 = float(st.session_state.get("cal_T_mad", 30.0))
            _t52_M0 = float(_mfac0_52.get("M_t", 0.0)) if isinstance(_mfac0_52, dict) else 0.0
            _t52_M1 = float(min(1.0, (1.0 / max(_t52_Tmad52, 1e-9)) ** 2))
            _t52_traj52 = st.session_state.get("rb_mu_traj_snapshot")
            _t52_traj_r0 = (
                _t52_traj52[_t52_traj52["t"].astype(int) == 0].iloc[0]
                if isinstance(_t52_traj52, pd.DataFrame)
                and not _t52_traj52.empty
                and "t" in _t52_traj52.columns
                else pd.Series(dtype=object)
            )
            _t52_V0_raw = _t52_traj_r0.get("V_t", "—") if not _t52_traj_r0.empty else "—"
            try:
                _t52_V0 = str(int(float(_t52_V0_raw)))
            except Exception:
                _t52_V0 = str(_t52_V0_raw)
            _t52_d0_raw = _t52_traj_r0.get("d_t", str(_d0_52)) if not _t52_traj_r0.empty else str(_d0_52)
            try:
                _t52_d0 = str(int(float(_t52_d0_raw)))
            except Exception:
                _t52_d0 = str(_t52_d0_raw)
            _t52_V0 = str(int(_V52_tau0_int))
            _t52_d0 = str(int(_d52_tau0_int))
            _t52_vals["V (voz)"]  = "—"
            _t52_vals["d (det.)"] = "—"
            _t52_vals["ι"]        = f"{_iota1_52:.4f}"
            _t52_vals["M_t"]      = f"{_t52_M1:.4f}"
            _t52_vals0["V (voz)"]  = _t52_V0
            _t52_vals0["d (det.)"] = _t52_d0
            _t52_vals0["ι"]        = f"{_iota_52:.4f}"
            _t52_vals0["M_t"]      = f"{_t52_M0:.4f}"
            _iota1_theta52 = max(_mu1_52, key=_mu1_52.get) if _mu1_52 else str(tipo_real)
            _iota1_parts52 = "".join(
                rf"<div>\(\mu_1({html.escape(str(_th_i52))})={float(_mu1_52.get(_th_i52, 0.0)):.4f}\)</div>"
                for _th_i52 in TIPOS_SECUESTRADOR
            )
            _t52_tips["ι"] = (
                r'<div class="t52h">'
                r'<div class="hdr">Precisión bayesiana en \(\tau=1\)</div>'
                r'<div class="sec">Definición</div>'
                r'<div>\(\iota_1=\max_{\theta\in\Theta_K}\mu_1(\theta)\)</div>'
                rf'<div class="sec">Máximo reportado</div>'
                rf'<div>\(\iota_1={_iota1_52:.4f}\), alcanzado por '
                rf'\(\theta={html.escape(str(_iota1_theta52))}\).</div>'
                rf'<div class="sec">Valores \(\mu_1\) usados</div>{_iota1_parts52}'
                r'</div>'
            )

            _t52_formulas: dict = {}
            _t52_formulas[_t52_gamma_row_label] = (
                r'<div>Auditoría formal del conjunto factible del Estado bajo valor esperado.</div>'
                r'<div>\(\Gamma_t(\mu_t)=1 \iff IR^K \land IC^K \land IR^F\).</div>'
                r'<div>Usa los gaps esperados guardados para \((a_S^*,\alpha_t^*,\gamma_t^*)\).</div>'
            )
            _t52_formulas[_t52_ir_true_row_label] = (
                rf'<div>Auditoría puntual de \(IR^K\) para el tipo verdadero '
                rf'\(\theta^*=\mathrm{{{html.escape(str(tipo_real))}}}\).</div>'
                r'<div>\(U^K_{rel}(\theta^*)-\max\{V^K_{cont}(\theta^*),U^K_{kill}(\theta^*)\}\ge0\).</div>'
                r'<div>Usa los instrumentos \((\alpha_t,\gamma_t)\) reportados para el Estado en esa columna.</div>'
            )
            for _th_mu52 in TIPOS_SECUESTRADOR:
                _mu_label52 = f"μ({_th_mu52})"
                _t52_formulas[_mu_label52] = (
                    rf'<div>\(\mu_1(\theta_K=\text{{{_th_mu52}}})'
                    rf'\propto \mu_0(\theta_K=\text{{{_th_mu52}}})'
                    rf'\mathcal{{L}}_0(\theta_K=\text{{{_th_mu52}}})\)</div>'
                    r'<div>τ=0 muestra el prior; para τ≥1 se lee la trayectoria μ reportada en Tabla 5.4.</div>'
                )

            def _t52_bench_rows(ref: dict[str, Any], key: str) -> str:
                bench = ref.get("bench", {}) if isinstance(ref, dict) else {}
                rows = []
                for th in TIPOS_SECUESTRADOR:
                    b = bench.get(str(th), {}) if isinstance(bench, dict) else {}
                    rows.append(
                        rf"\mu({th})={float(b.get('mu', 0.0)):.4f}\times "
                        rf"{float(b.get(key, 0.0)):.4f}"
                    )
                return r"\\ ".join(rows)

            _t52_formulas["α^μ (R)"] = (
                r'<div>Centro bayesiano \(\alpha_{t,R}^\mu=\sum_\theta\mu_t(\theta)\,\alpha_{t,R}^{\mathrm{bench}}(\theta)\).</div>'
                r'<div>\((\alpha_{t,R}^{\mathrm{bench}}(\theta),\gamma_{t,R}^{\mathrm{bench}}(\theta))'
                r'\in\arg\min_{\alpha,\gamma\in[0,1]}V_t^R(\iota_t,\hat\theta_t,\theta,\alpha,\gamma)\).</div>'
                rf'<div>τ=0: \({float(_ref_mu0_link52["alpha_R_mu"]):.4f}\) · '
                rf'τ=1: \({float(_ref_mu1_link52["alpha_R_mu"]):.4f}\).</div>'
                rf'<div class="ux">τ=0: \({_t52_bench_rows(_ref_mu0_link52, "alpha_R_bench")}\)</div>'
                rf'<div class="ux">τ=1: \({_t52_bench_rows(_ref_mu1_link52, "alpha_R_bench")}\)</div>'
            )
            _t52_formulas["γ^μ (R)"] = (
                r'<div>Centro bayesiano \(\gamma_{t,R}^\mu=\sum_\theta\mu_t(\theta)\,\gamma_{t,R}^{\mathrm{bench}}(\theta)\).</div>'
                r'<div>Benchmarks de rescate obtenidos minimizando \(V_t^R\) por tipo.</div>'
                rf'<div>τ=0: \({float(_ref_mu0_link52["gamma_R_mu"]):.4f}\) · '
                rf'τ=1: \({float(_ref_mu1_link52["gamma_R_mu"]):.4f}\).</div>'
                rf'<div class="ux">τ=0: \({_t52_bench_rows(_ref_mu0_link52, "gamma_R_bench")}\)</div>'
                rf'<div class="ux">τ=1: \({_t52_bench_rows(_ref_mu1_link52, "gamma_R_bench")}\)</div>'
            )
            _t52_formulas["α^μ (N)"] = (
                r'<div>Centro bayesiano \(\alpha_{t,N}^\mu=\sum_\theta\mu_t(\theta)\,\alpha_{t,N}^{\mathrm{bench}}(\theta)\).</div>'
                r'<div>\((\alpha_{t,N}^{\mathrm{bench}}(\theta),\gamma_{t,N}^{\mathrm{bench}}(\theta))'
                r'\in\arg\min_{\alpha,\gamma\in[0,1]}V_t^N(\theta,\alpha,\gamma)\).</div>'
                rf'<div>τ=0: \({float(_ref_mu0_link52["alpha_N_mu"]):.4f}\) · '
                rf'τ=1: \({float(_ref_mu1_link52["alpha_N_mu"]):.4f}\).</div>'
                rf'<div class="ux">τ=0: \({_t52_bench_rows(_ref_mu0_link52, "alpha_N_bench")}\)</div>'
                rf'<div class="ux">τ=1: \({_t52_bench_rows(_ref_mu1_link52, "alpha_N_bench")}\)</div>'
            )
            _t52_formulas["γ^μ (N)"] = (
                r'<div>Centro bayesiano \(\gamma_{t,N}^\mu=\sum_\theta\mu_t(\theta)\,\gamma_{t,N}^{\mathrm{bench}}(\theta)\).</div>'
                r'<div>Benchmarks de negociación obtenidos minimizando \(V_t^N\) por tipo.</div>'
                rf'<div>τ=0: \({float(_ref_mu0_link52["gamma_N_mu"]):.4f}\) · '
                rf'τ=1: \({float(_ref_mu1_link52["gamma_N_mu"]):.4f}\).</div>'
                rf'<div class="ux">τ=0: \({_t52_bench_rows(_ref_mu0_link52, "gamma_N_bench")}\)</div>'
                rf'<div class="ux">τ=1: \({_t52_bench_rows(_ref_mu1_link52, "gamma_N_bench")}\)</div>'
            )
            _t52_formulas["a_F*"] = (
                r'<div>\(\arg\max_{a^F}\,\mathbb{E}_{\theta_K|I_F}'
                r'[U_F(a^F|\theta_K,\theta_F)]\)</div>'
                r'<div style="margin-top:4px">Acción óptima de familia en \(\tau=1\).</div>'
            )
            _t52_formulas["ã_F"] = (
                r'<div>Fase 1 \(\mathcal{H}_t^F\): '
                r'\(\mathbb{P}_I^F(\tilde{a}_F \mid a_F^*)\)</div>'
                r'<div>Ley de Implementación de Tabla 7: '
                r'\(\exp(\mathbf{1}\{a=a_F^*\}/T_t)\), normalizada.</div>'
                r'<div>Acciones: [Cooperar, Coludir].</div>'
                r'<div>\(\tilde{a}_F\) = acción cuyo intervalo \([l_o,h_i)\) '
                r'contiene \(u_F \sim \mathcal{U}[0,1]\).</div>'
            )
            _t52_formulas[f"a_K* ({tipo_real})"] = (
                f'<div>\\(\\arg\\max_{{a^K}}\\,V_K^{{BW}}(a^K|\\theta_K='
                f'\\text{{{tipo_real}}},\\,\\tau=0)\\)</div>'
                f'<div style="margin-top:4px">Valor fijo: Tabla 15, fila \\(\\tau=0\\), columna 14.</div>'
            )
            _t52_formulas[f"ã_K ({tipo_real})"] = (
                r'<div>Fase 1 \(\mathcal{H}_t^K\): '
                r'\(\mathbb{P}_I^K(\tilde{a}_K \mid a_K^*)\)</div>'
                r'<div>Ley de Implementación de Tabla 7: '
                r'\(\exp(\mathbf{1}\{a=a_K^*\}/T_t)\), normalizada.</div>'
                r'<div>Acciones: [Continuar, Liberar, Matar].</div>'
                r'<div>\(\tilde{a}_K\) = acción cuyo intervalo \([l_o,h_i)\) '
                r'contiene \(u_K \sim \mathcal{U}[0,1]\).</div>'
            )
            _t52_formulas["a_S* óptima"] = (
                r'<div>\((a_t^{S*},\alpha_t^*,\gamma_t^*)\) por '
                r'\(\mathbf{1}\{V_t^{R,*}\le V_t^{N,*}\}\).</div>'
            )
            _t52_formulas["ã_S"] = (
                r'<div>Fase 1 \(\mathcal{H}_t^S\): '
                r'\(\mathbb{P}_I^S(\tilde{a}_S \mid a_S^*)\), Ley de Implementación de Tabla 7.</div>'
                r'<div>Acciones: [Rescatar, No Rescatar].</div>'
                r'<div>\(\tilde{a}_S\) = acción cuyo intervalo \([l_o,h_i)\) '
                r'contiene \(u_S \sim \mathcal{U}[0,1]\).</div>'
            )
            _t52_formulas["γ* Estado"] = (
                r'<div>En \(\tau=0\), reporta el valor inicial \(\gamma_0\) vinculado al estado inicial.</div>'
                r'<div>En \(\tau=1\), reporta \(\gamma_t^*\) de la rama óptima de Tabla 5.3.</div>'
                r'<div>Si \(a_S^*=\mathrm{Rescate}\), usa \(\gamma_R^*\); '
                r'si \(a_S^*=\mathrm{Negociar}\), usa \(\gamma_N^*\).</div>'
            )
            _t52_formulas["α* Estado"] = (
                r'<div>En \(\tau=0\), reporta el valor inicial \(\alpha_0\) vinculado al estado inicial.</div>'
                r'<div>En \(\tau=1\), reporta \(\alpha_t^*\) de la rama óptima de Tabla 5.3.</div>'
                r'<div>Si \(a_S^*=\mathrm{Rescate}\), usa \(\alpha_R^*\); '
                r'si \(a_S^*=\mathrm{Negociar}\), usa \(\alpha_N^*\).</div>'
            )
            _t52_formulas[_gamma_r_pi_label52] = (
                rf'<div>\(\gamma_R^{{{html.escape(str(tipo_real))},*}}\): presión de referencia PI en rescate.</div>'
                rf'<div>Se calcula con \(\mu^{{PI}}(\mathrm{{{html.escape(str(tipo_real))}}})=1\).</div>'
                r'<div>No alimenta el ciclo; solo expone el benchmark por rama.</div>'
            )
            _t52_formulas[_alpha_r_pi_label52] = (
                rf'<div>\(\alpha_R^{{{html.escape(str(tipo_real))},*}}\): bloqueo de referencia PI en rescate.</div>'
                rf'<div>Se calcula con \(\mu^{{PI}}(\mathrm{{{html.escape(str(tipo_real))}}})=1\).</div>'
                r'<div>No alimenta el ciclo; solo expone el benchmark por rama.</div>'
            )
            _t52_formulas[_gamma_n_pi_label52] = (
                rf'<div>\(\gamma_N^{{{html.escape(str(tipo_real))},*}}\): presión de referencia PI en negociación.</div>'
                rf'<div>Se calcula con \(\mu^{{PI}}(\mathrm{{{html.escape(str(tipo_real))}}})=1\).</div>'
                r'<div>No alimenta el ciclo; solo expone el benchmark por rama.</div>'
            )
            _t52_formulas[_alpha_n_pi_label52] = (
                rf'<div>\(\alpha_N^{{{html.escape(str(tipo_real))},*}}\): bloqueo de referencia PI en negociación.</div>'
                rf'<div>Se calcula con \(\mu^{{PI}}(\mathrm{{{html.escape(str(tipo_real))}}})=1\).</div>'
                r'<div>No alimenta el ciclo; solo expone el benchmark por rama.</div>'
            )
            _t52_formulas["m"] = (
                r'<div>\(\arg\max_j\,\mathbb{P}^E(m=j \mid \tilde{a}_K, \tilde{a}_F, '
                r'\tilde{a}_S,\alpha_t^*,\gamma_t^*,\iota_t,\theta_K)\)</div>'
            )
            _t52_formulas["V (voz)"] = (
                r'<div>\(V_t\in\{0,1\}\): señal de comunicación (voz/silencio).</div>'
                rf'<div>τ=0: \(V_0={html.escape(str(_t52_V0))}\) se define en Escenario base τ=0.</div>'
                r'<div>Entra en \(\mathcal{L}_C\): si \(V_t=0\), '
                r'\(\mathcal{L}_C=(1-\pi)^\omega\); si \(V_t=1\), \(\mathcal{L}_C=1\).</div>'
            )
            _t52_formulas["d (det.)"] = (
                r'<div>\(d_t\in\{0,1\}\): señal de detección de colusión.</div>'
                rf'<div>τ=0: \(d_0={html.escape(str(_t52_d0))}\) se define en Escenario base τ=0.</div>'
                r'<div>\(d_t\sim\mathrm{Bernoulli}(p_{\mathrm{det},t})\), '
                r'\(p_{\mathrm{det},t}=\Lambda(\eta_0+\eta_1\alpha_t^*+\eta_2\gamma_t^*)\).</div>'
                rf'<div>τ=0: \(p_{{\mathrm{{det}},0}}={_pdet52:.4f}\).</div>'
            )
            _t52_formulas["ι"] = (
                r'<div>\(\iota_t=\max_\theta\,\mu_t(\theta)\): índice de precisión bayesiana.</div>'
                rf'<div>τ=0: \(\iota_0={_iota_52:.4f}\) (de \(\mu_0\)).</div>'
                rf'<div>τ=1: \(\iota_1={_iota1_52:.4f}\) (de \(\mu_1\)).</div>'
                r'<div>Escala la distribución MDG y la probabilidad de supervivencia en rescate.</div>'
            )
            _t52_formulas["H(μ)"] = (
                r'<div>\(H(\mu_t)=-\sum_{\theta}\mu_t(\theta)\ln\mu_t(\theta)\).</div>'
                r'<div>Mide incertidumbre sobre el tipo del captor.</div>'
            )
            _t52_formulas["ΔH Estado"] = (
                r'<div>\(\Delta H=H(\mu_t)-\mathbb E_t[H(\mu_{t+1})]\).</div>'
                r'<div>El Estado evalúa los cinco desenlaces de Tabla 10 y aplica Bayes para cada uno.</div>'
                rf'<div>En el objetivo entra como \(-\psi_H\Delta H\), con \(\psi_H={_t52_entropy_weight:.1f}\).</div>'
            )
            _t52_formulas["M_t"] = (
                r'<div>\(M_t=\min\!\bigl(1,(t/T_{\mathrm{mad}})^2\bigr)\): filtro de maduración.</div>'
                r'<div>Escala todos los hazards competitivos \(\tilde{\lambda}_j(t)\).</div>'
                rf'<div>En la columna τ=0, \(m_0\) se calcula con la información de τ=0: '
                rf'\(t=0\), \(M_0={_t52_M0:.4f}\), '
                rf'\(\alpha_0^*={float(_policy0_52["alpha_star"]):.4f}\), '
                rf'\(\gamma_0^*={float(_policy0_52["gamma_star"]):.4f}\).</div>'
                rf'<div>Para τ=1 se reporta \(M_1={_t52_M1:.4f}\) con '
                rf'\(T_{{\mathrm{{mad}}}}={_t52_Tmad52:.1f}\).</div>'
            )

            def _t52_short_tip(title: str, formula: str, source: str = "") -> str:
                _src = f'<div class="ux">{source}</div>' if source else ""
                return (
                    f'<div class="t52h">'
                    f'<div class="hdr">{html.escape(title)}</div>'
                    f'<div>{formula}</div>'
                    f'{_src}'
                    f'</div>'
                )

            # Globos reducidos por celda del ciclo base.
            # setdefault conserva los globos detallados ya definidos arriba.
            for _th_mu52 in TIPOS_SECUESTRADOR:
                _mu_label52 = f"μ({_th_mu52})"
                _t52_tips0.setdefault(
                    _mu_label52,
                    _t52_short_tip(
                        f"{_mu_label52} en τ=0",
                        rf'\(\mu_0({_th_mu52})\): prior inicial normalizado.',
                        "Fuente: distribución inicial del incidente.",
                    ),
                )
                _t52_tips.setdefault(
                    _mu_label52,
                    _t52_short_tip(
                        f"{_mu_label52} en τ=1",
                        rf'\(\mu_1({_th_mu52})=\mu_0({_th_mu52})\mathcal{{L}}_0({_th_mu52})/Z_0\).',
                        "Fuente: posterior de Tabla 5.4.",
                    ),
                )
            _ir_f_symbol52 = r"\ge" if _ir_f_52 else "<"
            _t52_tips0.setdefault(
                "a_F*",
                _t52_short_tip(
                    "a_F* en τ=0",
                    rf'\(a_F^*=\mathrm{{{html.escape(str(_af52_star))}}}\) porque '
                    rf'\(U_{{coop}}={float(_u_coop_p3):.2f}\) '
                    rf'{_ir_f_symbol52} '
                    rf'\(U_{{col}}={float(_u_col_p3):.2f}\).',
                    "Fuente: pestaña 4 · Familia (F) — Maximización IR^F.",
                ),
            )
            _t52_tips.setdefault(
                "a_F*",
                _t52_short_tip(
                    "a_F* en τ=1",
                    r'\(a_F^*=\arg\max_{a^F}E[U_F(a^F)]\).',
                    "Queda latente hasta que se reporta el óptimo de F.",
                ),
            )
            _t52_tips0.setdefault(
                f"a_K* ({tipo_real})",
                _t52_short_tip(
                    f"a_K* ({tipo_real}) en τ=0",
                    rf'\(a_K^*=\mathrm{{{html.escape(str(_t52_clean_ak(_ak52_t0_raw)))}}}\).',
                    "Fuente: Tabla 15, fila τ=0, columna 14.",
                ),
            )
            _t52_tips.setdefault(
                f"a_K* ({tipo_real})",
                _t52_short_tip(
                    f"a_K* ({tipo_real}) fijo para τ≥1",
                    rf'\(a_K^*_\tau=a_K^*_0=\mathrm{{{html.escape(str(_t52_clean_ak(_ak52_t0_raw)))}}}\).',
                    "Fuente fija: Tabla 15, fila τ=0, columna 14.",
                ),
            )
            _t52_tips0.setdefault(
                "a_S* óptima",
                _t52_short_tip(
                    "a_S* en τ=0",
                    r'\(a_S^*\): regla inicial del Estado en el escenario base.',
                    "Valor inicial antes de la optimización de τ=1.",
                ),
            )
            _t52_tips0.setdefault(
                "γ* Estado",
                _t52_short_tip(
                    "γ Estado en τ=0",
                    rf'\(\gamma_0={_gamma_state_t0_52:.4f}\).',
                    "Valor inicial usado para cerrar el ciclo base.",
                ),
            )
            _t52_tips0.setdefault(
                "α* Estado",
                _t52_short_tip(
                    "α Estado en τ=0",
                    rf'\(\alpha_0={_alpha_state_t0_52:.4f}\).',
                    "Valor inicial usado para cerrar el ciclo base.",
                ),
            )
            _t52_tips0.setdefault(
                "ι",
                _t52_short_tip(
                    "ι en τ=0",
                    rf'\(\iota_0=\max_\theta\mu_0(\theta)={_iota_52:.4f}\).',
                    "Precisión bayesiana inicial.",
                ),
            )
            _t52_tips0.setdefault(
                "M_t",
                _t52_short_tip(
                    "M_t en τ=0",
                    rf'\(M_0=\min\{{1,(0/T_{{\mathrm{{mad}}}})^2\}}={_t52_M0:.4f}\).',
                    "El desenlace m de la columna τ=0 se calcula con la información de τ=0.",
                ),
            )
            _t52_tips.setdefault(
                "M_t",
                _t52_short_tip(
                    "M_t en τ=1",
                    rf'\(M_1=\min\{{1,(1/T_{{\mathrm{{mad}}}})^2\}}={_t52_M1:.4f}\).',
                    "Cierre del ciclo base.",
                ),
            )
            _t52_tips0.setdefault(
                "V (voz)",
                _t52_short_tip(
                    "Voz en τ=0",
                    rf'\(V_0={html.escape(str(_t52_V0))}\): valor definido en Escenario base τ=0.',
                    "Esta señal inicia el ciclo base y entra en la verosimilitud de comunicación.",
                ),
            )
            _t52_tips.setdefault(
                "V (voz)",
                _t52_short_tip(
                    "Voz en τ=1",
                    r'\(V_1\): inicio del ciclo siguiente; se llena al avanzar ciclo.',
                    "No cierra el ciclo base.",
                ),
            )
            _t52_tips0.setdefault(
                "d (det.)",
                _t52_short_tip(
                    "d en τ=0",
                    rf'\(d_0={html.escape(str(_t52_d0))}\): valor definido en Escenario base τ=0.',
                    rf'Entra en \(\mathcal{{L}}_d\); \(p_{{det,0}}={_pdet52:.4f}\).',
                ),
            )
            _t52_tips.setdefault(
                "d (det.)",
                _t52_short_tip(
                    "d en τ=1",
                    r'\(d_1\sim Bernoulli(p_{det,1})\).',
                    "Se genera al iniciar el ciclo 1.",
                ),
            )
            _t52_tips0.setdefault(
                "m",
                _t52_short_tip(
                    "m en τ=0",
                    r'\(m_0=\mathrm{Continuar}\): condición inicial del mecanismo.',
                    "Permite actualizar la posterior hacia τ=1.",
                ),
            )

            def _t52_ast_ready(value: Any) -> bool:
                return str(value).strip() not in ("", "—", "None", "nan")

            _t52_all_ast_ready = bool(
                _t52_ast_ready(_t52_vals.get("a_F*", "—"))
                and _t52_ast_ready(_t52_vals.get(f"a_K* ({tipo_real})", "—"))
                and _t52_ast_ready(_t52_vals.get("a_S* óptima", "—"))
            )

            _t52_row_order = [
                *[f"μ({_th52})" for _th52 in TIPOS_SECUESTRADOR],
                "α^μ (R)",
                "γ^μ (R)",
                "α^μ (N)",
                "γ^μ (N)",
                "a_S* óptima",
                "ã_S",
                "γ* Estado",
                "α* Estado",
                _gamma_r_pi_label52,
                _alpha_r_pi_label52,
                _gamma_n_pi_label52,
                _alpha_n_pi_label52,
                "ι",
                "H(μ)",
                "ΔH Estado",
                "M_t",
                "V (voz)",
                "d (det.)",
                "a_F*",
                "ã_F",
                f"a_K* ({tipo_real})",
                f"ã_K ({tipo_real})",
                "m",
                _t52_gamma_row_label,
                _t52_ir_true_row_label,
                *[f"−sgn(κh) ({_th52_kh})" for _th52_kh in TIPOS_SECUESTRADOR],
            ]

            _t52_cycle_vals: dict[str, str] = {}
            _t52_dynamic_cycles: list[dict[str, Any]] = []
            _t52_dynamic_cycle_version = 17
            _t52_dynamic_cycle_signature = {
                "version": int(_t52_dynamic_cycle_version),
                "theta_true": str(tipo_real),
                "family_capacity": str(f_capa),
                "state_type": str(s_tipo),
                "z_region": str(st.session_state.get("z_region", "")),
                "v_victim": str(st.session_state.get("v_victim", "")),
                "m_mode": str(st.session_state.get("t52_m_mode", "Sorteo")),
                "likelihood_policy_sensitivity": round(float(st.session_state.get("t52_likelihood_policy_sensitivity", 4.0)), 6),
                "entropy_info_weight": round(float(st.session_state.get("t52_entropy_info_weight", 25.0)), 6),
                "limite_dias": int(st.session_state.get("limite_dias", limite_dias)),
                "iric_top_k": int(st.session_state.get("t52_iric_top_k", 25)),
                "delta_h_top_m": int(st.session_state.get("t52_delta_h_top_m", 40)),
                "pi_benchmark_all_cycles": bool(st.session_state.get("t52_pi_benchmark_all_cycles", False)),
                "tau0_alpha": round(float(_t0_alpha_eff), 4),
                "tau0_gamma": round(float(_t0_gamma_eff), 4),
                "cal_T_mad": round(float(st.session_state.get("cal_T_mad", 30.0)), 8),
                "cal_lambda_4": round(float(st.session_state.get("cal_lambda_4", 0.0005)), 8),
                "eta0_by_theta": tuple(
                    (str(th), round(float(st.session_state.get(f"cal_eta0_pdet_{th}", _ETA0_PDET_DEFAULTS.get(th, -2.0))), 8))
                    for th in TIPOS_SECUESTRADOR
                ),
                "eta1": round(float(st.session_state.get("cal_eta1_pdet", 1.0)), 8),
                "eta2": round(float(st.session_state.get("cal_eta2_pdet", 1.0)), 8),
                "mu0_priors": tuple(
                    round(float(p), 4)
                    for p in st.session_state.get("final_priors", [25.0, 25.0, 25.0, 25.0])
                ),
                "tau0_aF": str(st.session_state.get("t52_tau0_aF_star", "")),
                "tau0_aK": str(_t52_clean_ak(str(_ak52_t0_raw))),
                "tau0_aS": str(st.session_state.get("t52_tau0_aS_star", "")),
                "tau0_V0": int(st.session_state.get("t52_tau0_V0", 0)),
                "tau0_d0": str(st.session_state.get("h0_d", "0")),
                "m_reroll": int(st.session_state.get("base_m_tau0_reroll_counter", 0)),
                "seed": int(st.session_state.get("global_semilla_rng", 123)),
                "run_counter": int(st.session_state.get("dynamic_current_run_counter", 0)),
                "voice_context_sig": _stable_json_signature(_voice_context_sig),
                "voice_params": _stable_json_signature({
                    "omega": st.session_state.get("cal_voz_omega", 0.2),
                    "pi_call": st.session_state.get("cal_voz_pi_call", {}),
                    "params": st.session_state.get("cal_voz_params", {}),
                }),
                "tab2_bundles": _stable_json_signature(_tab2_bundles52),
                "pcap_params": _stable_json_signature(st.session_state.get("cal_pcap_params", {})),
                "kidnapper_costs": _stable_json_signature({
                    th: {
                        "phi": st.session_state.get(f"tab3_phi_{th}", _TAB15_FIXED_COST_COEFFS.get(th, {}).get("phi", 1.0)),
                        "kc": st.session_state.get(f"tab3_kc_{th}", _TAB15_FIXED_COST_COEFFS.get(th, {}).get("kappa_c", 1.0)),
                        "nu": st.session_state.get(f"tab3_nu_{th}", _TAB15_FIXED_COST_COEFFS.get(th, {}).get("nu", 0.0)),
                    }
                    for th in TIPOS_SECUESTRADOR
                }),
                "ransom": round(float(R_escala), 8),
            }
            if (
                int(st.session_state.get("dynamic_cycles52_version", 0)) != _t52_dynamic_cycle_version
                or dict(st.session_state.get("dynamic_cycles52_signature", {})) != _t52_dynamic_cycle_signature
            ):
                st.session_state.pop("dynamic_cycles52", None)
                st.session_state.pop("dynamic_cycles_diag52", None)
                st.session_state.pop("dynamic_cycles_stop52", None)
                st.session_state.pop("first_cycle_tau1_52", None)
                st.session_state.pop("first_cycle_table52", None)
                st.session_state.pop("first_cycle_diag52", None)
                st.session_state.pop("first_cycle_post54", None)
                st.session_state["dynamic_cycles52_version"] = _t52_dynamic_cycle_version
                st.session_state["dynamic_cycles52_signature"] = dict(_t52_dynamic_cycle_signature)
            if bool(st.session_state.get("dynamic_cycles_requested", st.session_state.get("first_cycle_requested", False))):
                if not isinstance(st.session_state.get("dynamic_cycles52"), list):
                    try:
                        def _t52_family_star_for_cycle(
                            mu_v: dict[str, float],
                            alpha_v: float,
                            gamma_v: float,
                        ) -> tuple[str, float, float]:
                            _mu_f_cyc = {
                                th: max(0.0, float(dict(mu_v).get(th, 0.0)))
                                for th in TIPOS_SECUESTRADOR
                            }
                            _mu_f_sum = float(sum(_mu_f_cyc.values()))
                            _mu_f_cyc = (
                                {th: float(_mu_f_cyc.get(th, 0.0)) / _mu_f_sum for th in TIPOS_SECUESTRADOR}
                                if _mu_f_sum > 1e-12
                                else {th: 1.0 / len(TIPOS_SECUESTRADOR) for th in TIPOS_SECUESTRADOR}
                            )
                            _df_f_cyc, _ = compute_family_table(
                                modelo,
                                _mu_f_cyc,
                                float(gamma_v),
                                float(_p3_vl),
                                float(R_escala),
                                float(gamma_v),
                                float(_p3_phi_f),
                                float(_p3_kappa_f),
                                float(_p3_nu_f),
                                float(_p3_fcol),
                                float(_p3_pd0),
                                float(_p3_pda),
                                float(alpha_v),
                                float(cmh_alive),
                            )
                            _ucoop_cyc = float(
                                _df_f_cyc.loc[
                                    _df_f_cyc["Rama"].astype(str).str.startswith("Cooperar"),
                                    "EU ilustrativa",
                                ].iloc[0]
                            )
                            _ucol_cyc = float(
                                _df_f_cyc.loc[
                                    _df_f_cyc["Rama"].astype(str).str.startswith("Colusión"),
                                    "EU ilustrativa",
                                ].iloc[0]
                            )
                            return ("Cooperar" if _ucoop_cyc >= _ucol_cyc else "Coludir", _ucoop_cyc, _ucol_cyc)

                        def _t52_state_star_for_cycle(
                            mu_v: dict[str, float],
                            alpha_prev_v: float,
                            gamma_prev_v: float,
                            a_k_exec_v: str,
                            a_s_exec_v: str,
                            a_f_exec_v: str,
                        ) -> dict[str, Any]:
                            _mu_cyc = {
                                th: max(0.0, float(dict(mu_v).get(th, 0.0)))
                                for th in TIPOS_SECUESTRADOR
                            }
                            _sum_cyc = float(sum(_mu_cyc.values()))
                            if _sum_cyc <= 1e-12:
                                _mu_cyc = {th: 1.0 / len(TIPOS_SECUESTRADOR) for th in TIPOS_SECUESTRADOR}
                            else:
                                _mu_cyc = {th: float(_mu_cyc.get(th, 0.0)) / _sum_cyc for th in TIPOS_SECUESTRADOR}
                            _theta_hat_cyc = max(_mu_cyc, key=_mu_cyc.get) if _mu_cyc else str(tipo_real)
                            _iota_cyc = float(max(_mu_cyc.values())) if _mu_cyc else float(_iota_52)
                            _pkill_cyc: dict[str, float] = {}
                            _psurv_cyc: dict[str, float] = {}
                            for _th_cyc in TIPOS_SECUESTRADOR:
                                _psurv_cyc[_th_cyc] = _p_surv_precision_logit(
                                    _th_cyc, _iota_cyc, _theta_hat_cyc
                                )
                                _pkill_cyc[_th_cyc] = float(
                                    _outcome_probs_for_actions(
                                        str(_th_cyc),
                                        float(gamma_prev_v),
                                        float(_iota_cyc),
                                        str(a_k_exec_v),
                                        str(a_s_exec_v),
                                        str(a_f_exec_v),
                                    )["kill"]
                                )
                            _pkill_exp_cyc = float(
                                sum(_mu_cyc.get(_th, 0.0) * _pkill_cyc.get(_th, 0.0) for _th in TIPOS_SECUESTRADOR)
                            )
                            _psurv_exp_cyc = float(
                                sum(_mu_cyc.get(_th, 0.0) * _psurv_cyc.get(_th, 0.0) for _th in TIPOS_SECUESTRADOR)
                            )
                            _ops_mu_cyc = _state_weighted_cost_tuple(_p3_ops_by_type, _mu_cyc)
                            _mt_mu_cyc = _state_weighted_cost_tuple(_p3_mt_by_type, _mu_cyc)
                            _ref_mu_cyc = _state_reference_centers(_mu_cyc)

                            def _info_gain_vr_cyc(alpha_v: float, gamma_v: float) -> dict[str, Any]:
                                return _t52_expected_entropy_gain(
                                    _mu_cyc,
                                    int(_tau_start52),
                                    float(alpha_v),
                                    float(gamma_v),
                                    str(a_k_exec_v),
                                    "Rescatar",
                                    str(a_f_exec_v),
                                )

                            def _info_gain_vn_cyc(alpha_v: float, gamma_v: float) -> dict[str, Any]:
                                return _t52_expected_entropy_gain(
                                    _mu_cyc,
                                    int(_tau_start52),
                                    float(alpha_v),
                                    float(gamma_v),
                                    str(a_k_exec_v),
                                    "No Rescatar",
                                    str(a_f_exec_v),
                                )

                            _alpha_vr_cyc, _gamma_vr_cyc, _vstar_vr_cyc, _iric_vr_cyc = _t52_minimize_state_quadratic_iric(
                                const=float(
                                    _p3_omk * (1.0 - _psurv_exp_cyc)
                                    + _ops_mu_cyc[0]
                                    + _p3_chi_alpha * _ref_mu_cyc["alpha_R_mu"] ** 2
                                    + _p3_chi_gamma * _ref_mu_cyc["gamma_R_mu"] ** 2
                                ),
                                b_gamma=float(_ops_mu_cyc[1] - 2.0 * _p3_chi_gamma * _ref_mu_cyc["gamma_R_mu"]),
                                q_gamma=float(_ops_mu_cyc[2] + 2.0 * _p3_chi_gamma),
                                b_alpha=float(_ops_mu_cyc[3] - 2.0 * _p3_chi_alpha * _ref_mu_cyc["alpha_R_mu"]),
                                q_alpha=float(_ops_mu_cyc[4] + 2.0 * _p3_chi_alpha),
                                q_gamma_alpha=float(_ops_mu_cyc[5]),
                                mu_tau_v=_mu_cyc,
                                other_value_func=lambda _g, _a: 0.0,
                                info_bonus_func=_info_gain_vr_cyc,
                            )

                            def _vr_val_cyc(_g: float, _a: float) -> float:
                                return _t52_quad_value(
                                    _g,
                                    _a,
                                    float(
                                        _p3_omk * (1.0 - _psurv_exp_cyc)
                                        + _ops_mu_cyc[0]
                                        + _p3_chi_alpha * _ref_mu_cyc["alpha_R_mu"] ** 2
                                        + _p3_chi_gamma * _ref_mu_cyc["gamma_R_mu"] ** 2
                                    ),
                                    float(_ops_mu_cyc[1] - 2.0 * _p3_chi_gamma * _ref_mu_cyc["gamma_R_mu"]),
                                    float(_ops_mu_cyc[2] + 2.0 * _p3_chi_gamma),
                                    float(_ops_mu_cyc[3] - 2.0 * _p3_chi_alpha * _ref_mu_cyc["alpha_R_mu"]),
                                    float(_ops_mu_cyc[4] + 2.0 * _p3_chi_alpha),
                                    float(_ops_mu_cyc[5]),
                                )

                            _tau_cyc_eff = int(max(1, int(_tau_start52)))
                            _alpha_vn_cyc, _gamma_vn_cyc, _vstar_vn_cyc, _iric_vn_cyc = _t52_minimize_state_quadratic_iric(
                                const=float(
                                    _p3_omp * R_escala
                                    + _mt_mu_cyc[0]
                                    + _p3_chi_alpha * _ref_mu_cyc["alpha_N_mu"] ** 2
                                    + _p3_chi_gamma * _ref_mu_cyc["gamma_N_mu"] ** 2
                                ),
                                b_gamma=float(_mt_mu_cyc[1] - 2.0 * _p3_chi_gamma * _ref_mu_cyc["gamma_N_mu"]),
                                q_gamma=float(_mt_mu_cyc[2] + 2.0 * _p3_chi_gamma),
                                b_alpha=float(_mt_mu_cyc[3] - _p3_omp * R_escala - 2.0 * _p3_chi_alpha * _ref_mu_cyc["alpha_N_mu"]),
                                q_alpha=float(_mt_mu_cyc[4] + 2.0 * _p3_chi_alpha),
                                q_gamma_alpha=float(_mt_mu_cyc[5]),
                                mu_tau_v=_mu_cyc,
                                other_value_func=_vr_val_cyc,
                                info_bonus_func=_info_gain_vn_cyc,
                                extra_score_fn=lambda _av, _gv: _p3_omk * _t52_p_kill_exp_at(
                                    _av, _gv, _mu_cyc, _tau_cyc_eff, str(a_k_exec_v), str(a_f_exec_v)
                                ),
                            )
                            _vr_formal_cyc = bool(dict(_iric_vr_cyc or {}).get("feasible", False))
                            _vn_formal_cyc = bool(dict(_iric_vn_cyc or {}).get("feasible", False))
                            if _vr_formal_cyc and _vn_formal_cyc:
                                _rescue_cyc = bool(float(_vstar_vr_cyc) <= float(_vstar_vn_cyc))
                            elif _vr_formal_cyc:
                                _rescue_cyc = True
                            elif _vn_formal_cyc:
                                _rescue_cyc = False
                            else:
                                # No hay solución formal en Γ_t(μ_t); se conserva el
                                # menor valor irrestricto únicamente como diagnóstico.
                                _rescue_cyc = bool(float(_vstar_vr_cyc) <= float(_vstar_vn_cyc))
                            _gamma_formal_cyc = bool(_vr_formal_cyc or _vn_formal_cyc)
                            _info_vr_cyc = _info_gain_vr_cyc(float(_alpha_vr_cyc), float(_gamma_vr_cyc))
                            _info_vn_cyc = _info_gain_vn_cyc(float(_alpha_vn_cyc), float(_gamma_vn_cyc))
                            _info_sel_cyc = _info_vr_cyc if _rescue_cyc else _info_vn_cyc
                            return {
                                "a_s_star": (
                                    "Rescatar"
                                    if _gamma_formal_cyc and _rescue_cyc
                                    else ("No Rescatar" if _gamma_formal_cyc else "Γ vacío")
                                ),
                                "a_s_full": (
                                    "Rescate (a_res)"
                                    if _gamma_formal_cyc and _rescue_cyc
                                    else (
                                        "Negociar (a_neg)"
                                        if _gamma_formal_cyc
                                        else "Γ vacío (sin óptimo formal)"
                                    )
                                ),
                                "a_s_mdg_intent": "Rescatar" if _rescue_cyc else "No Rescatar",
                                "alpha_star": float(_alpha_vr_cyc if _rescue_cyc else _alpha_vn_cyc),
                                "gamma_star": float(_gamma_vr_cyc if _rescue_cyc else _gamma_vn_cyc),
                                "branch": "VR" if _rescue_cyc else "VN",
                                "Gamma_formal": bool(_gamma_formal_cyc),
                                "Gamma_VR": bool(_vr_formal_cyc),
                                "Gamma_VN": bool(_vn_formal_cyc),
                                "V_R": float(_vstar_vr_cyc),
                                "V_N": float(_vstar_vn_cyc),
                                "p_kill_exp": float(_t52_p_kill_exp_at(
                                    float(_alpha_vr_cyc if _rescue_cyc else _alpha_vn_cyc),
                                    float(_gamma_vr_cyc if _rescue_cyc else _gamma_vn_cyc),
                                    _mu_cyc, _tau_cyc_eff,
                                    str(a_k_exec_v), str(a_f_exec_v),
                                )),
                                "p_surv_exp": float(_psurv_exp_cyc),
                                "H_mu": float(_info_sel_cyc.get("H", _t52_shannon_entropy(_mu_cyc))),
                                "E_H_next": float(_info_sel_cyc.get("E_H_next", 0.0)),
                                "Delta_H": float(_info_sel_cyc.get("Delta_H", 0.0)),
                                "Delta_H_VR": float(_info_vr_cyc.get("Delta_H", 0.0)),
                                "Delta_H_VN": float(_info_vn_cyc.get("Delta_H", 0.0)),
                                "psi_H": float(_t52_entropy_weight),
                                "iota": float(_iota_cyc),
                                "theta_hat": str(_theta_hat_cyc),
                                "IRIC_VR": dict(_iric_vr_cyc or {}),
                                "IRIC_VN": dict(_iric_vn_cyc or {}),
                            }

                        _seed_visible52 = int(st.session_state.get("global_semilla_rng", 123))
                        _legacy_reset_counter52 = int(
                            dict(st.session_state.get("dynamic_seed_reset_counts", {})).get(
                                str(_seed_visible52), 0
                            )
                        )
                        _run_counter52 = int(st.session_state.get("dynamic_current_run_counter", 0))
                        if _run_counter52 <= 0:
                            _run_counter52 = int(
                                dict(st.session_state.get("dynamic_run_counter_by_seed", {})).get(
                                    str(_seed_visible52), 1
                                )
                            )
                        _seed_effective52 = int(
                            (
                                _seed_visible52
                                + 1000003 * int(_run_counter52)
                                + 9176 * int(_legacy_reset_counter52)
                            )
                            % (2**31 - 1)
                        )
                        _cycle_seed52 = int(_seed_effective52) + 101
                        _cycle_horizon52 = int(max(2, min(5000, int(st.session_state.get("limite_dias", limite_dias)))))
                        _pi_prior_c52, _pi_tilde_c52, _path_c52, _meta_voice_c52 = generate_incident_voice_scenario(
                            str(tipo_real),
                            _pi_call52,
                            _voz_params52,
                            t_max=_cycle_horizon52 + 1,
                            kappa=float(st.session_state.get("incident_voice_kappa", 30.0)),
                            seed=_cycle_seed52,
                        )
                        _rng_d_c52 = np.random.default_rng(_cycle_seed52 + 17)
                        _rng_m_c52 = np.random.default_rng(_cycle_seed52 + 7919)
                        _ak_star_tau0_fixed52 = _t52_clean_ak(str(_ak52_t0_raw))
                        if _ak_star_tau0_fixed52 not in ("Continuar", "Liberar", "Matar"):
                            _ak_star_tau0_fixed52 = _t52_clean_ak(str(_ak52_intent))
                        if _ak_star_tau0_fixed52 not in ("Continuar", "Liberar", "Matar"):
                            _ak_star_tau0_fixed52 = "Continuar"

                        _t52_traj_rows52: list[dict] = [
                            {
                                "t": 0,
                                "alpha_t": float(_policy0_52["alpha_star"]),
                                "gamma_t": float(_policy0_52["gamma_star"]),
                                **{f"mu_{_th_r}": float(_mu0_52.get(_th_r, 0.25)) for _th_r in TIPOS_SECUESTRADOR},
                            }
                        ]

                        _mu_prior_loop52 = {th: float(_mu1_52.get(th, 0.0)) for th in TIPOS_SECUESTRADOR}
                        # El primer ciclo dinamico (tau=1) debe tomar como base
                        # el cierre observado de tau=0; los ciclos siguientes usan
                        # el cierre del ciclo inmediatamente anterior.
                        _alpha_prev_loop52 = float(_policy0_52["alpha_star"])
                        _gamma_prev_loop52 = float(_policy0_52["gamma_star"])
                        _ak_exec_prev_loop52 = str(_atk0_m52)
                        _as_exec_prev_loop52 = str(_ats0_m52)
                        _af_exec_prev_loop52 = str(_atf0_m52)
                        _cycles52: list[dict[str, Any]] = []
                        _pi_ref_tau1_c52: Optional[dict[str, Any]] = None
                        _stop52: dict[str, Any] = {
                            "motivo": "horizonte",
                            "tau": int(_cycle_horizon52),
                            "m": "Continuar",
                        }
                        # Corre desde τ=1 hasta que m ≠ Continuar o se agote el horizonte.
                        for _tau_start52 in range(1, _cycle_horizon52 + 1):
                            _step_c52 = (
                                _path_c52[_tau_start52]
                                if len(_path_c52) > _tau_start52
                                else (_path_c52[-1] if _path_c52 else {})
                            )
                            _V_c52 = int(_step_c52.get("V_t", 0) or 0)
                            _x_c52_raw = _step_c52.get("x_obs")
                            _x_c52 = np.asarray(_x_c52_raw, dtype=float) if _x_c52_raw is not None else None
                            _af_star_c52, _ucoop_c52, _ucol_c52 = _t52_family_star_for_cycle(
                                _mu_prior_loop52,
                                float(_alpha_prev_loop52),
                                float(_gamma_prev_loop52),
                            )
                            _state_c52 = _t52_state_star_for_cycle(
                                _mu_prior_loop52,
                                float(_alpha_prev_loop52),
                                float(_gamma_prev_loop52),
                                _ak_exec_prev_loop52,
                                _as_exec_prev_loop52,
                                _af_exec_prev_loop52,
                            )
                            _as_star_c52 = str(_state_c52["a_s_star"])
                            _as_full_c52 = str(_state_c52["a_s_full"])
                            _as_mdg_intent_c52 = str(_state_c52.get("a_s_mdg_intent", _as_star_c52))
                            _alpha_star_c52 = float(_state_c52["alpha_star"])
                            _gamma_star_c52 = float(_state_c52["gamma_star"])
                            _pi_ref_compute_c52 = bool(
                                int(_tau_start52) <= 1
                                or st.session_state.get("t52_pi_benchmark_all_cycles", False)
                            )
                            if _pi_ref_compute_c52:
                                _pi_ref_c52 = _t52_perfect_info_state_reference(
                                    str(tipo_real),
                                    int(_tau_start52),
                                    float(_gamma_prev_loop52),
                                    str(_ak_exec_prev_loop52),
                                    str(_as_exec_prev_loop52),
                                    str(_af_exec_prev_loop52),
                                )
                                _pi_ref_c52["source_tau"] = int(_tau_start52)
                                if int(_tau_start52) == 1:
                                    _pi_ref_tau1_c52 = copy.deepcopy(dict(_pi_ref_c52))
                            else:
                                _pi_ref_c52 = (
                                    _t52_reuse_pi_reference(_pi_ref_tau1_c52, int(_tau_start52))
                                    if isinstance(_pi_ref_tau1_c52, dict)
                                    else _t52_blank_pi_reference(str(tipo_real), int(_tau_start52))
                                )
                            _iota_prior_c52 = float(_state_c52["iota"])
                            _theta_prior_c52 = str(_state_c52["theta_hat"])
                            _policy_c52 = _t52_screening_policy(
                                int(_tau_start52),
                                float(_alpha_star_c52),
                                float(_gamma_star_c52),
                            )
                            # Acumular estado actual para Tabla 14 post-loop
                            _t52_traj_rows52.append({
                                "t": int(_tau_start52),
                                "alpha_t": float(_alpha_prev_loop52),
                                "gamma_t": float(_gamma_prev_loop52),
                                **{f"mu_{_th_r}": float(_mu_prior_loop52.get(_th_r, 0.0)) for _th_r in TIPOS_SECUESTRADOR},
                            })
                            # Backward induction con μ_τ actualizado para a*_K(τ)
                            _ak_star_c52 = str(_ak_star_tau0_fixed52)
                            try:
                                _df_k15_cyc52 = st.session_state.get("tab15_k_params_calibrated")
                                _df_traj_base52 = st.session_state.get("rb_mu_traj_snapshot")
                                if (
                                    isinstance(_df_k15_cyc52, pd.DataFrame)
                                    and not _df_k15_cyc52.empty
                                    and isinstance(_df_traj_base52, pd.DataFrame)
                                    and not _df_traj_base52.empty
                                    and "t" in _df_traj_base52.columns
                                ):
                                    _df_traj_ov52 = _df_traj_base52.copy()
                                    _ov_cols52 = {
                                        "t": 0,
                                        "alpha_t": float(_alpha_prev_loop52),
                                        "gamma_t": float(_gamma_prev_loop52),
                                        **{f"mu_{_th_r}": float(_mu_prior_loop52.get(_th_r, 0.0)) for _th_r in TIPOS_SECUESTRADOR},
                                    }
                                    _mask_ov52 = _df_traj_ov52["t"].astype(int) == 0
                                    if _mask_ov52.any():
                                        for _k_ov52, _v_ov52 in _ov_cols52.items():
                                            if _k_ov52 in _df_traj_ov52.columns:
                                                _df_traj_ov52.loc[_mask_ov52, _k_ov52] = _v_ov52
                                    else:
                                        _df_traj_ov52 = pd.concat(
                                            [_df_traj_ov52, pd.DataFrame([_ov_cols52])], ignore_index=True
                                        )
                                    _df_ia_cyc52, _ = kidnapper_backward_induction_k_table(
                                        modelo,
                                        _df_traj_ov52,
                                        _df_k15_cyc52,
                                        tipo_real=str(tipo_real),
                                        beta_k=float(_p3_beta_k),
                                        R=float(R_escala),
                                        t_mad=float(st.session_state.get("cal_T_mad", 30.0)),
                                        T=int(_cycle_horizon52),
                                        alpha_fallback=float(_alpha_prev_loop52),
                                        gamma_fallback=float(_gamma_prev_loop52),
                                        alpha_tab12=float(_alpha_prev_loop52),
                                        ransom_tab12=float(R_escala),
                                    )
                                    if not _df_ia_cyc52.empty:
                                        _row_ia_cyc52 = _df_ia_cyc52.loc[
                                            _df_ia_cyc52["t"].astype(int) == 0
                                        ]
                                        if not _row_ia_cyc52.empty:
                                            _ak_raw_cyc52 = str(_row_ia_cyc52.iloc[0].get("opcion_BW", "—"))
                                            _ak_clean_cyc52 = _t52_clean_ak(_ak_raw_cyc52)
                                            if _ak_clean_cyc52 in ("Continuar", "Liberar", "Matar"):
                                                _ak_star_c52 = _ak_clean_cyc52
                            except Exception:
                                pass
                            _pdet_c52 = _pdet_logit_prob(
                                str(tipo_real),
                                float(_policy_c52["alpha_star"]),
                                float(_policy_c52["gamma_star"]),
                            )
                            _d_c52 = int(float(_rng_d_c52.random()) < max(0.0, min(1.0, _pdet_c52)))
                            _pf_c52 = _t52_p1(_af_star_c52, "F", _iota_prior_c52, _mu0_52, _mu_prior_loop52, tau=int(_tau_start52))
                            _pk_c52 = _t52_p1(_ak_star_c52, "K", _iota_prior_c52, _mu0_52, _mu_prior_loop52, tau=int(_tau_start52))
                            _ps_c52 = _t52_p1_s(_agent52, _as_mdg_intent_c52, _mu0_52, _mu_prior_loop52, tau=int(_tau_start52))
                            _u_f_c52 = float(_rng_m_c52.random())
                            _u_k_c52 = float(_rng_m_c52.random())
                            _u_s_c52 = float(_rng_m_c52.random())
                            _atf_c52 = _t52_realize(_pf_c52, _u_f_c52)
                            _atk_c52 = _t52_realize(_pk_c52, _u_k_c52)
                            _ats_c52 = _t52_realize(_ps_c52, _u_s_c52)
                            _pf_argmax_c52 = max(_pf_c52, key=_pf_c52.get)
                            _pk_argmax_c52 = max(_pk_c52, key=_pk_c52.get)
                            _ps_argmax_c52 = max(_ps_c52, key=_ps_c52.get)
                            _pm_c52, _mfac_c52 = _mechanism_m_probs_for_actions(
                                str(tipo_real),
                                int(_tau_start52),
                                float(_policy_c52["alpha_star"]),
                                float(_policy_c52["gamma_star"]),
                                float(_pdet_c52),
                                str(_atk_c52),
                                str(_ats_c52),
                                str(_atf_c52),
                                z_region=str(st.session_state.z_region),
                                v_victim=str(st.session_state.v_victim),
                                f_capa=str(f_capa),
                                s_tipo=str(s_tipo),
                                policy_sensitivity=float(st.session_state.get("t52_likelihood_policy_sensitivity", 4.0)),
                            )
                            _u_m_c52 = float(_rng_m_c52.random())
                            _m_mode_ignore_stop_c52 = str(st.session_state.get("t52_m_mode", "Sorteo")) == "Continuar"
                            _m_c52 = _t52_realize(_pm_c52, _u_m_c52)
                            _m_argmax_c52 = max(_pm_c52, key=_pm_c52.get)
                            _lambda_raw_c52 = dict(_mfac_c52.get("lam", {}))
                            _lambda_diag_c52 = {
                                "Liberación": float(_lambda_raw_c52.get("Exógeno", 0.0)),
                                "Rescate": float(_lambda_raw_c52.get("Rescate", 0.0)),
                                "Pago": float(_lambda_raw_c52.get("Pago", 0.0)),
                                "Muerte": float(_lambda_raw_c52.get("Muerte", 0.0)),
                                "Continuar": float(_mfac_c52.get("p_cont", 0.0)),
                            }
                            # κh por tipo para el ciclo corriente
                            _kh_signs_c52: dict[str, str] = {}
                            _mt_cyc52 = float(min(1.0, (float(_tau_start52) / max(float(st.session_state.get("cal_T_mad", 30.0)), 1e-9)) ** 2))
                            for _th_kh_c52 in TIPOS_SECUESTRADOR:
                                try:
                                    _zbj_kh_c = (_tab2_bundles52.get(str(_th_kh_c52), {}) or {}).get("zeta_by_j", {}) or {}
                                    _h_kh_c = modelo.calcular_hazards(
                                        int(_tau_start52),
                                        str(_th_kh_c52),
                                        float(_alpha_star_c52),
                                        maturity_mult=_mt_cyc52,
                                        z_region=str(st.session_state.get("z_region", "Andina")),
                                        v_victim=str(st.session_state.get("v_victim", "Privado")),
                                        alpha=float(_alpha_star_c52),
                                        gamma=float(_gamma_star_c52),
                                        zeta_by_j=_zbj_kh_c,
                                    )
                                    _kh_c_val = (
                                        float((_zbj_kh_c.get("Muerte") or {}).get("gamma", 0.0)) * float(_h_kh_c.get("Muerte", 0.0))
                                        + float((_zbj_kh_c.get("Rescate") or {}).get("gamma", 0.0)) * float(_h_kh_c.get("Rescate", 0.0))
                                        - float((_zbj_kh_c.get("Pago") or {}).get("gamma", 0.0)) * float(_h_kh_c.get("Pago", 0.0))
                                    )
                                    _kh_signs_c52[str(_th_kh_c52)] = str(-1 if _kh_c_val > 1e-12 else (1 if _kh_c_val < -1e-12 else 0))
                                except Exception:
                                    _kh_signs_c52[str(_th_kh_c52)] = "—"
                            _li_c52 = _build_t0_implementation_likelihood_by_theta(
                                _mu_prior_loop52,
                                presion_S=float(_policy_c52["gamma_star"]),
                                precision_iota=float(_iota_prior_c52),
                                alpha0=float(_policy_c52["alpha_star"]),
                                gamma0=float(_policy_c52["gamma_star"]),
                                ransom_scale=float(R_escala),
                                f_capa=str(f_capa),
                                estado_duro=(str(s_tipo) == "Duro"),
                                beta_k=float(_p3_beta_k),
                                atilde_F=str(_atf_c52),
                                atilde_K=str(_atk_c52),
                                atilde_S=str(_ats_c52),
                            )
                            _df_post_c52, _mu_post_c52, _meta_post_c52 = build_t0_bayesian_posterior_report(
                                modelo,
                                _mu_prior_loop52,
                                str(_m_c52),
                                int(_d_c52),
                                presion_S=float(_policy_c52["gamma_star"]),
                                z_region=str(st.session_state.z_region),
                                v_victim=str(st.session_state.v_victim),
                                alpha=float(_policy_c52["alpha_star"]),
                                gamma=float(_policy_c52["gamma_star"]),
                                p_det=float(_pdet_c52),
                                zeta_alpha=float(_zp52.get("zeta_alpha", 0.1)),
                                zeta_gamma=float(_zp52.get("zeta_gamma", 0.1)),
                                zeta_d=float(_zp52.get("zeta_d", 0.1)),
                                zeta_R=float(_zp52.get("zeta_R", 0.1)),
                                estado_rescata=str(_ats_c52).strip().lower().startswith("rescat"),
                                t_mad=float(st.session_state.get("cal_T_mad", 30.0)),
                                lambda4=float(st.session_state.get("cal_lambda_4", 0.0005)),
                                t_eval=int(_tau_start52),
                                omega_voz=float(_omega_voz52),
                                pi_call_by_theta=_pi_call52,
                                voz_params_by_theta=_voz_params52,
                                V_t=int(_V_c52),
                                x_obs=_x_c52,
                                atilde_F=_atf_c52,
                                atilde_K=_atk_c52,
                                atilde_S=_ats_c52,
                                implementation_likelihood_by_theta=_li_c52,
                                tab2_bundle_by_theta=_tab2_bundles52,
                                aggregate_unknown_theta=False,
                            )
                            _mu_post_c52 = {
                                th: float(_mu_post_c52.get(th, _mu_prior_loop52.get(th, 0.0)))
                                for th in TIPOS_SECUESTRADOR
                            }
                            _iota_post_c52 = float(max(_mu_post_c52.values())) if _mu_post_c52 else float(_iota_prior_c52)
                            _theta_post_c52 = max(_mu_post_c52, key=_mu_post_c52.get) if _mu_post_c52 else str(_theta_prior_c52)
                            _tau_end52 = int(_tau_start52 + 1)
                            _M_end_c52 = float(min(1.0, (float(_tau_end52) / max(_t52_Tmad52, 1e-9)) ** 2))
                            _M_start_c52 = float(min(1.0, (float(_tau_start52) / max(_t52_Tmad52, 1e-9)) ** 2))
                            _ref_mu_start_c52 = _state_reference_centers(_mu_prior_loop52)
                            _start_vals_c52 = {
                                **{f"μ({_th})": f"{float(_mu_prior_loop52.get(_th, 0.0)):.4f}" for _th in TIPOS_SECUESTRADOR},
                                "α^μ (R)": f"{float(_ref_mu_start_c52['alpha_R_mu']):.4f}",
                                "γ^μ (R)": f"{float(_ref_mu_start_c52['gamma_R_mu']):.4f}",
                                "α^μ (N)": f"{float(_ref_mu_start_c52['alpha_N_mu']):.4f}",
                                "γ^μ (N)": f"{float(_ref_mu_start_c52['gamma_N_mu']):.4f}",
                                "a_S* óptima": str(_as_full_c52),
                                "ã_S": f"{_ats_c52} (u={_u_s_c52:.4f}, p={_ps_c52[_ats_c52]:.4f})",
                                "γ* Estado": f"{float(_gamma_star_c52):.4f}",
                                "α* Estado": f"{float(_alpha_star_c52):.4f}",
                                _gamma_r_pi_label52: _t52_pi_display(_pi_ref_c52, "gamma_vr"),
                                _alpha_r_pi_label52: _t52_pi_display(_pi_ref_c52, "alpha_vr"),
                                _gamma_n_pi_label52: _t52_pi_display(_pi_ref_c52, "gamma_vn"),
                                _alpha_n_pi_label52: _t52_pi_display(_pi_ref_c52, "alpha_vn"),
                                "a_F*": str(_af_star_c52),
                                "ã_F": f"{_atf_c52} (u={_u_f_c52:.4f}, p={_pf_c52[_atf_c52]:.4f})",
                                f"a_K* ({tipo_real})": str(_ak_star_c52),
                                f"ã_K ({tipo_real})": f"{_atk_c52} (u={_u_k_c52:.4f}, p={_pk_c52[_atk_c52]:.4f})",
                                "V (voz)": str(_V_c52),
                                "d (det.)": str(_d_c52),
                                "ι": f"{float(_iota_prior_c52):.4f}",
                                "H(μ)": f"{float(_state_c52.get('H_mu', _t52_shannon_entropy(_mu_prior_loop52))):.4f}",
                                "ΔH Estado": f"{float(_state_c52.get('Delta_H', 0.0)):.4f}",
                                "M_t": f"{_M_start_c52:.4f}",
                                "m": f"{_m_c52} (u={_u_m_c52:.4f}, p={_pm_c52[_m_c52]:.4f})",
                            }
                            _end_vals_c52 = {
                                **{f"μ({_th})": f"{float(_mu_post_c52.get(_th, 0.0)):.4f}" for _th in TIPOS_SECUESTRADOR},
                                "γ* Estado": f"{float(_gamma_star_c52):.4f}",
                                "α* Estado": f"{float(_alpha_star_c52):.4f}",
                                _gamma_r_pi_label52: "—",
                                _alpha_r_pi_label52: "—",
                                _gamma_n_pi_label52: "—",
                                _alpha_n_pi_label52: "—",
                                "ι": f"{_iota_post_c52:.4f}",
                                "H(μ)": f"{float(_t52_shannon_entropy(_mu_post_c52)):.4f}",
                                "ΔH Estado": "—",
                                "M_t": f"{_M_end_c52:.4f}",
                                "V (voz)": "—",
                                "d (det.)": "—",
                                "a_F*": "—",
                                "ã_F": "—",
                                f"a_K* ({tipo_real})": "—",
                                f"ã_K ({tipo_real})": "—",
                                "a_S* óptima": "—",
                                "ã_S": "—",
                                "m": "—",
                            }
                            _cycles52.append(
                                {
                                    "cycle": int(_tau_start52),
                                    "tau_start": int(_tau_start52),
                                    "tau_end": int(_tau_end52),
                                    "start_vals": dict(_start_vals_c52),
                                    "end_vals": dict(_end_vals_c52),
                                    "post54": {
                                        "df": _df_post_c52.copy(),
                                        "meta": dict(_meta_post_c52),
                                        "mu_prior": dict(_mu_prior_loop52),
                                        "mu_post": dict(_mu_post_c52),
                                        "V": int(_V_c52),
                                        "d": int(_d_c52),
                                        "p_det": float(_pdet_c52),
                                    },
                                    "diag": {
                                        "alpha_usado": float(_alpha_prev_loop52),
                                        "gamma_usado": float(_gamma_prev_loop52),
                                        "alpha_prev_usado": float(_alpha_prev_loop52),
                                        "gamma_prev_usado": float(_gamma_prev_loop52),
                                        "alpha_optimo": float(_alpha_star_c52),
                                        "gamma_optimo": float(_gamma_star_c52),
                                        "V_nuevo": int(_V_c52),
                                        "d_nuevo": int(_d_c52),
                                        "p_det": float(_pdet_c52),
                                        "a_F_star": str(_af_star_c52),
                                        "a_K_star": str(_ak_star_c52),
                                        "a_S_star": str(_as_full_c52),
                                        "a_S_mdg_intent": str(_as_mdg_intent_c52),
                                        "Gamma_formal": bool(_state_c52.get("Gamma_formal", False)),
                                        "Gamma_VR": bool(_state_c52.get("Gamma_VR", False)),
                                        "Gamma_VN": bool(_state_c52.get("Gamma_VN", False)),
                                        "U_coop": float(_ucoop_c52),
                                        "U_col": float(_ucol_c52),
                                        "V_R": float(_state_c52.get("V_R", np.nan)),
                                        "V_N": float(_state_c52.get("V_N", np.nan)),
                                        "H_mu": float(_state_c52.get("H_mu", np.nan)),
                                        "E_H_next": float(_state_c52.get("E_H_next", np.nan)),
                                        "Delta_H": float(_state_c52.get("Delta_H", 0.0)),
                                        "Delta_H_VR": float(_state_c52.get("Delta_H_VR", 0.0)),
                                        "Delta_H_VN": float(_state_c52.get("Delta_H_VN", 0.0)),
                                        "psi_H": float(_state_c52.get("psi_H", _t52_entropy_weight)),
                                        "pi_ref": dict(_pi_ref_c52),
                                        "pi_ref_computed": bool(_pi_ref_compute_c52),
                                        "pi_ref_reused_from_tau": int(dict(_pi_ref_c52).get("reused_from_tau", 0) or 0),
                                        "alpha_R_pi_ref": float(_pi_ref_c52["alpha_vr"]),
                                        "gamma_R_pi_ref": float(_pi_ref_c52["gamma_vr"]),
                                        "alpha_N_pi_ref": float(_pi_ref_c52["alpha_vn"]),
                                        "gamma_N_pi_ref": float(_pi_ref_c52["gamma_vn"]),
                                        "V_R_pi_ref": float(_pi_ref_c52["V_R"]),
                                        "V_N_pi_ref": float(_pi_ref_c52["V_N"]),
                                        "iota_post": float(_iota_post_c52),
                                        "theta_modal_post": str(_theta_post_c52),
                                        "m": str(_m_c52),
                                        "p_m": float(_pm_c52[_m_c52]),
                                        "u_m": float(_u_m_c52),
                                        "m_mode": "Continuar" if _m_mode_ignore_stop_c52 else "Sorteo",
                                        "m_ignore_stop": bool(_m_mode_ignore_stop_c52),
                                        "iota_prior": float(_iota_prior_c52),
                                        "theta_prior": str(_theta_prior_c52),
                                        "theta_generator_hidden": str(tipo_real),
                                        "pf_probs": dict(_pf_c52),
                                        "pk_probs": dict(_pk_c52),
                                        "ps_probs": dict(_ps_c52),
                                        "pm_probs": dict(_pm_c52),
                                        "implementation_likelihood": dict(_li_c52),
                                        "m_factors": dict(_mfac_c52),
                                        "lambda_diag": dict(_lambda_diag_c52),
                                        "kh_signs": dict(_kh_signs_c52),
                                        "pf_argmax": str(_pf_argmax_c52),
                                        "pk_argmax": str(_pk_argmax_c52),
                                        "ps_argmax": str(_ps_argmax_c52),
                                        "m_argmax": str(_m_argmax_c52),
                                        "u_f": float(_u_f_c52),
                                        "u_k": float(_u_k_c52),
                                        "u_s": float(_u_s_c52),
                                        "atf": str(_atf_c52),
                                        "atk": str(_atk_c52),
                                        "ats": str(_ats_c52),
                                        "IRIC": dict(
                                            _t52_iric_status(
                                                float(_alpha_star_c52),
                                                float(_gamma_star_c52),
                                                dict(_mu_prior_loop52),
                                                float(_state_c52.get("V_R", np.nan)),
                                                float(_state_c52.get("V_N", np.nan)),
                                            )
                                        ),
                                    },
                                }
                            )
                            _mu_prior_loop52 = dict(_mu_post_c52)
                            _alpha_prev_loop52 = float(_policy_c52["alpha_star"])
                            _gamma_prev_loop52 = float(_policy_c52["gamma_star"])
                            _ak_exec_prev_loop52 = str(_atk_c52)
                            _as_exec_prev_loop52 = str(_ats_c52)
                            _af_exec_prev_loop52 = str(_atf_c52)
                            if (
                                str(_m_c52).strip().lower() != "continuar"
                                and not _m_mode_ignore_stop_c52
                            ):
                                _stop52 = {
                                    "motivo": "desenlace",
                                    "tau": int(_tau_start52),
                                    "m": str(_m_c52),
                                    "p_m": float(_pm_c52[_m_c52]),
                                    "u_m": float(_u_m_c52),
                                }
                                break
                        # ── Actualizar Tabla 14 con trayectoria real y refrescar Tabla 15 ──
                        if _t52_traj_rows52:
                            try:
                                _df_traj14_52 = pd.DataFrame(_t52_traj_rows52)
                                _df_traj14_52 = _rb_attach_mu_traj_epi_columns(
                                    _df_traj14_52,
                                    tipo_real=str(tipo_real),
                                    t0_gamma=float(_gamma_prev_loop52),
                                    t0_alpha=float(_alpha_prev_loop52),
                                    iota_t0=float(_iota_52),
                                    kc_k12=_kc_k12,
                                    ps_k12=_ps_k12,
                                    pf_k12=_pf_k12,
                                    p3_mdg_agent=_p3_mdg_agent,
                                )
                                st.session_state["rb_mu_traj_snapshot"] = _df_traj14_52
                                st.session_state["rb_mu_traj_sig"] = _mu_traj_sig
                                _run_kidnapper_backward_induction_cached.clear()
                            except Exception:
                                pass
                        st.session_state["dynamic_cycles52"] = list(_cycles52)
                        st.session_state["dynamic_cycles_diag52"] = [dict(c.get("diag", {})) for c in _cycles52]
                        st.session_state["dynamic_cycles_stop52"] = dict(_stop52)
                        st.session_state["dynamic_cycles_run_meta52"] = {
                            "seed_visible": int(_seed_visible52),
                            "run_counter": int(_run_counter52),
                            "reset_counter": int(_legacy_reset_counter52),
                            "seed_effective": int(_seed_effective52),
                            "saved": False,
                        }
                        if _cycles52:
                            _first_c52 = _cycles52[0]
                            st.session_state["first_cycle_tau1_52"] = dict(_first_c52.get("start_vals", {}))
                            st.session_state["first_cycle_table52"] = dict(_first_c52.get("end_vals", {}))
                            st.session_state["first_cycle_diag52"] = {
                                "ciclo": "primer ciclo",
                                "inicio": "τ=1 en V (voz)",
                                "cierre": "τ=2 en M_t",
                                **dict(_first_c52.get("diag", {})),
                            }
                            st.session_state["first_cycle_post54"] = dict(_first_c52.get("post54", {}))
                        st.session_state["first_cycle_voice_meta"] = dict(_meta_voice_c52)
                        if bool(st.session_state.pop("first_cycle_pending_rerun", False)):
                            st.rerun()
                    except Exception as _err_c52:
                        st.session_state.pop("first_cycle_tau1_52", None)
                        st.session_state["first_cycle_table52"] = {"__error__": str(_err_c52)}
                        st.session_state["dynamic_cycles52"] = [{"__error__": str(_err_c52)}]
                _t52_dynamic_cycles = st.session_state.get("dynamic_cycles52") or []
                _t52_cycle_vals = st.session_state.get("first_cycle_table52") or {}

            _t52_tau1_cycle_vals = st.session_state.get("first_cycle_tau1_52") or {}
            _t52_cycle_lookup: dict[int, dict[str, Any]] = {}
            if isinstance(_t52_dynamic_cycles, list):
                for _cy_lookup52 in _t52_dynamic_cycles:
                    if isinstance(_cy_lookup52, dict) and "__error__" not in _cy_lookup52:
                        try:
                            _t52_cycle_lookup[int(_cy_lookup52.get("tau_start", 0))] = _cy_lookup52
                        except Exception:
                            pass

            def _t52_dynamic_cell_tip(
                tau_v: int,
                var_v: str,
                val_v: Any,
                cy_v: Optional[dict[str, Any]] = None,
                prev_cy_v: Optional[dict[str, Any]] = None,
            ) -> str:
                _val_e = html.escape(str(val_v))
                _var_e = html.escape(str(var_v))
                _cy = cy_v if isinstance(cy_v, dict) else {}
                _prev = prev_cy_v if isinstance(prev_cy_v, dict) else {}
                _diag = dict(_cy.get("diag", {})) if isinstance(_cy.get("diag", {}), dict) else {}
                _prev_diag = dict(_prev.get("diag", {})) if isinstance(_prev.get("diag", {}), dict) else {}
                _hdr = f"{_var_e} en τ={int(tau_v)}"
                _source = "ciclo vigente"
                _formula = ""
                _detail = ""
                if var_v == "ã_F" and isinstance(_diag.get("pf_probs"), dict):
                    _eu_dyn_f = (
                        float(_diag["pf_probs"].get("Cooperar", 0.0)) * float(_diag.get("U_coop", 0.0))
                        + float(_diag["pf_probs"].get("Coludir", 0.0)) * float(_diag.get("U_col", 0.0))
                    )
                    return _t52_tip_html(
                        "F",
                        str(_diag.get("a_F_star", "Cooperar")),
                        dict(_diag.get("pf_probs", {})),
                        str(_diag.get("atf", val_v)).split(" ")[0],
                        str(_diag.get("pf_argmax", "—")),
                        float(_diag.get("iota_prior", 0.0)),
                        float(_eu_dyn_f),
                        float(_diag.get("u_f", 0.0)),
                        f"(U_coop={float(_diag.get('U_coop', 0.0)):.4f}, U_col={float(_diag.get('U_col', 0.0)):.4f})",
                    )
                if str(var_v).startswith("ã_K") and isinstance(_diag.get("pk_probs"), dict):
                    _eu_dyn_k = (
                        float(_diag["pk_probs"].get("Liberar", 0.0)) * float(_ur52t)
                        + float(_diag["pk_probs"].get("Matar", 0.0)) * float(_uk52t)
                        + float(_diag["pk_probs"].get("Continuar", 0.0)) * float(_vc52t)
                    )
                    return _t52_tip_html(
                        f"K({tipo_real})",
                        str(_diag.get("a_K_star", _ak52_intent)),
                        dict(_diag.get("pk_probs", {})),
                        str(_diag.get("atk", val_v)).split(" ")[0],
                        str(_diag.get("pk_argmax", "—")),
                        float(_diag.get("iota_prior", 0.0)),
                        float(_eu_dyn_k),
                        float(_diag.get("u_k", 0.0)),
                        f"(U_rel={float(_ur52t):.4f}, U_kill={float(_uk52t):.4f}, V_cont={float(_vc52t):.4f})",
                    )
                if var_v == "ã_S" and isinstance(_diag.get("ps_probs"), dict):
                    _eu_dyn_s = (
                        float(_diag["ps_probs"].get("Rescatar", 0.0)) * (-float(_diag.get("V_R", 0.0)))
                        + float(_diag["ps_probs"].get("No Rescatar", 0.0)) * (-float(_diag.get("V_N", 0.0)))
                    )
                    _s_intent_dyn = str(
                        _diag.get(
                            "a_S_mdg_intent",
                            "Rescatar" if "Rescate" in str(_diag.get("a_S_star", "")) else "No Rescatar",
                        )
                    )
                    return _t52_tip_html(
                        "S",
                        _s_intent_dyn,
                        dict(_diag.get("ps_probs", {})),
                        str(_diag.get("ats", val_v)).split(" ")[0],
                        str(_diag.get("ps_argmax", "—")),
                        float(_diag.get("iota_prior", 0.0)),
                        float(_eu_dyn_s),
                        float(_diag.get("u_s", 0.0)),
                        f"(−V_R={-float(_diag.get('V_R', 0.0)):.4f}, −V_N={-float(_diag.get('V_N', 0.0)):.4f})",
                    )
                if var_v == "m" and isinstance(_diag.get("pm_probs"), dict):
                    _pm_dyn = dict(_diag.get("pm_probs", {}))
                    _lambda_dyn = dict(_diag.get("lambda_diag", {}))
                    _mfac_dyn = dict(_diag.get("m_factors", {}))
                    _u_dyn = float(_diag.get("u_m", 0.0))
                    _m_dyn = str(_diag.get("m", val_v))
                    _argm_dyn = str(_diag.get("m_argmax", "—"))
                    _ignore_stop_dyn = bool(_diag.get("m_ignore_stop", False))
                    _psi_rows_dyn = ""
                    for _lbl_dyn, _p_dyn in _pm_dyn.items():
                        _pv_dyn = float(_lambda_dyn.get(_lbl_dyn, 0.0))
                        _is_arg_dyn = str(_lbl_dyn) == _argm_dyn
                        _wst_dyn = " font-weight:700;background:#2a3a2a;" if _is_arg_dyn else ""
                        _wmk_dyn = " &#9733;" if _is_arg_dyn else ""
                        _psi_rows_dyn += (
                            f"<tr style='{_wst_dyn}'>"
                            f"<td>{html.escape(str(_lbl_dyn))}{_wmk_dyn}</td>"
                            f"<td class='num'>{_pv_dyn:.8f}</td>"
                            f"<td class='num'>{float(_mfac_dyn.get('q', 0.0)):.4f}</td>"
                            f"<td class='num'>{float(_p_dyn):.4f}</td>"
                            f"</tr>"
                        )
                    _int_rows_dyn = ""
                    _cur_dyn = 0.0
                    _items_dyn = list(_pm_dyn.items())
                    for _i_dyn, (_lbl_dyn, _prob_dyn) in enumerate(_items_dyn):
                        _lo_dyn, _hi_dyn = _cur_dyn, _cur_dyn + float(_prob_dyn)
                        _last_dyn = _i_dyn == len(_items_dyn) - 1
                        _hit_dyn = (_lo_dyn <= _u_dyn < _hi_dyn) or (_last_dyn and _u_dyn >= _lo_dyn)
                        _hi_str_dyn = "1.0000" if _last_dyn else f"{_hi_dyn:.4f}"
                        _smk_dyn = '<span class="star">&#9733;</span>' if str(_lbl_dyn) == _argm_dyn else ""
                        _dmk_dyn = '<span class="dart">&#127919;</span>' if _hit_dyn else ""
                        _rc_dyn = ' class="hit"' if _hit_dyn else ""
                        _int_rows_dyn += (
                            f"<tr{_rc_dyn}>"
                            f"<td class='iv'>[{_lo_dyn:.4f},&nbsp;{_hi_str_dyn})</td>"
                            f"<td>{html.escape(str(_lbl_dyn))}</td>"
                            f"<td class='num'>{float(_prob_dyn):.4f}</td>"
                            f"<td>{_smk_dyn}{_dmk_dyn}</td></tr>"
                        )
                        _cur_dyn += float(_prob_dyn)
                    _m_mode_dyn_html = (
                        f'<div class="sec">Modo Continuar: parada desactivada</div>'
                        f'<div class="ux">El selector <b>Modo m</b> está en <b>Continuar</b>: '
                        f'\\(m_{{{int(tau_v)}}}\\) se sortea normalmente, entra en '
                        f'\\(\\mathcal{{L}}_H(m_{{{int(tau_v)}}})\\), pero no activa parada aunque '
                        f'\\(m_{{{int(tau_v)}}}\\ne\\mathrm{{Continuar}}\\).</div>'
                        f'<div class="sec">Intervalos acumulados \\([l_o,\\,h_i)\\)</div>'
                        f'<table class="int"><tr><td>Intervalo</td><td>m</td><td>P</td><td></td></tr>{_int_rows_dyn}</table>'
                        f'<div class="draw">\\(u_m={_u_dyn:.4f}\\) &rarr; '
                        f'\\(m_{{{int(tau_v)}}}=\\text{{{html.escape(_m_dyn)}}}\\) &#127919;</div>'
                        if _ignore_stop_dyn
                        else (
                            f'<div class="sec">Intervalos acumulados \\([l_o,\\,h_i)\\)</div>'
                            f'<table class="int"><tr><td>Intervalo</td><td>m</td><td>P</td><td></td></tr>{_int_rows_dyn}</table>'
                            f'<div class="draw">\\(u_m={_u_dyn:.4f}\\) &rarr; '
                            f'\\(m_{{{int(tau_v)}}}=\\text{{{html.escape(_m_dyn)}}}\\) &#127919;</div>'
                        )
                    )
                    return (
                        f'<div class="t52h">'
                        f'<div class="hdr">Desenlace \\(m\\) en \\(\\tau={int(tau_v)}\\) — Fase 2</div>'
                        f'<div class="sec">Tripleta ejecutada del ciclo</div>'
                        f'<div>\\((\\tilde a_F,\\tilde a_K,\\tilde a_S)=('
                        f'{html.escape(str(_diag.get("atf", "—")))}, '
                        f'{html.escape(str(_diag.get("atk", "—")))}, '
                        f'{html.escape(str(_diag.get("ats", "—")))})\\)</div>'
                        f'<div class="sec">Cálculo · Ecuaciones (28) y (29)</div>'
                        f'<div>\\(P^E(m_t=\\mathrm{{Cont}})=p_{{\\mathrm{{Cont}},t}}\\), '
                        f'\\(P^E(m_t=j)=h_j(t\\mid\\theta_K,\\mathcal C_t)\\).</div>'
                        f'<div class="ux">\\(\\theta_K={html.escape(str(_diag.get("theta_prior", "—")))}\\), '
                        f'\\(\\alpha^*={float(_diag.get("alpha_optimo", 0.0)):.4f}\\), '
                        f'\\(\\gamma^*={float(_diag.get("gamma_optimo", 0.0)):.4f}\\), '
                        f'\\(\\iota={float(_diag.get("iota_prior", 0.0)):.4f}\\).</div>'
                        f'{_t52_mechanism_m_tooltip_lines(_mfac_dyn, int(tau_v), float(_diag.get("alpha_optimo", 0.0)), float(_diag.get("gamma_optimo", 0.0)), float(_diag.get("p_det", 0.0)))}'
                        f'<div class="sec">Hazards competitivos</div>'
                        f'<table><tr><td>m</td><td>λ/pCont</td><td>q</td><td>P</td></tr>{_psi_rows_dyn}</table>'
                        f'{_m_mode_dyn_html}'
                        f'<div class="argm">argmax \\(P^E\\): <b>{html.escape(_argm_dyn)}</b> &#9733;</div>'
                        f'</div>'
                    )
                if str(var_v).startswith("μ("):
                    _source = "posterior heredada" if int(tau_v) >= 2 else "posterior del ciclo base"
                    _formula = (
                        rf'\(\mu_{{{int(tau_v)}}}(\theta)=\mu_{{{int(tau_v)-1}}}(\theta)'
                        rf'\mathcal{{L}}_F(\theta)\mathcal{{L}}_C(\theta)/Z_{{{int(tau_v)-1}}}\)'
                    )
                    _detail = (
                        "Se calcula después de observar el ciclo anterior y alimenta el ciclo vigente."
                    )
                elif var_v == "ι":
                    _source = "precisión posterior"
                    _formula = rf'\(\iota_{{{int(tau_v)}}}=\max_\theta\mu_{{{int(tau_v)}}}(\theta)\)'
                    _detail = "Resume qué tan concentrada quedó la creencia posterior."
                elif var_v in (_gamma_r_pi_label52, _alpha_r_pi_label52, _gamma_n_pi_label52, _alpha_n_pi_label52):
                    _source = "benchmark de información perfecta"
                    _is_gamma_pi = var_v in (_gamma_r_pi_label52, _gamma_n_pi_label52)
                    _is_rescue_pi = var_v in (_gamma_r_pi_label52, _alpha_r_pi_label52)
                    _sym_pi = r"\gamma" if _is_gamma_pi else r"\alpha"
                    _branch_pi = "R" if _is_rescue_pi else "N"
                    _diag_key = f'{"gamma" if _is_gamma_pi else "alpha"}_{_branch_pi}_pi_ref'
                    _val_pi = float(_diag.get(_diag_key, 0.0))
                    _formula = rf'\({_sym_pi}_{{{_branch_pi}}}^{{\theta^\ast,*}}=\arg\min_{{\alpha,\gamma}} V_{{{_branch_pi}}}(\alpha,\gamma;\theta^\ast)\), con \(\mu^{{PI}}(\theta^\ast)=1\)'
                    _detail = (
                        rf'\(\theta^\ast=\mathrm{{{html.escape(str(tipo_real))}}}\), '
                        rf'rama {_branch_pi}, '
                        rf'\(V_R^{{PI}}={float(_diag.get("V_R_pi_ref", 0.0)):.4f}\), '
                        rf'\(V_N^{{PI}}={float(_diag.get("V_N_pi_ref", 0.0)):.4f}\), '
                        rf'\({_sym_pi}_{{{_branch_pi}}}^{{\theta^\ast,*}}={_val_pi:.4f}\). '
                        r'Es referencia: no alimenta Tabla 10 ni la posterior.'
                    )
                elif var_v == "H(μ)":
                    _source = "entropía de Shannon"
                    _formula = rf'\(H(\mu_{{{int(tau_v)}}})=-\sum_\theta \mu_{{{int(tau_v)}}}(\theta)\ln\mu_{{{int(tau_v)}}}(\theta)\)'
                    _detail = rf'\(H={float(_diag.get("H_mu", 0.0)):.4f}\). Mide incertidumbre sobre el tipo del captor.'
                elif var_v == "ΔH Estado":
                    _source = "control dual informacional"
                    _formula = rf'\(\Delta H_{{{int(tau_v)}}}=H(\mu_{{{int(tau_v)}}})-\mathbb E_t[H(\mu_{{{int(tau_v)+1}}})]\)'
                    _detail = (
                        rf'\(E[H]={float(_diag.get("E_H_next", 0.0)):.4f}\), '
                        rf'\(\Delta H={float(_diag.get("Delta_H", 0.0)):.4f}\), '
                        rf'\(\psi_H={float(_diag.get("psi_H", _t52_entropy_weight)):.1f}\). '
                        r'Entra como descuento \(-\psi_H\Delta H\) en el argmin del Estado.'
                    )
                elif var_v == "M_t":
                    _source = "maduración temporal"
                    _formula = rf'\(M_{{{int(tau_v)}}}=\min\{{1,({int(tau_v)}/T_{{mad}})^2\}}\)'
                    _detail = "Escala los hazards competitivos del periodo."
                elif var_v == "V (voz)":
                    _source = "señal nueva del ciclo"
                    _formula = (
                        rf'\(V_{{{int(tau_v)}}}\sim Bernoulli(\sum_\theta '
                        rf'\mu_{{{int(tau_v)}}}(\theta)P(V=1\mid\theta,\alpha_{{{int(tau_v)}}},\gamma_{{{int(tau_v)}}}))\)'
                    )
                    _detail = "Se genera al iniciar el ciclo y entra en la verosimilitud de comunicación."
                elif var_v == "d (det.)":
                    _source = "señal nueva del ciclo"
                    _formula = rf'\(d_{{{int(tau_v)}}}\sim Bernoulli(P_{{det,{int(tau_v)}}})\)'
                    _detail = (
                        rf'\(P_{{det,{int(tau_v)}}}=\Lambda(\eta_0+\eta_1\alpha^*_{{{int(tau_v)}}}'
                        rf'+\eta_2\gamma^*_{{{int(tau_v)}}})\).'
                    )
                elif var_v == "a_F*":
                    _source = "óptimo de familia"
                    _formula = rf'\(a_F^\ast=\arg\max\{{U_{{{int(tau_v)}}}^F(a_{{coop}}),U_{{{int(tau_v)}}}^F(a_{{col}})\}}\)'
                    _detail = (
                        rf'\(U_{{coop}}={float(_diag.get("U_coop", 0.0)):.4f}\), '
                        rf'\(U_{{col}}={float(_diag.get("U_col", 0.0)):.4f}\).'
                    )
                elif var_v.startswith("a_K*"):
                    _source = "óptimo del secuestrador"
                    _formula = r'\(a_K^\ast\leftarrow\mathrm{Tabla\ 15},\ \tau=0,\ \mathrm{columna\ 14}\)'
                    _detail = "Por regla del modelo, se usa siempre el óptimo reportado allí."
                elif var_v == "a_S* óptima":
                    _source = "óptimo del Estado"
                    _formula = rf'\(a_S^\ast=\mathrm{{Rescate}}\ si\ V_R^\ast\le V_N^\ast;\ \mathrm{{Negociar}}\ si\ V_R^\ast>V_N^\ast\)'
                    _detail = (
                        rf'\(V_R^\ast={float(_diag.get("V_R", 0.0)):.4f}\), '
                        rf'\(V_N^\ast={float(_diag.get("V_N", 0.0)):.4f}\).'
                    )
                elif var_v == "ã_F":
                    _source = "implementación MDG de familia"
                    _formula = r'\(\tilde a_F\sim P_I^F(\tilde a_F\mid a_F^\ast)\)'
                    _detail = "Ley de Implementación de Tabla 7 con la creencia vigente del ciclo."
                elif var_v.startswith("ã_K"):
                    _source = "implementación MDG del secuestrador"
                    _formula = r'\(\tilde a_K\sim P_I^K(\tilde a_K\mid a_K^\ast)\)'
                    _detail = "Ley de Implementación de Tabla 7 con acciones [Continuar, Liberar, Matar]."
                elif var_v == "ã_S":
                    _source = "implementación MDG del Estado"
                    _formula = r'\(\tilde a_S\sim P_I^S(\tilde a_S\mid a_S^\ast)\)'
                    _detail = "Ley de Implementación de Tabla 7 con acciones [Rescatar, No Rescatar]."
                elif var_v == "m":
                    _source = "materialización del desenlace"
                    _formula = (
                        rf'\(m_{{{int(tau_v)}}}\sim P^E(m=j\mid '
                        rf'\tilde a_F,\tilde a_K,\tilde a_S,\alpha^*_{{{int(tau_v)}}},\gamma^*_{{{int(tau_v)}}},\iota_{{{int(tau_v)}}},\theta_K)\)'
                    )
                    _detail = (
                        rf'\(u_m={float(_diag.get("u_m", 0.0)):.4f}\), '
                        rf'\(p(m)={float(_diag.get("p_m", 0.0)):.4f}\).'
                    )
                elif var_v == _t52_gamma_row_label:
                    _iric_dyn = _diag.get("IRIC", {}) if isinstance(_diag.get("IRIC"), dict) else {}
                    _source = "auditoría formal del óptimo del Estado"
                    _formula = r'\(\Gamma_t(\mu_t)=1 \Longleftrightarrow IR^K \land IC^K \land IR^F\)'
                    _detail = (
                        rf'\(IR^K_E={float(_iric_dyn.get("IR_K_gap_E", float("nan"))):.4f}\), '
                        rf'\(IC^K_{{E,\min}}={float(_iric_dyn.get("IC_K_gap_E_min", float("nan"))):.4f}\), '
                        rf'\(IR^F_E={float(_iric_dyn.get("IR_F_gap_E", float("nan"))):.4f}\).'
                    )
                elif var_v == _t52_ir_true_row_label:
                    _iric_dyn = _diag.get("IRIC", {}) if isinstance(_diag.get("IRIC"), dict) else {}
                    _source = "auditoría puntual del tipo verdadero"
                    _formula = (
                        rf'\(IR^K(\theta^*)=1 \Longleftrightarrow '
                        rf'U^K_{{rel}}(\theta^*)-\max\{{V^K_{{cont}}(\theta^*),U^K_{{kill}}(\theta^*)\}}\ge0\)'
                    )
                    _detail = (
                        rf'\(\theta^*=\mathrm{{{html.escape(str(tipo_real))}}}\), '
                        rf'\(\alpha_t={float(_diag.get("alpha_optimo", _diag.get("alpha_prev_usado", float("nan")))):.4f}\), '
                        rf'\(\gamma_t={float(_diag.get("gamma_optimo", _diag.get("gamma_prev_usado", float("nan")))):.4f}\). '
                        rf'\(U^K_{{rel}}={float(_iric_dyn.get("IR_K_true_U_rel", float("nan"))):.4f}\), '
                        rf'\(V^K_{{cont}}={float(_iric_dyn.get("IR_K_true_V_cont", float("nan"))):.4f}\), '
                        rf'\(U^K_{{kill}}={float(_iric_dyn.get("IR_K_true_U_kill", float("nan"))):.4f}\), '
                        rf'\(gap={float(_iric_dyn.get("IR_K_true_gap", float("nan"))):.4f}\).'
                    )
                elif var_v == "γ* Estado":
                    _source = "instrumento óptimo heredado"
                    _formula = rf'\(\gamma_{{{int(tau_v)}}}^\ast\) es el óptimo matemático del Estado.'
                    _detail = (
                        rf'Óptimo calculado en el ciclo anterior: '
                        rf'\(\gamma^\ast={float(_prev_diag.get("gamma_optimo", _diag.get("gamma_prev_usado", 0.0))):.4f}\).'
                    )
                elif var_v == "α* Estado":
                    _source = "instrumento óptimo heredado"
                    _formula = rf'\(\alpha_{{{int(tau_v)}}}^\ast\) es el óptimo matemático del Estado.'
                    _detail = (
                        rf'Óptimo calculado en el ciclo anterior: '
                        rf'\(\alpha^\ast={float(_prev_diag.get("alpha_optimo", _diag.get("alpha_prev_usado", 0.0))):.4f}\).'
                    )
                elif str(var_v).startswith("−sgn(κh)"):
                    # κ_h tooltip for cycle columns — sign-only (components not stored in cycle diag)
                    _kh_th_c = str(var_v).replace("−sgn(κh) (", "").rstrip(")")
                    _kh_signs_c = _diag.get("kh_signs", {}) if isinstance(_diag.get("kh_signs"), dict) else {}
                    _kh_sgn_c = str(_kh_signs_c.get(_kh_th_c, val_v)).strip()
                    try:
                        _kh_sgn_int_c = int(_kh_sgn_c)
                    except Exception:
                        _kh_sgn_int_c = None
                    if _kh_sgn_int_c == -1:
                        _kh_interp_c = (
                            r"γ↑ empeora la posición del Estado: "
                            r"aumenta la hazard de Muerte y/o Rescate relativas a Pago."
                        )
                        _kh_color_c = "#c0392b"
                        _kh_sgn_label_c = "−1"
                    elif _kh_sgn_int_c == 1:
                        _kh_interp_c = (
                            r"γ↑ mejora la posición del Estado: "
                            r"la presión operacional reduce la hazard de Muerte relativa a Pago."
                        )
                        _kh_color_c = "#27ae60"
                        _kh_sgn_label_c = "+1"
                    else:
                        _kh_interp_c = (
                            r"El efecto neto de γ sobre los hazards ponderados es nulo "
                            r"o indeterminado en este período."
                        )
                        _kh_color_c = "#7f8c8d"
                        _kh_sgn_label_c = _kh_sgn_c if _kh_sgn_c else "—"
                    _mt_c_str = f"{float(min(1.0, (float(tau_v) / max(float(st.session_state.get('cal_T_mad', 30.0)), 1e-9)) ** 2)):.4f}"
                    return (
                        f'<div class="t52h">'
                        f'<div class="hdr">−sgn(κ_h) · θ={html.escape(_kh_th_c)} · τ={int(tau_v)}</div>'
                        f'<div class="sec">Fórmula</div>'
                        f'<div>\\(\\kappa_h(\\theta,t)=\\zeta_\\gamma^{{(2)}}\\tilde{{\\lambda}}_2'
                        f'+\\zeta_\\gamma^{{(3)}}\\tilde{{\\lambda}}_3-\\zeta_\\gamma^{{(1)}}\\tilde{{\\lambda}}_1\\)</div>'
                        f'<div class="sec">Cálculo · τ={int(tau_v)}</div>'
                        f'<div>\\(M_{{{int(tau_v)}}}={_mt_c_str}\\) '
                        f'(filtro maduración). '
                        f'Los hazards efectivos \\(\\tilde{{\\lambda}}_j=M_t\\cdot\\lambda_j\\) '
                        f'escalan con \\(M_t\\).</div>'
                        f'<div class="sec">Resultado</div>'
                        f'<div>\\(\\kappa_h\\) calculado con el tipo θ={html.escape(_kh_th_c)} '
                        f'y los parámetros óptimos de este ciclo.</div>'
                        f'<div style="margin:4px 0;font-size:1.1em;font-weight:700;'
                        f'color:{_kh_color_c};">−sgn(κ_h) = {html.escape(_kh_sgn_label_c)}</div>'
                        f'<div class="sec">Interpretación</div>'
                        f'<div class="ux">{html.escape(_kh_interp_c)}</div>'
                        f'<div class="ux">Fuente: <b>kh_signs</b> del diag del ciclo τ={int(tau_v)}.</div>'
                        f'</div>'
                    )
                else:
                    _formula = "Celda dinámica generada por el bloque de ciclos."
                    _detail = "Se actualiza al presionar Avanzar ciclos."
                return (
                    f'<div class="t52h">'
                    f'<div class="hdr">{_hdr}</div>'
                    f'<div class="sec">Fuente</div>'
                    f'<div>{html.escape(_source)}</div>'
                    f'<div class="sec">Cálculo</div>'
                    f'<div>{_formula}</div>'
                    f'<div class="sec">Valor reportado</div>'
                    f'<div><b>{_val_e}</b></div>'
                    f'<div class="ux">{_detail}</div>'
                    f'</div>'
                )

            if isinstance(_t52_tau1_cycle_vals, dict) and _t52_tau1_cycle_vals:
                for _key52, _val52 in _t52_tau1_cycle_vals.items():
                    if _key52 in _t52_vals:
                        _t52_vals[_key52] = str(_val52)
                        _t52_tips[_key52] = _t52_dynamic_cell_tip(
                            1, str(_key52), str(_val52), _t52_cycle_lookup.get(1)
                        )
                _t52_all_ast_ready = True

            _t52_extra_tau_vals: dict[int, dict[str, str]] = {}
            _t52_extra_tau_prev_cycle: dict[int, dict[str, Any]] = {}
            _t52_extra_tau_cycle: dict[int, dict[str, Any]] = {}
            if isinstance(_t52_dynamic_cycles, list):
                _cycles_clean52 = [
                    _cy52 for _cy52 in _t52_dynamic_cycles
                    if isinstance(_cy52, dict) and "__error__" not in _cy52
                ]
                for _idx_cy52, _cy52 in enumerate(_cycles_clean52):
                    if not isinstance(_cy52, dict) or "__error__" in _cy52:
                        continue
                    try:
                        _tau_start52 = int(_cy52.get("tau_start", 0))
                    except Exception:
                        continue
                    if _tau_start52 >= 1:
                        _col_vals52: dict[str, str] = {}
                        if _idx_cy52 > 0:
                            _prev_cy52 = _cycles_clean52[_idx_cy52 - 1]
                            try:
                                if int(_prev_cy52.get("tau_end", -1)) == _tau_start52:
                                    _col_vals52.update(
                                        {str(k): str(v) for k, v in dict(_prev_cy52.get("end_vals", {})).items()}
                                    )
                                    _t52_extra_tau_prev_cycle[_tau_start52] = _prev_cy52
                            except Exception:
                                pass
                        _col_vals52.update({str(k): str(v) for k, v in dict(_cy52.get("start_vals", {})).items()})
                        _diag_cy52 = _cy52.get("diag", {}) if isinstance(_cy52.get("diag"), dict) else {}
                        _iric_cy52 = _diag_cy52.get("IRIC", {}) if isinstance(_diag_cy52.get("IRIC"), dict) else {}
                        _col_vals52[_t52_gamma_row_label] = _t52_yes_no(
                            bool(_iric_cy52.get("Gamma_formal", False))
                        )
                        _col_vals52[_t52_ir_true_row_label] = _t52_yes_no(
                            bool(_iric_cy52.get("IR_K_true", False))
                        )
                        _kh_signs_cy52 = _diag_cy52.get("kh_signs", {}) if isinstance(_diag_cy52.get("kh_signs"), dict) else {}
                        for _th52_kh in TIPOS_SECUESTRADOR:
                            _col_vals52[f"−sgn(κh) ({_th52_kh})"] = str(_kh_signs_cy52.get(str(_th52_kh), "—"))
                        _t52_extra_tau_vals.setdefault(_tau_start52, {}).update(_col_vals52)
                        _t52_extra_tau_cycle[_tau_start52] = _cy52
            elif _t52_cycle_vals and "__error__" not in _t52_cycle_vals:
                _t52_extra_tau_vals[2] = {str(k): str(v) for k, v in dict(_t52_cycle_vals).items()}
            _t52_extra_taus = sorted(_t52_extra_tau_vals)

            _t52_vals = {
                _key52: _t52_vals[_key52]
                for _key52 in _t52_row_order
                if _key52 in _t52_vals
            }

            def _t52_attr(content: str) -> str:
                """Escapa HTML para embeber en atributo data-tiphtml."""
                return html.escape(str(content), quote=True)

            def _t52_var_tex(label: Any) -> str:
                """Etiqueta LaTeX para la columna Variable de Tabla 5.2."""
                _label = str(label)
                _tipo_tex = r"\mathrm{" + str(tipo_real).replace("_", r"\_") + "}"
                if _label.startswith("μ(") and _label.endswith(")"):
                    _th = _label[2:-1].replace("_", r"\_")
                    return r"\mu(\mathrm{" + _th + "})"
                _map = {
                    "α^μ (R)": r"\alpha_R^\mu",
                    "γ^μ (R)": r"\gamma_R^\mu",
                    "α^μ (N)": r"\alpha_N^\mu",
                    "γ^μ (N)": r"\gamma_N^\mu",
                    "a_S* óptima": r"a_S^\ast\ \mathrm{optima}",
                    "ã_S": r"\tilde a_S",
                    "γ* Estado": r"\gamma_t^\ast\ \mathrm{Estado}",
                    "α* Estado": r"\alpha_t^\ast\ \mathrm{Estado}",
                    _gamma_r_pi_label52: rf"\gamma_R^{{{_tipo_tex},\ast}}",
                    _alpha_r_pi_label52: rf"\alpha_R^{{{_tipo_tex},\ast}}",
                    _gamma_n_pi_label52: rf"\gamma_N^{{{_tipo_tex},\ast}}",
                    _alpha_n_pi_label52: rf"\alpha_N^{{{_tipo_tex},\ast}}",
                    "ι": r"\iota",
                    "H(μ)": r"H(\mu)",
                    "ΔH Estado": r"\Delta H\ \mathrm{Estado}",
                    "M_t": r"M_t",
                    "V (voz)": r"V\ \mathrm{(voz)}",
                    "d (det.)": r"d\ \mathrm{(det.)}",
                    "a_F*": r"a_F^\ast",
                    "ã_F": r"\tilde a_F",
                    f"a_K* ({tipo_real})": rf"a_K^\ast({_tipo_tex})",
                    f"ã_K ({tipo_real})": rf"\tilde a_K({_tipo_tex})",
                    _t52_gamma_row_label: r"\Gamma_t(\mu_t)\ \mathrm{bajo\ EV}",
                    _t52_ir_true_row_label: rf"IR^K({_tipo_tex})\ \mathrm{{tipo\ verdadero}}",
                    "m": r"m",
                    **{
                        f"−sgn(κh) ({_th52_kh})": (
                            r"-\operatorname{sgn}(\kappa_h(\mathrm{" + str(_th52_kh) + r"},t))"
                        )
                        for _th52_kh in TIPOS_SECUESTRADOR
                    },
                }
                return _map.get(_label, html.escape(_label))

            _t52_rows_html = ""
            for _vname52, _vval52 in _t52_vals.items():
                _form52    = _t52_formulas.get(_vname52, "")
                _valtip52  = _t52_tips.get(_vname52, "")
                _valtip0_52 = _t52_tips0.get(_vname52, "")
                _vname_e  = html.escape(str(_vname52))
                _vname_tex_e = html.escape(_t52_var_tex(_vname52), quote=True)
                _vname_html52 = (
                    f'<span class="t52-var-katex" data-katex="{_vname_tex_e}">{_vname_e}</span>'
                )
                _vval0_e  = html.escape(str(_t52_vals0.get(_vname52, "—")))
                _hide_tau1_exec52 = (
                    _vname52 in {"ã_F", f"ã_K ({tipo_real})"}
                    and not (
                        isinstance(_t52_tau1_cycle_vals, dict)
                        and _vname52 in _t52_tau1_cycle_vals
                    )
                )
                _vval_e   = "" if _hide_tau1_exec52 else html.escape(str(_vval52))

                if _form52:
                    _name_td = (
                        f'<td class="tc" data-tiphtml="{_t52_attr(_form52)}" '
                        f'style="padding:5px 10px;border:1px solid #ddd;'
                        f'white-space:nowrap;font-weight:600;">{_vname_html52}</td>'
                    )
                else:
                    _name_td = (
                        f'<td style="padding:5px 10px;border:1px solid #ddd;'
                        f'white-space:nowrap;font-weight:600;">{_vname_html52}</td>'
                    )

                if _valtip0_52:
                    _val0_td = (
                        f'<td class="tc" data-tiphtml="{_t52_attr(_valtip0_52)}" '
                        f'style="padding:5px 10px;border:1px solid #ddd;">{_vval0_e}</td>'
                    )
                else:
                    _val0_td = (
                        f'<td style="padding:5px 10px;border:1px solid #ddd;">{_vval0_e}</td>'
                    )

                if _valtip52 and not _hide_tau1_exec52:
                    _val_td = (
                        f'<td class="tc" data-tiphtml="{_t52_attr(_valtip52)}" '
                        f'style="padding:5px 10px;border:1px solid #ddd;">{_vval_e}</td>'
                    )
                else:
                    _val_td = (
                        f'<td style="padding:5px 10px;border:1px solid #ddd;">{_vval_e}</td>'
                    )

                _cycle_td = ""
                for _tau_extra52 in _t52_extra_taus:
                    _extra_vals52 = _t52_extra_tau_vals.get(_tau_extra52, {})
                    _vval2_e = html.escape(str(_extra_vals52.get(_vname52, "—")))
                    _cycle_tip52 = _t52_dynamic_cell_tip(
                        int(_tau_extra52),
                        str(_vname52),
                        _extra_vals52.get(_vname52, "—"),
                        _t52_extra_tau_cycle.get(_tau_extra52),
                        _t52_extra_tau_prev_cycle.get(_tau_extra52),
                    )
                    _cycle_td += (
                        f'<td class="tc" data-tiphtml="{_t52_attr(_cycle_tip52)}" '
                        f'style="padding:5px 10px;border:1px solid #ddd;">{_vval2_e}</td>'
                    )

                _t52_rows_html += f"<tr>{_name_td}{_val0_td}{_cycle_td}</tr>\n"

            _t52_comp_h = (len(_t52_vals) + 1) * 32 + 8   # filas + encab.; pegado a 5.3
            _t52_extra_headers = "".join(f"<th>τ = {_tau_extra52}</th>" for _tau_extra52 in _t52_extra_taus)
            _t52_total_cols = 2 + len(_t52_extra_taus)
            _t52_min_width_px = max(760, 210 + 138 * _t52_total_cols)

            _t52_component_html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<link rel="stylesheet" href="{_KATEX_BASE}/katex.min.css">
<script defer src="{_KATEX_BASE}/katex.min.js"></script>
<script defer src="{_KATEX_BASE}/contrib/auto-render.min.js"
  onload="renderMathInElement(document.body,{{delimiters:[{{left:'\\\\(',right:'\\\\)',display:false}}]}})">
</script>
<style>
  html, body {{ margin:0; padding:0; font-family:sans-serif; font-size:13px; overflow:hidden; }}
  .t52-scroll {{
    width:100%;
    overflow-x:auto;
    overflow-y:visible;
    padding-bottom:6px;
    box-sizing:border-box;
  }}
  .t52-scroll::-webkit-scrollbar {{ height:10px; }}
  .t52-scroll::-webkit-scrollbar-thumb {{ background:#b9c0cc; border-radius:999px; }}
  .t52-scroll::-webkit-scrollbar-track {{ background:#eef1f5; border-radius:999px; }}
  table {{ border-collapse:collapse; width:max-content; min-width:{_t52_min_width_px}px; }}
  th {{ padding:6px 10px; border:1px solid #ccc; background:#f0f2f6;
        text-align:left; font-size:0.92em; white-space:nowrap; }}
  td {{ padding:5px 10px; border:1px solid #ddd; vertical-align:middle; font-size:0.90em; white-space:nowrap; }}
  .t52-scroll > table th:first-child,
  .t52-scroll > table td:first-child {{
    min-width:180px;
    max-width:260px;
    width:210px;
  }}
  .t52-scroll > table th:first-child,
  .t52-scroll > table td:first-child {{
    position:sticky;
    left:0;
    z-index:2;
    background:#ffffff;
    box-shadow:1px 0 0 #ddd;
  }}
  .t52-scroll > table th:first-child {{ z-index:3; background:#f0f2f6; }}
  .linkref {{
    display:inline-block; margin-left:6px; padding:1px 6px;
    border-radius:10px; background:#eef3fb; color:#315b9a;
    font-size:0.76em; font-weight:600; white-space:nowrap;
  }}
  .t52-var-katex {{ display:inline-block; min-height:1.25em; }}
  .t52-var-katex .katex {{ font-size:1.02em; }}
  .tc {{ cursor:help; }}
  .tc:hover {{ background:#f7f9ff; }}
  #t52tip {{
    display:none; position:fixed;
    background:#1e1e2e; color:#e8e8f0;
    padding:10px 14px; border-radius:8px;
    border:1px solid #555;
    font-size:0.80em; line-height:1.65;
    max-width:480px; z-index:9999;
    box-shadow:0 6px 20px rgba(0,0,0,0.55);
    pointer-events:none;
  }}
  /* ── Tooltip inner styles ── */
  #t52tip .t52h {{ font-family:sans-serif; }}
  #t52tip .hdr  {{ font-size:0.88em; margin-bottom:5px; color:#b8c4e8; }}
  #t52tip .sec  {{ font-size:0.82em; color:#88ccff; margin:6px 0 2px; border-top:1px solid #444; padding-top:4px; }}
  #t52tip table {{ border-collapse:collapse; margin:2px 0; width:auto; min-width:0; background:transparent; }}
  #t52tip tr,
  #t52tip th,
  #t52tip td {{
    position:static !important;
    left:auto !important;
    z-index:auto !important;
    box-shadow:none !important;
    background:transparent;
  }}
  #t52tip th,
  #t52tip td    {{ padding:1px 7px; border:none; color:#e8e8f0; font-size:0.87em; white-space:nowrap; }}
  #t52tip td.num{{ font-family:monospace; text-align:right; }}
  #t52tip td.iv {{ font-family:monospace; white-space:nowrap; }}
  #t52tip tr.hit,
  #t52tip tr.hit td{{ background:rgba(100,200,120,0.18) !important; }}
  #t52tip tr[style] td{{ background:inherit; }}
  #t52tip .katex {{ color:#e8e8f0; }}
  #t52tip .star {{ color:#f5c842; }}
  #t52tip .dart {{ color:#ff9; }}
  #t52tip .draw {{ margin-top:6px; color:#b8ffb8; }}
  #t52tip .argm {{ color:#e8c060; font-size:0.85em; }}
  #t52tip .eu   {{ margin-top:4px; color:#88ddff; font-size:0.87em; }}
  #t52tip .ux   {{ color:#aaa; font-size:0.80em; margin-top:2px; }}
</style>
</head><body>
<div id="t52tip"></div>
<div class="t52-scroll">
<table>
<thead><tr>
  <th>Variable</th>
  <th>τ = 0</th>
  {_t52_extra_headers}
</tr></thead>
<tbody>
{_t52_rows_html}
</tbody>
</table>
</div>
<script>
(function(){{
  var tip = document.getElementById('t52tip');
  function renderVarLabels(){{
    if(!window.katex) return;
    document.querySelectorAll('.t52-var-katex').forEach(function(el){{
      var src = el.getAttribute('data-katex') || el.textContent || '';
      try {{
        katex.render(src, el, {{ displayMode:false, throwOnError:false }});
      }} catch(e) {{}}
    }});
  }}
  function renderKatex(el){{
    if(window.renderMathInElement){{
      renderMathInElement(el,{{delimiters:[{{left:'\\\\(',right:'\\\\)',display:false}}]}});
    }}
  }}
  function show(e){{
    var raw = e.currentTarget.getAttribute('data-tiphtml') || '';
    if(!raw) return;
    tip.innerHTML = raw;
    renderKatex(tip);
    tip.style.display = 'block';
    position(e);
  }}
      function position(e){{
        var w = tip.offsetWidth  || 480;
        var h = tip.offsetHeight || 260;
        var tbl = document.querySelector('.t52-scroll') || document.querySelector('table');
        var r = tbl ? tbl.getBoundingClientRect() : {{left:0, top:0, width:window.innerWidth, height:window.innerHeight}};
        var x = r.left + (r.width / 2) - (w / 2);
        var y = r.top + (r.height / 2) - (h / 2);
        x = Math.max(8, Math.min(window.innerWidth - w - 8, x));
        y = Math.max(8, Math.min(window.innerHeight - h - 8, y));
        tip.style.left = x + 'px';
        tip.style.top  = y + 'px';
      }}
  function hide(){{ tip.style.display = 'none'; }}
      document.querySelectorAll('.tc').forEach(function(el){{
        el.addEventListener('mouseenter', show);
        el.addEventListener('mouseleave', hide);
      }});
      if(document.readyState === 'loading') {{
        document.addEventListener('DOMContentLoaded', renderVarLabels);
      }} else {{
        setTimeout(renderVarLabels, 0);
      }}
}})();
</script>
</body></html>"""
            components.html(_t52_component_html, height=_t52_comp_h, scrolling=False)
            st.markdown(
                '<div style="margin-top:-22px;margin-bottom:-10px;"></div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                "**Tabla 5.3 · Probabilidades requeridas para calcular la pérdida del Estado**"
            )
            _stop52_view = st.session_state.get("dynamic_cycles_stop52")
            _cycles52_view = st.session_state.get("dynamic_cycles52")
            if isinstance(_stop52_view, dict) and isinstance(_cycles52_view, list) and _cycles52_view:
                _n_ciclos52 = len(_cycles52_view)
                _tau_stop52 = int(_stop52_view.get("tau", 0))
                _m_stop52   = str(_stop52_view.get("m", "—"))
                _pm_stop52  = float(_stop52_view.get("p_m", 0.0))
                _um_stop52  = float(_stop52_view.get("u_m", 0.0))
                if str(_stop52_view.get("motivo", "")).lower() == "desenlace":
                    _desenlace_labels = {
                        "Liberación": "Liberación exógena",
                        "Rescate":    "Rescate por el Estado",
                        "Pago":       "Pago de rescate",
                        "Muerte":     "Muerte del rehén",
                    }
                    _m_label52 = _desenlace_labels.get(_m_stop52, _m_stop52)
                    st.success(
                        f"**Simulación terminada** · {_n_ciclos52} ciclo(s) generados  ·  "
                        f"Último ciclo: **τ = {_tau_stop52}**  ·  "
                        f"Desenlace que detuvo la simulación: **{_m_label52}**  ·  "
                        f"p(m={_m_stop52}) = {_pm_stop52:.4f}, u = {_um_stop52:.4f}"
                    )
                else:
                    _ignore_stop_msg52 = (
                        " La parada por m estaba desactivada; cada m sorteado entró en la verosimilitud."
                        if str(st.session_state.get("t52_m_mode", "Sorteo")) == "Continuar"
                        else ""
                    )
                    st.info(
                        f"**Simulación finalizada por horizonte máximo** · "
                        f"{_n_ciclos52} ciclo(s) generados hasta τ = {_tau_stop52 - 1}; "
                        "m = Continuar en todos los períodos."
                        f"{_ignore_stop_msg52}"
                    )

            def _t53_norm_mu(mu_raw: dict[str, float]) -> dict[str, float]:
                vals = {str(k): max(0.0, float(v)) for k, v in dict(mu_raw).items()}
                total = float(sum(vals.values()))
                if total <= 1e-12:
                    return {th: 1.0 / len(TIPOS_SECUESTRADOR) for th in TIPOS_SECUESTRADOR}
                return {th: float(vals.get(th, 0.0)) / total for th in TIPOS_SECUESTRADOR}

            def _t53_state_exec_label(a_s: str) -> str:
                return "Rescatar" if str(a_s).strip().lower().startswith("rescat") else "No Rescatar"

            def _t53_row_inputs(tau: int) -> tuple[dict[str, float], float, str, float, float]:
                if int(tau) <= 1:
                    mu_tau0 = _t53_norm_mu(dict(_mu0_52))
                    iota_tau0 = float(max(mu_tau0.values())) if mu_tau0 else float(_iota_52)
                    theta_hat_tau0 = max(mu_tau0, key=mu_tau0.get) if mu_tau0 else str(tipo_real)
                    return (
                        mu_tau0,
                        iota_tau0,
                        theta_hat_tau0,
                        float(_t0_gamma_eff),
                        float(_t0_alpha_eff),
                    )
                df_mu = st.session_state.get("tab15_mu_snapshot")
                if not isinstance(df_mu, pd.DataFrame) or df_mu.empty:
                    df_mu = st.session_state.get("rb_mu_traj_snapshot")
                mu_tau = dict(_mu0_52)
                gamma_tau = float(_t0_gamma_eff)
                alpha_tau = float(_t0_alpha_eff)
                if isinstance(df_mu, pd.DataFrame) and not df_mu.empty and "t" in df_mu.columns:
                    _rw = df_mu[df_mu["t"].astype(int) == int(tau)]
                    if not _rw.empty:
                        row = _rw.iloc[0]
                        mu_tau = {
                            th: float(row.get(f"mu_{th}", _mu0_52.get(th, 0.0)))
                            for th in TIPOS_SECUESTRADOR
                        }
                        gamma_tau = float(row.get("gamma_t", gamma_tau))
                        alpha_tau = float(row.get("alpha_t", alpha_tau))
                mu_tau = _t53_norm_mu(mu_tau)
                iota_tau = float(max(mu_tau.values())) if mu_tau else float(_iota_52)
                theta_hat_tau = max(mu_tau, key=mu_tau.get) if mu_tau else str(tipo_real)
                return mu_tau, iota_tau, theta_hat_tau, gamma_tau, alpha_tau

            def _t53_table10_probs(
                mu_tau: dict[str, float],
                iota_tau: float,
                gamma_tau: float,
                alpha_tau: float,
                a_k_exec: str,
                a_s_exec: str,
                a_f_exec: str,
            ) -> tuple[dict[str, float], dict[str, float]]:
                """Pkill física sin expectativa Qneg y p_surv primitiva, ambas por tipo."""
                _ = alpha_tau
                _theta_hat53 = max(mu_tau, key=lambda k: float(mu_tau.get(k, 0.0)))
                _iota_c53 = float(max(0.0, min(1.0, iota_tau)))
                _pkill_by_type53: dict[str, float] = {}
                _psurv_by_type53: dict[str, float] = {}
                for _th_surv53 in TIPOS_SECUESTRADOR:
                    _psurv_by_type53[_th_surv53] = _p_surv_precision_logit(
                        _th_surv53, _iota_c53, _theta_hat53
                    )
                    _pkill_by_type53[_th_surv53] = float(
                        _outcome_probs_for_actions(
                            str(_th_surv53),
                            float(gamma_tau),
                            float(iota_tau),
                            str(a_k_exec),
                            str(a_s_exec),
                            str(a_f_exec),
                        )["kill"]
                    )
                return (
                    {
                        th: float(max(0.0, min(1.0, _pkill_by_type53.get(th, 0.0))))
                        for th in TIPOS_SECUESTRADOR
                    },
                    {
                        th: float(max(0.0, min(1.0, _psurv_by_type53.get(th, 0.0))))
                        for th in TIPOS_SECUESTRADOR
                    },
                )

            def _t53_minimize_state_quadratic(
                const: float,
                b_gamma: float,
                q_gamma: float,
                b_alpha: float,
                q_alpha: float,
                q_gamma_alpha: float,
                info_bonus_func=None,
            ) -> tuple[float, float, float]:
                """Minimiza const + b_g*g + .5*q_g*g^2 + b_a*a + .5*q_a*a^2 + q_ga*g*a en [0,1]^2."""
                candidates: set[tuple[float, float]] = {
                    (0.0, 0.0),
                    (0.0, 1.0),
                    (1.0, 0.0),
                    (1.0, 1.0),
                }
                if float(_t52_entropy_weight) > 0.0 and info_bonus_func is not None:
                    for _g_grid in np.linspace(0.0, 1.0, 21):
                        for _a_grid in np.linspace(0.0, 1.0, 21):
                            candidates.add((round(float(_g_grid), 8), round(float(_a_grid), 8)))

                det = float(q_gamma * q_alpha - q_gamma_alpha * q_gamma_alpha)
                if abs(det) > 1e-12:
                    gamma_int = float((-b_gamma * q_alpha + q_gamma_alpha * b_alpha) / det)
                    alpha_int = float((q_gamma_alpha * b_gamma - q_gamma * b_alpha) / det)
                    if 0.0 <= gamma_int <= 1.0 and 0.0 <= alpha_int <= 1.0:
                        candidates.add((round(gamma_int, 8), round(alpha_int, 8)))

                for alpha_fix in (0.0, 1.0):
                    if abs(q_gamma) > 1e-12:
                        gamma_b = float(-(b_gamma + q_gamma_alpha * alpha_fix) / q_gamma)
                    else:
                        gamma_b = 0.0 if b_gamma + q_gamma_alpha * alpha_fix >= 0.0 else 1.0
                    candidates.add((round(float(min(1.0, max(0.0, gamma_b))), 8), alpha_fix))

                for gamma_fix in (0.0, 1.0):
                    if abs(q_alpha) > 1e-12:
                        alpha_b = float(-(b_alpha + q_gamma_alpha * gamma_fix) / q_alpha)
                    else:
                        alpha_b = 0.0 if b_alpha + q_gamma_alpha * gamma_fix >= 0.0 else 1.0
                    candidates.add((gamma_fix, round(float(min(1.0, max(0.0, alpha_b))), 8)))

                def _raw_val(gamma_v: float, alpha_v: float) -> float:
                    return float(
                        const
                        + b_gamma * gamma_v
                        + 0.5 * q_gamma * gamma_v * gamma_v
                        + b_alpha * alpha_v
                        + 0.5 * q_alpha * alpha_v * alpha_v
                        + q_gamma_alpha * gamma_v * alpha_v
                    )

                def _score(gamma_v: float, alpha_v: float) -> float:
                    bonus = 0.0
                    if float(_t52_entropy_weight) > 0.0 and info_bonus_func is not None:
                        try:
                            bonus = float(info_bonus_func(float(alpha_v), float(gamma_v)).get("Delta_H", 0.0))
                        except Exception:
                            bonus = 0.0
                    return float(_raw_val(gamma_v, alpha_v) - float(_t52_entropy_weight) * bonus)

                gamma_star, alpha_star = min(candidates, key=lambda x: _score(x[0], x[1]))
                return float(alpha_star), float(gamma_star), _score(gamma_star, alpha_star)

            def _t53_quad_value(
                gamma_v: float,
                alpha_v: float,
                const: float,
                b_gamma: float,
                q_gamma: float,
                b_alpha: float,
                q_alpha: float,
                q_gamma_alpha: float,
            ) -> float:
                return float(
                    const
                    + b_gamma * gamma_v
                    + 0.5 * q_gamma * gamma_v * gamma_v
                    + b_alpha * alpha_v
                    + 0.5 * q_alpha * alpha_v * alpha_v
                    + q_gamma_alpha * gamma_v * alpha_v
                )

            def _t53_bool_label(ok: bool) -> str:
                return "Sí" if bool(ok) else "No"

            def _t53_iric_status(
                alpha_v: float,
                gamma_v: float,
                mu_tau_v: dict[str, float],
                V_R_v: float,
                V_N_v: float,
            ) -> dict[str, Any]:
                try:
                    _df_k_cand = refresh_kidnapper_endogenous_columns(
                        _df_p3_k_params.copy(),
                        modelo,
                        float(gamma_v),
                        float(gamma_v),
                        alpha=float(alpha_v),
                    )
                    _df_util_cand = kidnapper_util_df_from_param_df(
                        _df_k_cand,
                        modelo,
                        float(gamma_v),
                        float(alpha_v),
                        float(gamma_v),
                        float(R_escala),
                        str(tipo_real),
                        float(_p3_beta_k),
                    )
                    _mu_ir_raw = {str(k): max(0.0, float(v)) for k, v in dict(mu_tau_v).items()}
                    _mu_ir_sum = float(sum(_mu_ir_raw.values()))
                    _mu_ir = (
                        {th: float(_mu_ir_raw.get(th, 0.0)) / _mu_ir_sum for th in TIPOS_SECUESTRADOR}
                        if _mu_ir_sum > 1e-12
                        else {th: 1.0 / len(TIPOS_SECUESTRADOR) for th in TIPOS_SECUESTRADOR}
                    )
                    _ir_gap = 0.0
                    for _, _ri_ir in _df_util_cand.iterrows():
                        _theta_ir = str(_ri_ir["theta_K"])
                        _outside_criminal = max(float(_ri_ir["V_cont"]), float(_ri_ir["U_kill"]))
                        _ir_gap += float(_mu_ir.get(_theta_ir, 0.0)) * (
                            float(_ri_ir["U_rel"]) - _outside_criminal
                        )
                    _ir_k = bool(not _df_util_cand.empty and _ir_gap >= -1e-9)
                    _mu_ic_raw = {str(k): max(0.0, float(v)) for k, v in dict(mu_tau_v).items()}
                    _mu_ic_sum = float(sum(_mu_ic_raw.values()))
                    _mu_ic = (
                        {th: float(_mu_ic_raw.get(th, 0.0)) / _mu_ic_sum for th in TIPOS_SECUESTRADOR}
                        if _mu_ic_sum > 1e-12
                        else {th: 1.0 / len(TIPOS_SECUESTRADOR) for th in TIPOS_SECUESTRADOR}
                    )
                    _ic_k_expected: list[bool] = []
                    _util_cols = {
                        "Liberar (a_rel)": "U_rel",
                        "Matar (a_kill)": "U_kill",
                        "Continuar (a_cont)": "V_cont",
                    }
                    for _, _rj in _df_util_cand.iterrows():
                        _gain_j = 0.0
                        for _, _ri in _df_util_cand.iterrows():
                            _theta_i = str(_ri["theta_K"])
                            _best_i = max(float(_ri["U_rel"]), float(_ri["U_kill"]), float(_ri["V_cont"]))
                            _col_j = _util_cols.get(str(_rj["rama_optima"]), "V_cont")
                            _gain_j += float(_mu_ic.get(_theta_i, 0.0)) * (_best_i - float(_ri[_col_j]))
                        _ic_k_expected.append(_gain_j >= -1e-9)
                    _ic_k = bool(all(_ic_k_expected) if _ic_k_expected else True)
                except Exception:
                    _ir_k = False
                    _ic_k = False

                try:
                    _df_f_cand, _ = compute_family_table(
                        modelo,
                        _t53_norm_mu(mu_tau_v),
                        float(gamma_v),
                        float(_p3_vl),
                        float(R_escala),
                        float(gamma_v),
                        float(_p3_phi_f),
                        float(_p3_kappa_f),
                        float(_p3_nu_f),
                        float(_p3_fcol),
                        float(_p3_pd0),
                        float(_p3_pda),
                        float(alpha_v),
                        float(cmh_alive),
                    )
                    _ucoop = float(_df_f_cand.loc[_df_f_cand["Rama"].str.startswith("Cooperar"), "EU ilustrativa"].iloc[0])
                    _ucol = float(_df_f_cand.loc[_df_f_cand["Rama"].str.startswith("Colusión"), "EU ilustrativa"].iloc[0])
                    _ir_f = bool(_ucoop >= _ucol)
                    _ic_f = bool(np.isfinite(_ucoop) and np.isfinite(_ucol))
                except Exception:
                    _ucoop = float("nan")
                    _ucol = float("nan")
                    _ir_f = False
                    _ic_f = False

                _ic_s = bool(np.isfinite(float(V_R_v)) and np.isfinite(float(V_N_v)))
                _gamma_formal = bool(_ir_k and _ic_k and _ir_f)
                return {
                    "IR_K": _ir_k,
                    "IC_K": _ic_k,
                    "IR_F": _ir_f,
                    "IC_F": _ic_f,
                    "IC_S": _ic_s,
                    "U_coop": _ucoop,
                    "U_col": _ucol,
                    "Gamma_formal": _gamma_formal,
                    "feasible": _gamma_formal,
                    "audit_feasible": bool(_gamma_formal and _ic_f and _ic_s),
                }

            def _t53_minimize_state_quadratic_iric(
                const: float,
                b_gamma: float,
                q_gamma: float,
                b_alpha: float,
                q_alpha: float,
                q_gamma_alpha: float,
                mu_tau_v: dict[str, float],
                other_value_func,
                info_bonus_func=None,
            ) -> tuple[float, float, float, dict[str, Any]]:
                """Minimiza sobre Γ_t; si Γ_t queda vacío, retorna un diagnóstico no factible."""
                _cands: set[tuple[float, float]] = set()
                _grid = np.linspace(0.0, 1.0, 21)
                for _g in _grid:
                    for _a in _grid:
                        _cands.add((round(float(_g), 8), round(float(_a), 8)))
                det = float(q_gamma * q_alpha - q_gamma_alpha * q_gamma_alpha)
                if abs(det) > 1e-12:
                    _g_int = float((-b_gamma * q_alpha + q_gamma_alpha * b_alpha) / det)
                    _a_int = float((q_gamma_alpha * b_gamma - q_gamma * b_alpha) / det)
                    if 0.0 <= _g_int <= 1.0 and 0.0 <= _a_int <= 1.0:
                        _cands.add((round(_g_int, 8), round(_a_int, 8)))
                for _a_fix in (0.0, 1.0):
                    _g_b = (
                        float(-(b_gamma + q_gamma_alpha * _a_fix) / q_gamma)
                        if abs(q_gamma) > 1e-12
                        else (0.0 if b_gamma + q_gamma_alpha * _a_fix >= 0.0 else 1.0)
                    )
                    _cands.add((round(float(min(1.0, max(0.0, _g_b))), 8), round(_a_fix, 8)))
                for _g_fix in (0.0, 1.0):
                    _a_b = (
                        float(-(b_alpha + q_gamma_alpha * _g_fix) / q_alpha)
                        if abs(q_alpha) > 1e-12
                        else (0.0 if b_alpha + q_gamma_alpha * _g_fix >= 0.0 else 1.0)
                    )
                    _cands.add((round(_g_fix, 8), round(float(min(1.0, max(0.0, _a_b))), 8)))

                _best_any: Optional[tuple[float, float, float, dict[str, Any]]] = None
                _best_feas: Optional[tuple[float, float, float, dict[str, Any]]] = None
                for _g_v, _a_v in _cands:
                    _val_v = _t53_quad_value(_g_v, _a_v, const, b_gamma, q_gamma, b_alpha, q_alpha, q_gamma_alpha)
                    _bonus_v = 0.0
                    if float(_t52_entropy_weight) > 0.0 and info_bonus_func is not None:
                        try:
                            _bonus_v = float(info_bonus_func(float(_a_v), float(_g_v)).get("Delta_H", 0.0))
                        except Exception:
                            _bonus_v = 0.0
                    _score_v = float(_val_v - float(_t52_entropy_weight) * _bonus_v)
                    _other_v = float(other_value_func(_g_v, _a_v))
                    _st_v = _t53_iric_status(_a_v, _g_v, mu_tau_v, _val_v, _other_v)
                    _rec_v = (float(_a_v), float(_g_v), float(_score_v), _st_v)
                    if _best_any is None or _score_v < _best_any[2]:
                        _best_any = _rec_v
                    if bool(_st_v.get("feasible", False)) and (_best_feas is None or _score_v < _best_feas[2]):
                        _best_feas = _rec_v
                # Γ_t(μ_t) = ∅ si no existe punto con IR^K, IC^K e IR^F.
                # Se devuelve el mínimo irrestricto solo como diagnóstico, no como óptimo formal.
                return _best_feas if _best_feas is not None else _best_any  # type: ignore[return-value]

            _rows53 = []

            def _fmt_t53_thousands_1(value: float) -> str:
                return _fmt_es_num(float(value), 1)

            for _tau53 in [1]:
                _mu53, _iota53, _thhat53, _gamma53, _alpha53 = _t53_row_inputs(_tau53)
                _p_kill_by_type53, _p_surv_by_type53 = _t53_table10_probs(
                    _mu53, _iota53, _gamma53, _alpha53, _atk52_real, _s_exec52, _atf52
                )
                _p_kill_exp_mu1_53 = float(
                    sum(
                        float(_mu1_52.get(_th53, 0.0))
                        * float(_p_kill_by_type53.get(_th53, 0.0))
                        for _th53 in TIPOS_SECUESTRADOR
                    )
                )
                _p_surv_exp_mu1_53 = float(
                    sum(
                        float(_mu1_52.get(_th53, 0.0))
                        * float(_p_surv_by_type53.get(_th53, 0.0))
                        for _th53 in TIPOS_SECUESTRADOR
                    )
                )
                _ops_mu53 = _state_weighted_cost_tuple(_p3_ops_by_type, _mu53)
                _mt_mu53 = _state_weighted_cost_tuple(_p3_mt_by_type, _mu53)
                _ref_mu53 = _state_reference_centers(_mu53)
                _vr_const53 = float(
                    _p3_omk * (1.0 - _p_surv_exp_mu1_53)
                    + _ops_mu53[0]
                    + _p3_chi_alpha * _ref_mu53["alpha_R_mu"] ** 2
                    + _p3_chi_gamma * _ref_mu53["gamma_R_mu"] ** 2
                )
                _vr_bg53 = float(_ops_mu53[1] - 2.0 * _p3_chi_gamma * _ref_mu53["gamma_R_mu"])
                _vr_qg53 = float(_ops_mu53[2] + 2.0 * _p3_chi_gamma)
                _vr_ba53 = float(_ops_mu53[3] - 2.0 * _p3_chi_alpha * _ref_mu53["alpha_R_mu"])
                _vr_qa53 = float(_ops_mu53[4] + 2.0 * _p3_chi_alpha)
                _vr_qga53 = float(_ops_mu53[5])
                _vn_const53 = float(
                    _p3_omp * R_escala
                    + _p3_omk * _p_kill_exp_mu1_53
                    + _mt_mu53[0]
                    + _p3_chi_alpha * _ref_mu53["alpha_N_mu"] ** 2
                    + _p3_chi_gamma * _ref_mu53["gamma_N_mu"] ** 2
                )
                _vn_bg53 = float(_mt_mu53[1] - 2.0 * _p3_chi_gamma * _ref_mu53["gamma_N_mu"])
                _vn_qg53 = float(_mt_mu53[2] + 2.0 * _p3_chi_gamma)
                _vn_ba53 = float(_mt_mu53[3] - _p3_omp * R_escala - 2.0 * _p3_chi_alpha * _ref_mu53["alpha_N_mu"])
                _vn_qa53 = float(_mt_mu53[4] + 2.0 * _p3_chi_alpha)
                _vn_qga53 = float(_mt_mu53[5])

                def _vn_val_at53(_g: float, _a: float) -> float:
                    return _t53_quad_value(_g, _a, _vn_const53, _vn_bg53, _vn_qg53, _vn_ba53, _vn_qa53, _vn_qga53)

                def _vr_val_at53(_g: float, _a: float) -> float:
                    return _t53_quad_value(_g, _a, _vr_const53, _vr_bg53, _vr_qg53, _vr_ba53, _vr_qa53, _vr_qga53)

                def _info_gain_vr53(alpha_v: float, gamma_v: float) -> dict[str, Any]:
                    return _t52_expected_entropy_gain(
                        _mu53,
                        int(_tau53),
                        float(alpha_v),
                        float(gamma_v),
                        _atk52_real,
                        "Rescatar",
                        _atf52,
                    )

                def _info_gain_vn53(alpha_v: float, gamma_v: float) -> dict[str, Any]:
                    return _t52_expected_entropy_gain(
                        _mu53,
                        int(_tau53),
                        float(alpha_v),
                        float(gamma_v),
                        _atk52_real,
                        "No Rescatar",
                        _atf52,
                    )

                _alpha_vr53, _gamma_vr53, _vstar_vr53, _iric_vr53 = _t53_minimize_state_quadratic_iric(
                    const=_vr_const53,
                    b_gamma=_vr_bg53,
                    q_gamma=_vr_qg53,
                    b_alpha=_vr_ba53,
                    q_alpha=_vr_qa53,
                    q_gamma_alpha=_vr_qga53,
                    mu_tau_v=_mu53,
                    other_value_func=_vn_val_at53,
                    info_bonus_func=_info_gain_vr53,
                )
                _alpha_vn53, _gamma_vn53, _vstar_vn53, _iric_vn53 = _t53_minimize_state_quadratic_iric(
                    const=_vn_const53,
                    b_gamma=_vn_bg53,
                    q_gamma=_vn_qg53,
                    b_alpha=_vn_ba53,
                    q_alpha=_vn_qa53,
                    q_gamma_alpha=_vn_qga53,
                    mu_tau_v=_mu53,
                    other_value_func=_vr_val_at53,
                    info_bonus_func=_info_gain_vn53,
                )
                _vr_formal53 = bool(dict(_iric_vr53 or {}).get("feasible", False))
                _vn_formal53 = bool(dict(_iric_vn53 or {}).get("feasible", False))
                if _vr_formal53 and _vn_formal53:
                    _as_star53_is_rescue = bool(float(_vstar_vr53) <= float(_vstar_vn53))
                elif _vr_formal53:
                    _as_star53_is_rescue = True
                elif _vn_formal53:
                    _as_star53_is_rescue = False
                else:
                    # Γ_t(μ_t) vacío: conservar el menor valor irrestricto como diagnóstico.
                    _as_star53_is_rescue = bool(float(_vstar_vr53) <= float(_vstar_vn53))
                _gamma_formal53 = bool(_vr_formal53 or _vn_formal53)
                _info_vr53 = _info_gain_vr53(float(_alpha_vr53), float(_gamma_vr53))
                _info_vn53 = _info_gain_vn53(float(_alpha_vn53), float(_gamma_vn53))
                _info_sel53 = _info_vr53 if _as_star53_is_rescue else _info_vn53
                # IRIC de la rama elegida; IC_S verifica que ambos pisos sean finitos
                _iric_chosen53 = dict(_iric_vr53 if _as_star53_is_rescue else _iric_vn53)
                _iric_sel53 = dict(_iric_chosen53)
                _iric_sel53["IC_S"] = bool(np.isfinite(_vstar_vr53) and np.isfinite(_vstar_vn53))
                _iric_sel53["Gamma_formal"] = bool(_gamma_formal53 and _iric_sel53.get("Gamma_formal", False))
                _iric_sel53["feasible"] = bool(
                    _iric_sel53.get("IR_K", False)
                    and _iric_sel53.get("IC_K", False)
                    and _iric_sel53.get("IR_F", False)
                )
                _as_star53 = (
                    "Rescate (a_res)"
                    if _gamma_formal53 and _as_star53_is_rescue
                    else ("Negociar (a_neg)" if _gamma_formal53 else "Γ vacío (sin óptimo formal)")
                )
                _as_star53_latex = (
                    r"\mathrm{Rescate}\ (a_{\mathrm{res}})"
                    if _gamma_formal53 and _as_star53_is_rescue
                    else (
                        r"\mathrm{Negociar}\ (a_{\mathrm{neg}})"
                        if _gamma_formal53
                        else r"\Gamma_t(\mu_t)=\varnothing"
                    )
                )
                _row53 = {
                    "tau": int(_tau53),
                    "a_S_star": _as_star53,
                }
                for _th53 in TIPOS_SECUESTRADOR:
                    _row53[f"ptilde_kill_{_th53}"] = f"{float(_p_kill_by_type53.get(_th53, 0.0)):.4f}"
                _row53["p_kill_exp_mu1"] = f"{_p_kill_exp_mu1_53:.4f}"
                for _th53 in TIPOS_SECUESTRADOR:
                    _row53[f"p_surv_{_th53}"] = f"{float(_p_surv_by_type53.get(_th53, 0.0)):.4f}"
                _row53["p_surv_exp_mu1"] = f"{_p_surv_exp_mu1_53:.4f}"
                _row53["alpha_star_VR"] = f"{_alpha_vr53:.4f}"
                _row53["gamma_star_VR"] = f"{_gamma_vr53:.4f}"
                _row53["V_star_R"] = _fmt_t53_thousands_1(_vstar_vr53)
                _row53["alpha_star_VN"] = f"{_alpha_vn53:.4f}"
                _row53["gamma_star_VN"] = f"{_gamma_vn53:.4f}"
                _row53["V_star_N"] = _fmt_t53_thousands_1(_vstar_vn53)
                _row53["H_mu"] = f"{float(_info_sel53.get('H', 0.0)):.4f}"
                _row53["Delta_H"] = f"{float(_info_sel53.get('Delta_H', 0.0)):.4f}"
                _row53["psi_H"] = f"{float(_t52_entropy_weight):.1f}"
                _row53["IR_K"] = _t53_bool_label(bool(_iric_sel53.get("IR_K", False)))
                _row53["IC_K"] = _t53_bool_label(bool(_iric_sel53.get("IC_K", False)))
                _row53["IR_F"] = _t53_bool_label(bool(_iric_sel53.get("IR_F", False)))
                _row53["IC_F"] = _t53_bool_label(bool(_iric_sel53.get("IC_F", False)))
                _row53["IC_S"] = _t53_bool_label(bool(_iric_sel53.get("IC_S", False)))
                _row53["Gamma_factible"] = _t53_bool_label(bool(_iric_sel53.get("feasible", False)))
                _rows53.append(_row53)
                # Persist τ=1 optimal instruments so Table 5.2 can mirror them on the next run
                _t53_gamma_sel = float(_gamma_vr53 if _as_star53_is_rescue else _gamma_vn53)
                _t53_alpha_sel = float(_alpha_vr53 if _as_star53_is_rescue else _alpha_vn53)
                st.session_state["t53_tau1_gamma_star"] = _t53_gamma_sel
                st.session_state["t53_tau1_alpha_star"] = _t53_alpha_sel
                st.session_state["t53_tau1_is_rescue"] = bool(_as_star53_is_rescue)
                st.session_state["t53_tau1_vstar_vr"] = float(_vstar_vr53)
                st.session_state["t53_tau1_vstar_vn"] = float(_vstar_vn53)
                st.session_state["t53_tau1_gamma_vr"] = float(_gamma_vr53)
                st.session_state["t53_tau1_alpha_vr"] = float(_alpha_vr53)
                st.session_state["t53_tau1_gamma_vn"] = float(_gamma_vn53)
                st.session_state["t53_tau1_alpha_vn"] = float(_alpha_vn53)
            _df53 = pd.DataFrame(_rows53)
            _hdr53 = [r"\tau", r"a_S^\ast"] + [
                rf"p_{{\mathrm{{kill}}}}({{{_th53}}})"
                for _th53 in TIPOS_SECUESTRADOR
            ] + [
                r"\mathbb{E}_{\mu_1}[p_{\mathrm{kill}}]"
            ] + [
                rf"p_{{\mathrm{{surv}}}}({{{_th53}}})"
                for _th53 in TIPOS_SECUESTRADOR
            ] + [
                r"\mathbb{E}_{\mu_1}[p_{\mathrm{surv}}]"
            ] + [
                r"\alpha_R^\ast",
                r"\gamma_R^\ast",
                r"V_R^\ast",
                r"\alpha_N^\ast",
                r"\gamma_N^\ast",
                r"V_N^\ast",
            ]
            _tip53 = [
                "Periodo reportado en Tabla 5.3.",
                "Acción óptima del Estado en tau=1: Rescate si V_R* <= V_N*, Negociar si no.",
            ] + [
                "" for _ in TIPOS_SECUESTRADOR
            ] + [
                "Promedio ponderado de las columnas p_kill por tipo usando los mu de tau=1 en Tabla 5.2."
            ] + [
                "" for _ in TIPOS_SECUESTRADOR
            ] + [
                "Promedio ponderado de las columnas p_surv por tipo usando los mu de tau=1 en Tabla 5.2."
            ] + [
                "Bloqueo financiero del piso V_R* en [0,1]^2, con penalización Π_R.",
                "Presión operativa del piso V_R* en [0,1]^2, con penalización Π_R.",
                "Valor del piso V_R* usado en la regla discreta del Estado.",
                "Bloqueo financiero del piso V_N* en [0,1]^2, con penalización Π_N y auditoría Γ.",
                "Presión operativa del piso V_N* en [0,1]^2, con penalización Π_N y auditoría Γ.",
                "Valor del piso V_N* usado en la regla discreta; si Γ no tiene punto factible en la grilla, reporta el mínimo no factible.",
                "IR^K evaluada en el óptimo de V_N.",
                "IC^K evaluada en el óptimo de V_N.",
                "IR^F evaluada en el óptimo de V_N.",
                "IC^F evaluada en el óptimo de V_N.",
                "IC^S: comparación discreta entre V_R* y V_N*.",
                "Γ factible: la tripleta reportada satisface IC^K, IR^K e IR^F en la auditoría de implementabilidad.",
            ]
            _theta_hat_tip53 = str(_thhat53)
            _beta_r_tip53 = float(st.session_state.get("cal_surv_beta_R", 1.0))
            _ltip53 = [
                r"\tau=1",
                (
                    r"\begin{gathered}"
                    rf"a_S^\ast={_as_star53_latex}\\"
                    rf"V_R^\ast={_fmt_t53_thousands_1(_vstar_vr53)},\quad "
                    rf"V_N^\ast={_fmt_t53_thousands_1(_vstar_vn53)}\\"
                    r"a_S^\ast=\mathrm{Rescate}\ \mathrm{si}\ V_R^\ast\le V_N^\ast;"
                    r"\ \mathrm{Negociar}\ \mathrm{si}\ V_R^\ast>V_N^\ast"
                    r"\end{gathered}"
                ),
            ] + [
                (
                    r"\begin{gathered}"
                    rf"p_{{\mathrm{{kill}}}}({{{_th53}}})={float(_p_kill_by_type53.get(_th53, 0.0)):.4f}\\"
                    r"p_{\mathrm{kill}}(\theta_K)=\frac{\exp(\Psi_4(\theta_K))}"
                    r"{\sum_{\ell=1}^{5}\exp(\Psi_\ell(\theta_K))}\\"
                    r"\Psi_j=\delta_j+\gamma_{K,j}\mathbf 1\{j=1,\tilde a_K=\mathrm{Liberar}\}"
                    r"+\gamma_{S,j}\mathbf 1\{j=2,\tilde a_S=\mathrm{Rescatar}\}\\"
                    r"\quad+\gamma_{F,j}\mathbf 1\{j=3,\tilde a_F=\mathrm{Coludir}\}"
                    r"+\gamma_{K,j}\mathbf 1\{j=4,\tilde a_K=\mathrm{Matar}\}\\"
                    r"\quad+\phi_{\gamma,j}\gamma_1+\phi_{\theta,j}'\theta_K+\kappa_j\iota_1\\"
                    rf"\tilde a_K={{{_atk52_real}}},\ \tilde a_S={{{_s_exec52}}},\ "
                    rf"\tilde a_F={{{_atf52}}},\ \gamma_1={float(_gamma53):.4f},\ "
                    rf"\iota_1={float(_iota53):.4f}"
                    r"\end{gathered}"
                )
                for _th53 in TIPOS_SECUESTRADOR
            ] + [
                (
                    r"\begin{gathered}"
                    rf"\mathbb{{E}}_{{\mu_1}}[p_{{\mathrm{{kill}}}}]={_p_kill_exp_mu1_53:.4f}\\"
                    r"=\sum_{\theta\in\Theta_K}\mu_1(\theta)\,"
                    r"p_{\mathrm{kill}}(\theta)\\"
                    + r"\\ ".join(
                        [
                            rf"\mu_1({{{_th53}}})={float(_mu1_52.get(_th53, 0.0)):.4f},\ "
                            rf"p_{{\mathrm{{kill}}}}({{{_th53}}})="
                            rf"{float(_p_kill_by_type53.get(_th53, 0.0)):.4f}"
                            for _th53 in TIPOS_SECUESTRADOR
                        ]
                    )
                    + r"\end{gathered}"
                )
            ] + [
                (
                    r"\begin{gathered}"
                    rf"p_{{\mathrm{{surv}}}}({{{_th53}}})={float(_p_surv_by_type53.get(_th53, 0.0)):.4f}\\"
                    r"=\Lambda\!\left(\alpha_{\mathrm{leth}}("
                    rf"{{{_th53}}}"
                    r")+\beta_R\,\iota_1\,\mathbf 1\{\hat\theta_1="
                    rf"{{{_th53}}}"
                    r"\}\right)\\"
                    rf"\alpha_{{\mathrm{{leth}}}}({{{_th53}}})="
                    rf"{float(st.session_state.get('cal_surv_alpha0', {}).get(_th53, -1.0)):.4f},\ "
                    rf"\beta_R={_beta_r_tip53:.4f},\ "
                    rf"\iota_1={float(_iota53):.4f}\\"
                    rf"\hat\theta_1={{{_theta_hat_tip53}}},\ "
                    rf"\mathbf 1\{{\hat\theta_1={{{_th53}}}\}}="
                    rf"{1 if _theta_hat_tip53 == str(_th53) else 0}"
                    r"\end{gathered}"
                )
                for _th53 in TIPOS_SECUESTRADOR
            ] + [
                (
                    r"\begin{gathered}"
                    rf"\mathbb{{E}}_{{\mu_1}}[p_{{\mathrm{{surv}}}}]={_p_surv_exp_mu1_53:.4f}\\"
                    r"=\sum_{\theta\in\Theta_K}\mu_1(\theta)\,"
                    r"p_{\mathrm{surv}}(\theta)\\"
                    + r"\\ ".join(
                        [
                            rf"\mu_1({{{_th53}}})={float(_mu1_52.get(_th53, 0.0)):.4f},\ "
                            rf"p_{{\mathrm{{surv}}}}({{{_th53}}})="
                            rf"{float(_p_surv_by_type53.get(_th53, 0.0)):.4f}"
                            for _th53 in TIPOS_SECUESTRADOR
                        ]
                    )
                    + r"\end{gathered}"
                )
            ] + [
                rf"\alpha_R^\ast={_alpha_vr53:.4f}",
                rf"\gamma_R^\ast={_gamma_vr53:.4f}",
                rf"V_R^\ast={_fmt_t53_thousands_1(_vstar_vr53)}",
                rf"\alpha_N^\ast={_alpha_vn53:.4f}",
                rf"\gamma_N^\ast={_gamma_vn53:.4f}",
                rf"V_N^\ast={_fmt_t53_thousands_1(_vstar_vn53)}",
                r"IR^K:\ \mathbb E_{\theta\sim\mu_t}[U_{\mathrm{rel}}^K(\theta)-\max\{V_{\mathrm{cont}}^K(\theta),U_{\mathrm{kill}}^K(\theta)\}]\ge0",
                r"IC^K:\ \mathbb E_{\theta\sim\mu_t}[V^K(a^\ast(\theta)\mid\theta)-V^K(a^\ast(\theta_j)\mid\theta)]\ge0\ \forall j",
                r"IR^F:\ U_F(\mathrm{Cooperar})\ge U_F(\mathrm{Coludir})",
                r"IC^F:\ U_F(\mathrm{Cooperar}),U_F(\mathrm{Coludir})\ \mathrm{bien\ definidos}",
                r"IC^S:\ a_S^\ast=\arg\min\{V_R^\ast,V_N^\ast\}",
                r"\Gamma_t(\mu_t)\ \mathrm{factible}\iff IC^K\land IR^K\land IR^F\quad(\mathrm{auditoría:\ }IC^F,IC^S)",
            ]
            _hdr53 += [
                r"IR^K_N",
                r"IC^K_N",
                r"IR^F_N",
                r"IC^F_N",
                r"IC^S",
                r"\Gamma_N",
            ]
            # Transponer: una fila por variable, una columna por período τ
            _df53_row0 = _df53.iloc[0] if not _df53.empty else pd.Series(dtype=object)
            _cols53_T = list(_df53.columns)

            def _ktx53_cell(s: str) -> str:
                return f'<span class="math">{html.escape(s, quote=False)}</span>'

            _rows53_T = []
            _row_tips_53T = []
            for _col53T, _lhdr53T, _tip53T in zip(_cols53_T, _hdr53, _tip53):
                _val53_T = str(_df53_row0.get(_col53T, "—")) if not _df53.empty else "—"
                _rows53_T.append({"Variable": _ktx53_cell(_lhdr53T), "tau1": _val53_T})
                _row_tips_53T.append(_tip53T)
            _df53_T = pd.DataFrame(_rows53_T)
            _c1_post_for53 = st.session_state.get("first_cycle_post54")
            _c1_vals_for53 = st.session_state.get("first_cycle_table52") or {}
            _c1_tau1_vals_for53 = st.session_state.get("first_cycle_tau1_52") or {}
            if (
                isinstance(_c1_post_for53, dict)
                and isinstance(_c1_post_for53.get("mu_post"), dict)
                and isinstance(_c1_vals_for53, dict)
                and "__error__" not in _c1_vals_for53
                and not _df53_T.empty
            ):
                try:
                    _mu_c1_53 = _t53_norm_mu(dict(_c1_post_for53.get("mu_post", {})))
                    _iota_c1_53 = float(max(_mu_c1_53.values())) if _mu_c1_53 else float(_iota1_52)
                    _theta_c1_53 = max(_mu_c1_53, key=_mu_c1_53.get) if _mu_c1_53 else str(tipo_real)
                    _gamma_c1_53 = float(_c1_vals_for53.get("γ* Estado", _gamma_state_star52))
                    _alpha_c1_53 = float(_c1_vals_for53.get("α* Estado", _alpha_state_star52))
                    _ak_c1_53 = str(_c1_tau1_vals_for53.get(f"ã_K ({tipo_real})", _atk52_real)).split(" ")[0]
                    _af_c1_53 = str(_c1_tau1_vals_for53.get("ã_F", _atf52)).split(" ")[0]
                    _as_c1_raw53 = str(
                        _c1_tau1_vals_for53.get("ã_S")
                        or _c1_vals_for53.get("ã_S")
                        or _ats52
                    ).split(" ")[0]
                    _as_c1_53 = _t53_state_exec_label(_as_c1_raw53)
                    _pkill_c1_53, _psurv_c1_53 = _t53_table10_probs(
                        _mu_c1_53,
                        _iota_c1_53,
                        _gamma_c1_53,
                        _alpha_c1_53,
                        _ak_c1_53,
                        _as_c1_53,
                        _af_c1_53,
                    )
                    _pkill_exp_c1_53 = float(sum(_mu_c1_53.get(_th, 0.0) * _pkill_c1_53.get(_th, 0.0) for _th in TIPOS_SECUESTRADOR))
                    _psurv_exp_c1_53 = float(sum(_mu_c1_53.get(_th, 0.0) * _psurv_c1_53.get(_th, 0.0) for _th in TIPOS_SECUESTRADOR))
                    _c1_vals_53 = [
                        r"\tau=2",
                        str(
                            _c1_tau1_vals_for53.get("a_S* óptima")
                            or _c1_vals_for53.get("a_S* óptima")
                            or _s52_full
                        ),
                    ] + [
                        f"{float(_pkill_c1_53.get(_th, 0.0)):.4f}" for _th in TIPOS_SECUESTRADOR
                    ] + [
                        f"{_pkill_exp_c1_53:.4f}",
                    ] + [
                        f"{float(_psurv_c1_53.get(_th, 0.0)):.4f}" for _th in TIPOS_SECUESTRADOR
                    ] + [
                        f"{_psurv_exp_c1_53:.4f}",
                        "—", "—", "—", "—", "—", "—",
                        "—", "—", "—", "—", "—", "—",
                    ]
                    _df53_T["tau2"] = _c1_vals_53[: len(_df53_T)]
                except Exception:
                    _df53_T["tau2"] = "—"
            render_generic_katex_table(
                _df53_T,
                [r"\text{Variable}", r"\tau=1", r"\tau=2"] if "tau2" in _df53_T.columns else [r"\text{Variable}", r"\tau=1"],
                height=120 + 34 * (len(_df53_T) + 1),
                compact=True,
                relaxed_compact=True,
                header_nowrap=True,
                row_tooltips=_row_tips_53T,
            )
            with st.expander(
                "Funciones de pérdida del Estado con probabilidades de Tabla 5.3",
                expanded=False,
            ):
                def _fmt_latex_num(value: float, nd: int = 2) -> str:
                    return _fmt_es_num(float(value), nd)

                def _fmt_signed_latex(value: float, suffix: str) -> str:
                    sign = "+" if float(value) >= 0.0 else "-"
                    return rf"{sign}{_fmt_latex_num(abs(float(value)))}{suffix}"
                _sg_ops_g = _fmt_signed_latex(_ops_mu53[1], "\\gamma")
                _sg_ops_a = _fmt_signed_latex(_ops_mu53[3], "\\alpha")
                _sg_ops_ga = _fmt_signed_latex(_ops_mu53[5], "\\gamma\\alpha")
                _sg_mt_g = _fmt_signed_latex(_mt_mu53[1], "\\gamma")
                _sg_mt_a = _fmt_signed_latex(_mt_mu53[3], "\\alpha")
                _sg_mt_ga = _fmt_signed_latex(_mt_mu53[5], "\\gamma\\alpha")
                st.latex(
                    rf"""
                    \small
                    \begin{{aligned}}
                    \widetilde V_R(\gamma,\alpha)
                    &= ({_fmt_latex_num(_p3_omk)})(1-{_fmt_latex_num(_p_surv_exp_mu1_53)})
                    + {_fmt_latex_num(_ops_mu53[0])}{_sg_ops_g}
                    +\frac{{{_fmt_latex_num(_ops_mu53[2])}}}{{2}}\gamma^2
                    {_sg_ops_a}
                    +\frac{{{_fmt_latex_num(_ops_mu53[4])}}}{{2}}\alpha^2
                    {_sg_ops_ga}
                    +{_fmt_latex_num(_p3_chi_alpha)}(\alpha-{_fmt_latex_num(_ref_mu53["alpha_R_mu"])})^2
                    +{_fmt_latex_num(_p3_chi_gamma)}(\gamma-{_fmt_latex_num(_ref_mu53["gamma_R_mu"])})^2
                    -{_fmt_latex_num(float(st.session_state.get("t52_entropy_info_weight", 25.0)))}\Delta H_R(\alpha,\gamma)
                    \end{{aligned}}
                    """
                )
                st.latex(
                    rf"""
                    \small
                    \begin{{aligned}}
                    \widetilde V_N(\gamma,\alpha)
                    &= ({_fmt_latex_num(_p3_omp)})({_fmt_latex_num(R_escala)})(1-\alpha)
                    +({_fmt_latex_num(_p3_omk)})({_fmt_latex_num(_p_kill_exp_mu1_53)})\\
                    &\quad+{_fmt_latex_num(_mt_mu53[0])}{_sg_mt_g}
                    +\frac{{{_fmt_latex_num(_mt_mu53[2])}}}{{2}}\gamma^2
                    {_sg_mt_a}
                    +\frac{{{_fmt_latex_num(_mt_mu53[4])}}}{{2}}\alpha^2
                    {_sg_mt_ga}
                    +{_fmt_latex_num(_p3_chi_alpha)}(\alpha-{_fmt_latex_num(_ref_mu53["alpha_N_mu"])})^2
                    +{_fmt_latex_num(_p3_chi_gamma)}(\gamma-{_fmt_latex_num(_ref_mu53["gamma_N_mu"])})^2
                    -{_fmt_latex_num(float(st.session_state.get("t52_entropy_info_weight", 25.0)))}\Delta H_N(\alpha,\gamma)
                    \end{{aligned}}
                    """
                )
                st.markdown(
                    "Implementabilidad según `Mechanism.tex`: la política ejecutada debe pertenecer a "
                    "$\\Gamma_t(\\mu_t)$, que reúne $IC^K$, $IR^K$ e $IR^F$. El término "
                    "$-\\psi_H\\Delta H$ es el motivo informacional; si $\\psi_H=0$, se recupera "
                    "el problema sin exploración por entropía."
                )
                st.latex(
                    r"""
                    \small
                    \begin{aligned}
                    \mathcal{L}_t^{S\ast}
                    &=
                    \min\left\{
                    V_t^{R,\ast}(\iota_t,\widehat{\theta}_t,\mu_t),
                    V_t^{N,\ast}(\mu_t)
                    \right\},\\
                    (a_t^{S\ast},\alpha_t^\ast,\gamma_t^\ast)
                    &\in
                    \arg\min_{(a_t^S,\alpha_t,\gamma_t)\in\Gamma_t(\mu_t)}
                    \left[
                    \sum_{\theta\in\Theta_K}\mu_t(\theta)\,
                    L_t(a_t^S,\alpha_t,\gamma_t,\theta,\iota_t)
                    +\Pi_{t,a^S}^S(\alpha_t,\gamma_t;\mu_t)
                    \right],\\
                    \Gamma_t(\mu_t)
                    &=
                    \left\{
                    (a_t^S,\alpha_t,\gamma_t)\in
                    \mathcal A^{S,\mathrm{disc}}\times[0,1]^2:
                    \operatorname{IC}^K,\ \operatorname{IR}^K,\ \operatorname{IR}^F
                    \right\}.
                    \end{aligned}
                    """
                )
                st.latex(
                    rf"""
                    \small
                    \begin{{aligned}}
                    IR^K &: \mathbb E_{{\theta\sim\mu_t}}\!\left[
                    U_{{rel}}^K(\theta)-
                    \max\{{V_{{cont}}^K(\theta,\alpha^\ast,\gamma^\ast),U_{{kill}}^K(\theta)\}}
                    \right]\ge0,\\
                    IC^K &: \mathbb E_{{\theta\sim\mu_t}}\!\left[
                    V^K(a^\ast(\theta)\mid\theta)-
                    V^K(a^\ast(\theta_j)\mid\theta)\right]\ge0
                    \quad \forall \theta_j,\\
                    IR^F &: U_F(\mathrm{{Cooperar}};\phi_F={_fmt_latex_num(_p3_phi_f)},\kappa_F={_fmt_latex_num(_p3_kappa_f)},\nu_F={_fmt_latex_num(_p3_nu_f)})
                    \ge U_F(\mathrm{{Coludir}};F_{{col}}={_fmt_latex_num(_p3_fcol)}),\\
                    IC^F &: \bigl(U_F(\mathrm{{Cooperar}}),U_F(\mathrm{{Coludir}})\bigr)
                    \text{{ bien definidos}},\\
                    IC^S &: a_S^\ast=\arg\min\bigl(V_R^\ast,V_N^\ast\bigr).
                    \end{{aligned}}
                    """
                )
                st.caption(
                    "Las utilidades de K usan los coeficientes calibrados por tipo de secuestrador "
                    "en las tablas previas; IR^F/IC^F usan los coeficientes calibrados de familia."
                )

            st.markdown(
                '<div style="margin-top:0.15rem;margin-bottom:2px;font-weight:700;">'
                'Tabla 5.4 · Componentes de la posteriori desde τ = 0'
                '</div>',
                unsafe_allow_html=True,
            )
            _rows54: list[dict[str, str]] = []
            _row_tips_54: list[str] = []
            _tab14_row1_54 = pd.Series(dtype=object)
            _df14_54 = st.session_state.get("tab15_mu_snapshot")
            if not isinstance(_df14_54, pd.DataFrame) or _df14_54.empty:
                _df14_54 = st.session_state.get("rb_mu_traj_snapshot")
            if isinstance(_df14_54, pd.DataFrame) and not _df14_54.empty and "t" in _df14_54.columns:
                _r14_54 = _df14_54[_df14_54["t"].astype(int) == 1]
                if not _r14_54.empty:
                    _tab14_row1_54 = _r14_54.iloc[0]

            def _t54_fmt_row14(col: str, fallback: Any = "—", nd: int = 6) -> str:
                if not isinstance(_tab14_row1_54, pd.Series) or _tab14_row1_54.empty:
                    return "—" if fallback == "—" else f"{float(fallback):.{nd}f}"
                try:
                    val = _tab14_row1_54.get(col, fallback)
                    if val in ("—", "", None):
                        return "—"
                    return f"{float(val):.{nd}f}"
                except Exception:
                    return "—"

            def _ktx54(s: str, disp: bool = False) -> str:
                disp_a = ' data-katex-disp' if disp else ''
                return f'<span data-katex="{html.escape(s, quote=True)}"{disp_a}></span>'

            def _ktx54_cell(s: str) -> str:
                return f'<span class="math">{html.escape(s, quote=False)}</span>'

            def _th_ltx54(th: Any) -> str:
                return r"\mathrm{" + re.sub(r"([%#_{}&])", r"\\\1", str(th)) + "}"

            def _tip_hdr(title: str) -> str:
                return f'<div style="font-weight:600;color:#b8c4e8;margin-bottom:4px;">{title}</div>'

            if isinstance(_df_post52, pd.DataFrame) and not _df_post52.empty:
                _dfp54 = _df_post52.copy()
                _denom54 = float(_meta_post52.get("denom", _dfp54["denom_Z"].iloc[0] if "denom_Z" in _dfp54.columns else 0.0))
                _lh54: dict[str, float] = {}
                _ld54: dict[str, float] = {}
                _lf54: dict[str, float] = {}
                _lc54: dict[str, float] = {}
                _lt54: dict[str, float] = {}
                _mu0_54: dict[str, float] = {}
                _mu1_54: dict[str, float] = {}
                # Precompute KaTeX spans (Python 3.9: no backslash inside f-string {})
                _k_mu0_th = _ktx54(r"\mu_0(\theta)")
                _k_lF_th = _ktx54(r"\mathcal{L}_F(\theta)")
                _k_lC_th = _ktx54(r"\mathcal{L}_C(\theta)")
                _k_Zt_th = _ktx54(r"Z_t")
                _k_mu1_th = _ktx54(r"\mu_1(\theta)")
                _k_Ldt = _ktx54(r"\mathcal{L}_{d,t}")
                for _th54 in TIPOS_SECUESTRADOR:
                    _rp54 = _dfp54[_dfp54["theta_K"].astype(str) == str(_th54)]
                    if _rp54.empty:
                        continue
                    _r54 = _rp54.iloc[0]
                    _mu0_v54 = float(_r54.get("mu_0", _mu0_52.get(_th54, 0.0)))
                    _mu1_raw54 = float(_r54.get("mu_1", _mu1_52.get(_th54, 0.0)))
                    _lh54[_th54] = float(_r54.get("L_H", 1.0))
                    _ld54[_th54] = float(_r54.get("L_d", 1.0))
                    _lf54[_th54] = float(_r54.get("L_F", 1.0))
                    _lc54[_th54] = float(_r54.get("L_C", 1.0))
                    _lt54[_th54] = float(_r54.get("L_total", _lf54[_th54] * _lc54[_th54]))
                    _mu1_v54 = (
                        float(_mu0_v54 * _lt54[_th54] / _denom54)
                        if _denom54 > 1e-15
                        else _mu1_raw54
                    )
                    _mu0_54[_th54] = _mu0_v54
                    _mu1_54[_th54] = _mu1_v54
                    _rows54.extend([
                        {
                            "Componente": _ktx54_cell(rf"\mu(\theta={_th_ltx54(_th54)})"),
                            "tau0": f"{_mu0_v54:.6f}",
                            "tau1": f"{_mu1_v54:.6f}",
                        }
                    ])
                    _lf_v54 = _lf54[_th54]
                    _lc_v54 = _lc54[_th54]
                    _tip_mu = (
                        _tip_hdr(f"μ(θ={_th54})")
                        + '<div style="font-size:0.85em;color:#9ab;margin-top:4px;">'
                        + _ktx54(r"\mu_1=\mu_0\cdot\mathcal{L}_F\cdot\mathcal{L}_C\,/\,Z_t")
                        + "</div>"
                        + '<div>La posterior τ=1 se calcula con la verosimilitud de la columna τ=0, evaluada en t=0.</div>'
                    )
                    _row_tips_54.append(_tip_mu)
                _m_obs54 = str(_meta_post52.get("m_obs", "—"))
                _d_obs54 = int(_meta_post52.get("d_obs", 0))
                _p_det54 = float(_meta_post52.get("p_det", 0.0))
                _lh_exp_cont54 = float(_meta_post52.get("expected_p_cont_mu0", 0.0))
                _lh_uses_exp54 = bool(_meta_post52.get("LH_uses_expected_p_cont", False))
                _lc_exp_sil54 = float(_meta_post52.get("expected_lc_silence_mu0", 1.0))
                _lc_uses_exp_sil54 = bool(_meta_post52.get("LC_uses_expected_silence", False))
                _ld_common54 = float(_meta_post52.get("L_d_common", list(_ld54.values())[0] if _ld54 else 1.0))
                # ── Promedios ponderados por μ₀ (Estado no conoce θ verdadero) ──
                _lh0_mu = sum(_mu0_54.get(_th, 0.0) * _lh54.get(_th, 0.0) for _th in TIPOS_SECUESTRADOR)
                _lf0_mu = sum(_mu0_54.get(_th, 0.0) * _lf54.get(_th, 0.0) for _th in TIPOS_SECUESTRADOR)
                _lc0_mu = sum(_mu0_54.get(_th, 0.0) * _lc54.get(_th, 0.0) for _th in TIPOS_SECUESTRADOR)
                # Z_t = Σ_θ μ₀·ℒ_total (único promedio ponderado — denominador de Bayes)
                _zt54 = float(_denom54)
                # Verosimilitudes per-θ* (no ponderadas)
                _rp54_star = _dfp54[_dfp54["theta_K"].astype(str) == str(tipo_real)]
                _r54_star = _rp54_star.iloc[0] if not _rp54_star.empty else pd.Series(dtype=object)
                _lh_star = float(_r54_star.get("L_H", 0.0)) if not _r54_star.empty else 0.0
                _lf_star = float(_r54_star.get("L_F", 0.0)) if not _r54_star.empty else 0.0
                _lc_star = float(_r54_star.get("L_C", 0.0)) if not _r54_star.empty else 0.0
                # ── Tooltips por fila ──
                _row_tips_54.append(
                    _tip_hdr(f"ℒ_H(θ*={tipo_real}) — Prob. continuación")
                    + '<div style="margin:3px 0;">'
                    + (
                        _ktx54(
                            r"\mathcal{L}_H=\mathbb{E}_{\mu_0}[p_{\mathrm{cont}}(\theta)]"
                            r"=\sum_{\theta}\mu_0(\theta)p_{\mathrm{cont}}(\theta)",
                            disp=True,
                        )
                        if _lh_uses_exp54
                        else _ktx54(r"\mathcal{L}_H(\theta^*)=p_{\mathrm{cont}}(t=0\mid\theta^*,\alpha_0,\gamma_0)", disp=True)
                    )
                    + "</div>"
                    + (
                        "<div>Como el Estado no conoce θ, se usa la expectativa con μ₀.</div>"
                        if _lh_uses_exp54
                        else "<div>Se evalúa la probabilidad primitiva de continuación para el tipo θ*.</div>"
                    )
                )
                _row_tips_54.append(
                    _tip_hdr("ℒ_d — Detección (igual ∀θ)")
                    + '<div style="margin:3px 0;">'
                    + _ktx54(r"\mathcal{L}_d=p_{\mathrm{det}}^{d_0}(1-p_{\mathrm{det}})^{1-d_0}", disp=True)
                    + "</div>"
                    + "<div>Es la verosimilitud Bernoulli del indicador de detección.</div>"
                )
                _row_tips_54.append(
                    _tip_hdr(f"ℒ_F(θ*={tipo_real}) = ℒ_H · ℒ_d")
                    + '<div style="margin:3px 0;">'
                    + _ktx54(r"\mathcal{L}_F(\theta^*)=\mathcal{L}_H(\theta^*)\cdot\mathcal{L}_d", disp=True)
                    + "</div>"
                    + "<div>Combina la evidencia física de continuación con la detección.</div>"
                )
                _row_tips_54.append(
                    _tip_hdr(f"Urgencia de comunicación · ℒ_C(θ*={tipo_real})")
                    + '<div style="margin:3px 0;">'
                    + _ktx54(rf"\mathcal{{L}}_C(\theta^\ast={_th_ltx54(tipo_real)})", disp=True)
                    + "</div>"
                    + '<div style="margin:3px 0;">'
                    + (
                        _ktx54(
                            r"\mathcal{L}_C=\left(\mathbb{E}_{\mu_0}[1-\pi_{call}(\theta)]\right)^{\omega_{voz}}",
                            disp=True,
                        )
                        if _lc_uses_exp_sil54
                        else _ktx54(
                            r"\mathcal{L}_C(\theta\mid V_t)=\begin{cases}"
                            r"(\mathcal{L}_{voz}(\theta)\pi_{call}(\theta))^{\omega_{voz}},&V_t=1\\"
                            r"(1-\pi_{call}(\theta))^{\omega_{voz}},&V_t=0"
                            r"\end{cases}",
                            disp=True,
                        )
                    )
                    + "</div>"
                    + (
                        "<div>Si no hay voz observada, se usa el silencio esperado ponderado por μ₀.</div>"
                        if _lc_uses_exp_sil54
                        else "<div>Si hay voz, pondera la señal de voz por π_call; si no hay voz, usa 1−π_call.</div>"
                    )
                )
                _row_tips_54.append(
                    _tip_hdr("Z_t — Denominador bayesiano (ponderado por μ₀)")
                    + '<div style="margin:3px 0;">'
                    + _ktx54(r"Z_t=\sum_\theta\mu_0(\theta)\,\mathcal{L}_F(\theta)\,\mathcal{L}_C(\theta)", disp=True)
                    + "</div>"
                    + "<div>Normaliza la distribución posterior para que las probabilidades sumen uno.</div>"
                )
                _rows54.extend(
                    [
                        {"Componente": _ktx54_cell(rf"\mathcal{{L}}_H(\theta^\ast={_th_ltx54(tipo_real)})"), "tau0": f"{_lh_star:.6f}", "tau1": "—"},
                        {"Componente": _ktx54_cell(r"\mathcal{L}_d"), "tau0": f"{_ld_common54:.6f}", "tau1": "—"},
                        {"Componente": _ktx54_cell(rf"\mathcal{{L}}_F(\theta^\ast={_th_ltx54(tipo_real)})"), "tau0": f"{_lf_star:.6f}", "tau1": "—"},
                        {"Componente": _ktx54_cell(rf"\mathcal{{L}}_C(\theta^\ast={_th_ltx54(tipo_real)})"), "tau0": f"{_lc_star:.6f}", "tau1": "—"},
                        {"Componente": _ktx54_cell(r"Z_t=\sum_{\theta}\mu_0(\theta)\mathcal{L}_F(\theta)\mathcal{L}_C(\theta)"), "tau0": f"{_zt54:.6f}", "tau1": "—"},
                    ]
                )
            else:
                for _th54 in TIPOS_SECUESTRADOR:
                    _rows54.append(
                        {
                            "Componente": _ktx54_cell(rf"\mu(\theta={_th_ltx54(_th54)})"),
                            "tau0": f"{float(_mu0_52.get(_th54, 0.0)):.6f}",
                            "tau1": _t54_fmt_row14(f"mu_{_th54}", _mu1_52.get(_th54, 0.0)),
                        }
                    )
                    _row_tips_54.append("")
                _rows54.extend(
                    [
                        {"Componente": _ktx54_cell(r"\mathcal{L}_H"), "tau0": _t54_fmt_row14("L_H"), "tau1": "—"},
                        {"Componente": _ktx54_cell(r"\mathcal{L}_d"), "tau0": _t54_fmt_row14("L_d"), "tau1": "—"},
                        {"Componente": _ktx54_cell(r"\mathcal{L}_F=\mathcal{L}_H\mathcal{L}_d"), "tau0": _t54_fmt_row14("L_F"), "tau1": "—"},
                        {"Componente": _ktx54_cell(r"\mathcal{L}_C"), "tau0": _t54_fmt_row14("L_C"), "tau1": "—"},
                        {"Componente": _ktx54_cell(r"\mathcal{L}_{total}=\mathcal{L}_F\mathcal{L}_C"), "tau0": _t54_fmt_row14("L_bayes"), "tau1": "—"},
                        {"Componente": _ktx54_cell(r"Z_t"), "tau0": _t54_fmt_row14("Z_t"), "tau1": "—"},
                    ]
                )
            _df54 = pd.DataFrame(_rows54)
            _c1_post54 = st.session_state.get("first_cycle_post54")
            _has_c1_54 = isinstance(_c1_post54, dict) and isinstance(_c1_post54.get("df"), pd.DataFrame)
            if _has_c1_54 and not _df54.empty:
                _df_c1_54 = _c1_post54["df"].copy()
                _mu_post_c1 = dict(_c1_post54.get("mu_post", {}))
                _mu_prior_c1 = dict(_c1_post54.get("mu_prior", {}))
                _df54["tau2"] = "—"
                for _idx_c1, _th_c1 in enumerate(TIPOS_SECUESTRADOR):
                    if _idx_c1 < len(_df54):
                        _df54.loc[_idx_c1, "tau2"] = f"{float(_mu_post_c1.get(_th_c1, _mu_prior_c1.get(_th_c1, 0.0))):.6f}"
                _rstar_c1 = _df_c1_54[_df_c1_54["theta_K"].astype(str) == str(tipo_real)]
                if not _rstar_c1.empty:
                    _rs_c1 = _rstar_c1.iloc[0]
                    _start_extra_c1 = len(TIPOS_SECUESTRADOR)
                    _extra_vals_c1_full = [
                        f"{float(_rs_c1.get('L_H', 1.0)):.6f}",
                        f"{float(_rs_c1.get('L_d', 1.0)):.6f}",
                        f"{float(_rs_c1.get('L_F', 1.0)):.6f}",
                        f"{float(_rs_c1.get('L_C', 1.0)):.6f}",
                        f"{float(_rs_c1.get('L_total', 1.0)):.6f}",
                        f"{float((_c1_post54.get('meta') or {}).get('denom', _rs_c1.get('denom_Z', 0.0))):.6f}",
                    ]
                    _n_extra_rows_c1 = max(0, len(_df54) - _start_extra_c1)
                    _extra_vals_c1 = (
                        _extra_vals_c1_full
                        if _n_extra_rows_c1 >= 6
                        else _extra_vals_c1_full[:4] + [_extra_vals_c1_full[-1]]
                    )
                    for _off_c1, _val_c1 in enumerate(_extra_vals_c1):
                        _rid_c1 = _start_extra_c1 + _off_c1
                        if _rid_c1 < len(_df54):
                            _df54.loc[_rid_c1, "tau1"] = _val_c1
            render_generic_katex_table(
                _df54,
                (
                    [
                        r"\text{Componente posterior}",
                        r"\tau=0",
                        r"\tau=1",
                        r"\tau=2",
                    ]
                    if "tau2" in _df54.columns
                    else [
                    r"\text{Componente posterior}",
                    r"\tau=0",
                    r"\tau=1",
                    ]
                ),
                height=120 + 34 * (len(_df54) + 1),
                compact=True,
                relaxed_compact=True,
                header_nowrap=True,
                row_tooltips=_row_tips_54,
            )
            if _has_c1_54:
                st.caption(
                    f"Ciclo 1 integrado: τ=1 registra V={int(_c1_post54.get('V', 0))} "
                    f"y d={int(_c1_post54.get('d', 0))}; τ=2 reporta la posterior resultante. "
                    f"p_det={float(_c1_post54.get('p_det', 0.0)):.4f}."
                )
            _s_rule_icon = "✅" if _vr_lt_vn else "ℹ️"
            st.markdown(
                f"**Regla discreta**: "
                f"$V^R$ = **{float(_vr_p3):.2f}** "
                f"{'≤' if _vr_lt_vn else '>'} "
                f"$V^N$ = **{float(_vn_p3):.2f}** "
                f"{_s_rule_icon}  →  **a_S* = {_s_rule_p3}**"
            )

            # ── Conjunto de información pública h_0 + I^S_0 (Mechanism.tex) ──────────
            _h0_af_v      = str(st.session_state.get("h0_Atilde_F", "—"))
            _h0_ak_v      = str(st.session_state.get("h0_Atilde_K", "—"))
            _h0_as_v      = str(st.session_state.get("h0_Atilde_S", "No Rescatar"))
            _h0_d_v       = str(st.session_state.get("h0_d", "0"))
            _h0_m_v       = str(_t52_vals0.get("m", "Continuar"))
            _h0_z_v       = str(st.session_state.get("z_region", "—"))
            _h0_tv_v      = str(st.session_state.get("v_victim", "—"))
            _h0_tf_v      = str(f_capa)
            _h0_ts_v      = str(s_tipo)
            _h0_R_v       = float(R_escala)
            _h0_iota_v    = float(_iota_52)
            _h0_thetahat_v = str(_theta_hat52)
            _h0_V0_v      = 0  # silencio en τ=0 (no hay señal de voz inicial)
            _h0_Tmad      = float(st.session_state.get("cal_T_mad", 30.0))
            _h0_M0_v      = float(min(1.0, (0.0 / max(_h0_Tmad, 1e-9)) ** 2))  # = 0 en t=0
            _h0_pdet_v    = float(_pdet52)
            _h0_mu0_rows  = "".join(
                f"| $\\mu_{{0}}(\\theta_K=\\text{{{th}}})$ | {float(_mu0_52.get(th, 0.0)):.4f} |\n"
                for th in TIPOS_SECUESTRADOR
            )
            st.markdown("---")
            st.markdown(
                r"**Tabla 5.5 · Conjunto de información** $\mathcal{I}_0^S = (h_0,\,z,\,\theta_F,\,\theta_V,\,\theta_S,\,\mu_0)$"
                " · Mechanism.tex · τ=0 &nbsp;&nbsp; _(⁺ = fijo en todo el mecanismo)_"
            )
            st.markdown(
                "| Componente | Realización τ=0 |\n"
                "|:---|:---:|\n"
                "| **— h_t: historia pública —** | |\n"
                "| $t$ | 0 |\n"
                f"| $\\alpha_{{0}}$ | {_t0_alpha_eff:.4f} |\n"
                f"| $\\gamma_{{0}}$ | {_t0_gamma_eff:.4f} |\n"
                f"| $\\tilde{{a}}^F_{{0}}$ | {_h0_af_v} |\n"
                f"| $\\tilde{{a}}^K_{{0}}$ ({tipo_real}) | {_h0_ak_v} |\n"
                f"| $\\tilde{{a}}^S_{{0}}$ | {_h0_as_v} |\n"
                f"| $m_{{0}}$ | {_h0_m_v} |\n"
                f"| $d_{{0}}$ | {_h0_d_v} |\n"
                "| **— $\\mathcal{{I}}^S$ adicionales —** | |\n"
                f"| $z^+$ | {_h0_z_v} |\n"
                f"| $\\theta_F^+$ | {_h0_tf_v} |\n"
                f"| $\\theta_V^+$ | {_h0_tv_v} |\n"
                f"| $\\theta_S^+$ | {_h0_ts_v} |\n"
                "| **— Derivadas / parámetros —** | |\n"
                f"| $\\iota_{{0}}$ | {_h0_iota_v:.4f} |\n"
                f"| $\\hat{{\\theta}}_{{0}}$ | {_h0_thetahat_v} |\n"
                f"| $R^+$ | {_h0_R_v:.2f} |\n"
                f"| $V_{{0}}$ | {_h0_V0_v} |\n"
                f"| $M_{{0}}$ | {_h0_M0_v:.4f} |\n"
                f"| $p_{{\\mathrm{{det}},0}}$ | {_h0_pdet_v:.4f} |\n"
                "| **— Creencia posterior —** | |\n"
                + _h0_mu0_rows.rstrip("\n")
            )
            st.caption(
                "⁺ Fijo al abrir el episodio: z, θ_F, θ_V, θ_S, R. "
                "Formalmente h₀ = (0, ∅, …, ∅) — la tabla muestra las realizaciones de τ=0 que alimentan h₁. "
                "V₀=0 (silencio); M₀=0 siempre en t=0."
            )


with tab_dyn:
    st.markdown("## Pestaña 6 · Visualización dinámica del mecanismo")
    st.caption(
        "Paneles construidos a partir de **Avanzar ciclos** (Tabla 5.2). "
        "Cada gráfica va seguida de una lectura breve anclada en los números de la corrida "
        "y en **Mechanism.tex** (programa del Estado, MDG, IR/IC y aprendizaje bayesiano)."
    )

    _tipo_dyn = str(st.session_state.get("global_tipo_real", TIPOS_SECUESTRADOR[0]))
    _TIPO_COLORS = {"DC": "#e45c2b", "PAR": "#2563eb", "ELN": "#16a34a", "FARC": "#7c3aed"}

    def _dyn_mu_norm(mu_raw: dict[str, Any]) -> dict[str, float]:
        vals = {th: max(0.0, float(dict(mu_raw).get(th, 0.0))) for th in TIPOS_SECUESTRADOR}
        total = float(sum(vals.values()))
        if total <= 1e-12:
            return {th: 1.0 / len(TIPOS_SECUESTRADOR) for th in TIPOS_SECUESTRADOR}
        return {th: float(vals.get(th, 0.0)) / total for th in TIPOS_SECUESTRADOR}

    def _dyn_comment(lines: list[str]) -> None:
        st.markdown(
            "<div style='font-size:0.92rem;line-height:1.45;margin:0.15rem 0 1.35rem 0'>"
            + "<br>".join(str(x) for x in lines[:3])
            + "</div>",
            unsafe_allow_html=True,
        )

    def _dyn_clean_s(a_s: str) -> str:
        s = str(a_s).strip()
        if "Rescat" in s or s == "Rescate":
            return "Rescate"
        if "Neg" in s:
            return "Negociar"
        return s

    def _dyn_clean_f(a_f: str) -> str:
        s = str(a_f).strip()
        return "Cooperar" if "Coop" in s else ("Coludir" if "Col" in s else s)

    def _dyn_clean_k(a_k: str) -> str:
        s = str(a_k).strip()
        for lab in ("Continuar", "Liberar", "Matar"):
            if lab in s:
                return lab
        return s

    def _dyn_u_rel_at(
        theta_k: str, gamma_v: float, alpha_v: float, ransom: float, beta_k: float
    ) -> float:
        try:
            _snap = st.session_state.get("rb_k_params_snapshot")
            if isinstance(_snap, pd.DataFrame) and not _snap.empty:
                _df_k = refresh_kidnapper_endogenous_columns(
                    _snap.copy(), modelo, float(gamma_v), float(gamma_v), alpha=float(alpha_v)
                )
            else:
                _df_k = build_kidnapper_params_df(modelo, R_base=float(ransom))
                _df_k = refresh_kidnapper_endogenous_columns(
                    _df_k, modelo, float(gamma_v), float(gamma_v), alpha=float(alpha_v)
                )
            _df_u = kidnapper_util_df_from_param_df(
                _df_k, modelo, float(gamma_v), float(alpha_v), float(gamma_v),
                float(ransom), str(theta_k), float(beta_k),
            )
            _row = _df_u[_df_u["theta_K"].astype(str) == str(theta_k)]
            if not _row.empty:
                return float(_row["U_rel"].iloc[0])
        except Exception:
            pass
        return float("nan")

    _cycles_dyn = st.session_state.get("dynamic_cycles52", [])
    _cycles_dyn = [
        c for c in _cycles_dyn
        if isinstance(c, dict) and "__error__" not in c
    ] if isinstance(_cycles_dyn, list) else []

    if not _cycles_dyn:
        st.info(
            "No hay ciclos dinámicos. Presione **Avanzar ciclos** (panel superior) tras **Iniciar proceso**."
        )
    else:
        _dyn_cycles_render_sig = _stable_json_signature({
            "signature": st.session_state.get("dynamic_cycles52_signature", {}),
            "stop": st.session_state.get("dynamic_cycles_stop52", {}),
            "n_cycles": len(_cycles_dyn),
        })
        if st.session_state.get("dyn_render_sig") != _dyn_cycles_render_sig:
            st.session_state["dyn_render_requested"] = False
            st.session_state["dyn_render_sig"] = _dyn_cycles_render_sig
        _render_dyn_now = st.button(
            "Cargar gráficas de pestaña 6",
            key="btn_load_dyn_tab6",
            type="primary",
            use_container_width=True,
            help=(
                "Renderiza las gráficas usando los ciclos ya calculados por Tabla 5.2. "
                "No recalcula ni modifica los resultados del mecanismo."
            ),
        )
        if _render_dyn_now:
            st.session_state["dyn_render_requested"] = True
        if not bool(st.session_state.get("dyn_render_requested", False)):
            _stop_preview = st.session_state.get("dynamic_cycles_stop52") or {}
            st.success(
                f"Ciclos calculados: {len(_cycles_dyn)}. "
                f"Parada: τ={_stop_preview.get('tau', '—')}, m={_stop_preview.get('m', '—')}."
            )
            st.info(
                "Para acelerar el flujo después de **Avanzar ciclos**, las gráficas pesadas no se "
                "renderizan automáticamente. Use el botón para ver la pestaña 6 completa con los "
                "mismos resultados guardados."
            )
            st.stop()
        _mu0_dyn = {
            th: float(st.session_state.final_priors[i]) / 100.0
            for i, th in enumerate(TIPOS_SECUESTRADOR)
        }
        _mu_rows_dyn: list[dict[str, Any]] = [{"tau": 0, **_dyn_mu_norm(_mu0_dyn)}]
        _seen_mu_tau: set[int] = {0}
        for _cy in _cycles_dyn:
            _tau_c = int(_cy.get("tau_start", 0))
            _post = _cy.get("post54", {}) if isinstance(_cy.get("post54"), dict) else {}
            _mu_pr = _post.get("mu_prior", {})
            if _tau_c >= 1 and isinstance(_mu_pr, dict) and _tau_c not in _seen_mu_tau:
                _mu_rows_dyn.append({"tau": _tau_c, **_dyn_mu_norm(_mu_pr)})
                _seen_mu_tau.add(_tau_c)
        _last = _cycles_dyn[-1]
        _post_last = _last.get("post54", {}) if isinstance(_last.get("post54"), dict) else {}
        _tau_end = int(_last.get("tau_end", int(_last.get("tau_start", 0)) + 1))
        if isinstance(_post_last.get("mu_post"), dict) and _tau_end not in _seen_mu_tau:
            _mu_rows_dyn.append({"tau": _tau_end, **_dyn_mu_norm(_post_last["mu_post"])})
        _df_mu_dyn = pd.DataFrame(_mu_rows_dyn).sort_values("tau")

        _R_dyn = float(st.session_state.get("tab3_R_override", 20_000_000.0))
        _beta_k_dyn = float(st.session_state.get("cal_beta_k", 1.0))
        _cycle_rows: list[dict[str, Any]] = []
        for _cy in _cycles_dyn:
            _d = _cy.get("diag", {}) if isinstance(_cy.get("diag"), dict) else {}
            _post = _cy.get("post54", {}) if isinstance(_cy.get("post54"), dict) else {}
            _mu_pr = _dyn_mu_norm(_post.get("mu_prior", _mu0_dyn))
            _tau_c = int(_cy.get("tau_start", 0))
            _iric = dict(_d.get("IRIC", {}))
            _gam = float(_d.get("gamma_optimo", np.nan))
            _alp = float(_d.get("alpha_optimo", np.nan))
            _cycle_rows.append({
                "tau": _tau_c,
                **{th: float(_mu_pr.get(th, 0.0)) for th in TIPOS_SECUESTRADOR},
                "iota": float(_d.get("iota_prior", np.nan)),
                "iota_post": float(_d.get("iota_post", np.nan)),
                "alpha": _alp,
                "gamma": _gam,
                "alpha_R": float(_d.get("alpha_R_pi_ref", np.nan)),
                "gamma_R": float(_d.get("gamma_R_pi_ref", np.nan)),
                "alpha_N": float(_d.get("alpha_N_pi_ref", np.nan)),
                "gamma_N": float(_d.get("gamma_N_pi_ref", np.nan)),
                "Delta_H": float(_d.get("Delta_H", np.nan)),
                "V": int(_d.get("V_nuevo", 0)),
                "d": int(_d.get("d_nuevo", 0)),
                "a_S": _dyn_clean_s(_d.get("a_S_star", "—")),
                "a_S_t": _dyn_clean_s(_d.get("ats", "—")),
                "a_F": _dyn_clean_f(_d.get("a_F_star", "—")),
                "a_F_t": _dyn_clean_f(_d.get("atf", "—")),
                "a_K": _dyn_clean_k(_d.get("a_K_star", "—")),
                "a_K_t": _dyn_clean_k(_d.get("atk", "—")),
                "match_S": _dyn_clean_s(_d.get("a_S_star", "")) == _dyn_clean_s(_d.get("ats", "")),
                "match_F": _dyn_clean_f(_d.get("a_F_star", "")) == _dyn_clean_f(_d.get("atf", "")),
                "match_K": _dyn_clean_k(_d.get("a_K_star", "")) == _dyn_clean_k(_d.get("atk", "")),
                "U_rel": _dyn_u_rel_at(_tipo_dyn, _gam, _alp, _R_dyn, _beta_k_dyn),
                "IR_K": bool(_iric.get("IR_K", False)),
                "IC_K": bool(_iric.get("IC_K", False)),
                "IR_F": bool(_iric.get("IR_F", False)),
                "Gamma_formal": bool(_iric.get("Gamma_formal", False)),
                "IR_K_gap_E": float(_iric.get("IR_K_gap_E", np.nan)),
                "IC_K_gap_E_min": float(_iric.get("IC_K_gap_E_min", np.nan)),
                "IR_F_gap_E": float(_iric.get("IR_F_gap_E", np.nan)),
                "IR_K_true": bool(_iric.get("IR_K_true", False)),
                "IR_K_true_gap": float(_iric.get("IR_K_true_gap", np.nan)),
                "IR_K_true_U_rel": float(_iric.get("IR_K_true_U_rel", np.nan)),
                "IR_K_true_V_cont": float(_iric.get("IR_K_true_V_cont", np.nan)),
                "IR_K_true_U_kill": float(_iric.get("IR_K_true_U_kill", np.nan)),
                "IC_F": bool(_iric.get("IC_F", False)),
                "IC_S": bool(_iric.get("IC_S", False)),
                "m": str(_d.get("m", "—")),
                **{
                    f"kh({_th})": str(
                        (dict(_d.get("kh_signs", {})) if isinstance(_d.get("kh_signs"), dict) else {}).get(str(_th), "—")
                    )
                    for _th in TIPOS_SECUESTRADOR
                },
            })
        _df_cyc = pd.DataFrame(_cycle_rows).sort_values("tau")
        _stop = st.session_state.get("dynamic_cycles_stop52") or {}
        _tau_stop = int(_stop.get("tau", 0)) if str(_stop.get("motivo", "")) == "desenlace" else None
        _m_stop = str(_stop.get("m", "")) if _tau_stop is not None else ""
        from figuras_plot import terminal_tau_from_frame

        _tau_band_dyn, _m_band_dyn = terminal_tau_from_frame(_df_cyc, tau_col="tau", m_col="m")
        if _tau_band_dyn is not None:
            _tau_stop, _m_stop = _tau_band_dyn, _m_band_dyn

        def _plotly_tau_band(fig, tau: int, m_tau: str) -> None:
            from figuras_plot import is_continuar_desenlace

            if is_continuar_desenlace(m_tau):
                return
            fig.add_vrect(
                x0=tau - 0.5,
                x1=tau + 0.5,
                fillcolor="rgba(232,232,232,0.92)",
                layer="below",
                line_width=0,
            )

        # ── 1. Solo μ ───────────────────────────────────────────────────────
        st.markdown("### 1 · Creencias μ_τ(θ)")
        _df_mu_long = _df_mu_dyn.melt(
            id_vars="tau", value_vars=TIPOS_SECUESTRADOR, var_name="Tipo", value_name="μ"
        )
        _fig_mu = px.line(
            _df_mu_long, x="tau", y="μ", color="Tipo", markers=True,
            color_discrete_map=_TIPO_COLORS,
            labels={"tau": "τ", "μ": "μ_τ(θ)"},
        )
        _fig_mu.update_yaxes(range=[0, 1.05], tickformat=".2f")
        _fig_mu.update_xaxes(dtick=1)
        if _tau_stop is not None:
            _plotly_tau_band(_fig_mu, _tau_stop, _m_stop)
        _fig_mu.update_layout(margin=dict(t=20, b=10), legend_title="")
        st.plotly_chart(_fig_mu, use_container_width=True)
        _mu_t0 = {th: float(_df_mu_dyn.iloc[0][th]) for th in TIPOS_SECUESTRADOR}
        _mu_last_row = _df_mu_dyn.iloc[-1]
        _mu_tf = {th: float(_mu_last_row[th]) for th in TIPOS_SECUESTRADOR}
        _th_dom0 = max(_mu_t0, key=_mu_t0.get)
        _th_domf = max(_mu_tf, key=_mu_tf.get)
        _dyn_comment([
            f"En τ=0 la masa modal cae en **{_th_dom0}** (μ={_mu_t0[_th_dom0]:.3f}); al cierre del horizonte domina **{_th_domf}** (μ={_mu_tf[_th_domf]:.3f}), tipo generador **{_tipo_dyn}**.",
            "La trayectoria ilustra el filtro bayesiano de **Mechanism.tex**: cada ciclo repondera tipos según verosimilitudes ℒ_H, ℒ_F y ℒ_C condicionadas a (V_t, d_t, m_t).",
            f"La concentración sobre {_th_domf} alimenta ι_t= max_θ μ_t(θ) usado en la rama de rescate (eq. p-surv-rescue-logit-ajustado) y en la regla discreta a_S^*.",
        ])

        # ── 2. α* vs benchmarks R y N ───────────────────────────────────────
        st.markdown("### 2 · Política óptima α* frente a benchmarks de rescate (R) y negociación (N)")
        _fig_a = go.Figure()
        _fig_a.add_trace(go.Scatter(
            x=_df_cyc["tau"], y=_df_cyc["alpha"], mode="lines+markers", name="α* (óptimo)",
            line=dict(color="#1d4ed8", width=2.5),
        ))
        _fig_a.add_trace(go.Scatter(
            x=_df_cyc["tau"], y=_df_cyc["alpha_R"], mode="lines+markers", name="α^R (piso rescate)",
            line=dict(color="#60a5fa", dash="dash"),
        ))
        _fig_a.add_trace(go.Scatter(
            x=_df_cyc["tau"], y=_df_cyc["alpha_N"], mode="lines+markers", name="α^N (piso negociación)",
            line=dict(color="#f59e0b", dash="dot"),
        ))
        _fig_a.update_yaxes(range=[0, 1.05], title="α", tickformat=".2f")
        _fig_a.update_xaxes(dtick=1, title="τ")
        if _tau_stop is not None:
            _plotly_tau_band(_fig_a, _tau_stop, _m_stop)
        _fig_a.update_layout(margin=dict(t=20, b=10), legend_title="")
        st.plotly_chart(_fig_a, use_container_width=True)
        _a_mean = float(_df_cyc["alpha"].mean())
        _dyn_comment([
            f"El α* promedio en la corrida es **{_a_mean:.3f}**, con separación respecto a los pisos π-referencia α^R y α^N de **eq. state-expected-loss**.",
            "Cuando α* se acerca a α^N el Estado internaliza mayor costo de transferencia bloqueada (ω_p R(1−α)) en la rama de negociación; cercanía a α^R anticipa rescate focal.",
            "La forma cuadrática C_ops(α,γ;θ_K) y C_maint en **Mechanism.tex** explica por qué los tres trazos no coinciden punto a punto bajo μ_t heterogéneo.",
        ])

        # ── 3. γ* vs benchmarks ───────────────────────────────────────────────
        st.markdown("### 3 · Política óptima γ* frente a benchmarks de rescate (R) y negociación (N)")
        _fig_g = go.Figure()
        _fig_g.add_trace(go.Scatter(
            x=_df_cyc["tau"], y=_df_cyc["gamma"], mode="lines+markers", name="γ* (óptimo)",
            line=dict(color="#15803d", width=2.5),
        ))
        _fig_g.add_trace(go.Scatter(
            x=_df_cyc["tau"], y=_df_cyc["gamma_R"], mode="lines+markers", name="γ^R (piso rescate)",
            line=dict(color="#86efac", dash="dash"),
        ))
        _fig_g.add_trace(go.Scatter(
            x=_df_cyc["tau"], y=_df_cyc["gamma_N"], mode="lines+markers", name="γ^N (piso negociación)",
            line=dict(color="#fb923c", dash="dot"),
        ))
        _fig_g.update_yaxes(range=[0, 1.05], title="γ", tickformat=".2f")
        _fig_g.update_xaxes(dtick=1, title="τ")
        if _tau_stop is not None:
            _plotly_tau_band(_fig_g, _tau_stop, _m_stop)
        _fig_g.update_layout(margin=dict(t=20, b=10), legend_title="")
        st.plotly_chart(_fig_g, use_container_width=True)
        _g_mean = float(_df_cyc["gamma"].mean())
        _dyn_comment([
            f"La presión óptima media es γ*≈**{_g_mean:.3f}**, comparada con γ^R y γ^N calculados bajo el principio de partición del programa del Estado.",
            "Incrementos de γ* elevan hazards de cierre (eq. xi) y la probabilidad de detección p_det=Λ(η_0+η_1 α+η_2 γ), coherente con los ciclos donde d_t=1.",
            "La divergencia entre γ* y los benchmarks π refleja el trade-off humano-financiero: más γ no siempre reduce L^S* si sube el riesgo de muerte en cautiverio.",
        ])

        # ── 4. Frecuencias a* vs ã ───────────────────────────────────────────
        st.markdown("### 4 · Frecuencia de decisiones óptimas (a*) vs. ejecución MDG (ã)")
        _n_cyc = max(1, len(_df_cyc))
        _freq_rows = []
        for _agent, _col_s, _col_t, _match in (
            ("Estado S", "a_S", "a_S_t", "match_S"),
            ("Familia F", "a_F", "a_F_t", "match_F"),
            ("Secuestrador K", "a_K", "a_K_t", "match_K"),
        ):
            for _lab, _src in (("Óptimo a*", _col_s), ("Ejecutado ã", _col_t)):
                _vc = _df_cyc[_src].value_counts(normalize=True)
                for _act, _p in _vc.items():
                    _freq_rows.append({
                        "Agente": _agent, "Serie": _src, "Acción": str(_act),
                        "Frecuencia": float(_p) * 100.0,
                    })
            _freq_rows.append({
                "Agente": _agent, "Serie": "Coincidencia a*=ã",
                "Acción": "Match",
                "Frecuencia": 100.0 * float(_df_cyc[_match].mean()),
            })
        _df_freq = pd.DataFrame(_freq_rows)
        _fig_bar_act = px.bar(
            _df_freq, x="Agente", y="Frecuencia", color="Acción", facet_col="Serie",
            barmode="group", text_auto=".1f",
        )
        _fig_bar_act.update_yaxes(title="% ciclos")
        _fig_bar_act.update_layout(margin=dict(t=20, b=10), showlegend=True)
        st.plotly_chart(_fig_bar_act, use_container_width=True)
        _mS = 100.0 * float(_df_cyc["match_S"].mean())
        _mF = 100.0 * float(_df_cyc["match_F"].mean())
        _mK = 100.0 * float(_df_cyc["match_K"].mean())
        _dyn_comment([
            f"Coincidencia óptimo–ejecutado: S **{_mS:.0f}%**, F **{_mF:.0f}%**, K **{_mK:.0f}%** de los ciclos (ley de implementación P_I^j en el bloque MDG).",
            "Desviaciones ã≠a* cuantifican la capa MDG (mano temblorosa): el Principal diseña a^* pero el registro público observa ã_t, base de las verosimilitudes en Tabla 14.",
            "Baja coincidencia en K o F tensiona IC^K / IR^F de **subsec:ir-ic**; alta coincidencia en S indica alineación entre V^R, V^N y la regla discreta.",
        ])

        # ── 5. Voz ────────────────────────────────────────────────────────────
        st.markdown("### 5 · Frecuencia de la señal de voz V_τ")
        _vc_v = _df_cyc["V"].value_counts().reindex([0, 1], fill_value=0)
        _df_vb = pd.DataFrame({
            "Señal": ["Silencio (0)", "Voz (1)"],
            "Pct": [
                100.0 * float(_vc_v.get(0, 0)) / _n_cyc,
                100.0 * float(_vc_v.get(1, 0)) / _n_cyc,
            ],
        })
        _fig_v = px.bar(
            _df_vb, x="Señal", y="Pct", color="Señal",
            color_discrete_map={"Silencio (0)": "#94a3b8", "Voz (1)": "#2563eb"},
            text_auto=".1f",
        )
        _fig_v.update_layout(showlegend=False, margin=dict(t=20, b=10), yaxis_title="% ciclos")
        st.plotly_chart(_fig_v, use_container_width=True)
        _pv1 = 100.0 * float(_vc_v.get(1, 0)) / _n_cyc
        _dyn_comment([
            f"La señal V=1 aparece en **{_pv1:.0f}%** de los ciclos; el resto es silencio, estado que en ℒ_C usa el factor (1−π_call)^ω_voz (eq. LC / Lvoz-diag).",
            "V=1 acelera la separación de tipos cuando ω_voz>0 porque eleva la verosimilitud de comunicación del tipo con π_call alto.",
            f"Para θ={_tipo_dyn}, la calibración de π_call en pestaña 2 fija la tasa base con la que V discrimina entre DC, PAR, ELN y FARC.",
        ])

        # ── 6. Detección ─────────────────────────────────────────────────────
        st.markdown("### 6 · Frecuencia de la señal de detección d_τ")
        _vc_d = _df_cyc["d"].value_counts().reindex([0, 1], fill_value=0)
        _df_db = pd.DataFrame({
            "Señal": ["No detección (0)", "Detección (1)"],
            "Pct": [
                100.0 * float(_vc_d.get(0, 0)) / _n_cyc,
                100.0 * float(_vc_d.get(1, 0)) / _n_cyc,
            ],
        })
        _fig_d = px.bar(
            _df_db, x="Señal", y="Pct", color="Señal",
            color_discrete_map={"No detección (0)": "#cbd5e1", "Detección (1)": "#f59e0b"},
            text_auto=".1f",
        )
        _fig_d.update_layout(showlegend=False, margin=dict(t=20, b=10), yaxis_title="% ciclos")
        st.plotly_chart(_fig_d, use_container_width=True)
        _pd1 = 100.0 * float(_vc_d.get(1, 0)) / _n_cyc
        _dyn_comment([
            f"Detección colusión d=1 en **{_pd1:.0f}%** de los ciclos; d_t ~ Bernoulli(p_det) con p_det=Λ(η_0+η_1 α*+η_2 γ*) del mecanismo.",
            "Ciclos con d=1 suelen coincidir con α* o γ* elevados, consistente con la política de asfixia financiera y presión perimetral del Estado.",
            "La señal d entra en el vector público h_t y condiciona la actualización bayesiana junto con m_t y V_t (eq. historia-publica).",
        ])

        # ── 7. ΔH ─────────────────────────────────────────────────────────────
        st.markdown("### 7 · Ganancia esperada de información ΔH del Estado")
        _fig_dh = px.bar(
            _df_cyc, x="tau", y="Delta_H", text_auto=".3f",
            labels={"tau": "τ", "Delta_H": "ΔH"},
            color_discrete_sequence=["#7c3aed"],
        )
        _fig_dh.update_layout(margin=dict(t=20, b=10), showlegend=False)
        st.plotly_chart(_fig_dh, use_container_width=True)
        _dh_mean = float(_df_cyc["Delta_H"].mean())
        _dh_max = float(_df_cyc["Delta_H"].max())
        _dyn_comment([
            f"ΔH promedio **{_dh_mean:.3f}** y máximo **{_dh_max:.3f}** en la corrida; mide E[H(μ_{{t+1}})]−H(μ_t) bajo contrafactuales de m (eq. expected-information-gain).",
            "Valores positivos indican que la política (α*,γ*) elegida en el piso activo (rescate o negociación) redujo entropía esperada antes de observar la señal.",
            "El peso ψ_H en el objetivo del Estado traduce esta ganancia en incentivo exploratorio sin recurrir a temblores continuos sobre los instrumentos.",
        ])

        # ── 8. γ* vs ι ────────────────────────────────────────────────────────
        st.markdown("### 8 · γ* óptimo vs. precisión modal ι")
        _fig_gi = px.scatter(
            _df_cyc, x="gamma", y="iota", text="tau",
            labels={"gamma": "γ*", "iota": "ι = max_θ μ(θ)"},
        )
        _fig_gi.update_traces(textposition="top center", marker=dict(size=11, color="#15803d"))
        _fig_gi.update_xaxes(range=[0, 1.05])
        _fig_gi.update_yaxes(range=[0, 1.05])
        st.plotly_chart(_fig_gi, use_container_width=True)
        _dyn_comment([
            f"Dispersión γ*–ι: al subir ι (máxima creencia modal) la política puede modular γ* sin confundir identificación con presión militar (Proposición óptima bajo μ).",
            "ι_t resume certidumbre institucional (eq. iota-precision-mode) y alimenta la logística de rescate vía β_R·ι·1{θ̂=θ}.",
            "Correlaciones visuales entre γ* e ι guían si el Estado intensifica presión cuando ya identificó al tipo dominante.",
        ])

        # ── 9. α* vs ι ────────────────────────────────────────────────────────
        st.markdown("### 9 · α* óptimo vs. precisión modal ι")
        _fig_ai = px.scatter(
            _df_cyc, x="alpha", y="iota", text="tau",
            labels={"alpha": "α*", "iota": "ι"},
        )
        _fig_ai.update_traces(textposition="top center", marker=dict(size=11, color="#1d4ed8"))
        _fig_ai.update_xaxes(range=[0, 1.05])
        _fig_ai.update_yaxes(range=[0, 1.05])
        st.plotly_chart(_fig_ai, use_container_width=True)
        _dyn_comment([
            f"α* vs ι muestra cómo el bloqueo financiero óptimo responde a la concentración de μ_t: α* medio **{float(_df_cyc['alpha'].mean()):.3f}** con ι medio **{float(_df_cyc['iota'].mean()):.3f}**.",
            "En la rama de negociación, α* aparece en el término ω_p R(1−α) (eq. negotiation-cost); en rescate, α* entra en C_ops y en p_det.",
            "Trayectorias planas sugieren esquinas de la grilla factible Γ_t(μ_t); saltos indican cambio de piso Rescate/Negociación.",
        ])

        # ── 10. γ* vs U_rel ─────────────────────────────────────────────────────
        st.markdown("### 10 · γ* óptimo vs. utilidad de liberar U^K_rel(θ)")
        _df_ur = _df_cyc.dropna(subset=["U_rel", "gamma"])
        _fig_ug = px.scatter(
            _df_ur, x="gamma", y="U_rel", text="tau",
            labels={"gamma": "γ*", "U_rel": f"U_rel({_tipo_dyn})"},
        )
        _fig_ug.update_traces(textposition="top center", marker=dict(size=11, color="#b45309"))
        st.plotly_chart(_fig_ug, use_container_width=True)
        _ur_mean = float(_df_ur["U_rel"].mean()) if not _df_ur.empty else float("nan")
        _dyn_comment([
            f"U_rel({_tipo_dyn}) promedio **{_ur_mean:.2f}** frente a γ* (utilidad de liberar a_rel, eq. ir-K: U_rel > max{{V_cont, U_kill}}).",
            "Pendiente negativa de U_rel en γ indica que mayor presión militar mejora la salida pacífica relativa al continuar, canal de disuasión del secuestrador.",
            "La comparación visual apoya verificar IR^K ciclo a ciclo: liberar debe dominar matar y continuar bajo la política vigente.",
        ])

        # ── 11. IR / IC ─────────────────────────────────────────────────────────
        st.markdown("### 11 · Verificación IR / IC por ciclo")
        _iric_labels = [
            "IR^K secuestrador",
            "IC^K secuestrador",
            "IR^F familia",
        ]
        _iric_cols = ["IR_K", "IC_K", "IR_F"]
        _iric_pct = [
            100.0 * float(_df_cyc[c].mean()) for c in _iric_cols
        ]
        _df_ir = pd.DataFrame({"Restricción": _iric_labels, "Pct": _iric_pct})
        _fig_ir = px.bar(
            _df_ir, x="Restricción", y="Pct", color="Restricción",
            color_discrete_sequence=px.colors.qualitative.Set2,
            text_auto=".0f",
        )
        _fig_ir.update_yaxes(range=[0, 105], title="% ciclos")
        _fig_ir.update_xaxes(title="Restricciones formales evaluadas en el óptimo del Estado")
        _fig_ir.update_layout(showlegend=False, margin=dict(t=30, b=45))
        st.plotly_chart(_fig_ir, use_container_width=True)
        _iric_txt = ", ".join(f"{lab} {pct:.0f}%" for lab, pct in zip(_iric_labels, _iric_pct))
        _dyn_comment([
            f"Cumplimiento medio en la corrida: {_iric_txt} (eq. ir-K, ic-kidnapper e ir-family).",
            "La gráfica usa solo el IR/IC guardado para el candidato elegido por el Estado en cada ciclo: a_S^*, α_t^*, γ_t^*.",
            "Se excluyen IC_F e IC_S porque son chequeos operativos de la app, no restricciones formales de Γ_t(μ_t) en Mechanism.tex.",
        ])

        # Construcción anticipada de frecuencias κ_h (usada en expander y en sección 12)
        _kh_sign_labels = {"-1": "−1 (γ↑ empeora Estado)", "0": "0 (nulo/indeterminado)", "1": "+1 (γ↑ mejora Estado)"}
        _kh_sign_colors = {
            "−1 (γ↑ empeora Estado)": "#c0392b",
            "0 (nulo/indeterminado)": "#7f8c8d",
            "+1 (γ↑ mejora Estado)": "#27ae60",
        }
        _kh_freq_rows: list[dict[str, Any]] = []
        for _th_kh in TIPOS_SECUESTRADOR:
            _cnt_kh: dict[str, int] = {"-1": 0, "0": 0, "1": 0}
            for _cy_kh in _cycles_dyn:
                _d_kh = _cy_kh.get("diag", {}) if isinstance(_cy_kh.get("diag"), dict) else {}
                _khs = _d_kh.get("kh_signs", {}) if isinstance(_d_kh.get("kh_signs"), dict) else {}
                _s_kh = str(_khs.get(str(_th_kh), "")).strip()
                if _s_kh in _cnt_kh:
                    _cnt_kh[_s_kh] += 1
            _total_kh = sum(_cnt_kh.values())
            for _sign_kh, _count_kh in _cnt_kh.items():
                _kh_freq_rows.append({
                    "Tipo": str(_th_kh),
                    "−sgn(κ_h)": _kh_sign_labels[_sign_kh],
                    "Ciclos": _count_kh,
                    "Porcentaje (%)": round(100.0 * _count_kh / max(_total_kh, 1), 1),
                })
        _df_kh_freq = pd.DataFrame(_kh_freq_rows)
        # Pivot: filas = tipo, columnas = signo (para vista tabular compacta)
        if not _df_kh_freq.empty:
            _df_kh_pivot = _df_kh_freq.pivot_table(
                index="Tipo",
                columns="−sgn(κ_h)",
                values=["Ciclos", "Porcentaje (%)"],
                aggfunc="sum",
            ).reindex(TIPOS_SECUESTRADOR)
            _df_kh_pivot.columns = [
                f"{col[0]} · {col[1]}" for col in _df_kh_pivot.columns
            ]
            _df_kh_pivot = _df_kh_pivot.reset_index()
        else:
            _df_kh_pivot = pd.DataFrame()

        st.markdown("### 12 · Frecuencia de −sgn(κ_h) por tipo en los ciclos dinámicos")
        st.caption(
            r"Conteo de ciclos donde −sgn(κ_h(θ,t)) = −1, 0 ó +1 para cada tipo de secuestrador. "
            r"−1: γ↑ empeora la posición del Estado (Muerte/Rescate dominan sobre Pago). "
            r"+1: γ↑ mejora la posición del Estado (Pago domina). "
            r"κ_h(θ,t) = ζ_γ^(2)·λ̃₂ + ζ_γ^(3)·λ̃₃ − ζ_γ^(1)·λ̃₁."
        )
        if not _df_kh_freq.empty and _df_kh_freq["Ciclos"].sum() > 0:
            _fig_kh = px.bar(
                _df_kh_freq,
                x="Tipo",
                y="Ciclos",
                color="−sgn(κ_h)",
                barmode="group",
                color_discrete_map=_kh_sign_colors,
                text="Ciclos",
                hover_data={"Porcentaje (%)": True, "Ciclos": True},
                category_orders={
                    "Tipo": TIPOS_SECUESTRADOR,
                    "−sgn(κ_h)": list(_kh_sign_labels.values()),
                },
            )
            _fig_kh.update_traces(textposition="outside", cliponaxis=False)
            _fig_kh.update_yaxes(title="Número de ciclos", rangemode="tozero")
            _fig_kh.update_xaxes(title="Tipo de secuestrador (θ)")
            _fig_kh.update_layout(
                legend_title="−sgn(κ_h)",
                margin=dict(t=35, b=55),
            )
            st.plotly_chart(_fig_kh, use_container_width=True)
            _kh_dom = {
                _th_kh: max(
                    [r for r in _kh_freq_rows if r["Tipo"] == _th_kh],
                    key=lambda r: r["Ciclos"],
                )["−sgn(κ_h)"]
                for _th_kh in TIPOS_SECUESTRADOR
            }
            _kh_n1_types = [th for th in TIPOS_SECUESTRADOR if _kh_dom[th].startswith("−1")]
            _kh_p1_types = [th for th in TIPOS_SECUESTRADOR if _kh_dom[th].startswith("+1")]
            _dyn_comment([
                f"En {len(_cycles_dyn)} ciclos: signo dominante por tipo — "
                + ", ".join(f"{th}: {_kh_dom[th]}" for th in TIPOS_SECUESTRADOR) + ".",
                ("Tipos con γ↑ adverso (−1 dominante): " + ", ".join(_kh_n1_types)) if _kh_n1_types
                else "Ningún tipo muestra signo −1 dominante en esta corrida.",
                "Leer junto con γ* (sección 3): cuando −sgn=−1 prevalece, el óptimo del Estado tiende a moderar la presión operacional.",
            ])
        else:
            st.info("No hay datos de κ_h en los ciclos actuales. Presione **Avanzar ciclos** primero.")

        with st.expander("Datos agregados por ciclo (tabla)", expanded=False):
            st.dataframe(_df_cyc, hide_index=True, use_container_width=True)
            st.dataframe(_df_mu_dyn, hide_index=True, use_container_width=True)
            if not _df_kh_pivot.empty:
                st.markdown("**Frecuencia de −sgn(κ_h) por tipo**")
                st.dataframe(_df_kh_pivot, hide_index=True, use_container_width=True)
                st.dataframe(
                    _df_kh_freq.sort_values(["Tipo", "−sgn(κ_h)"]),
                    hide_index=True,
                    use_container_width=True,
                )

        def _export_app_apa_figures_for_current_run() -> dict[str, Any]:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib.ticker import MultipleLocator

            out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figuras_calibracion"))
            os.makedirs(out_dir, exist_ok=True)
            theta_now = str(_tipo_dyn)

            from figuras_plot import (
                annotate_tau_in_legend,
                draw_tau_desenlace_band,
                terminal_tau_from_frame,
            )

            cyc = _df_cyc.copy()
            mu_df = _df_mu_dyn.copy()
            cyc["theta_true"] = theta_now
            mu_df["theta_true"] = theta_now
            _stop_exp = st.session_state.get("dynamic_cycles_stop52") or {}
            if str(_stop_exp.get("motivo", "")).lower() == "desenlace":
                tau_hyp = int(_stop_exp.get("tau", 0))
                m_tau = str(_stop_exp.get("m", "Continuar"))
            else:
                tau_hyp, m_tau = terminal_tau_from_frame(cyc, tau_col="tau", m_col="m")
                if tau_hyp is None:
                    tau_hyp = int(cyc["tau"].max())
                    m_tau = "Continuar"
            cyc["tau_hyp"] = tau_hyp
            cyc["m_tau"] = m_tau
            mu_df["tau_hyp"] = tau_hyp
            mu_df["m_tau"] = m_tau

            cyc_path = os.path.join(out_dir, f"app_export_cycles_{theta_now}.csv")
            mu_path = os.path.join(out_dir, f"app_export_mu_{theta_now}.csv")
            cyc.to_csv(cyc_path, index=False)
            mu_df.to_csv(mu_path, index=False)

            plt.rcParams.update({
                "font.family": "serif",
                "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                "axes.edgecolor": "0.15",
                "axes.linewidth": 0.6,
                "pdf.fonttype": 42,
                "ps.fonttype": 42,
            })

            types_pair = ("DC", "FARC")

            def _read_pair(prefix: str) -> Optional[pd.DataFrame]:
                frames = []
                for th in types_pair:
                    pth = os.path.join(out_dir, f"app_export_{prefix}_{th}.csv")
                    if not os.path.exists(pth):
                        return None
                    frames.append(pd.read_csv(pth))
                return pd.concat(frames, ignore_index=True)

            def _setup_axis(ax):
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.grid(True, axis="y", color="0.88", linewidth=0.6)
                ax.tick_params(axis="both", labelsize=8)

            def _time_panel(df: pd.DataFrame, filename: str, yspec: list[tuple[str, str, str, str]], ylabel: str, ylim=(0.0, 1.05)):
                fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.05), sharex=False, sharey=False)
                for ax, th, letter in zip(axes, types_pair, ("a", "b")):
                    d = df[df["theta_true"].astype(str) == th].copy().sort_values("tau")
                    if d.empty:
                        continue
                    tau_t, m_tau = terminal_tau_from_frame(
                        d, tau_col="tau", m_col="m", tau_hyp_col="tau_hyp", m_tau_col="m_tau"
                    )
                    for col, label, linestyle, color in yspec:
                        ax.plot(d["tau"], d[col], linestyle=linestyle, color=color, linewidth=1.15, label=label)
                    _setup_axis(ax)
                    ax.set_title(f"({letter}) {th}", fontsize=9, loc="left", pad=9)
                    ax.set_xlabel(r"Periodo $\tau$", fontsize=8)
                    ax.set_ylabel(ylabel, fontsize=8)
                    xmax = int(max(1, d["tau"].max()))
                    ax.set_xlim(0, xmax)
                    ax.xaxis.set_major_locator(MultipleLocator(max(1, int(np.ceil(xmax / 8)))))
                    ax.xaxis.set_minor_locator(MultipleLocator(1))
                    if ylim is not None:
                        ax.set_ylim(*ylim)
                    if tau_t is not None:
                        draw_tau_desenlace_band(ax, tau_t, m_tau)
                handles, labels = axes[0].get_legend_handles_labels()
                fig.legend(handles, labels, loc="lower center", ncol=min(3, len(labels)), frameon=False, fontsize=7)
                fig.tight_layout(rect=(0, 0.17, 1, 0.98), w_pad=2.1)
                fig.savefig(os.path.join(out_dir, filename), bbox_inches="tight")
                plt.close(fig)

            def _iota_panel(df: pd.DataFrame, filename: str, ycol: str, ylabel: str):
                fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.05), sharex=True, sharey=True)
                for ax, th, letter in zip(axes, types_pair, ("a", "b")):
                    d = df[df["theta_true"].astype(str) == th].copy().sort_values("tau")
                    if d.empty:
                        continue
                    tau_t, m_tau = terminal_tau_from_frame(
                        d, tau_col="tau", m_col="m", tau_hyp_col="tau_hyp", m_tau_col="m_tau"
                    )
                    ax.plot(d["iota"], d[ycol], color="0.0", linewidth=1.0, marker="o", markersize=2.2, label="Trayectoria")
                    if tau_t is not None:
                        dt = d[d["tau"] == tau_t]
                        if not dt.empty:
                            ax.plot(
                                dt["iota"],
                                dt[ycol],
                                marker="o",
                                markersize=6.2,
                                markerfacecolor="white",
                                markeredgecolor="0.0",
                                linestyle="None",
                                label=annotate_tau_in_legend(m_tau),
                            )
                    _setup_axis(ax)
                    ax.set_title(f"({letter}) {th}", fontsize=9, loc="left", pad=9)
                    ax.set_xlabel(r"Precisión posterior $\iota_\tau$", fontsize=8)
                    ax.set_ylabel(ylabel, fontsize=8)
                    ax.set_xlim(0, 1.05)
                    ax.set_ylim(0, 1.05)
                handles, labels = axes[0].get_legend_handles_labels()
                fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=7)
                fig.tight_layout(rect=(0, 0.17, 1, 0.98), w_pad=2.1)
                fig.savefig(os.path.join(out_dir, filename), bbox_inches="tight")
                plt.close(fig)

            pair_cyc = _read_pair("cycles")
            pair_mu = _read_pair("mu")
            made_combined = False
            if pair_cyc is not None and pair_mu is not None:
                mu_plot_rows = []
                for _, rr in pair_mu.iterrows():
                    th = str(rr["theta_true"])
                    mu_plot_rows.append({
                        "theta_true": th,
                        "tau": int(rr["tau"]),
                        "mu_true": float(rr.get(th, np.nan)),
                        "tau_hyp": int(rr["tau_hyp"]),
                        "m_tau": str(rr["m_tau"]),
                    })
                pair_mu_true = pd.DataFrame(mu_plot_rows)
                _time_panel(pair_mu_true, "fig_mu_dc_farc.pdf", [("mu_true", r"$\mu_\tau(\theta_K^\ast)$", "-", "0.0")], "Posterior", (0.0, 1.05))
                _time_panel(pair_cyc, "fig_alpha_dc_farc.pdf", [("alpha", r"$\alpha_\tau^\ast$", "-", "0.0"), ("alpha_R", r"$\alpha^R$", ":", "0.35"), ("alpha_N", r"$\alpha^N$", "-.", "0.35")], r"Bloqueo financiero $\alpha$", (0.0, 1.05))
                _time_panel(pair_cyc, "fig_gamma_dc_farc.pdf", [("gamma", r"$\gamma_\tau^\ast$", "-", "0.0"), ("gamma_R", r"$\gamma^R$", ":", "0.35"), ("gamma_N", r"$\gamma^N$", "-.", "0.35")], r"Presión operativa $\gamma$", (0.0, 1.05))
                _iota_panel(pair_cyc, "fig_iota_alpha_dc_farc.pdf", "alpha", r"Bloqueo financiero $\alpha_\tau^\ast$")
                _iota_panel(pair_cyc, "fig_iota_gamma_dc_farc.pdf", "gamma", r"Presión operativa $\gamma_\tau^\ast$")
                dh_max = float(max(1e-9, pair_cyc["Delta_H"].max()))
                _time_panel(pair_cyc, "fig_deltaH_dc_farc.pdf", [("Delta_H", r"$\Delta H_\tau$", "-", "0.0")], r"Ganancia de entropía", (0.0, dh_max * 1.10))

                rows_tab = []
                for th in types_pair:
                    dc = pair_cyc[pair_cyc["theta_true"].astype(str) == th].copy()
                    dm = pair_mu_true[pair_mu_true["theta_true"].astype(str) == th].copy()
                    tau_tab, m_tab = terminal_tau_from_frame(
                        dc, tau_col="tau", m_col="m", tau_hyp_col="tau_hyp", m_tau_col="m_tau"
                    )
                    rows_tab.append({
                        "theta": th,
                        "mu0": float(dm["mu_true"].iloc[0]),
                        "muT": float(dm["mu_true"].iloc[-1]),
                        "alpha_bar": float(dc["alpha"].mean()),
                        "gamma_bar": float(dc["gamma"].mean()),
                        "iota_bar": float(dc["iota"].mean()),
                        "deltaH_bar": float(dc["Delta_H"].mean()),
                        "tau_hyp": -1 if tau_tab is None else int(tau_tab),
                        "m_tau": str(m_tab),
                    })
                pd.DataFrame(rows_tab).to_csv(os.path.join(out_dir, "tabla_resumen_calibracion_app.csv"), index=False)
                made_combined = True

            return {
                "out_dir": out_dir,
                "theta": theta_now,
                "tau_hyp": tau_hyp,
                "m_tau": m_tau,
                "combined": made_combined,
            }

        _exp_col1, _exp_col2 = st.columns([1, 2])
        with _exp_col1:
            if st.button("Exportar figuras APA", key="btn_export_app_apa_figs", use_container_width=True):
                try:
                    _export_meta = _export_app_apa_figures_for_current_run()
                    st.success(
                        f"Exportada corrida {_export_meta['theta']} "
                        f"(τ^hyp={_export_meta['tau_hyp']}, m={_export_meta['m_tau']})."
                    )
                    if _export_meta["combined"]:
                        st.info("Ya existen DC y FARC: se generaron las figuras combinadas que usa `Mechanism_2.tex`.")
                    else:
                        st.info("Falta exportar el otro tipo (DC o FARC) para crear las figuras combinadas.")
                except Exception as _apa_exc:
                    st.error(f"No se pudieron exportar las figuras APA: {_apa_exc}")
        with _exp_col2:
            st.caption(
                "Este exportador usa exactamente los datos de la app (`dynamic_cycles52`): "
                "primero exporte DC, luego FARC, y la app arma las figuras combinadas."
            )

        st.divider()
        st.markdown("### 13 · Informe PDF e infografía (generación en la app)")
        st.caption(
            "Sin NotebookLM: se arma un informe descargable y un panel-resumen a partir de la corrida "
            "actual, con lectura alineada a **Mechanism.tex** (Estado, MDG, señales y IR/IC)."
        )

        _dyn_narrative = [
            ("1 · Creencias μ_τ(θ)", [
                f"En τ=0 domina {_th_dom0} (μ={_mu_t0[_th_dom0]:.3f}); al cierre {_th_domf} (μ={_mu_tf[_th_domf]:.3f}); tipo generador {_tipo_dyn}.",
                "Filtro bayesiano: reponderación con ℒ_H, ℒ_F y ℒ_C dado (V_t, d_t, m_t).",
                f"La concentración en {_th_domf} alimenta ι_t y la regla discreta del Estado.",
            ]),
            ("2 · Política α*", [
                f"α* promedio {_a_mean:.3f} frente a pisos α^R y α^N (programa por pisos).",
                "Cercanía a α^N internaliza costo ω_p R(1−α) en negociación; a α^R anticipa rescate.",
                "C_ops y C_maint explican divergencia bajo μ_t heterogéneo.",
            ]),
            ("3 · Política γ*", [
                f"γ* medio {_g_mean:.3f} vs benchmarks γ^R y γ^N.",
                "Mayor γ eleva hazards de cierre y p_det=Λ(η_0+η_1 α+η_2 γ).",
                "Trade-off humano-financiero: más presión no siempre minimiza L^S*.",
            ]),
            ("4 · Decisiones a* vs ã", [
                f"Coincidencia S/F/K: {_mS:.0f}% / {_mF:.0f}% / {_mK:.0f}%.",
                "ã≠a* refleja la capa MDG (P_I^j); base de verosimilitudes Tabla 14.",
                "Desalineación en K o F tensiona IC^K e IR^F.",
            ]),
            ("5–6 · Señales V y d", [
                f"V=1 en {_pv1:.0f}% de ciclos; d=1 en {_pd1:.0f}%.",
                "V discrimina vía (1−π_call)^ω_voz; d ~ Bernoulli(p_det(α*,γ*)).",
                "Ambas entran en h_t y la actualización de μ.",
            ]),
            ("7 · Ganancia ΔH", [
                f"ΔH promedio {_dh_mean:.3f}, máximo {_dh_max:.3f}.",
                "Mide ganancia esperada de información antes de observar la señal.",
                "ψ_H traduce exploración sin temblor continuo en (α,γ).",
            ]),
            ("8–11 · ι, U_rel e IR/IC", [
                f"ι medio {float(_df_cyc['iota'].mean()):.3f}; U_rel({_tipo_dyn}) medio {_ur_mean:.2f}.",
                f"Cumplimiento IR/IC: {_iric_txt}.",
                "Verificar IR^K, IC^K, IR^F, IC^F e IC^S ciclo a ciclo bajo la política óptima.",
            ]),
            ("12 · Frecuencia −sgn(κ_h)", [
                "Frecuencia de −sgn(κ_h(θ,t)) por tipo en los ciclos dinámicos.",
                "−1: γ↑ empeora posición del Estado; +1: la mejora; 0: efecto nulo.",
                "Leer junto con γ* para interpretar el óptimo de presión operacional.",
            ]),
        ]
        _dyn_meta = {
            "n_cyc": _n_cyc,
            "semilla": st.session_state.get("dynamic_seed_effective52", st.session_state.get("semilla", "—")),
            "tau_stop": _tau_stop,
            "m_stop": str(_stop.get("m", "—")) if isinstance(_stop, dict) else "—",
            "motivo_stop": str(_stop.get("motivo", "—")) if isinstance(_stop, dict) else "—",
            "alpha_mean": _a_mean,
            "gamma_mean": _g_mean,
            "dh_mean": _dh_mean,
            "dh_max": _dh_max,
            "match_S": _mS,
            "match_F": _mF,
            "match_K": _mK,
            "pct_v1": _pv1,
            "pct_d1": _pd1,
            "R_escala": _R_dyn,
        }

        _rep_col1, _rep_col2 = st.columns([1, 1])
        with _rep_col1:
            if st.button(
                "Generar informe e infografía",
                type="primary",
                key="dyn_btn_gen_report",
            ):
                with st.spinner("Construyendo PDF e infografía…"):
                    try:
                        _pdf_b, _fig_inf, _png_inf = build_full_dyn_report(
                            _df_cyc,
                            _df_mu_dyn,
                            tipo_real=_tipo_dyn,
                            tipo_colors=_TIPO_COLORS,
                            tipos=tuple(TIPOS_SECUESTRADOR),
                            tau_stop=_tau_stop,
                            th_dom0=_th_dom0,
                            th_domf=_th_domf,
                            mu_t0=_mu_t0,
                            mu_tf=_mu_tf,
                            meta=_dyn_meta,
                            iric_labels=_iric_labels,
                            iric_pct=_iric_pct,
                            narrative_sections=_dyn_narrative,
                        )
                        st.session_state["dyn_report_pdf"] = _pdf_b
                        st.session_state["dyn_report_fig"] = _fig_inf
                        st.session_state["dyn_report_png"] = _png_inf
                        st.session_state["dyn_report_ready"] = True
                    except Exception as _rep_exc:
                        st.session_state["dyn_report_ready"] = False
                        st.error(f"No se pudo generar el informe: {_rep_exc}")
        with _rep_col2:
            if st.session_state.get("dyn_report_ready"):
                st.download_button(
                    "Descargar informe PDF",
                    data=st.session_state["dyn_report_pdf"],
                    file_name=f"informe_mecanismo_{_tipo_dyn}_{_n_cyc}ciclos.pdf",
                    mime="application/pdf",
                    key="dyn_dl_pdf",
                )
                if st.session_state.get("dyn_report_png"):
                    st.download_button(
                        "Descargar infografía PNG",
                        data=st.session_state["dyn_report_png"],
                        file_name=f"infografia_mecanismo_{_tipo_dyn}.png",
                        mime="image/png",
                        key="dyn_dl_png",
                    )

        if st.session_state.get("dyn_report_ready") and st.session_state.get("dyn_report_fig"):
            st.plotly_chart(
                st.session_state["dyn_report_fig"],
                use_container_width=True,
                key="dyn_infographic_chart",
            )
            if not st.session_state.get("dyn_report_png"):
                st.info(
                    "Para incrustar gráficas en el PDF e exportar PNG, instale **kaleido** "
                    "(`pip install kaleido`). La infografía interactiva arriba sigue disponible."
                )
if False:
    st.markdown("## Simulación diaria y proceso MDG")
    st.markdown("### Bitácora de semillas dinámicas")
    _seed_log52 = st.session_state.get("dynamic_seed_run_log", [])
    if isinstance(_seed_log52, list) and _seed_log52:
        _df_seed_log52 = pd.DataFrame(_seed_log52)
        _seed_cols52 = [
            "Semilla visible",
            "Reinicio",
            "Semilla efectiva",
            "Periodo de parada",
            "τ parada",
            "m parada",
            "Motivo",
        ]
        _df_seed_log52 = _df_seed_log52[
            [c for c in _seed_cols52 if c in _df_seed_log52.columns]
        ].copy()
        st.dataframe(
            _df_seed_log52,
            width="stretch",
            height=_glide_full_height_px(_st_table_row_count(_df_seed_log52)),
            hide_index=True,
        )
    else:
        st.info(
            "No hay corridas dinámicas registradas. Presione Avanzar ciclos para guardar "
            "la semilla, el desenlace de parada y el τ correspondiente."
        )
    st.caption(
        "El botón ↻ junto a Semillas (RNG) borra solo la corrida dinámica τ≥1 de la "
        "semilla visible y aumenta su reinicio interno. El ciclo base no se modifica."
    )

    st.caption(
        "Temperatura híbrida T_t, entropía H(μ), filtro M(t) y riesgos h_j,t con **λ₀** y **β** del estado de sesión (Mechanism.tex)."
    )
    mu_mdg = {
        t: float(st.session_state.final_priors[i]) / 100.0
        for i, t in enumerate(TIPOS_SECUESTRADOR)
    }
    H_mu = shannon_entropy(mu_mdg)
    g1, g2 = st.columns(2)
    T0_m = g1.slider("T₀ (escala temperatura)", 0.2, 3.0, 1.0, 0.05, key="mdg_T0")
    eta_cal_m = g2.slider("η_cal (decaimiento temporal)", 0.0, 0.3, 0.05, 0.005, key="mdg_eta_cal")
    g3, g4 = st.columns(2)
    c_bar_m = g3.slider("c̄ (piso de decaimiento)", 0.01, 0.5, 0.1, 0.01, key="mdg_c_bar")
    eps0_m = g4.slider("ε₀ (ruido base MDG)", 0.02, 0.6, 0.2, 0.02, key="mdg_eps0")
    H0_log_n = float(np.log(len(TIPOS_SECUESTRADOR)))  # H máx uniforme sobre 4 tipos
    _t5_day_prev = int(st.session_state.get("mdg_tday", 1))
    T_t = hybrid_temperature(H_mu, T0_m, H0=H0_log_n, eta_cal=eta_cal_m, t=_t5_day_prev, c_bar=c_bar_m)
    eps_t = mdg_execution_noise(eps0_m, T_t)
    align = float(max(0.0, min(1.0, 1.0 - 2.5 * eps_t)))
    mA, mB, mC, mD = st.columns(4)
    mA.metric("H(μ)", f"{H_mu:.2f}")
    mB.metric("T_t", f"{T_t:.2f}")
    mC.metric("ε_t = ε₀·T_t", f"{eps_t:.2f}")
    mD.metric("Coherencia Ã↔A*", f"{align:.2f}")
    st.latex(
        r"T_t = T_0\max\!\left\{\frac{H(\mu_t)}{H(\mu_0)}e^{-\eta_{\mathrm{cal}}\,t},\;\bar c\right\}"
    )
    st.latex(r"H(\mu_t) = -\sum_{\theta\in\Theta_K} \mu_t(\theta)\,\log \mu_t(\theta)")
    st.markdown(
        r"A mayor entropía $H(\mu_t)$ (mayor incertidumbre sobre $\theta_K$), el término "
        r"$(H/H_0)e^{-\eta_{\mathrm{cal}} t}$ es mayor y, mientras no ligue $\bar c$, $T_t$ y $\varepsilon_t$ aumentan — "
        r"la mezcla MDG aleja la **acción ejecutada** $\tilde A_t$ de la **intención** $A^*_t$ "
        r"(más ruido operativo cuando el Estado no ha identificado al secuestrador). "
        r"Conforme las creencias se concentran ($H(\mu_t)\downarrow$), $T_t$ decrece hacia $T_0\bar c$; "
        r"con $\bar c>0$ el soporte MDG permanece completo y la probabilidad de la acción óptima no converge a uno "
        r"(Bernal_H.tex, ec. temperatura)."
    )

    st.markdown("### Riesgos competitivos y maduración M(t)")
    st.latex(r"M(t)=\min\!\left\{1,\left(\frac{t}{T_{\mathrm{mad}}}\right)^{\!2}\right\}")
    t_day = st.number_input("Día t", min_value=1, max_value=500, value=1, key="mdg_tday")
    _t5_Tmad = float(st.session_state.get("cal_T_mad", 30.0))
    M_t = float(min(1.0, (int(t_day) / _t5_Tmad) ** 2)) if _t5_Tmad > 0 else 0.0
    st.caption(rf"T_mad = {_t5_Tmad:.1f} (Tabla 2 · Prior)")
    st.metric("M(t)", f"{M_t:.2f}")
    # Instrumentos de política desde Tab 4 y parámetros de detección desde Tab 3
    _theta_modal5 = max(mu_mdg, key=lambda k: float(mu_mdg.get(k, 0.0)))
    _p_det5 = _pdet_logit_prob(str(_theta_modal5), float(alpha_star), float(presion_S))
    _zp5 = _focus_cmh_endogenous_tentatives(_theta_modal5)
    _za5 = float(_zp5.get("zeta_alpha", 0.1))
    _zg5 = float(_zp5.get("zeta_gamma", 0.1))
    _zd5 = float(_zp5.get("zeta_d", 0.1))
    _zR5 = float(_zp5.get("zeta_R", 0.1))
    h_blend = blend_hazards(
        modelo, mu_mdg, 1, presion_S, maturity_mult=M_t,
        z_region=str(st.session_state.z_region), v_victim=str(st.session_state.v_victim),
        alpha=alpha_star, gamma=presion_S, p_det=_p_det5,
        zeta_alpha=_za5, zeta_gamma=_zg5, zeta_d=_zd5, zeta_R=_zR5,
    )
    h_df = pd.DataFrame(
        [{"Desenlace": k, "h_j": round(float(v), 2)} for k, v in h_blend.items()]
    )
    rb_katex_grid_header(RB_LATEX_HEADER_H_BLEND, height=48)
    st.dataframe(
        h_df,
        width="stretch",
        height=_glide_full_height_px(_st_table_row_count(h_df)),
        hide_index=True,
    )
    with st.expander("Por tipo θ_K (mismo día y M(t))"):
        rows_t = []
        for th in TIPOS_SECUESTRADOR:
            _zp5th = _focus_cmh_endogenous_tentatives(th)
            hh = modelo.calcular_hazards(
                int(t_day), th, presion_S, maturity_mult=M_t,
                z_region=str(st.session_state.z_region), v_victim=str(st.session_state.v_victim),
                alpha=alpha_star, gamma=presion_S, p_det=_p_det5,
                zeta_alpha=float(_zp5th.get("zeta_alpha", 0.1)),
                zeta_gamma=float(_zp5th.get("zeta_gamma", 0.1)),
                zeta_d=float(_zp5th.get("zeta_d", 0.1)),
                zeta_R=float(_zp5th.get("zeta_R", 0.1)),
            )
            rows_t.append({**{"θ_K": th}, **{k: round(float(hh[k]), 2) for k in hh}})
        rb_katex_grid_header(RB_LATEX_HEADER_H_TIPO, height=48)
        _df_ht = pd.DataFrame(rows_t)
        st.dataframe(
            _df_ht,
            width="stretch",
            height=_glide_full_height_px(_st_table_row_count(_df_ht)),
            hide_index=True,
        )

    st.markdown("### Sorteo MDG (transformada inversa, panel superior)")
    if st.button("GENERAR SORTEO (una realización)", use_container_width=True, key="mdg_draw"):
        des = modelo.simular_proceso_mdg("Rescate", "Coludir", "Continuar", precision_iota)
        st.success(f"Desenlace simulado: **{des}**")

    st.markdown("### Voz y silencio — verosimilitud ℒ_C (Mahalanobis diagonal)")
    st.caption(
        "Señal vectorial ζ; verosimilitud log-concava bajo normal diagonal (precisión σ_k dada)."
    )
    use_sig = st.checkbox("Activar cálculo de log-verosimilitud", value=False, key="mdg_use_sig")
    if use_sig:
        c_sig = st.columns(3)
        zeta = np.array(
            [
                c_sig[0].number_input("ζ₁", -10.0, 10.0, 0.0, 0.1, key="mdg_z1"),
                c_sig[1].number_input("ζ₂", -10.0, 10.0, 0.0, 0.1, key="mdg_z2"),
                c_sig[2].number_input("ζ₃", -10.0, 10.0, 0.0, 0.1, key="mdg_z3"),
            ]
        )
        mean = np.array([0.0, 0.0, 0.0])
        sigma = np.array([1.0, 1.0, 1.0])
        ll = mahalanobis_diagonal_loglik(zeta, mean, sigma)
        st.metric("log ℒ_C (± constante)", f"{ll:.2f}")

if False:
    st.markdown("## Aprendizaje y convergencia (Teorema 7.9)")
    st.caption(
        "Panel de creencias hasta el tiempo de parada τ; colapso cualitativo hacia θ* cuando el desenlace es terminal (m_t ∈ M_term)."
    )
    st.latex(r"\mathcal{M}_{\mathrm{term}}:\ \text{desenlaces absorbentes del mecanismo.}")
    if st.button("Simular trayectoria (guarda última corrida)", use_container_width=True, key="learn_btn"):
        mu_0 = {t: p / 100.0 for t, p in zip(TIPOS_SECUESTRADOR, st.session_state.final_priors)}
        _theta_modal6 = max(mu_0, key=lambda k: float(mu_0.get(k, 0.0)))
        _p_det6 = _pdet_logit_prob(str(_theta_modal6), float(alpha_star), float(presion_S))
        _zp6 = _focus_cmh_endogenous_tentatives(_theta_modal6)
        ev, hist = modelo.simular_trayectoria_cautiverio(
            tipo_real, mu_0, int(limite_dias), presion_S,
            alpha=alpha_star, gamma=presion_S, p_det=_p_det6,
            zeta_alpha=float(_zp6.get("zeta_alpha", 0.1)),
            zeta_gamma=float(_zp6.get("zeta_gamma", 0.1)),
            zeta_d=float(_zp6.get("zeta_d", 0.1)),
            zeta_R=float(_zp6.get("zeta_R", 0.1)),
        )
        st.session_state["learn_events"] = ev
        st.session_state["learn_hist"] = hist

    if st.session_state.get("learn_hist") is not None and len(st.session_state["learn_hist"]) > 0:
        hist = st.session_state["learn_hist"]
        ev = st.session_state.get("learn_events", [])
        tau = len(hist) - 1
        df_mu = pd.DataFrame(hist).round(2)
        df_mu.insert(0, "t", range(len(df_mu)))
        fig_mu = px.line(
            df_mu,
            x="t",
            y=TIPOS_SECUESTRADOR,
            title=f"Evolución de μ (creencias) hasta τ = {tau}",
        )
        st.plotly_chart(fig_mu, use_container_width=True)
        df_Ht = trajectory_entropy_series(hist)
        fig_H = px.line(df_Ht, x="t", y="H_mu", title="H(μ_t) — contracción ilustrativa")
        st.plotly_chart(fig_H, use_container_width=True)

        st.markdown("### Justificación empírica: señales públicas ζ_t")
        st.latex(r"\mu_{t+1}(\theta)\propto \mu_t(\theta)\,\ell(y_t\mid \theta)")
        st.info(
            "Cada desenlace observable actualiza las creencias: los tipos inconsistentes con la verosimilitud pierden masa; "
            "H(μ_t) suele caer tras señales informativas (simulación ilustrativa; **Teorema 7.9** en **Mechanism.tex**)."
        )
        if ev:
            _df_ev = pd.DataFrame({"t": list(range(1, len(ev) + 1)), "Desenlace": ev})
            st.dataframe(
                _df_ev,
                width="stretch",
                height=_glide_full_height_px(_st_table_row_count(_df_ev)),
                hide_index=True,
            )
        term = [x for x in ev if x != "Continuar"]
        if term:
            st.markdown(f"**Estado terminal observado:** `{term[-1]}` · **τ** = {tau}")
        ok_c, p_star, msg_abs = absorption_posterior_check(hist, tipo_real, 0.85)
        st.markdown("### Resultado posterior vs. θ* (tipo verdadero del panel)")
        st.markdown(msg_abs)
        if ok_c:
            st.success("En esta corrida, la masa en θ* supera el umbral 0.85 (ilustrativo).")
        else:
            st.warning("En esta corrida el umbral 0.85 no se alcanza; pruebe otra semilla o más días.")
