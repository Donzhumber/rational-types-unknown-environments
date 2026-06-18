"""
Ilustraciones numéricas del comportamiento racional (Mechanism.tex, §5).
Momentos descriptivos: Data_CMH.csv. Pesos estructurales: calibración tipo main.tex / tablero.
"""
from __future__ import annotations

import copy
import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from model_logic import DESENLACES, TIPOS_SECUESTRADOR, ModeloSecuestro

# Claves de β_{K,j} en ModeloSecuestro (coinciden con lambdas_0; "Liberación por Pago" → "Liberación").
BETA_OUTCOME_KEYS = ("Liberación", "Rescate", "Pago", "Muerte")


def _row_float_or_nan(row: Optional[pd.Series], key: str) -> float:
    if row is None:
        return float("nan")
    try:
        return float(row.get(key, float("nan")))
    except (TypeError, ValueError, KeyError, AttributeError):
        return float("nan")


def _tab14_pay_for_theta(row: Optional[pd.Series], theta: str) -> float:
    return _row_float_or_nan(row, f"Epi_pay_Qcont_{str(theta)}")


def _tab14_pcap_for_theta(row: Optional[pd.Series], theta: str) -> float:
    return _row_float_or_nan(row, f"Epi_pcap_Qcap_{str(theta)}")


def family_institutional_cost_e(
    gamma: float,
    phi_F: float,
    kappa_F: float,
    nu_F: float,
) -> float:
    """
    Costo institucional de cooperación ``e_t(γ; θ_F)`` — **Mechanism.tex**,
    ec. ``family-institutional-cost`` (Mechanism.tex), análoga a ``cost-function-kidnapper``:
    ``e_t = φ_F exp(κ_F γ) + ν_F``, con ``γ`` en ``[0,1]``.
    """
    g = float(max(0.0, min(1.0, gamma)))
    ph = float(max(1e-12, phi_F))
    ka = float(kappa_F)
    nu = float(nu_F)
    return float(ph * np.exp(ka * g) + nu)


def map_grupo_responsable_to_theta(g: Any) -> Optional[str]:
    """Mapea texto de Data_CMH (`Grupo_Responsable`) a θ ∈ {DC, PAR, ELN, FARC}."""
    if pd.isna(g):
        return None
    s = str(g).strip().upper()
    if "FARC" in s:
        return "FARC"
    if "ELN" in s:
        return "ELN"
    if "PARAMILITAR" in s:
        return "PAR"
    if "DELINCUENCIA" in s or "OTRO" in s:
        return "DC"
    return None


def map_cmh_y_resultado_to_outcome_weights(y: Any) -> Optional[Dict[str, float]]:
    """Distribución sobre {Liberación, Rescate, Pago, Muerte} para una fila CMH."""
    if pd.isna(y):
        return None
    s = str(y).strip().lower()
    if "muerte" in s or "muert" in s:
        return {"Liberación": 0.0, "Rescate": 0.0, "Pago": 0.0, "Muerte": 1.0}
    if "fuga" in s or "liberación" in s or "liberacion" in s:
        return {"Liberación": 1.0, "Rescate": 0.0, "Pago": 0.0, "Muerte": 0.0}
    if "pago" in s:
        return {"Liberación": 0.0, "Rescate": 0.0, "Pago": 1.0, "Muerte": 0.0}
    if "rescate" in s:
        return {"Liberación": 0.0, "Rescate": 1.0, "Pago": 0.0, "Muerte": 0.0}
    return None

try:
    import streamlit as st

    _cache_data = st.cache_data
except Exception:  # import fuera de Streamlit

    def _cache_data(func):
        return func


@_cache_data
def load_cmh_outcome_moments() -> Optional[Dict[str, Any]]:
    path = os.path.join(os.path.dirname(__file__), "Data_CMH.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if "Y_Resultado" not in df.columns:
        return {"n": len(df), "outcomes": {}}
    vc = df["Y_Resultado"].value_counts(normalize=True)
    return {"n": len(df), "outcomes": vc.to_dict()}


def quadratic_cost(
    gamma: float,
    c0: float,
    c1: float,
    c2: float,
    c3: float = 0.0,
    c4: float = 0.0,
    c5: float = 0.0,
    alpha: float = 0.0,
) -> float:
    """Costo bivariado de Mechanism.tex: c0+c1γ+c2γ²/2+c3α+c4α²/2+c5γα."""
    g = float(gamma)
    a = float(alpha)
    return float(
        c0
        + c1 * g
        + 0.5 * c2 * g**2
        + c3 * a
        + 0.5 * c4 * a**2
        + c5 * g * a
    )


def blend_hazards(
    modelo: ModeloSecuestro,
    mu: Dict[str, float],
    t: int,
    presion_S: float,
    maturity_mult: float = 1.0,
    z_region: str = "Metropolitana",
    v_victim: str = "Privado",
    alpha: float = 0.0,
    gamma: float = None,
    p_det: float = 0.0,
    zeta_alpha: float = None,
    zeta_gamma: float = None,
    zeta_d: float = None,
    zeta_R: float = 0.0,
    estado_rescata: bool = False,
) -> Dict[str, float]:
    """Esperanza de intensidades bajo creencias mu (coherente con pestaña Priors).

    Pasa los instrumentos alpha, gamma y p_det al modelo para que las intensidades
    cause-specific se calculen exactamente segun Mechanism.tex.  Cuando gamma es
    None se usa presion_S como valor efectivo de gamma (compatibilidad).
    """
    gamma_eff = float(gamma) if gamma is not None else float(presion_S)
    acc = {k: 0.0 for k in DESENLACES}
    for theta in TIPOS_SECUESTRADOR:
        h = modelo.calcular_hazards(
            t,
            theta,
            presion_S,
            maturity_mult=maturity_mult,
            z_region=z_region,
            v_victim=v_victim,
            alpha=alpha,
            gamma=gamma_eff,
            p_det=p_det,
            zeta_alpha=zeta_alpha,
            zeta_gamma=zeta_gamma,
            zeta_d=zeta_d,
            zeta_R=zeta_R,
            estado_rescata=estado_rescata,
        )
        w = float(mu.get(theta, 0.0))
        for k in DESENLACES:
            acc[k] += w * h[k]
    return acc


def maturation_filter(t: int, rho: float = 0.04) -> float:
    """Filtro de maduracion M(t) segun Mechanism.tex, eq. (956).

    La ecuacion del mecanismo define:
        M(t) = min{1, (t / T_mad)^2}

    Esta funcion conserva la firma original (t, rho) por compatibilidad, pero
    cuando se llama sin T_mad explicito usa rho como T_mad aproximado via la
    relacion T_mad = 1/sqrt(rho) (que invierte la escala cuadratica).  Para
    calculos directos en app.py, usar la formula cerrada min(1, (t/T_mad)^2).

    NOTA: el parametro rho se conserva en la firma para no romper llamadas
    existentes, pero el computo sigue la forma cuadratica de Mechanism.tex.
    """
    tt = max(0, int(t))
    rr = max(1e-6, float(rho))
    # T_mad equivalente: la raiz cuadrada de 1/rho da la misma curvatura que
    # la forma cuadratica del mecanismo cuando M(T_mad)=1.
    T_mad_equiv = 1.0 / (rr ** 0.5)
    return float(min(1.0, (tt / T_mad_equiv) ** 2))


def shannon_entropy(mu: Dict[str, float]) -> float:
    """Entropía H(μ) en nats."""
    h = 0.0
    for p in mu.values():
        pp = float(max(1e-12, min(1.0, p)))
        h -= pp * np.log(pp)
    return float(h)


def hybrid_temperature(
    H_mu: float,
    T0: float = 1.0,
    H0: float = 1.0,
    eta_cal: float = 0.05,
    t: int = 0,
    c_bar: float = 0.1,
) -> float:
    """
    Temperatura híbrida (Mechanism.tex eq.693):
        T_t = T_0 · (H(μ_t)/H(μ_0)) · max{e^{-η_cal·t}, c̄}
    Mayor H(μ_t) → mayor T_t → más ruido operativo MDG (acción ejecutada menos alineada con A*_t).
    H0=0 se trata como 1 para evitar división por cero (uniforme).
    """
    ratio_H = H_mu / max(1e-12, H0)
    decay = max(c_bar, np.exp(-eta_cal * max(0, t)))
    return float(T0 * ratio_H * decay)


def mdg_execution_noise(eps0: float, T_t: float) -> float:
    """Ruido ε_t = ε_0 · T_t acotado a [0, 0.5]."""
    return float(min(0.5, max(0.0, eps0 * T_t)))


def mahalanobis_diagonal_loglik(
    zeta: np.ndarray,
    mean: np.ndarray,
    sigma: np.ndarray,
) -> float:
    """Verosimilitud log ℒ_C bajo normal diagonal: -½ Σ (ζ_k-μ_k)²/σ_k² (constante omitida)."""
    z = np.asarray(zeta, dtype=float)
    m = np.asarray(mean, dtype=float)
    s = np.maximum(1e-8, np.asarray(sigma, dtype=float))
    return float(-0.5 * np.sum((z - m) ** 2 / s**2))


def voice_effective_sigma(
    sigma_L: Sequence[float],
    sigma_S: Sequence[float],
) -> np.ndarray:
    """$\tilde{\sigma}_i=\sqrt{\sigma_{L,i}^2+\sigma_{S,i}^2}$ (eq. Lvoz-diag / Medición de voz)."""
    sl = np.asarray(sigma_L, dtype=float)
    ss = np.asarray(sigma_S, dtype=float)
    return np.sqrt(sl**2 + ss**2)


def Lvoz_diagonal_likelihood(
    x_obs: np.ndarray,
    theta: str,
    voz_params_by_theta: Dict[str, Dict[str, Any]],
) -> float:
    """
    $\mathcal{L}_{\mathrm{voz},t}(\theta_K)\propto\exp(-D_M^2/2)$ con Mahalanobis diagonal
    (Mechanism.tex, eq. Lvoz-diag; pestaña 2, §5 Medición de voz).
    """
    vp = voz_params_by_theta.get(str(theta))
    if vp is None:
        return 1e-300
    xb = np.asarray(vp["x"], dtype=float)
    sig = voice_effective_sigma(vp["sigma_L"], vp["sigma_S"])
    loglk = mahalanobis_diagonal_loglik(
        np.asarray(x_obs, dtype=float), xb, sig
    )
    return float(max(1e-300, np.exp(loglk)))


def sample_voice_observation(
    theta_emit: str,
    voz_params_by_theta: Dict[str, Dict[str, Any]],
    rng: np.random.Generator,
) -> np.ndarray:
    """$x_t^{obs}=\bar{x}(\theta)+\varepsilon_L+\varepsilon_S$ (§5 Medición de voz)."""
    vp = voz_params_by_theta[str(theta_emit)]
    xb = np.asarray(vp["x"], dtype=float)
    sL = np.asarray(vp["sigma_L"], dtype=float)
    sS = np.asarray(vp["sigma_S"], dtype=float)
    return xb + rng.normal(0.0, sL, size=xb.shape) + rng.normal(0.0, sS, size=xb.shape)


def draw_voice_indicator(
    pi_call: float,
    rng: np.random.Generator,
) -> int:
    """$V_t\in\{0,1\}$ con $\mathbb{P}(V_t=1)=\pi_{\mathrm{call}}$ del emisor del periodo."""
    p = float(np.clip(float(pi_call), 1e-9, 1.0 - 1e-9))
    return int(rng.random() < p)


def _pi_call_for_theta(
    theta: str,
    pi_call: Any,
) -> float:
    if isinstance(pi_call, dict):
        return float(np.clip(float(pi_call.get(str(theta), 0.2)), 1e-9, 1.0 - 1e-9))
    return float(np.clip(float(pi_call), 1e-9, 1.0 - 1e-9))


def sample_incident_pi_call_realized(
    pi_call_prior: Dict[str, float],
    *,
    kappa: float = 30.0,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, float]:
    """
    Frecuencia de llamadas realizada por tipo en el incidente:
    $\\tilde{\\pi}_{\\mathrm{call}}(\\theta)\\sim\\mathrm{Beta}(\\kappa\\pi,\\kappa(1-\\pi))$
    anclada al prior de pestaña 2.
    """
    rng_gen = rng if rng is not None else np.random.default_rng()
    kappa_f = float(max(2.0, kappa))
    out: Dict[str, float] = {}
    for th in TIPOS_SECUESTRADOR:
        p = _pi_call_for_theta(th, pi_call_prior)
        a = float(max(1e-6, kappa_f * p))
        b = float(max(1e-6, kappa_f * (1.0 - p)))
        out[str(th)] = float(np.clip(rng_gen.beta(a, b), 1e-6, 1.0 - 1e-6))
    return out


def build_incident_voice_path(
    theta_incident: str,
    pi_call_realized: Dict[str, float],
    voz_params_by_theta: Dict[str, Dict[str, Any]],
    *,
    t_max: int,
    rng: Optional[np.random.Generator] = None,
) -> List[Dict[str, Any]]:
    """
    Trayectoria diaria $(V_t, x_t^{obs})$ del captor verdadero $\\theta^{\\ast}$:
    $V_t\\sim\\mathrm{Bern}(\\tilde{\\pi}_{\\mathrm{call}}(\\theta^{\\ast}))$;
    si $V_t=1$, $x_t^{obs}$ vía §5 Medición de voz.
    """
    th_star = str(theta_incident)
    rng_gen = rng if rng is not None else np.random.default_rng()
    p_emit = _pi_call_for_theta(th_star, pi_call_realized)
    t_max = int(max(0, t_max))
    path: List[Dict[str, Any]] = []
    for t in range(t_max):
        v_t = draw_voice_indicator(p_emit, rng_gen)
        x_obs = None
        if int(v_t) == 1:
            x_obs = sample_voice_observation(th_star, voz_params_by_theta, rng_gen)
        path.append(
            {
                "t": int(t),
                "V_t": int(v_t),
                "x_obs": np.asarray(x_obs, dtype=float).tolist() if x_obs is not None else None,
                "emisor_voz": th_star,
                "pi_tilde_emit": round(float(p_emit), 4),
            }
        )
    return path


def generate_incident_voice_scenario(
    theta_incident: str,
    pi_call_prior: Dict[str, float],
    voz_params_by_theta: Dict[str, Dict[str, Any]],
    *,
    t_max: int = 10,
    kappa: float = 30.0,
    seed: Optional[int] = None,
) -> Tuple[Dict[str, float], Dict[str, float], List[Dict[str, Any]], Dict[str, Any]]:
    """Paquete completo: prior $\\pi$, realizada $\\tilde{\\pi}$ por tipo y trayectoria $(V_t,x^{obs})$."""
    rng = np.random.default_rng(int(seed) if seed is not None else None)
    pi_prior = {str(k): float(_pi_call_for_theta(k, pi_call_prior)) for k in TIPOS_SECUESTRADOR}
    pi_tilde = sample_incident_pi_call_realized(pi_call_prior, kappa=kappa, rng=rng)
    path = build_incident_voice_path(
        theta_incident,
        pi_tilde,
        voz_params_by_theta,
        t_max=int(t_max),
        rng=rng,
    )
    n_call = sum(int(s["V_t"]) for s in path)
    meta = {
        "theta_incident": str(theta_incident),
        "kappa": float(kappa),
        "seed": int(seed) if seed is not None else None,
        "t_max": int(t_max),
        "pi_emit": float(_pi_call_for_theta(theta_incident, pi_tilde)),
        "n_calls": int(n_call),
        "days_voice": [int(s["t"]) for s in path if int(s["V_t"]) == 1],
    }
    return pi_prior, pi_tilde, path, meta


def voice_path_step_at_t(
    voice_path: Optional[Sequence[Dict[str, Any]]],
    t: int,
) -> Tuple[Optional[int], Optional[np.ndarray], str]:
    """Lee $(V_t, x^{obs}, \\mathrm{emisor})$ del paso ``t`` en la trayectoria del incidente."""
    if voice_path is None or t < 0 or t >= len(voice_path):
        return None, None, "—"
    step = voice_path[t]
    v_raw = step.get("V_t")
    v_t = int(v_raw) if v_raw is not None and str(v_raw) not in ("—", "") else None
    x_raw = step.get("x_obs")
    if x_raw is None:
        x_obs = None
    else:
        x_obs = np.asarray(x_raw, dtype=float)
    emisor = str(step.get("emisor_voz", "—"))
    return v_t, x_obs, emisor


def compute_voice_likelihood_trajectory(
    voice_path: Sequence[Dict[str, Any]],
    theta_focus: str,
    *,
    omega_voz: float,
    pi_call_by_theta: Dict[str, float],
    voz_params_by_theta: Dict[str, Dict[str, Any]],
) -> pd.DataFrame:
    """
    Verosimilitudes de voz por periodo (eq. LC, Lvoz-diag) alineadas con Tabla 14:
    mismos $(V_t, x^{obs})$ del incidente; columnas $\\mathcal{L}_{C,t}$, $\\mathcal{L}_{\\mathrm{voz},t}$
    en el foco $\\theta^\\ast$; desglose $\\mathcal{L}_{C}$ por tipo en columnas auxiliares.
    """
    th_star = str(theta_focus)
    rows: List[Dict[str, Any]] = []
    for step in voice_path:
        t = int(step.get("t", 0))
        v_raw = step.get("V_t")
        v_t = int(v_raw) if v_raw is not None and str(v_raw) not in ("—", "") else None
        x_raw = step.get("x_obs")
        x_obs = np.asarray(x_raw, dtype=float) if x_raw is not None else None
        lc_by: Dict[str, float] = {}
        lv_by: Dict[str, float] = {}
        for th in TIPOS_SECUESTRADOR:
            lc, lv = communication_likelihood_LC(
                th,
                V_t=v_t,
                omega_voz=omega_voz,
                pi_call=pi_call_by_theta,
                x_obs=x_obs,
                voz_params_by_theta=voz_params_by_theta,
            )
            lc_by[str(th)] = float(lc)
            lv_by[str(th)] = float(lv)
        lc_f, lv_f = lc_by.get(th_star, 1.0), lv_by.get(th_star, 1.0)
        rows.append(
            {
                "t": t,
                "V_t": v_t if v_t is not None else "—",
                "Llamada": "Sí" if v_t == 1 else ("No" if v_t == 0 else "—"),
                "L_voz": round(lv_f, 6) if v_t == 1 else "—",
                "L_C": round(lc_f, 6),
                "L_C_DC": round(lc_by.get("DC", 1.0), 6),
                "L_C_PAR": round(lc_by.get("PAR", 1.0), 6),
                "L_C_ELN": round(lc_by.get("ELN", 1.0), 6),
                "L_C_FARC": round(lc_by.get("FARC", 1.0), 6),
            }
        )
    return pd.DataFrame(rows)


@_cache_data
def load_cmh_delta_eta_empirical() -> Optional[Dict[str, Any]]:
    """
    Frecuencias empíricas en Data_CMH: δ̂_θ (log-ratio vs referencia) y tabla η̂_{θ,z}
    (desviación log del tipo θ dentro de la zona z respecto al marginal).
    """
    path = os.path.join(os.path.dirname(__file__), "Data_CMH.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if "Grupo_Responsable" not in df.columns or "Zona_Geografica" not in df.columns:
        return None

    df = df.copy()
    df["theta_hat"] = df["Grupo_Responsable"].map(map_grupo_responsable_to_theta)
    df = df.dropna(subset=["theta_hat"])
    n = len(df)
    if n == 0:
        return None

    counts = df["theta_hat"].value_counts()
    p = {t: float(counts.get(t, 0)) / n for t in TIPOS_SECUESTRADOR}
    p_ref = max(1e-6, p.get("FARC", 0.25))
    delta_hat = {t: float(np.log(max(1e-6, p[t]) / p_ref)) for t in TIPOS_SECUESTRADOR}

    zones = df["Zona_Geografica"].fillna("Sin región").astype(str)
    eta_rows: List[Dict[str, Any]] = []
    for z in sorted(zones.unique()):
        sub = df[zones == z]
        nz = len(sub)
        if nz == 0:
            continue
        cz = sub["theta_hat"].value_counts()
        pz = {t: float(cz.get(t, 0)) / nz for t in TIPOS_SECUESTRADOR}
        row: Dict[str, Any] = {"Zona": z, "n": nz}
        for t in TIPOS_SECUESTRADOR:
            row[f"P({t}|z)"] = round(pz[t], 2)
            # Nombre ASCII para que Streamlit/DataFrame muestre bien el encabezado
            row[f"eta_hat_{t}"] = round(float(np.log(max(1e-6, pz[t]) / max(1e-6, p[t]))), 2)
        eta_rows.append(row)

    return {
        "n": n,
        "p_theta": {k: round(v, 2) for k, v in p.items()},
        "delta_hat_log_ratio": {k: round(v, 2) for k, v in delta_hat.items()},
        "eta_theta_zona": pd.DataFrame(eta_rows),
    }


@_cache_data
def compute_cmh_beta_calibration_tables() -> Optional[Dict[str, Any]]:
    """
    Frecuencias empíricas de desenlaces competitivos por θ (CMH), para alinear β_{K,j}
    con proporciones observadas (riesgos proporcionales, Mechanism.tex).
    """
    path = os.path.join(os.path.dirname(__file__), "Data_CMH.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if "Grupo_Responsable" not in df.columns or "Y_Resultado" not in df.columns:
        return None

    rows: List[Dict[str, Any]] = []
    for _, r in df.iterrows():
        th = map_grupo_responsable_to_theta(r.get("Grupo_Responsable"))
        w = map_cmh_y_resultado_to_outcome_weights(r.get("Y_Resultado"))
        if th is None or w is None:
            continue
        rows.append({"theta": th, **w})

    if not rows:
        return None

    dff = pd.DataFrame(rows)
    n_eff = len(dff)
    marg = {j: float(dff[j].mean()) for j in BETA_OUTCOME_KEYS}
    by_theta = dff.groupby("theta", observed=True)[list(BETA_OUTCOME_KEYS)].mean()
    by_theta = by_theta.reindex(TIPOS_SECUESTRADOR)
    return {
        "n_mapped": n_eff,
        "marginal_outcome": marg,
        "shares_by_theta": by_theta.round(4),
        "note": (
            "Pago CMH → Liberación (pago); Rescate → Rescate; Muerte → Muerte; "
            "Fuga o Liberación → Liberación."
        ),
    }


def suggest_betas_from_cmh_tables(
    base_betas: Dict[str, Dict[str, float]],
    tables: Dict[str, Any],
    *,
    weight: float = 1.0,
    eps: float = 0.02,
    max_abs_adj: float = 1.5,
    beta_clip: Tuple[float, float] = (-3.0, 3.0),
) -> Tuple[Dict[str, Dict[str, float]], pd.DataFrame]:
    """
    Ajuste tipo log-ratio frente al marginal global, centrado en j para cada θ (identificabilidad local).
    """
    marg = tables["marginal_outcome"]
    agg = tables["shares_by_theta"]
    wgt = float(max(0.0, weight))
    out = copy.deepcopy(base_betas)
    diag: List[Dict[str, Any]] = []

    for theta in TIPOS_SECUESTRADOR:
        ser = agg.loc[theta]
        if not np.isfinite(ser.values).all():
            for j in BETA_OUTCOME_KEYS:
                diag.append(
                    {
                        "θ": theta,
                        "j": j,
                        "p_theta": None,
                        "adj": 0.0,
                        "beta_new": round(out[theta][j], 4),
                        "note": "sin filas CMH",
                    }
                )
            continue

        raw: List[float] = []
        for j in BETA_OUTCOME_KEYS:
            p_t = float(ser[j])
            p_m = float(max(eps, marg.get(j, eps)))
            raw.append(float(np.log(max(eps, p_t) / p_m)))

        arr = np.array(raw, dtype=float)
        arr = arr - float(arr.mean())
        for idx, j in enumerate(BETA_OUTCOME_KEYS):
            adj = float(np.clip(wgt * arr[idx], -max_abs_adj, max_abs_adj))
            lo, hi = beta_clip
            out[theta][j] = float(np.clip(base_betas[theta][j] + adj, lo, hi))
            p_t = float(ser[j])
            diag.append(
                {
                    "θ": theta,
                    "j": j,
                    "p_theta": round(p_t, 4),
                    "adj": round(adj, 4),
                    "beta_new": round(out[theta][j], 4),
                    "note": "",
                }
            )

    return out, pd.DataFrame(diag)


def hazards_to_mechanism_quantities(h: Dict[str, float]) -> Dict[str, float]:
    """
    Del bloque discreto del simulador a los objetos de **Mechanism.tex** (aprox. ecuaciones (35)–(36)):
    p_{Cont,t}, q(t) y ξ_j(t) para j=1,2,3,4 (liberación, rescate, pago, muerte).
    """
    p_cont = float(h["Continuar"])
    q = float(max(0.0, min(1.0, 1.0 - p_cont)))
    h1 = float(h["Liberación"])
    h2 = float(h["Rescate"])
    h3 = float(h["Pago"])
    h4 = float(h["Muerte"])
    if q > 1e-12:
        xi1, xi2, xi3, xi4 = h1 / q, h2 / q, h3 / q, h4 / q
    else:
        xi1 = xi2 = xi3 = xi4 = 0.0
    return {
        "p_Cont": p_cont,
        "q": q,
        "xi_1": xi1,
        "xi_2": xi2,
        "xi_3": xi3,
        "xi_4": xi4,
        "h1": h1,
        "h2": h2,
        "h3": h3,
        "h4": h4,
    }


def longitudinal_S_and_Fj(
    modelo: ModeloSecuestro,
    theta: str,
    presion_S: float,
    rho: float,
    t_max: int,
) -> pd.DataFrame:
    """
    Ecuación (39): S(t)=∏ p_{Cont,t'}, F_j(t)=∑ h_j(t')S(t'-1).
    """
    S_prev = 1.0
    f1 = f2 = f3 = f4 = 0.0
    rows: List[Dict[str, Any]] = []
    for tt in range(1, max(1, int(t_max)) + 1):
        M_t = maturation_filter(tt, rho)
        h = modelo.calcular_hazards(tt, theta, presion_S, maturity_mult=M_t)
        p_cont = float(h["Continuar"])
        h1 = float(h["Liberación"])
        h2 = float(h["Rescate"])
        h3 = float(h["Pago"])
        h4 = float(h["Muerte"])
        f1 += h1 * S_prev
        f2 += h2 * S_prev
        f3 += h3 * S_prev
        f4 += h4 * S_prev
        s_new = S_prev * p_cont
        rows.append(
            {
                "t": tt,
                "S(t)": round(s_new, 6),
                "F_1 Lib": round(f1, 6),
                "F_2 Res": round(f2, 6),
                "F_3 Pag": round(f3, 6),
                "F_4 Mue": round(f4, 6),
                "p_Cont,t": round(p_cont, 6),
            }
        )
        S_prev = s_new
    return pd.DataFrame(rows)


def compute_state_VR_VN(
    mu: Dict[str, float],
    modelo: ModeloSecuestro,
    presion_S: float,
    iota: float,
    omega_k: float,
    omega_p: float,
    omega_G: float,
    alpha: float,
    gamma: float,
    R: float,
    c_ops: Tuple[float, ...],
    c_maint: Tuple[float, ...],
    c_inst: Tuple[float, float, float],
    cmh_p_kill: float,
    cmh_p_surv_proxy: float,
    maturity_mult: float = 1.0,
) -> Tuple[float, float, float, float]:
    """Devuelve V_R, V_N, P_surv_rescue, p_kill_neg."""
    h_mu = blend_hazards(modelo, mu, 1, presion_S, maturity_mult=maturity_mult)
    p_kill_neg = float(0.5 * h_mu["Muerte"] + 0.5 * cmh_p_kill)
    P_surv_rescue = float(min(0.99, cmh_p_surv_proxy * (0.5 + 0.5 * iota)))
    cops = quadratic_cost(gamma, *tuple(c_ops), alpha=alpha)
    cmaint = quadratic_cost(gamma, *tuple(c_maint), alpha=alpha)
    g_inst = c_inst[0] * alpha**2 + c_inst[1] * gamma**2 + c_inst[2] * alpha * gamma
    V_R = omega_k * (1.0 - P_surv_rescue) + cops + omega_G * g_inst
    V_N = omega_p * R * (1.0 - alpha) + omega_k * p_kill_neg + cmaint + omega_G * g_inst
    return V_R, V_N, P_surv_rescue, p_kill_neg


def recursive_equilibrium_path(
    modelo: ModeloSecuestro,
    tipo_verdadero: str,
    mu_0: Dict[str, float],
    presion_S: float,
    alpha_0: float,
    gamma_0: float,
    iota_0: float,
    R: float,
    p_cap_base: float,
    estado_duro: bool,
    beta_k: float,
    V_L: float,
    phi_F: float,
    kappa_F: float,
    nu_F: float,
    F_col: float,
    p_det_base: float,
    p_det_alpha: float,
    omega_k: float,
    omega_p: float,
    omega_G: float,
    c_ops: Tuple[float, float, float],
    c_maint: Tuple[float, float, float],
    c_inst: Tuple[float, float, float],
    T_mad: float,
    T0: float,
    eta_cal: float,
    c_bar: float,
    eps0: float,
    max_t: int = 60,
    grid_n: int = 15,
    seed: int = 42,
    z_region: str = "Metropolitana",
    v_victim: str = "Privado",
    zeta_alpha: float = 0.1,
    zeta_gamma: float = 0.1,
    zeta_d: float = 0.1,
    zeta_R: float = 0.1,
) -> Dict[str, Any]:
    """
    Equilibrio bayesiano perfecto recursivo — 3 jugadores (Mechanism.tex §2).

    En cada periodo t:
      1. S minimiza min(V^R_t, V^N_t) sobre (α,γ)∈[0,1]² → (α_t*, γ_t*)
      2. K maximiza bajo θ_K_true (privado): a_K* = argmax{U_rel, U_kill, V_next}
      3. F maximiza bajo creencias μ_t: a_F* = argmax{EU_coop, EU_col}
      4. S elige a_S* por regla discreta (eq. state-discrete-rule)
      5. MDG aplica temperatura híbrida T_t → acciones ejecutadas (ã_K, ã_S, ã_F)
      6. m_t ~ riesgos competitivos bajo θ_K_true y (α_t*, γ_t*, ã_S)
      7. μ_{t+1}(θ) ∝ μ_t(θ)·h(m_t|θ) (actualización bayesiana, ec. eq:historia-publica)
    El loop se detiene cuando m_t ≠ "Continuar" o se alcanza max_t.
    """
    rng = np.random.default_rng(int(seed))
    H0 = max(1e-12, shannon_entropy(mu_0))
    mu_t = {k: float(v) for k, v in mu_0.items()}

    kpar_K = derive_kidnapper_structural_params(modelo, tipo_verdadero, p_cap_base, estado_duro)

    def _mdg_execute(action: str, all_actions: List[str], eps: float) -> str:
        if len(all_actions) <= 1 or eps <= 1e-9:
            return action
        p_star = max(0.0, 1.0 - eps)
        p_other = (1.0 - p_star) / (len(all_actions) - 1)
        probs = [p_star if a == action else p_other for a in all_actions]
        s = sum(probs)
        probs = [p / s for p in probs]
        return str(rng.choice(all_actions, p=probs))

    ACTIONS_K = ["Liberar (a_rel)", "Matar (a_kill)", "Continuar (a_cont)"]
    ACTIONS_S = ["Rescate", "Negociar"]
    ACTIONS_F = ["Cooperar (a_coop)", "Colusión (a_col)"]

    rows: List[Dict[str, Any]] = []

    for t in range(1, max_t + 1):
        M_t = float(min(1.0, (t / max(1e-9, T_mad)) ** 2))
        H_t = shannon_entropy(mu_t)
        T_t = hybrid_temperature(H_t, T0, H0, eta_cal, t, c_bar)
        eps_t = mdg_execution_noise(eps0, T_t)
        p_det_t = min(0.99, p_det_base + p_det_alpha * alpha_0)

        # ── 1. Estado: optimiza (α*, γ*) ────────────────────────────────────
        opt = optimize_state_instruments(
            mu_t, modelo, presion_S, iota_0,
            omega_k, omega_p, omega_G, R,
            c_ops, c_maint, c_inst,
            cmh_p_kill=0.05, cmh_p_surv_proxy=0.90,
            grid_n=grid_n, maturity_mult=M_t,
        )
        alpha_t = float(opt["alpha"])
        gamma_t = float(opt["gamma"])
        V_R_t   = float(opt["V_R"])
        V_N_t   = float(opt["V_N"])
        p_det_t = min(0.99, p_det_base + p_det_alpha * alpha_t)

        # ── 2. Secuestrador bajo θ_K_true (información privada) ─────────────
        h_K = modelo.calcular_hazards(
            t, tipo_verdadero, presion_S,
            maturity_mult=M_t, z_region=z_region, v_victim=v_victim,
            alpha=alpha_t, gamma=gamma_t, p_det=p_det_t,
            zeta_alpha=zeta_alpha, zeta_gamma=zeta_gamma,
            zeta_d=zeta_d, zeta_R=zeta_R,
        )
        p_pay_K = float(h_K.get("Pago", 0.0))
        Cg_K    = kidnapper_cost_c(gamma_t, kpar_K["phi"], kpar_K["kappa_c"], kpar_K["nu"])
        pc_K    = kpar_K["p_cap"]
        u_rel_K  = float(-kpar_K["kappa_rel"])
        u_kill_K = float((1.0 - pc_K) * kpar_K["eta"] - pc_K * kpar_K["F_cap"])
        flow_K   = float(p_pay_K * R * (1.0 - alpha_t) - Cg_K - pc_K * kpar_K["F_cap"])
        v_cont_K = float(kidnapper_V_cont_branch(u_rel_K, u_kill_K, flow_K, beta_k, pc_K))
        utils_K  = {
            "Liberar (a_rel)":   u_rel_K,
            "Matar (a_kill)":    u_kill_K,
            "Continuar (a_cont)": v_cont_K,
        }
        a_K_star = str(max(utils_K, key=lambda k: utils_K[k]))

        # ── 3. Familia bajo creencias μ_t ────────────────────────────────────
        h_mu = blend_hazards(
            modelo, mu_t, t, presion_S, maturity_mult=M_t,
            alpha=alpha_t, gamma=gamma_t, p_det=p_det_t,
            zeta_alpha=zeta_alpha, zeta_gamma=zeta_gamma,
            zeta_d=zeta_d, zeta_R=zeta_R,
        )
        p_death_F   = float(h_mu.get("Muerte", 0.0))
        p_surv_coop = float(max(0.0, min(1.0, 1.0 - p_death_F)))
        e_t_F       = float(family_institutional_cost_e(gamma_t, phi_F, kappa_F, nu_F))
        p_det_F     = float(min(0.99, p_det_base + p_det_alpha * alpha_t))
        u_coop_F    = float(p_surv_coop * V_L - e_t_F)
        u_col_F     = float(p_surv_coop * V_L - R - p_det_F * F_col)
        a_F_star    = "Cooperar (a_coop)" if u_coop_F >= u_col_F else "Colusión (a_col)"

        # ── 4. Estado acción discreta ────────────────────────────────────────
        a_S_star = "Rescate" if V_R_t <= V_N_t else "Negociar"

        # ── 5. MDG: materialización estocástica ──────────────────────────────
        a_K_exec = _mdg_execute(a_K_star, ACTIONS_K, eps_t)
        a_S_exec = _mdg_execute(a_S_star, ACTIONS_S, eps_t)
        a_F_exec = _mdg_execute(a_F_star, ACTIONS_F, eps_t)

        # ── 6. Desenlace físico m_t (riesgos competitivos bajo θ_K_true) ─────
        h_true = modelo.calcular_hazards(
            t, tipo_verdadero, presion_S,
            maturity_mult=M_t, z_region=z_region, v_victim=v_victim,
            alpha=alpha_t, gamma=gamma_t, p_det=p_det_t,
            zeta_alpha=zeta_alpha, zeta_gamma=zeta_gamma,
            zeta_d=zeta_d, zeta_R=zeta_R,
            estado_rescata=(a_S_exec == "Rescate"),
        )
        probs_m = [max(0.0, float(h_true.get(d, 0.0))) for d in DESENLACES]
        tot_pm  = sum(probs_m)
        probs_m = [p / tot_pm for p in probs_m] if tot_pm > 1e-12 else [1.0 / len(DESENLACES)] * len(DESENLACES)
        m_t = str(rng.choice(DESENLACES, p=probs_m))

        # ── 7. Actualización bayesiana ────────────────────────────────────────
        mu_next: Dict[str, float] = {}
        ev_total = 0.0
        for theta in TIPOS_SECUESTRADOR:
            h_th = modelo.calcular_hazards(
                t, theta, presion_S, maturity_mult=M_t,
                alpha=alpha_t, gamma=gamma_t, p_det=p_det_t,
                zeta_alpha=zeta_alpha, zeta_gamma=zeta_gamma,
                zeta_d=zeta_d, zeta_R=zeta_R,
            )
            lk = float(max(1e-15, h_th.get(m_t, 1e-15)))
            mu_next[theta] = float(mu_t.get(theta, 0.0)) * lk
            ev_total += mu_next[theta]
        if ev_total > 1e-15:
            for theta in TIPOS_SECUESTRADOR:
                mu_next[theta] /= ev_total
        else:
            mu_next = dict(mu_t)

        rows.append({
            "t": t,
            "M(t)": round(M_t, 3),
            "H(μ)": round(H_t, 3),
            "T_t": round(T_t, 3),
            "ε_t": round(eps_t, 3),
            "α_t*": round(alpha_t, 3),
            "γ_t*": round(gamma_t, 3),
            "a_K*": a_K_star,
            "ã_K": a_K_exec,
            "U_rel(K)": round(u_rel_K, 3),
            "U_kill(K)": round(u_kill_K, 3),
            "V_cont(K)": round(v_cont_K, 3),
            "a_F*": a_F_star,
            "ã_F": a_F_exec,
            "EU_coop(F)": round(u_coop_F, 3),
            "EU_col(F)": round(u_col_F, 3),
            "V_R(S)": round(V_R_t, 3),
            "V_N(S)": round(V_N_t, 3),
            "a_S*": a_S_star,
            "ã_S": a_S_exec,
            "m_t": m_t,
            "μ_FARC": round(mu_t.get("FARC", 0.0), 3),
            "μ_ELN":  round(mu_t.get("ELN",  0.0), 3),
            "μ_PAR":  round(mu_t.get("PAR",  0.0), 3),
            "μ_DC":   round(mu_t.get("DC",   0.0), 3),
        })

        mu_t = mu_next
        if m_t != "Continuar":
            break

    df_traj = pd.DataFrame(rows)
    return {
        "trajectory":      df_traj,
        "desenlace_final": str(df_traj["m_t"].iloc[-1]) if not df_traj.empty else "—",
        "t_final":         int(df_traj["t"].iloc[-1])   if not df_traj.empty else 0,
        "mu_final":        mu_t,
    }


def optimize_state_instruments(
    mu: Dict[str, float],
    modelo: ModeloSecuestro,
    presion_S: float,
    iota: float,
    omega_k: float,
    omega_p: float,
    omega_G: float,
    R: float,
    c_ops: Tuple[float, float, float],
    c_maint: Tuple[float, float, float],
    c_inst: Tuple[float, float, float],
    cmh_p_kill: float,
    cmh_p_surv_proxy: float,
    grid_n: int = 25,
    maturity_mult: float = 1.0,
) -> Dict[str, Any]:
    """(α*, γ*) que minimizan min(V^R, V^N) bajo la regla discreta (ilustrativo, grilla uniforme)."""
    best = {"alpha": 0.25, "gamma": 0.25, "loss": 1e99, "V_R": 0.0, "V_N": 0.0}
    g = max(3, int(grid_n))
    alphas = np.linspace(0.05, 0.95, g)
    gammas = np.linspace(0.05, 0.95, g)
    for a in alphas:
        for gam in gammas:
            V_R, V_N, _, _ = compute_state_VR_VN(
                mu,
                modelo,
                presion_S,
                iota,
                omega_k,
                omega_p,
                omega_G,
                float(a),
                float(gam),
                R,
                c_ops,
                c_maint,
                c_inst,
                cmh_p_kill,
                cmh_p_surv_proxy,
                maturity_mult=maturity_mult,
            )
            L = min(V_R, V_N)
            if L < best["loss"]:
                best = {"alpha": float(a), "gamma": float(gam), "loss": float(L), "V_R": V_R, "V_N": V_N}
    return best


def kidnapper_cost_c(gamma: float, phi: float, kappa_c: float, nu: float) -> float:
    """Eq. (cost-function-kidnapper): φ exp(κ_c γ) + ν."""
    return float(phi * np.exp(kappa_c * gamma) + nu)


def kidnapper_equilibrium_continuation_value(
    u_rel: float, u_kill: float, flow: float, beta: float, p_cap: float
) -> float:
    """
    Aproxima la esperanza en E[V^K_{t+1}] del eq. kidnapper-cont (Mechanism.tex) con el valor fijo V*
    que resuelve V* = max(U_rel, U_kill, flow + beta*(1-p_cap)*V*).
    """
    b = float(np.clip(beta, 0.0, 0.9999))
    pc = float(np.clip(p_cap, 0.0, 1.0))
    disc = b * (1.0 - pc)
    v = float(flow)
    for _ in range(500):
        vn = max(u_rel, u_kill, flow + disc * v)
        if abs(vn - v) < 1e-9:
            return float(vn)
        v = vn
    return float(vn)


def kidnapper_V_cont_branch(
    u_rel: float, u_kill: float, flow: float, beta: float, p_cap: float
) -> float:
    """Rama continuar: flujo corriente + beta*(1-p_cap)*V* (eq. kidnapper-cont)."""
    v_star = kidnapper_equilibrium_continuation_value(u_rel, u_kill, flow, beta, p_cap)
    b = float(np.clip(beta, 0.0, 0.9999))
    pc = float(np.clip(p_cap, 0.0, 1.0))
    return float(flow + b * (1.0 - pc) * v_star)


# Fracción objetivo de C(γ,θ) respecto a R·rel(θ) (Tabla 12); se recalibran vía
# ``calibrate_kidnapper_type_scales`` (tipos con menos infraestructura: costo ↑ y pendiente en γ).
KIDNAPPER_COST_FRAC_OF_R: Dict[str, float] = {
    "DC": 0.08,
    "PAR": 0.095,
    "ELN": 0.115,
    "FARC": 0.105,
}
# Escala rescate relativa a R_base (DC mayor infraestructura → mayor R efectivo).
KIDNAPPER_RANSOM_REL: Dict[str, float] = {
    "DC": 3.50,
    "PAR": 3.85,
    "ELN": 5.80,
    "FARC": 7.50,
}
# κ_c por tipo: pendiente de C en γ (FARC ↑ → abandono de continuar más rápido).
KIDNAPPER_KAPPA_C_BY_TYPE: Dict[str, float] = {
    "DC": 2.25,
    "PAR": 2.55,
    "ELN": 2.85,
    "FARC": 3.15,
}
# Objetivos de calibración (margen continuar − max(kill, rel)) en γ bajo / γ alto.
KIDNAPPER_MARGIN_CONT_LO = 3.0
KIDNAPPER_MARGIN_VS_REL_LO = 2.5
KIDNAPPER_MARGIN_COL14_LO = 0.75
KIDNAPPER_MARGIN_CONT_HI_DC = 0.5
KIDNAPPER_MARGIN_CONT_HI_FARC = -4.0
KIDNAPPER_CALIB_T_REF = 100
# τ de referencia donde DC/PAR dejan de continuar antes que ELN/FARC.
KIDNAPPER_SWITCH_TAU_REF: Dict[str, int] = {
    "DC": 12,
    "PAR": 25,
    "ELN": 50,
    "FARC": 80,
}
# Atenúa η(θ) en tipos con más infraestructura → matar menos atractivo al inicio.
KIDNAPPER_ETA_INFRA_MULT: Dict[str, float] = {
    "DC": 0.05,
    "PAR": 0.06,
    "ELN": 0.10,
    # 0.10: garantiza U_kill_FARC < 0 para ba ≤ 1 (condición de propiedad single-crossing).
    "FARC": 0.10,
}
KIDNAPPER_RANSOM_REL_CAP = 5.5
# Multiplicador rel(FARC) usado como fallback cuando no hay R_escala en el df.
# El valor absoluto ya no es un piso rígido en la calibración (que usa R_base * rel).
KIDNAPPER_FARC_R_ESCALA_FIXED = 3.30   # relativo: R_escala_fallback = R_base * 3.30
KIDNAPPER_FARC_R_ESCALA_MIN   = 0.3    # piso mínimo como múltiplo de R_base


def kidnapper_ransom_rel_cap(modelo: ModeloSecuestro, tipo: str) -> float:
    """Techo de ``rel(θ)``; tipos con menos infra (FARC) pueden requerir ``R`` más alto."""
    z = float(kidnapper_infra_z(modelo, str(tipo)))
    cap = float(KIDNAPPER_RANSOM_REL_CAP * (0.82 + 1.55 * (1.0 - z)))
    if z < 0.30:
        cap = float(max(cap, 13.5))
    return cap
KIDNAPPER_KREL_CAP = 32.0
# Piso de κ_rel: U_rel = −κ_rel; más alto → Liberar menos atractivo en col. 14.
KIDNAPPER_KREL_MIN: Dict[str, float] = {
    "DC": 28.0,
    "PAR": 26.0,
    "ELN": 22.0,
    "FARC": 18.0,
}
# κ_rel alto → U_rel muy negativo → liberar peor que continuar al comparar col. 14.
KIDNAPPER_KREL_INFRA_MULT: Dict[str, float] = {
    "DC": 1.40,
    "PAR": 1.38,
    "ELN": 1.35,
    "FARC": 1.32,
}
# Boost extra por calibración numérica (por tipo).
KIDNAPPER_KREL_CAL_BOOST: Dict[str, float] = {
    th: 1.0 for th in TIPOS_SECUESTRADOR
}
def kidnapper_infrastructure_index(modelo: ModeloSecuestro, tipo: str) -> float:
    """Índice de infraestructura / capacidad de negociación (DC alto, FARC bajo)."""
    b = modelo.betas[str(tipo)]
    pago = float(b.get("Pago", 0.0))
    muerte = float(b.get("Muerte", 0.0))
    liber = float(b.get("Liberación", 0.0))
    return float(pago - 0.35 * muerte + 0.15 * liber)


def kidnapper_infra_z(modelo: ModeloSecuestro, tipo: str) -> float:
    """z ∈ [0,1]: 0 = FARC (baja infra), 1 = DC (alta infra)."""
    scores = {
        th: kidnapper_infrastructure_index(modelo, th) for th in TIPOS_SECUESTRADOR
    }
    smin = float(min(scores.values()))
    smax = float(max(scores.values()))
    if smax <= smin:
        return 0.5
    return float((scores[str(tipo)] - smin) / max(smax - smin, 1e-9))


def kidnapper_cost_tau_pressure(
    modelo: ModeloSecuestro, tipo: str, tau: int, T_eff: int
) -> float:
    """
    Multiplica C(γ,θ): casi plano al inicio del horizonte; pendiente fuerte solo
    en la cola (DC/PAR antes que ELN/FARC en col. 14).
    """
    z = float(kidnapper_infra_z(modelo, str(tipo)))
    tau_frac = float(max(0.0, int(tau))) / max(float(T_eff), 1.0)
    tau_soft = float(0.38 + 0.42 * (1.0 - z))
    if tau_frac <= tau_soft:
        return float(1.0 + 0.12 * tau_frac / max(tau_soft, 1e-9))
    adj = float((tau_frac - tau_soft) / max(1.0 - tau_soft, 1e-9))
    slope = float(0.18 + 0.72 * z)
    return float(min(2.15, 1.12 + slope * adj))


def _kidnapper_calibration_p_pay(modelo: ModeloSecuestro, tipo: str) -> float:
    """Cuota de pago en t=1: betas + piso por infraestructura (DC negocia más)."""
    b = modelo.betas[str(tipo)]
    scores = {
        j: float(np.exp(float(b.get(j, 0.0))))
        for j in ("Liberación", "Rescate", "Pago", "Muerte")
    }
    tot = float(sum(scores.values()))
    base = float(scores["Pago"] / tot) if tot > 1e-15 else 0.12
    scores_infra = {
        th: kidnapper_infrastructure_index(modelo, th) for th in TIPOS_SECUESTRADOR
    }
    smin = float(min(scores_infra.values()))
    smax = float(max(scores_infra.values()))
    z = (
        float((scores_infra[str(tipo)] - smin) / max(smax - smin, 1e-9))
        if smax > smin
        else 0.5
    )
    floor = float(0.11 + 0.14 * z)
    return float(np.clip(max(floor, base * (0.75 + 0.55 * z)), 0.08, 0.55))


def _kidnapper_continue_margin(
    modelo: ModeloSecuestro,
    tipo: str,
    par: Dict[str, float],
    *,
    R_eff: float,
    gamma: float,
    alpha: float,
    beta_k: float,
    presion_S: float,
    t_hazard: int = 1,
) -> float:
    """V_cont − max(U_rel, U_kill); ``p_pay`` calibrado vía betas del tipo."""
    p_pay = float(_kidnapper_calibration_p_pay(modelo, str(tipo)))
    cg = kidnapper_cost_c(
        float(gamma), float(par["phi"]), float(par["kappa_c"]), float(par["nu"])
    )
    pc = float(par["p_cap"])
    u_rel = float(-par["kappa_rel"])
    u_kill = float((1.0 - pc) * par["eta"] - pc * par["F_cap"])
    flow = float(p_pay * R_eff * (1.0 - alpha) - cg - pc * par["F_cap"])
    v_cont = float(kidnapper_V_cont_branch(u_rel, u_kill, flow, beta_k, pc))
    return float(v_cont - max(u_rel, u_kill))


def _kidnapper_table15_col14_margin(
    modelo: ModeloSecuestro,
    tipo: str,
    par: Dict[str, float],
    *,
    R_eff: float,
    gamma: float,
    alpha: float,
    beta_k: float,
    tau: int,
    T_eff: int,
    p_pay: Optional[float] = None,
) -> Tuple[float, float]:
    """
    Margen col. 14 − max(col. 7, col. 8) con col. 13 = flujos + V_next proxy
    (misma lógica que Tabla 15: cols. 9–12).
    """
    th = str(tipo)
    pc = float(np.clip(float(par["p_cap"]), 0.0, 1.0))
    u_rel = float(-par["kappa_rel"])
    u_kill = float((1.0 - pc) * par["eta"] - pc * par["F_cap"])
    C = float(
        kidnapper_cost_c(
            float(gamma), float(par["phi"]), float(par["kappa_c"]), float(par["nu"])
        )
    )
    pp = float(
        p_pay
        if p_pay is not None
        else _kidnapper_calibration_p_pay(modelo, th)
    )
    flow_rev = float(pp * float(R_eff) * (1.0 - float(alpha)))
    flow_cost = float(-C)
    flow_cap = float(-pc * par["F_cap"])
    flow_net = float(flow_rev + flow_cost + flow_cap)
    v_branch = float(
        kidnapper_V_cont_branch(u_rel, u_kill, flow_net, float(beta_k), pc)
    )
    disc = float(beta_k) * (1.0 - pc)
    v_next = float(disc * v_branch) if int(tau) < int(T_eff) else 0.0
    v13 = float(flow_rev + flow_cost + flow_cap + v_next)
    return float(v13 - max(u_rel, u_kill)), float(v13)


def _kidnapper_margin_vs_liberar(
    modelo: ModeloSecuestro,
    tipo: str,
    par: Dict[str, float],
    *,
    R_eff: float,
    gamma: float,
    alpha: float,
    beta_k: float,
    presion_S: float,
) -> float:
    """V_cont − U_rel (solo rama liberar)."""
    p_pay = float(_kidnapper_calibration_p_pay(modelo, str(tipo)))
    cg = kidnapper_cost_c(
        float(gamma), float(par["phi"]), float(par["kappa_c"]), float(par["nu"])
    )
    pc = float(par["p_cap"])
    u_rel = float(-par["kappa_rel"])
    u_kill = float((1.0 - pc) * par["eta"] - pc * par["F_cap"])
    flow = float(p_pay * R_eff * (1.0 - alpha) - cg - pc * par["F_cap"])
    v_cont = float(kidnapper_V_cont_branch(u_rel, u_kill, flow, beta_k, pc))
    return float(v_cont - u_rel)


def _kidnapper_fixed_cost_kappa_maps(
    modelo: ModeloSecuestro,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Costos estructurales fijos (no se recalibran; solo ``R`` por tipo)."""
    scores = {
        th: kidnapper_infrastructure_index(modelo, th) for th in TIPOS_SECUESTRADOR
    }
    smin = float(min(scores.values()))
    smax = float(max(scores.values()))
    cost_frac: Dict[str, float] = {}
    kappa_c_map: Dict[str, float] = {}
    for th in TIPOS_SECUESTRADOR:
        z = (
            float((scores[th] - smin) / max(smax - smin, 1e-9))
            if smax > smin
            else 0.5
        )
        cost_frac[th] = float(
            KIDNAPPER_COST_FRAC_OF_R.get(str(th), 0.055 + 0.042 * z)
        )
        kappa_c_map[th] = float(
            KIDNAPPER_KAPPA_C_BY_TYPE.get(str(th), 1.85 + 1.05 * z)
        )
    return cost_frac, kappa_c_map


def _kidnapper_initial_ransom_rel(modelo: ModeloSecuestro) -> Dict[str, float]:
    scores = {
        th: kidnapper_infrastructure_index(modelo, th) for th in TIPOS_SECUESTRADOR
    }
    smin = float(min(scores.values()))
    smax = float(max(scores.values()))
    out: Dict[str, float] = {}
    for th in TIPOS_SECUESTRADOR:
        z = (
            float((scores[th] - smin) / max(smax - smin, 1e-9))
            if smax > smin
            else 0.5
        )
        out[th] = float(KIDNAPPER_RANSOM_REL.get(str(th), 0.92 + 0.72 * z))
    return out


def calibrate_kidnapper_type_scales(
    modelo: ModeloSecuestro,
    *,
    R_base: float,
    gamma_lo: float,
    gamma_hi: float,
    alpha: float,
    beta_k: float,
    p_cap_base: float,
    estado_duro: bool,
    presion_S: float,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    """
    Calibra **solo** ``ransom_rel`` (``R(θ)=R_base·rel(θ)``) por tipo.
    ``cost_frac`` y ``kappa_c`` quedan fijos (estructura por infraestructura).
    Objetivo: col.~14 = Continuar en $\\tau=1$; orden de cambio DC $<$ PAR $<$ ELN $<$ FARC.
    """
    del presion_S  # reservado; margen vía proxy col.~14
    cost_frac, kappa_c_map = _kidnapper_fixed_cost_kappa_maps(modelo)
    ransom_rel = _kidnapper_initial_ransom_rel(modelo)
    T_ref = int(KIDNAPPER_CALIB_T_REF)

    for th in TIPOS_SECUESTRADOR:
        for _ in range(48):
            r_eff = float(R_base) * float(ransom_rel[th])
            par = derive_kidnapper_structural_params(
                modelo,
                th,
                p_cap_base,
                estado_duro,
                R_scale=r_eff,
                gamma_oper=float(gamma_lo),
                cost_frac_override=float(cost_frac[th]),
                kappa_c_override=float(kappa_c_map[th]),
                krel_boost=1.0,
            )
            m_c14, _ = _kidnapper_table15_col14_margin(
                modelo,
                th,
                par,
                R_eff=r_eff,
                gamma=float(gamma_lo),
                alpha=float(alpha),
                beta_k=float(beta_k),
                tau=1,
                T_eff=T_ref,
            )
            if m_c14 >= KIDNAPPER_MARGIN_COL14_LO:
                break
            ransom_rel[th] = float(
                min(
                    kidnapper_ransom_rel_cap(modelo, th),
                    ransom_rel[th] * 1.07,
                )
            )

    for _pass in range(10):
        margins_hi: Dict[str, float] = {}
        for th in TIPOS_SECUESTRADOR:
            r_eff = float(R_base) * float(ransom_rel[th])
            par = derive_kidnapper_structural_params(
                modelo,
                th,
                p_cap_base,
                estado_duro,
                R_scale=r_eff,
                gamma_oper=float(gamma_lo),
                cost_frac_override=float(cost_frac[th]),
                kappa_c_override=float(kappa_c_map[th]),
                krel_boost=1.0,
            )
            tau_sw = int(KIDNAPPER_SWITCH_TAU_REF.get(str(th), 40))
            m_sw, _ = _kidnapper_table15_col14_margin(
                modelo,
                th,
                par,
                R_eff=r_eff,
                gamma=float(gamma_hi),
                alpha=float(alpha),
                beta_k=float(beta_k),
                tau=int(tau_sw),
                T_eff=T_ref,
            )
            margins_hi[th] = float(m_sw)
        changed = False
        if margins_hi.get("PAR", 0.0) > margins_hi.get("ELN", 0.0) + 0.4:
            ransom_rel["PAR"] = float(max(0.45, ransom_rel["PAR"] * 0.96))
            ransom_rel["ELN"] = float(
                min(kidnapper_ransom_rel_cap(modelo, "ELN"), ransom_rel["ELN"] * 1.04)
            )
            changed = True
        if margins_hi.get("FARC", 0.0) > margins_hi.get("ELN", 0.0) + 0.35:
            ransom_rel["FARC"] = float(max(0.40, ransom_rel["FARC"] * 0.95))
            ransom_rel["ELN"] = float(
                min(kidnapper_ransom_rel_cap(modelo, "ELN"), ransom_rel["ELN"] * 1.03)
            )
            changed = True
        if margins_hi.get("DC", 0.0) < margins_hi.get("PAR", 0.0) - 0.5:
            ransom_rel["DC"] = float(
                min(kidnapper_ransom_rel_cap(modelo, "DC"), ransom_rel["DC"] * 1.04)
            )
            changed = True
        if not changed:
            break

    for th in TIPOS_SECUESTRADOR:
        ransom_rel[th] = round(float(ransom_rel[th]), 4)
        cost_frac[th] = round(float(cost_frac[th]), 4)
        kappa_c_map[th] = round(float(kappa_c_map[th]), 3)
    return ransom_rel, cost_frac, kappa_c_map


def kidnapper_R_eff_for_type(
    R_base: float,
    tipo: str,
    *,
    rel_map: Optional[Dict[str, float]] = None,
) -> float:
    """Escala rescate efectiva R(θ) = R_base · rel(θ)."""
    src = rel_map if rel_map is not None else KIDNAPPER_RANSOM_REL
    rel = float(src.get(str(tipo), 1.0))
    return float(max(1e-6, float(R_base) * rel))


def apply_kidnapper_scale_calibration(
    df_params: pd.DataFrame,
    modelo: ModeloSecuestro,
    *,
    R_base: float,
    gamma_oper: float,
    p_cap_base: float,
    estado_duro: bool,
    presion_S: float,
    alpha: float,
    beta_k: float,
    gamma_lo: Optional[float] = None,
    gamma_hi: Optional[float] = None,
    df_mu_traj: Optional[pd.DataFrame] = None,
    T_horizon: Optional[int] = None,
    finalize: bool = True,
) -> pd.DataFrame:
    """
    Recalibra φ, ν y κ_c por fila de Tabla 12 usando escalas por tipo.
    ``R_escala`` queda común para todos los tipos e igual a ``R_base``.
    Actualiza los dicts globales ``KIDNAPPER_*`` con la última calibración.
    """
    global KIDNAPPER_RANSOM_REL, KIDNAPPER_COST_FRAC_OF_R, KIDNAPPER_KAPPA_C_BY_TYPE
    g_lo = float(gamma_lo if gamma_lo is not None else gamma_oper)
    g_hi = float(
        gamma_hi if gamma_hi is not None else min(0.95, float(gamma_oper) + 0.38)
    )
    rel_r, rel_c, rel_k = calibrate_kidnapper_type_scales(
        modelo,
        R_base=float(R_base),
        gamma_lo=g_lo,
        gamma_hi=g_hi,
        alpha=float(alpha),
        beta_k=float(beta_k),
        p_cap_base=float(p_cap_base),
        estado_duro=bool(estado_duro),
        presion_S=float(presion_S),
    )
    KIDNAPPER_RANSOM_REL.update(rel_r)
    KIDNAPPER_COST_FRAC_OF_R.update(rel_c)
    KIDNAPPER_KAPPA_C_BY_TYPE.update(rel_k)

    out = df_params.copy()
    if "R_escala" not in out.columns:
        out["R_escala"] = float(R_base)
    if "beta_k" not in out.columns:
        out["beta_k"] = round(float(beta_k), 4)
    for i, row in out.iterrows():
        th = str(row["theta_K"])
        r_eff = float(R_base)
        par = derive_kidnapper_structural_params(
            modelo,
            th,
            float(p_cap_base),
            bool(estado_duro),
            R_scale=float(r_eff),
            gamma_oper=float(gamma_oper),
            cost_frac_override=float(rel_c.get(th, 0.10)),
            kappa_c_override=float(rel_k.get(th, 2.5)),
            krel_boost=float(KIDNAPPER_KREL_CAL_BOOST.get(str(th), 1.0)),
        )
        out.at[i, "kappa_rel"] = round(float(par["kappa_rel"]), 3)
        out.at[i, "eta"] = round(float(par["eta"]), 3)
        out.at[i, "F_cap"] = round(float(par["F_cap"]), 3)
        out.at[i, "phi"] = round(float(par["phi"]), 4)
        out.at[i, "kappa_c"] = round(float(par["kappa_c"]), 3)
        out.at[i, "nu"] = round(float(par["nu"]), 4)
        out.at[i, "p_cap_tilde"] = round(float(par["p_cap"]), 4)
        out.at[i, "R_escala"] = round(float(R_base), 2)
    if not finalize:
        return out
    KIDNAPPER_RANSOM_REL.update(_kidnapper_initial_ransom_rel(modelo))
    T_ref = int(T_horizon) if T_horizon is not None else int(KIDNAPPER_CALIB_T_REF)
    return _finalize_kidnapper_table12_col14(
        out,
        modelo,
        R_base=float(R_base),
        gamma_lo=float(g_lo),
        gamma_hi=float(g_hi),
        alpha=float(alpha),
        beta_k=float(beta_k),
        p_cap_base=float(p_cap_base),
        estado_duro=bool(estado_duro),
        presion_S=float(presion_S),
        T_check=int(T_ref),
        df_mu_traj=df_mu_traj,
    )


@_cache_data(show_spinner=False)
def apply_kidnapper_scale_calibration_cached(
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
    """Versión cacheada para Streamlit (evita recalibrar en cada rerun)."""
    df_in = pd.DataFrame(list(df_params_records), columns=list(df_columns))
    df_mu = None
    if df_mu_records is not None and df_mu_columns is not None:
        df_mu = pd.DataFrame(list(df_mu_records), columns=list(df_mu_columns))
    betas = {k: dict(v) for k, v in betas_items}
    lambdas = {str(k): float(v) for k, v in lambdas_items}
    modelo = ModeloSecuestro(betas=betas, lambdas_0=lambdas)
    return apply_kidnapper_scale_calibration(
        df_in,
        modelo,
        R_base=float(R_base),
        gamma_oper=float(gamma_oper),
        p_cap_base=float(p_cap_base),
        estado_duro=bool(estado_duro),
        presion_S=float(presion_S),
        alpha=float(alpha),
        beta_k=float(beta_k),
        gamma_lo=float(gamma_lo),
        gamma_hi=float(gamma_hi),
        df_mu_traj=df_mu,
        T_horizon=int(T_horizon),
        finalize=bool(finalize),
    )


def _synthetic_mu_traj_col14_calib(
    T: int,
    *,
    gamma_lo: float,
    gamma_hi: float,
    alpha: float,
    p_pay: float = 0.18,
    p_cap: float = 0.12,
) -> pd.DataFrame:
    """Trayectoria μ_t sintética (γ crece linealmente) para validar col. 14."""
    rows: List[Dict[str, Any]] = []
    T_eff = int(max(1, T))
    for t in range(T_eff + 1):
        frac = float(t) / float(T_eff)
        g = float(gamma_lo + (gamma_hi - gamma_lo) * frac)
        rows.append(
            {
                "t": int(t),
                "alpha_t": float(alpha),
                "gamma_t": float(g),
                "mu_DC": 0.25,
                "mu_PAR": 0.25,
                "mu_ELN": 0.25,
                "mu_FARC": 0.25,
                "Epi_pay_Qcont_mu": float(p_pay),
                "Epi_pcap_Qcap": float(p_cap),
            }
        )
    return pd.DataFrame(rows)


def _set_kidnapper_r_escala_only(
    df_params: pd.DataFrame, row_index: int, r_eff: float
) -> None:
    """Actualiza solo ``R_escala`` (sin recalibrar φ, ν, κ_rel por escala de costos)."""
    df_params.at[int(row_index), "R_escala"] = round(float(r_eff), 2)


def apply_farc_r_escala_fixed(
    df_params: pd.DataFrame,
    modelo: ModeloSecuestro,
    *,
    R_farc: Optional[float] = None,
    R_base: float,
    gamma_oper: float,
    alpha: float,
    p_cap_base: float,
    estado_duro: bool,
    rel_map: Optional[Dict[str, float]] = None,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Fija ``R_escala`` de FARC (solo columna R; costos estructurales intactos)."""
    global KIDNAPPER_RANSOM_REL
    th = "FARC"
    out = df_params.copy()
    rel_out: Dict[str, float] = {}
    if rel_map:
        rel_out = {str(k): float(v) for k, v in rel_map.items()}
    else:
        for th2 in TIPOS_SECUESTRADOR:
            if (
                "R_escala" in out.columns
                and not out[out["theta_K"].astype(str) == str(th2)].empty
            ):
                rel_out[str(th2)] = float(
                    out.loc[out["theta_K"].astype(str) == str(th2), "R_escala"].iloc[0]
                ) / max(float(R_base), 1e-9)
            else:
                rel_out[str(th2)] = float(KIDNAPPER_RANSOM_REL.get(str(th2), 1.0))

    # Piso relativo a R_base; R_farc explícito tiene prioridad si se pasa
    r_floor_rel = float(max(1.0, float(R_base) * float(KIDNAPPER_FARC_R_ESCALA_MIN)))
    r_default = float(R_base) * float(KIDNAPPER_FARC_R_ESCALA_FIXED)
    r_eff = float(max(r_floor_rel, float(R_farc) if R_farc is not None else r_default))
    rel_out[str(th)] = float(r_eff / max(float(R_base), 1e-9))
    KIDNAPPER_RANSOM_REL[str(th)] = float(rel_out[str(th)])

    idx = out[out["theta_K"].astype(str) == str(th)].index
    if len(idx) == 0:
        return out, rel_out
    _set_kidnapper_r_escala_only(out, int(idx[0]), r_eff)
    return out, rel_out


def calibrate_farc_ransom_rel_only(
    df_params: pd.DataFrame,
    modelo: ModeloSecuestro,
    *,
    R_base: float,
    gamma_lo: float,
    alpha: float,
    beta_k: float,
    p_cap_base: float,
    estado_duro: bool,
    df_mu_traj: pd.DataFrame,
    T_check: int,
    rel_map: Dict[str, float],
    switch_target: Optional[int] = None,
    switch_min: int = 3,
    switch_tol: int = 350,
) -> Tuple[pd.DataFrame, Dict[str, float], Dict[str, Any]]:
    """
    Ajusta **solo** ``R_escala`` de FARC (sin recalibrar φ, ν, κ_rel con ``R_scale``).

    Objetivo: col.~14 = Continuar en $\\tau=1$ y cambio de rama en $\\tau>1$.
    Los ``rel(θ)`` de DC/PAR/ELN en ``rel_map`` no se modifican.
    """
    _ = (p_cap_base, estado_duro)
    global KIDNAPPER_RANSOM_REL
    th = "FARC"
    out = df_params.copy()
    rel_out = {str(k): float(v) for k, v in rel_map.items()}
    meta_out: Dict[str, Any] = {
        "ok_tau1_continuar": False,
        "primer_tau_switch": None,
        "opcion_tau1": "",
        "R_escala_FARC": float("nan"),
        "rel_FARC": float(rel_out.get(th, KIDNAPPER_RANSOM_REL.get(th, 7.0))),
    }
    if th not in rel_out:
        rel_out[th] = float(KIDNAPPER_RANSOM_REL.get(th, 7.0))

    if df_mu_traj is None or df_mu_traj.empty:
        return out, rel_out, meta_out

    idx_f = out[out["theta_K"].astype(str) == str(th)].index
    if len(idx_f) == 0:
        return out, rel_out, meta_out
    ii_f = int(idx_f[0])

    tgt_sw = int(
        switch_target
        if switch_target is not None
        else KIDNAPPER_SWITCH_TAU_REF.get(th, 40)
    )
    cap_rel = float(kidnapper_ransom_rel_cap(modelo, th))
    r_cap = float(R_base) * cap_rel
    # Piso relativo: 1.5 × R_base; permite encontrar R_escala bajo si R_base es pequeño.
    r_floor = float(max(1.0, float(R_base) * float(KIDNAPPER_FARC_R_ESCALA_MIN)))
    r_cap = float(max(r_cap, r_floor * 4.0))

    def _set_farc_r(r_abs: float) -> float:
        r_use = float(max(r_floor, min(r_cap, float(r_abs))))
        _set_kidnapper_r_escala_only(out, ii_f, r_use)
        # Recalibrate phi/nu/kappa_c so C(gamma_lo) = cost_frac * R_new
        _par = derive_kidnapper_structural_params(
            modelo,
            th,
            float(p_cap_base),
            bool(estado_duro),
            R_scale=r_use,
            gamma_oper=float(gamma_lo),
            cost_frac_override=float(KIDNAPPER_COST_FRAC_OF_R.get(th, 0.105)),
            kappa_c_override=float(KIDNAPPER_KAPPA_C_BY_TYPE.get(th, 3.15)),
        )
        out.at[ii_f, "phi"] = round(float(_par["phi"]), 4)
        out.at[ii_f, "nu"] = round(float(_par["nu"]), 4)
        out.at[ii_f, "kappa_c"] = round(float(_par["kappa_c"]), 4)
        rel_out[th] = float(r_use / max(float(R_base), 1e-9))
        return r_use

    def _bi_farc() -> Tuple[bool, Optional[int], str]:
        r_eff = _set_farc_r(float(R_base) * float(rel_out[th]))
        df_ia, meta = kidnapper_backward_induction_k_table(
            modelo,
            df_mu_traj,
            out,
            tipo_real=str(th),
            beta_k=float(beta_k),
            R=float(R_base),
            t_mad=5.0,
            T=int(T_check),
            alpha_fallback=float(alpha),
            gamma_fallback=float(gamma_lo),
            alpha_tab12=float(alpha),
            ransom_tab12=float(r_eff),
        )
        sw = meta.get("primer_tau_backward")
        sub = df_ia.loc[df_ia["t"].astype(int) == 1]
        if sub.empty:
            return False, sw, ""
        row = sub.iloc[0]
        opt = str(row["opcion_BW"])
        ok_t1 = opt == "Continuar (a_cont)"
        return bool(ok_t1), sw, opt

    r_lo, r_hi = float(r_floor), float(r_cap)
    r_best = r_hi
    for _ in range(24):
        r_mid = 0.5 * (r_lo + r_hi)
        rel_out[th] = float(r_mid / max(float(R_base), 1e-9))
        ok_t1, _, _ = _bi_farc()
        if ok_t1:
            r_best = r_mid
            r_hi = r_mid
        else:
            r_lo = r_mid

    T_eff_cal = int(max(1, T_check))
    r_star = float(r_best)
    rel_out[th] = float(r_star / max(float(R_base), 1e-9))
    ok_t1, sw0, opt0 = _bi_farc()
    if not ok_t1:
        r_star = float(r_cap)
        rel_out[th] = float(r_star / max(float(R_base), 1e-9))
        ok_t1, sw0, opt0 = _bi_farc()

    # sw_i: τ donde ocurre el cambio de rama; T_eff_cal+1 = sin cambio en el horizonte
    sw_i = int(sw0) if sw0 is not None else T_eff_cal + 1
    if ok_t1 and sw_i < int(switch_min):
        for _ in range(28):
            r_star = float(min(r_cap, r_star * 1.08))
            rel_out[th] = float(r_star / max(float(R_base), 1e-9))
            ok_t1, sw0, opt0 = _bi_farc()
            sw_i = int(sw0) if sw0 is not None else T_eff_cal + 1
            if not ok_t1:
                break
            if sw_i >= int(switch_min):
                break

    # Bucle bidireccional: acerca el punto de cambio al target
    # sw0=None (sin cambio en horizonte) → sw_i=T+1 (demasiado tarde) → bajar R
    for _ in range(60):
        ok_t1, sw0, opt0 = _bi_farc()
        sw_i = int(sw0) if sw0 is not None else T_eff_cal + 1
        if not ok_t1:
            r_star = float(min(r_cap, r_star * 1.06))
            rel_out[th] = float(r_star / max(float(R_base), 1e-9))
            continue
        tgt_lo = int(max(int(switch_min), tgt_sw - int(switch_tol)))
        tgt_hi = int(min(tgt_sw + int(switch_tol), T_eff_cal))
        if tgt_lo <= sw_i <= tgt_hi:
            break
        if sw_i > tgt_hi:
            # cambio demasiado tarde o sin cambio → bajar R para que V_cont caiga antes
            r_new = float(max(r_floor, r_star * 0.97))
            if r_new >= r_star - 0.01:
                break
            r_star = r_new
        else:
            # cambio demasiado temprano → subir R para retrasarlo
            r_star = float(min(r_cap, r_star * 1.05))
        rel_out[th] = float(r_star / max(float(R_base), 1e-9))

    r_eff_f = _set_farc_r(float(r_star))
    ok_f, sw_f, opt_f = _bi_farc()
    rel_out[th] = round(float(r_eff_f / max(float(R_base), 1e-9)), 4)
    KIDNAPPER_RANSOM_REL[th] = float(rel_out[th])
    meta_out = {
        "ok_tau1_continuar": bool(ok_f),
        "primer_tau_switch": int(sw_f) if sw_f is not None else None,
        "opcion_tau1": str(opt_f or opt0 or ""),
        "R_escala_FARC": round(float(r_eff_f), 2),
        "rel_FARC": float(rel_out[th]),
    }
    return out, rel_out, meta_out


def verify_farc_tab15_backward_pattern(
    modelo: ModeloSecuestro,
    df_mu_traj: pd.DataFrame,
    df_k_params: pd.DataFrame,
    *,
    R_base: float,
    beta_k: float,
    alpha: float,
    gamma_lo: float,
    T_check: int,
) -> Dict[str, Any]:
    """Comprueba Tabla 15 (θ*=FARC): τ=1 Continuar y cambio de rama después."""
    th = "FARC"
    if (
        df_mu_traj is None
        or df_mu_traj.empty
        or df_k_params is None
        or df_k_params.empty
    ):
        return {"ok": False, "reason": "sin trayectoria o Tabla 12"}
    sub = df_k_params[df_k_params["theta_K"].astype(str) == str(th)]
    if sub.empty or "R_escala" not in sub.columns:
        return {"ok": False, "reason": "sin fila FARC en Tabla 12"}
    r_eff = float(sub.iloc[0]["R_escala"])
    df_ia, meta = kidnapper_backward_induction_k_table(
        modelo,
        df_mu_traj,
        df_k_params,
        tipo_real=str(th),
        beta_k=float(beta_k),
        R=float(R_base),
        t_mad=5.0,
        T=int(T_check),
        alpha_fallback=float(alpha),
        gamma_fallback=float(gamma_lo),
        alpha_tab12=float(alpha),
        ransom_tab12=float(r_eff),
    )
    row1 = df_ia.loc[df_ia["t"].astype(int) == 1]
    if row1.empty:
        return {"ok": False, "reason": "sin fila τ=1"}
    r1 = row1.iloc[0]
    opt1 = str(r1["opcion_BW"])
    ok_t1 = opt1 == "Continuar (a_cont)"
    sw = meta.get("primer_tau_backward")
    sw_i = int(sw) if sw is not None else None
    min_continue_tau = 6 if str(th).upper() in {"PAR", "ELN", "FARC", "LEN"} else 2
    ok_switch = (
        sw_i is not None
        and min_continue_tau < sw_i <= int(T_check)
        and sw_i < min(500, int(T_check) + 1)
    )
    return {
        "ok": bool(ok_t1 and ok_switch),
        "ok_tau1_continuar": bool(ok_t1),
        "opcion_tau1": opt1,
        "primer_tau_switch": sw_i,
        "R_escala_FARC": r_eff,
    }


def validate_tab15_all_types(
    df_k_params: pd.DataFrame,
    modelo: ModeloSecuestro,
    df_mu_traj: pd.DataFrame,
    *,
    R_base: float,
    beta_k: float,
    alpha: float,
    gamma_lo: float,
    T_check: int,
) -> Dict[str, Dict[str, Any]]:
    """
    Ejecuta inducción hacia atrás para cada tipo y verifica:
    - ``ok_tau1``: col.~14 = "Continuar" en $\\tau=1$.
    - ``ok_switch``: existe un cambio con el mínimo por tipo antes de 500.
    """
    results: Dict[str, Dict[str, Any]] = {}
    for th in TIPOS_SECUESTRADOR:
        sub = df_k_params[df_k_params["theta_K"].astype(str) == str(th)]
        if sub.empty or df_mu_traj is None or df_mu_traj.empty:
            results[th] = {
                "ok_tau1": False,
                "ok_switch": False,
                "primer_tau": None,
                "opcion_tau1": "sin datos",
                "R_escala": None,
            }
            continue
        r_eff = float(sub.iloc[0]["R_escala"]) if "R_escala" in sub.columns else float(R_base)
        df_ia, meta = kidnapper_backward_induction_k_table(
            modelo,
            df_mu_traj,
            df_k_params,
            tipo_real=str(th),
            beta_k=float(beta_k),
            R=float(R_base),
            t_mad=5.0,
            T=int(T_check),
            alpha_fallback=float(alpha),
            gamma_fallback=float(gamma_lo),
            alpha_tab12=float(alpha),
            ransom_tab12=float(r_eff),
        )
        row1 = df_ia.loc[df_ia["t"].astype(int) == 1]
        opt1 = str(row1.iloc[0]["opcion_BW"]) if not row1.empty else "—"
        ok_t1 = opt1 == "Continuar (a_cont)"
        sw = meta.get("primer_tau_backward")
        sw_i = int(sw) if sw is not None else None
        min_continue_tau = 6 if str(th).upper() in {"PAR", "ELN", "FARC", "LEN"} else 2
        ok_sw = (
            sw_i is not None
            and min_continue_tau < sw_i <= int(T_check)
            and sw_i < min(500, int(T_check) + 1)
        )
        results[th] = {
            "ok_tau1": bool(ok_t1),
            "ok_switch": bool(ok_sw),
            "ok_switch_before_500": bool(ok_sw),
            "primer_tau": sw_i,
            "opcion_tau1": opt1,
            "R_escala": round(float(r_eff), 2),
        }
    return results


def _finalize_kidnapper_table12_col14(
    df_params: pd.DataFrame,
    modelo: ModeloSecuestro,
    *,
    R_base: float,
    gamma_lo: float,
    gamma_hi: float,
    alpha: float,
    beta_k: float,
    p_cap_base: float,
    estado_duro: bool,
    presion_S: float,
    T_check: int = KIDNAPPER_CALIB_T_REF,
    early_taus: Tuple[int, ...] = (1,),
    df_mu_traj: Optional[pd.DataFrame] = None,
    switch_tol: int = 6,
) -> pd.DataFrame:
    """
    Ajuste fino con BI (Tabla 15): **solo** ``R_escala`` / ``ransom_rel`` por θ_K.
    Col.~14 = Continuar en $\\tau=1$; cambio de rama DC $<$ PAR $<$ ELN $<$ FARC.
    """
    del presion_S
    global KIDNAPPER_RANSOM_REL
    out = df_params.copy()
    cost_frac, kappa_c_map = _kidnapper_fixed_cost_kappa_maps(modelo)
    p_pay_cal = 0.24
    if (
        df_mu_traj is not None
        and not df_mu_traj.empty
        and "Epi_pay_Qcont_mu" in df_mu_traj.columns
    ):
        try:
            _pp = float(np.nanmedian(df_mu_traj["Epi_pay_Qcont_mu"].astype(float)))
            if np.isfinite(_pp) and _pp > 0.05:
                p_pay_cal = float(_pp)
        except (TypeError, ValueError):
            pass
    if df_mu_traj is not None and not df_mu_traj.empty:
        df_mu = df_mu_traj.copy()
    else:
        df_mu = _synthetic_mu_traj_col14_calib(
            int(T_check),
            gamma_lo=float(gamma_lo),
            gamma_hi=float(gamma_hi),
            alpha=float(alpha),
            p_pay=float(p_pay_cal),
        )

    if float(p_pay_cal) < 0.10:
        pay_boost = 1.65
    else:
        pay_boost = float(np.clip(0.17 / max(float(p_pay_cal), 0.06), 1.0, 2.85))

    def _apply_row(th: str, ii: int, r_eff: float) -> None:
        par = derive_kidnapper_structural_params(
            modelo,
            str(th),
            float(p_cap_base),
            bool(estado_duro),
            R_scale=float(r_eff),
            gamma_oper=float(gamma_lo),
            cost_frac_override=float(cost_frac[str(th)]),
            kappa_c_override=float(kappa_c_map[str(th)]),
            krel_boost=1.0,
        )
        out.at[ii, "kappa_rel"] = round(float(par["kappa_rel"]), 3)
        out.at[ii, "eta"] = round(float(par["eta"]), 3)
        out.at[ii, "F_cap"] = round(float(par["F_cap"]), 3)
        out.at[ii, "phi"] = round(float(par["phi"]), 4)
        out.at[ii, "kappa_c"] = round(float(par["kappa_c"]), 3)
        out.at[ii, "nu"] = round(float(par["nu"]), 4)
        out.at[ii, "p_cap_tilde"] = round(float(par["p_cap"]), 4)
        out.at[ii, "R_escala"] = round(float(r_eff), 2)
        h_pay = modelo.calcular_hazards(
            1,
            str(th),
            float(gamma_lo),
            maturity_mult=1.0,
            alpha=float(alpha),
        )
        out.at[ii, "h_LibPago"] = round(float(h_pay["Pago"]), 4)
        out.at[ii, "C_gamma_theta"] = round(
            float(
                kidnapper_cost_c(
                    float(gamma_lo),
                    float(par["phi"]),
                    float(par["kappa_c"]),
                    float(par["nu"]),
                )
            ),
            2,
        )

    others_rel_baseline = _kidnapper_initial_ransom_rel(modelo)
    rel_final: Dict[str, float] = {}

    def _apply_all_rows_for_bi(focal_th: str, r_eff_focal: float) -> None:
        """Fija el resto de tipos en R basal para no contaminar V_next del focal."""
        for th2 in TIPOS_SECUESTRADOR:
            idx2 = out[out["theta_K"].astype(str) == str(th2)].index
            if len(idx2) == 0:
                continue
            ii2 = int(idx2[0])
            if str(th2) == str(focal_th):
                r_use = float(r_eff_focal)
            else:
                r_use = float(
                    float(R_base)
                    * float(others_rel_baseline.get(str(th2), 1.0))
                )
            _apply_row(str(th2), ii2, r_use)

    def _run_bi(th: str, r_eff: float) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        _apply_all_rows_for_bi(str(th), float(r_eff))
        return kidnapper_backward_induction_k_table(
            modelo,
            df_mu,
            out,
            tipo_real=str(th),
            beta_k=float(beta_k),
            R=float(R_base),
            t_mad=5.0,
            T=int(T_check),
            alpha_fallback=float(alpha),
            gamma_fallback=float(gamma_lo),
            alpha_tab12=float(alpha),
            ransom_tab12=float(r_eff),
        )

    def _eval_early(
        th: str,
        r_eff: float,
        *,
        check_taus: Optional[Tuple[int, ...]] = None,
    ) -> Tuple[bool, Optional[str], Optional[int], pd.DataFrame]:
        df_ia, meta = _run_bi(th, r_eff)
        sw = meta.get("primer_tau_backward")
        fail_opt: Optional[str] = None
        ok = True
        taus_chk = check_taus if check_taus is not None else early_taus
        for tau_e in taus_chk:
            sub_t = df_ia.loc[df_ia["t"].astype(int) == int(tau_e)]
            if sub_t.empty:
                ok = False
                fail_opt = ""
                break
            row = sub_t.iloc[0]
            opt = str(row["opcion_BW"])
            vc = float(row["V_cont"])
            ur = float(row["U_rel"])
            uk = float(row["U_kill"])
            if opt != "Continuar (a_cont)" or vc <= max(ur, uk) + 0.25:
                ok = False
                fail_opt = opt
                break
        return ok, fail_opt, sw, df_ia

    for th in TIPOS_SECUESTRADOR:
        idx = out[out["theta_K"].astype(str) == str(th)].index
        if len(idx) == 0:
            continue
        ii = int(idx[0])
        tgt_sw = int(KIDNAPPER_SWITCH_TAU_REF.get(str(th), 40))
        rel_cap_th = float(kidnapper_ransom_rel_cap(modelo, str(th)))
        rel_lo = float(0.55)
        rel_hi = float(rel_cap_th)
        rel_best = float(rel_hi)

        for _ in range(22):
            rel_mid = float(0.5 * (rel_lo + rel_hi))
            r_eff = float(rel_mid * float(R_base))
            ok_t1, _, _, _ = _eval_early(str(th), r_eff, check_taus=(1,))
            if ok_t1:
                rel_best = rel_mid
                rel_hi = rel_mid
            else:
                rel_lo = rel_mid

        rel_th = float(min(rel_cap_th, rel_best))
        if float(p_pay_cal) < 0.14:
            rel_th = float(min(rel_cap_th, rel_th * max(1.0, pay_boost)))
        r_eff_chk = float(rel_th * float(R_base))
        ok_t1_final, _, _, _ = _eval_early(str(th), r_eff_chk, check_taus=(1,))
        if not ok_t1_final:
            rel_th = float(rel_cap_th)

        rel_final[str(th)] = round(float(rel_th), 4)

    for th in TIPOS_SECUESTRADOR:
        if str(th) not in rel_final:
            continue
        KIDNAPPER_RANSOM_REL[str(th)] = float(rel_final[str(th)])

    def _apply_all_rows_combined() -> None:
        for th2 in TIPOS_SECUESTRADOR:
            idx2 = out[out["theta_K"].astype(str) == str(th2)].index
            if len(idx2) == 0:
                continue
            r_use = kidnapper_R_eff_for_type(
                float(R_base), str(th2), rel_map=rel_final
            )
            _apply_row(str(th2), int(idx2[0]), float(r_use))

    def _eval_combined(
        th: str, *, check_taus: Tuple[int, ...] = (1,)
    ) -> Tuple[bool, Optional[str], Optional[int]]:
        _apply_all_rows_combined()
        r_eff = kidnapper_R_eff_for_type(float(R_base), str(th), rel_map=rel_final)
        df_ia, meta = kidnapper_backward_induction_k_table(
            modelo,
            df_mu,
            out,
            tipo_real=str(th),
            beta_k=float(beta_k),
            R=float(R_base),
            t_mad=5.0,
            T=int(T_check),
            alpha_fallback=float(alpha),
            gamma_fallback=float(gamma_lo),
            alpha_tab12=float(alpha),
            ransom_tab12=float(r_eff),
        )
        sw = meta.get("primer_tau_backward")
        for tau_e in check_taus:
            sub_t = df_ia.loc[df_ia["t"].astype(int) == int(tau_e)]
            if sub_t.empty:
                return False, "", sw
            row = sub_t.iloc[0]
            if str(row["opcion_BW"]) != "Continuar (a_cont)":
                return False, str(row["opcion_BW"]), sw
            vc = float(row["V_cont"])
            if vc <= max(float(row["U_rel"]), float(row["U_kill"])) + 0.25:
                return False, str(row["opcion_BW"]), sw
        return True, None, sw

    for th in TIPOS_SECUESTRADOR:
        if str(th) not in rel_final:
            continue
        cap_th = float(kidnapper_ransom_rel_cap(modelo, str(th)))
        for _ in range(40):
            ok_c, _, _ = _eval_combined(str(th), check_taus=(1,))
            if ok_c:
                break
            rel_final[str(th)] = float(
                min(cap_th, float(rel_final[str(th)]) * 1.08)
            )
        else:
            rel_final[str(th)] = float(cap_th)
        KIDNAPPER_RANSOM_REL[str(th)] = float(rel_final[str(th)])

    if df_mu is not None and not df_mu.empty:
        out, rel_final, _farc_meta = calibrate_farc_ransom_rel_only(
            out,
            modelo,
            R_base=float(R_base),
            gamma_lo=float(gamma_lo),
            alpha=float(alpha),
            beta_k=float(beta_k),
            p_cap_base=float(p_cap_base),
            estado_duro=bool(estado_duro),
            df_mu_traj=df_mu,
            T_check=int(T_check),
            rel_map=rel_final,
        )

    # Re-verificación tras calibración de FARC: cualquier tipo cuyo V_cont(1)
    # haya bajado por cambio en V_next (contribución de FARC) recupera R.
    for th in TIPOS_SECUESTRADOR:
        if str(th) not in rel_final:
            continue
        cap_th = float(kidnapper_ransom_rel_cap(modelo, str(th)))
        for _ in range(15):
            ok_rv, _, _ = _eval_combined(str(th), check_taus=(1,))
            if ok_rv:
                break
            rel_final[str(th)] = float(min(cap_th, float(rel_final[str(th)]) * 1.08))
            KIDNAPPER_RANSOM_REL[str(th)] = float(rel_final[str(th)])

    # Ajuste final: todos los tipos deben iniciar con Continuar en τ=1 y
    # cambiar de rama dentro de la tabla antes del período 100. Si el cambio
    # queda demasiado tarde, se aproxima R_escala al menor valor que conserva
    # Continuar en τ=1; eso adelanta el cruce sin tocar costos de otros tipos.
    def _eval_combined_status(th: str) -> Tuple[bool, Optional[int], str]:
        _apply_all_rows_combined()
        r_eff = kidnapper_R_eff_for_type(float(R_base), str(th), rel_map=rel_final)
        df_ia, meta = kidnapper_backward_induction_k_table(
            modelo,
            df_mu,
            out,
            tipo_real=str(th),
            beta_k=float(beta_k),
            R=float(R_base),
            t_mad=5.0,
            T=int(T_check),
            alpha_fallback=float(alpha),
            gamma_fallback=float(gamma_lo),
            alpha_tab12=float(alpha),
            ransom_tab12=float(r_eff),
        )
        row1 = df_ia.loc[df_ia["t"].astype(int) == 1]
        opt1 = str(row1.iloc[0]["opcion_BW"]) if not row1.empty else ""
        ok_t1 = opt1 == "Continuar (a_cont)"
        sw = meta.get("primer_tau_backward")
        sw_i = int(sw) if sw is not None else None
        return bool(ok_t1), sw_i, opt1

    tau_switch_limit = int(min(max(2, int(T_check)), 99))
    for _pass in range(3):
        changed_any = False
        for th in TIPOS_SECUESTRADOR:
            if str(th) not in rel_final:
                continue
            ok_t1, sw_i, _ = _eval_combined_status(str(th))
            ok_sw = sw_i is not None and 1 < int(sw_i) <= tau_switch_limit
            if ok_t1 and ok_sw:
                continue

            cap_th = float(kidnapper_ransom_rel_cap(modelo, str(th)))
            rel_cur = float(min(cap_th, max(0.02, float(rel_final[str(th)]))))

            if not ok_t1:
                for _ in range(25):
                    rel_cur = float(min(cap_th, rel_cur * 1.08))
                    rel_final[str(th)] = rel_cur
                    KIDNAPPER_RANSOM_REL[str(th)] = rel_cur
                    ok_t1, sw_i, _ = _eval_combined_status(str(th))
                    if ok_t1:
                        break

            lo, hi = 0.02, float(rel_cur)
            rel_min_t1 = float(hi)
            for _ in range(28):
                mid = float(0.5 * (lo + hi))
                rel_final[str(th)] = mid
                KIDNAPPER_RANSOM_REL[str(th)] = mid
                ok_mid, _, _ = _eval_combined_status(str(th))
                if ok_mid:
                    rel_min_t1 = mid
                    hi = mid
                else:
                    lo = mid

            # Pequeño colchón sobre el umbral de τ=1: conserva Continuar, pero
            # evita empujar el cambio hacia el final del horizonte.
            rel_new = float(min(cap_th, rel_min_t1 * 1.015))
            rel_final[str(th)] = rel_new
            KIDNAPPER_RANSOM_REL[str(th)] = rel_new
            changed_any = True
        if not changed_any:
            break

    for th in TIPOS_SECUESTRADOR:
        idx = out[out["theta_K"].astype(str) == str(th)].index
        if len(idx) == 0:
            continue
        ii = int(idx[0])
        r_eff = kidnapper_R_eff_for_type(
            float(R_base), str(th), rel_map=rel_final
        )
        _apply_row(str(th), ii, float(r_eff))

    return out


def derive_kidnapper_structural_params(
    modelo: ModeloSecuestro,
    tipo: str,
    p_cap_base: float,
    estado_duro: bool,
    *,
    R_scale: Optional[float] = None,
    gamma_oper: Optional[float] = None,
    cost_frac_override: Optional[float] = None,
    kappa_c_override: Optional[float] = None,
    krel_boost: float = 1.0,
) -> Dict[str, float]:
    """
    Mapea las betas del riesgo proporcional (model_logic / Mechanism, heterogeneidad en θ_K)
    a φ(θ), κ_c(θ), ν(θ), κ_rel(θ), η(θ), F_cap y p̃_cap(θ), θ_S).

    Si ``R_scale`` y ``gamma_oper`` están definidos, calibra φ y ν para que
    $C(\\gamma,\\theta)=\\phi\\,e^{\\kappa_c\\gamma}+\\nu$ quede en
    ``KIDNAPPER_COST_FRAC_OF_R[θ]·R`` (entre 5\\% y 15\\% de $R$ por tipo).
    """
    b = modelo.betas[tipo]
    ba = float(b["Muerte"])
    bl = float(b["Liberación"])
    br = float(b["Rescate"])
    # η(θ): mayor propensión a asesinato estructural → mayor beneficio marginal reputacional del kill
    eta = (18.0 + 28.0 * ba) * float(
        KIDNAPPER_ETA_INFRA_MULT.get(str(tipo), 1.0)
    )
    # κ_rel(θ): mayor orientación a pago (bl alto) → menor desutilidad relativa de liberar sin rescate
    kappa_rel = float(max(5.0, 24.0 - 18.0 * (bl + 0.45))) * float(
        KIDNAPPER_KREL_INFRA_MULT.get(str(tipo), 1.0)
    ) * float(
        KIDNAPPER_KREL_CAL_BOOST.get(str(tipo), 1.0) * max(1e-6, float(krel_boost))
    )
    kappa_rel = float(min(float(KIDNAPPER_KREL_CAP), kappa_rel))
    # κ_c: single-crossing en γ (Mechanism, eq. cost-function-kidnapper)
    kappa_c = float(2.20 + 1.40 * ba + 0.70 * max(0.0, -bl))
    nu_struct = float(0.70 + 0.55 * ba + 0.28 * abs(bl))
    phi_struct = float(0.45 + 0.35 * (0.5 + 0.5 * np.tanh(br)) + 0.10 * ba)

    if (
        R_scale is not None
        and gamma_oper is not None
        and float(R_scale) > 1e-6
    ):
        frac = float(
            cost_frac_override
            if cost_frac_override is not None
            else KIDNAPPER_COST_FRAC_OF_R.get(str(tipo), 0.10)
        )
        frac = float(np.clip(frac, 0.05, 0.25))
        c_tgt = float(frac * float(R_scale))
        kappa_c = float(
            kappa_c_override
            if kappa_c_override is not None
            else KIDNAPPER_KAPPA_C_BY_TYPE.get(str(tipo), 2.50)
        )
        exp_kg = float(np.exp(kappa_c * float(gamma_oper)))
        phi = float(c_tgt / max(exp_kg, 1e-9))
        nu = float(max(0.0, c_tgt - phi * exp_kg))
    else:
        phi = phi_struct
        nu = nu_struct
    F_cap = float(38.0 + 24.0 * ba + (10.0 if estado_duro else 0.0))
    p_cap = float(p_cap_base * (0.72 + 0.55 * ba))
    if estado_duro:
        p_cap = min(0.5, p_cap * 1.14)
        F_cap *= 1.06
    p_cap = float(min(0.5, max(0.01, p_cap)))

    # ── Garantía condición (v) Teorema 4.17 (implementabilidad) ─────────────
    # IR^K se cumple en (α=1, γ=1) ↔ U_rel ≥ U_kill ↔ kappa_rel ≤ |U_kill|.
    # El código ya garantiza U_kill < 0 para todos los tipos (ETA_INFRA_MULT).
    # Si kappa_rel > |U_kill|, ningún (α,γ)∈[0,1]² satisface IR^K y Γ_t = ∅.
    u_kill_check = float((1.0 - p_cap) * eta - p_cap * F_cap)
    if u_kill_check < -1e-9:
        # Techo: kappa_rel ≤ |U_kill| − 0.10  (margen estricto)
        kappa_rel_ceiling = float(abs(u_kill_check) - 0.10)
        if kappa_rel > kappa_rel_ceiling:
            kappa_rel = float(max(0.10, kappa_rel_ceiling))

    return {
        "kappa_rel": kappa_rel,
        "eta": eta,
        "F_cap": F_cap,
        "phi": phi,
        "kappa_c": kappa_c,
        "nu": nu,
        "p_cap": p_cap,
    }


def refresh_kidnapper_endogenous_columns(
    df_params: pd.DataFrame,
    modelo: ModeloSecuestro,
    presion_S: float,
    gamma: float,
    alpha: float = 0.0,
) -> pd.DataFrame:
    """
    Actualiza columnas endógenas por θ_K: h(Liberación|Pago) desde hazards;
    C(γ,θ) desde φ, κ_c, ν calibrados (filas del editor).
    """
    out = df_params.copy().reset_index(drop=True)
    for i in range(len(out)):
        tipo = str(out.at[i, "theta_K"])
        # maturity_mult=1.0: hazard estructural maduro (M=1), independiente de t.
        # El backward induction (Tabla 15) recomputa con el M_t correcto por período.
        h = modelo.calcular_hazards(1, tipo, presion_S, maturity_mult=1.0, alpha=alpha, gamma=gamma)
        p_pay = float(h["Pago"])
        phi = float(out.at[i, "phi"])
        kc = float(out.at[i, "kappa_c"])
        nu = float(out.at[i, "nu"])
        out.at[i, "h_LibPago"] = round(p_pay, 2)
        out.at[i, "C_gamma_theta"] = round(kidnapper_cost_c(gamma, phi, kc, nu), 2)
    return out


def kidnapper_u_kill_u_rel_from_tab12(
    row: pd.Series,
    p_cap_tilde: Optional[float] = None,
) -> Tuple[float, float]:
    """
    U^K_rel = -κ_rel; U^K_kill = (1-p̃_cap)η - p̃_cap F_cap (eq. kidnapper-kill).

    ``p_cap_tilde`` puede ser ``Epi_pcap_Qcap`` de Tabla 14; si es ``None``, se usa la fila Tabla 12.
    """
    pc = float(row["p_cap_tilde"] if p_cap_tilde is None else p_cap_tilde)
    pc = float(np.clip(pc, 0.0, 1.0))
    kr = float(row["kappa_rel"])
    et = float(row["eta"])
    fc = float(row["F_cap"])
    u_rel = float(-kr)
    u_kill = float((1.0 - pc) * et - pc * fc)
    return u_rel, u_kill


def kidnapper_r_escala_tab12_for_type(
    df_k_params: pd.DataFrame,
    tipo: str,
    R_fallback: float,
) -> float:
    """``R_escala`` de Tabla 12; si falta la fila, usa el ``R_fallback`` común."""
    th = str(tipo)
    if df_k_params is not None and not df_k_params.empty and "R_escala" in df_k_params.columns:
        sub = df_k_params[df_k_params["theta_K"].astype(str) == th]
        if not sub.empty:
            try:
                rv = float(sub.iloc[0]["R_escala"])
                if np.isfinite(rv) and rv > 0.0:
                    return float(rv)
            except (TypeError, ValueError, KeyError):
                pass
    return float(R_fallback)


def kidnapper_tab15_flow_rev_col9(
    p_pay: float,
    R_tab12: float,
    alpha_tab12: float,
) -> float:
    """Col. 9 Tabla 15: $\\tilde p_{pay} \\cdot R(\\theta^\\ast) \\cdot (1-\\alpha)$."""
    pp = float(np.clip(float(p_pay), 0.0, 1.0))
    r = float(max(1e-9, float(R_tab12)))
    a = float(np.clip(float(alpha_tab12), 0.0, 1.0))
    return float(pp * r * (1.0 - a))


def kidnapper_tab15_argmax_opcion_bw(
    u_kill: float,
    u_rel: float,
    v_cont: float,
) -> Tuple[str, float]:
    """
    Col.~14 Tabla 15: $a_K^\\ast=\\arg\\max\\{U_{\\mathrm{kill}},U_{\\mathrm{rel}},\\bar V_{\\mathrm{cont}}\\}$
    con $\\bar V_{\\mathrm{cont}}$ = col.~13 (suma cols.~9–12).
    """
    cand = (
        ("Matar (a_kill)", float(u_kill)),
        ("Liberar (a_rel)", float(u_rel)),
        ("Continuar (a_cont)", float(v_cont)),
    )
    return max(cand, key=lambda x: float(x[1]))


def kidnapper_r_escala_by_theta(
    df_k_params: pd.DataFrame,
    *,
    R_fallback: float,
    tipo_real: str,
    ransom_tab12: Optional[float] = None,
) -> Dict[str, float]:
    """
    ``R`` común por fila de Tabla 12 (``R_escala``).

    Para ``θ*`` (``tipo_real``), si se pasa ``ransom_tab12`` (col. 9 Tabla 15),
    ese valor tiene prioridad sobre el respaldo global ``R_fallback``.
    """
    out: Dict[str, float] = {}
    th_star = str(tipo_real)
    for th in TIPOS_SECUESTRADOR:
        sub = df_k_params[df_k_params["theta_K"].astype(str) == str(th)]
        rv = float("nan")
        if not sub.empty and "R_escala" in sub.columns:
            try:
                rv = float(sub.iloc[0]["R_escala"])
            except (TypeError, ValueError, KeyError):
                rv = float("nan")
        if str(th) == th_star and ransom_tab12 is not None:
            rv = float(ransom_tab12)
        if not np.isfinite(rv) or rv <= 0.0:
            rv = float(R_fallback)
        out[str(th)] = float(rv)
    return out


def kidnapper_branch_payoffs_from_tab12_row(
    row: pd.Series,
    modelo: ModeloSecuestro,
    *,
    t_hazard: int,
    presion_S: float,
    alpha: float,
    gamma: float,
    maturity_mult: float,
    R: float,
) -> Tuple[float, float, float]:
    """
    Utilidades corrientes del secuestrador por tipo (ramas Liberar/Matar/flujo continuación).

    Igualdad algebraica que **Tabla 12** / ``kidnapper_util_df_from_param_df`` —
    ecuaciones para ``U_rel``, ``U_kill`` y ``flow`` con columnas ``kappa_rel``, ``eta``,
    ``F_cap``, ``p_cap_tilde``, ``phi``, ``kappa_c``, ``nu`` de la tabla de parámetros;
    ``C(\\gamma,\\theta)=`` ``kidnapper_cost_c`` y ``\\tilde p_{pay}`` vía hazards del día
    ``t_hazard``, presión ``S`` y madurez ``maturity_mult``.
    """
    tipo = str(row["theta_K"])
    fc = float(row["F_cap"])
    pc = float(row["p_cap_tilde"])
    phi = float(row["phi"])
    kc = float(row["kappa_c"])
    nu = float(row["nu"])
    u_rel, u_kill = kidnapper_u_kill_u_rel_from_tab12(row, pc)
    cg = kidnapper_cost_c(gamma, phi, kc, nu)
    h = modelo.calcular_hazards(
        int(t_hazard),
        tipo,
        float(presion_S),
        maturity_mult=float(maturity_mult),
        alpha=float(alpha),
        gamma=float(gamma),
    )
    p_pay = float(h["Pago"])
    r_eff = float(row["R_escala"]) if "R_escala" in row.index else float(R)
    flow = float(p_pay * r_eff * (1.0 - alpha) - cg - pc * fc)
    return u_rel, u_kill, flow


def kidnapper_util_df_from_param_df(
    df_full: pd.DataFrame,
    _modelo: ModeloSecuestro,
    _presion_S: float,
    alpha: float,
    gamma: float,
    R: float,
    tipo_incidente: str,
    beta_k: float = 0.92,
) -> pd.DataFrame:
    """Utilidades por rama a partir de la tabla de parámetros (con h y C ya actualizados)."""
    util_rows: List[Dict[str, Any]] = []
    for _, row in df_full.iterrows():
        tipo = str(row["theta_K"])
        p_pay = float(row["h_LibPago"])
        Cg = float(row["C_gamma_theta"])
        pc = float(row["p_cap_tilde"])
        Fc = float(row["F_cap"])
        u_rel, u_kill = kidnapper_u_kill_u_rel_from_tab12(row, pc)
        r_eff = float(row["R_escala"]) if "R_escala" in row.index else float(R)
        flow = p_pay * r_eff * (1.0 - alpha) - Cg - pc * Fc
        beta_eff = float(row["beta_k"]) if "beta_k" in row.index else float(beta_k)
        v_cont = kidnapper_V_cont_branch(u_rel, u_kill, flow, beta_eff, pc)
        branches = [
            ("Liberar (a_rel)", u_rel),
            ("Matar (a_kill)", u_kill),
            ("Continuar (a_cont)", v_cont),
        ]
        best_name, _ = max(branches, key=lambda x: x[1])
        util_rows.append(
            {
                "theta_K": tipo,
                "U_rel": round(u_rel, 2),
                "U_kill": round(u_kill, 2),
                "V_cont": round(v_cont, 2),
                "rama_optima": best_name,
                "tipo_panel": "Si" if tipo == tipo_incidente else "-",
            }
        )
    return pd.DataFrame(util_rows)


def build_kidnapper_params_df(
    modelo: ModeloSecuestro,
    *,
    R_base: float,
    gamma_oper: float,
    p_cap_base: float,
    estado_duro: bool,
    beta_by_type: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    Tabla 12 sin ``calibrate_kidnapper_type_scales`` (carga rápida).
    Use ``apply_kidnapper_scale_calibration`` / «Recalibrar» para afinar escalas y col. 14.
    """
    rows: List[Dict[str, Any]] = []
    for tipo in TIPOS_SECUESTRADOR:
        r_eff = float(R_base)
        r_escala = float(r_eff)
        r_cost = float(r_eff)
        par = derive_kidnapper_structural_params(
            modelo,
            str(tipo),
            float(p_cap_base),
            bool(estado_duro),
            R_scale=float(r_cost),
            gamma_oper=float(gamma_oper),
            cost_frac_override=float(KIDNAPPER_COST_FRAC_OF_R.get(str(tipo), 0.10)),
            kappa_c_override=float(KIDNAPPER_KAPPA_C_BY_TYPE.get(str(tipo), 2.5)),
            krel_boost=float(KIDNAPPER_KREL_CAL_BOOST.get(str(tipo), 1.0)),
        )
        rows.append(
            {
                "theta_K": str(tipo),
                "kappa_rel": round(float(par["kappa_rel"]), 3),
                "eta": round(float(par["eta"]), 3),
                "F_cap": round(float(par["F_cap"]), 3),
                "phi": round(float(par["phi"]), 4),
                "kappa_c": round(float(par["kappa_c"]), 3),
                "nu": round(float(par["nu"]), 4),
                "p_cap_tilde": round(float(par["p_cap"]), 4),
                "R_escala": round(float(r_escala), 2),
                "beta_k": round(
                    float(
                        (beta_by_type or {}).get(
                            str(tipo),
                            (beta_by_type or {}).get(str(tipo).upper(), 0.92),
                        )
                    ),
                    4,
                ),
                "h_LibPago": 0.0,
                "C_gamma_theta": 0.0,
            }
        )
    return pd.DataFrame(rows)


def compute_kidnapper_by_type_tables(
    modelo: ModeloSecuestro,
    mu: Dict[str, float],
    presion_S: float,
    alpha: float,
    gamma: float,
    R: float,
    p_cap_base: float,
    estado_duro: bool,
    tipo_incidente: str,
    beta_k: float = 0.92,
) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    """
    Para cada θ_K ∈ Θ_K: parámetros que dependen del tipo, costo C(γ,θ), hazards propios
    y utilidades U_rel, U_kill, V_cont (rama continuar con término futuro, eq. kidnapper-cont).
    """
    h_mu = blend_hazards(modelo, mu, 1, presion_S)
    param_rows: List[Dict[str, Any]] = []
    util_rows: List[Dict[str, Any]] = []

    ransom_rel, cost_frac, kappa_map = calibrate_kidnapper_type_scales(
        modelo,
        R_base=float(R),
        gamma_lo=float(gamma),
        gamma_hi=float(min(0.95, float(gamma) + 0.38)),
        alpha=float(alpha),
        beta_k=float(beta_k),
        p_cap_base=float(p_cap_base),
        estado_duro=bool(estado_duro),
        presion_S=float(presion_S),
    )
    KIDNAPPER_RANSOM_REL.update(ransom_rel)
    KIDNAPPER_COST_FRAC_OF_R.update(cost_frac)
    KIDNAPPER_KAPPA_C_BY_TYPE.update(kappa_map)

    for tipo in TIPOS_SECUESTRADOR:
        r_eff = float(R)
        par = derive_kidnapper_structural_params(
            modelo,
            tipo,
            p_cap_base,
            estado_duro,
            R_scale=float(r_eff),
            gamma_oper=float(gamma),
            cost_frac_override=float(cost_frac.get(str(tipo), 0.10)),
            kappa_c_override=float(kappa_map.get(str(tipo), 2.5)),
            krel_boost=float(KIDNAPPER_KREL_CAL_BOOST.get(str(tipo), 1.0)),
        )
        p_pay = float(_kidnapper_calibration_p_pay(modelo, str(tipo)))
        Cg = kidnapper_cost_c(gamma, par["phi"], par["kappa_c"], par["nu"])
        pc = par["p_cap"]
        u_rel = -par["kappa_rel"]
        u_kill = (1.0 - pc) * par["eta"] - pc * par["F_cap"]
        flow = p_pay * r_eff * (1.0 - alpha) - Cg - pc * par["F_cap"]
        v_cont = kidnapper_V_cont_branch(u_rel, u_kill, flow, beta_k, pc)
        branches = [
            ("Liberar (a_rel)", u_rel),
            ("Matar (a_kill)", u_kill),
            ("Continuar (a_cont)", v_cont),
        ]
        best_name, _ = max(branches, key=lambda x: x[1])

        param_rows.append(
            {
                "theta_K": tipo,
                "kappa_rel": round(par["kappa_rel"], 2),
                "eta": round(par["eta"], 2),
                "F_cap": round(par["F_cap"], 2),
                "phi": round(par["phi"], 2),
                "kappa_c": round(par["kappa_c"], 2),
                "nu": round(par["nu"], 2),
                "p_cap_tilde": round(pc, 2),
                "h_LibPago": round(p_pay, 2),
                "C_gamma_theta": round(Cg, 2),
                "R_escala": round(float(r_eff), 2),
                "beta_k": round(float(beta_k), 4),
            }
        )
        util_rows.append(
            {
                "theta_K": tipo,
                "U_rel": round(u_rel, 2),
                "U_kill": round(u_kill, 2),
                "V_cont": round(v_cont, 2),
                "rama_optima": best_name,
                "tipo_panel": "Si" if tipo == tipo_incidente else "-",
            }
        )

    df_p = pd.DataFrame(param_rows)
    df_u = pd.DataFrame(util_rows)

    row_ui = next((r for r in util_rows if r["theta_K"] == tipo_incidente), util_rows[0])
    ha = float(h_mu["Muerte"])
    note = (
        r"Parámetros por $\theta_K$ ($\phi$, $\kappa_c$, $\nu$, $\kappa_{\mathrm{rel}}$, $\eta$, "
        r"$F_{\mathrm{cap}}$, $\tilde{p}_{\mathrm{cap}}$) desde betas y **Mechanism.tex** "
        r"(**cost-function-kidnapper**). "
        f"**Tipo panel:** {tipo_incidente} $\\to$ rama óptima: **{row_ui['rama_optima']}**. "
        fr"$V_{{\mathrm{{cont}}}}$ usa $\beta_K(1-\tilde{{p}}_{{\mathrm{{cap}}}})V^\ast$ "
        fr"(**kidnapper-cont**; $V^\ast$ por punto fijo). $\beta_K={beta_k:.2f}$. "
        fr"$S={presion_S:.2f}$; bajo $\mu$ mezcla, $h_{{\mathrm{{kill}}}}\approx {ha:.2f}$."
    )
    return df_p, df_u, note


def simulate_tau_K_sim(
    modelo: ModeloSecuestro,
    tipo: str,
    presion_S: float,
    alpha: float,
    gamma: float,
    R: float,
    p_cap_base: float,
    estado_duro: bool,
    beta_k: float = 0.92,
    rho: float = 0.04,
    T_max: int = 200,
    use_maturation: bool = False,
    alpha_drift: float = 0.0,
) -> Dict[str, Any]:
    """
    Simula τ_K^{sim} (Mechanism.tex, eq. tau-K-pure-simulation) para un tipo θ_K dado.

    En cada t ≥ 1 calcula V^K_{cont,t} y detiene en el primer t donde
    V_cont,t ≤ max(U_rel, U_kill).

    Parámetros:
        use_maturation : si True aplica maturation_filter(t, rho) a los hazards;
                         si False usa M_t = 1 (hazards en estado estacionario), que es
                         coherente con compute_kidnapper_by_type_tables.
        alpha_drift    : incremento de α por periodo (simula presión estatal creciente).
                         Permite obtener τ > 1 con parámetros realistas.

    Retorna:
        tau_sim   : primer t donde V_cont,t ≤ umbral  (T_max si no se detiene)
        winner    : "Liberar" | "Matar" | "No termina" | "Nunca empieza" (si V_cont < umbral ya en t=1)
        trajectory: DataFrame con t, alpha_t, p_pay, flow, V_cont, U_rel, U_kill, umbral, continua
    """
    par = derive_kidnapper_structural_params(modelo, tipo, p_cap_base, estado_duro)
    u_rel = -par["kappa_rel"]
    u_kill = (1.0 - par["p_cap"]) * par["eta"] - par["p_cap"] * par["F_cap"]
    umbral = max(u_rel, u_kill)
    Cg = kidnapper_cost_c(gamma, par["phi"], par["kappa_c"], par["nu"])

    tau_sim: int = T_max
    winner: str = "No termina"
    rows: List[Dict[str, Any]] = []
    started = False  # True en cuanto V_cont supera umbral al menos una vez

    for t in range(1, T_max + 1):
        M_t = maturation_filter(t, rho) if use_maturation else 1.0
        h = modelo.calcular_hazards(t, tipo, presion_S, maturity_mult=M_t)
        p_pay_t = float(h["Pago"])
        alpha_t = float(min(1.0, alpha + alpha_drift * (t - 1)))
        flow_t = p_pay_t * R * (1.0 - alpha_t) - Cg - par["p_cap"] * par["F_cap"]
        v_cont_t = kidnapper_V_cont_branch(u_rel, u_kill, flow_t, beta_k, par["p_cap"])

        continua = v_cont_t > umbral
        if continua:
            started = True

        rows.append({
            "t": t,
            "alpha_t": round(alpha_t, 4),
            "p_pay": round(p_pay_t, 4),
            "flow": round(flow_t, 4),
            "V_cont": round(v_cont_t, 4),
            "U_rel": round(u_rel, 4),
            "U_kill": round(u_kill, 4),
            "umbral": round(umbral, 4),
            "continua": continua,
        })

        # Solo detener cuando la continuación cae DESPUÉS de haber dominado
        if started and not continua:
            tau_sim = t
            winner = "Liberar" if u_rel >= u_kill else "Matar"
            break

    if not started:
        winner = "Nunca empieza"
        tau_sim = 1

    return {
        "tau_sim": tau_sim,
        "winner": winner,
        "U_rel": round(u_rel, 4),
        "U_kill": round(u_kill, 4),
        "umbral": round(umbral, 4),
        "trajectory": pd.DataFrame(rows),
    }


def compute_family_table(
    modelo: ModeloSecuestro,
    mu: Dict[str, float],
    presion_S: float,
    V_L: float,
    R: float,
    gamma: float,
    phi_F: float,
    kappa_F: float,
    nu_F: float,
    F_col: float,
    p_det_base: float,
    p_det_alpha: float,
    alpha: float,
    cmh_p_alive_closure: float,
    *,
    p_surv_mu_tab10: Optional[float] = None,
    p_det_logit_tab10: Optional[float] = None,
    f1_nested_triple: Optional[Tuple[float, float, float]] = None,
) -> Tuple[pd.DataFrame, str]:
    """Cooperación vs colusión; coherente con eq. family-utility-coop / family-utility-col.

    Overrides opcionales alineados a **Tabla 10** (misma sesión): ver ``family_calibrated_vs_endogenous``.
    Con ``f1_nested_triple=(p_rel, p_det_nested, p_s1)`` (app, **Tabla F.1**): utilidades con las tres
    esperanzas anidadas de **Mechanism.tex** (col / detección / coop); se ignoran ``p_surv_mu_tab10`` y la
    mezcla CMH en ``u_coop`` / ``u_col``.
    """
    if p_det_logit_tab10 is not None:
        p_det = float(max(0.0, min(0.99, float(p_det_logit_tab10))))
    else:
        p_det = min(0.99, p_det_base + p_det_alpha * alpha)
    h_mu = blend_hazards(
        modelo, mu, 1, presion_S, alpha=alpha, gamma=gamma, p_det=p_det
    )
    p_death = float(h_mu["Muerte"])
    p_surv_inst = max(0.0, min(1.0, 1.0 - p_death))
    if p_surv_mu_tab10 is not None:
        p_surv_inst = float(max(0.0, min(1.0, float(p_surv_mu_tab10))))
    p_surv_coop = float(0.5 * p_surv_inst + 0.5 * cmh_p_alive_closure)
    e_t = float(family_institutional_cost_e(gamma, phi_F, kappa_F, nu_F))
    if f1_nested_triple is not None:
        p_rel_e, p_det_e, p_s1_e = (
            float(f1_nested_triple[0]),
            float(f1_nested_triple[1]),
            float(f1_nested_triple[2]),
        )
        u_coop = float(p_s1_e) * V_L - e_t
        u_col = float(p_rel_e) * V_L - R - float(p_det_e) * F_col
    else:
        u_coop = p_surv_coop * V_L - e_t
        p_rel_col = cmh_p_alive_closure
        u_col = p_rel_col * V_L - R - p_det * F_col
    rows = [
        {"Rama": "Cooperar (a_coop)", "EU ilustrativa": round(u_coop, 2), "Ref.": "U_F_coop"},
        {"Rama": "Colusión (a_col)", "EU ilustrativa": round(u_col, 2), "Ref.": "U_F_col"},
    ]
    df = pd.DataFrame(rows)
    pref = "Cooperar (a_coop)" if u_coop >= u_col else "Colusión (a_col)"
    if f1_nested_triple is not None:
        prn, pde, psn = (
            float(f1_nested_triple[0]),
            float(f1_nested_triple[1]),
            float(f1_nested_triple[2]),
        )
        note = (
            fr"Bajo creencias $\mu$: P(asesinato|mezcla) $\approx$ **{p_death:.2f}**; "
            fr"$e(\gamma)={e_t:.2f}$. **Tabla F.1 (Mechanism):** "
            fr"$\mathbb{{E}}_{{\theta_K\mid\mathcal{{I}}^F}}\mathbb{{E}}_{{\tilde A\mid\mathcal{{Q}}^{{\mathrm{{Coop}}}}}}[P_E(s{{=}}1)]\approx {psn:.3f}$, "
            fr"$\mathbb{{E}}_{{\theta_K\mid\mathcal{{I}}^F}}\mathbb{{E}}_{{\tilde A\mid\mathcal{{Q}}^{{\mathrm{{Col}}}}}}[P_E(m{{=}}\mathrm{{rel}})]\approx {prn:.3f}$, "
            fr"$\mathbb{{E}}_{{\tilde a^F\mid\mathcal{{Q}}^F}}[P_E(d{{=}}1)]\approx {pde:.3f}$. "
            fr"Preferencia: **{pref}**."
        )
    else:
        note = (
            fr"Bajo creencias $\mu$ del tablero: P(asesinato|mezcla) $\approx$ **{p_death:.2f}**; "
            fr"$e(\gamma)={e_t:.2f}$. Preferencia ilustrativa: **{pref}**."
        )
    return df, note


def family_calibrated_vs_endogenous(
    modelo: ModeloSecuestro,
    mu: Dict[str, float],
    presion_S: float,
    V_L: float,
    R: float,
    gamma: float,
    phi_F: float,
    kappa_F: float,
    nu_F: float,
    F_col: float,
    p_det_base: float,
    p_det_alpha: float,
    alpha: float,
    cmh_p_alive_closure: float,
    *,
    p_surv_mu_tab10: Optional[float] = None,
    p_det_logit_tab10: Optional[float] = None,
    f1_nested_triple: Optional[Tuple[float, float, float]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Tablas separadas: parámetros calibrables vs objetos endógenos/inducidos (Familia).

    Incluye en ``df_cal`` las probabilidades que entran en **family-utility-coop/col**.
    Con ``p_surv_mu_tab10`` / ``p_det_logit_tab10`` (app, **Tabla 10**): ``\\sum_\\theta \\mu(\\theta)\\,p_{\\mathrm{surv},0}(\\theta)``
    y ``p_{\\mathrm{det},0}=\\Lambda(\\eta_0+\\eta_1\\alpha_0+\\eta_2\\gamma_0)`` en lugar del afín ``p_{\\det,0}+\\Delta\\,\\alpha``.
    Con ``f1_nested_triple``: ``df_cal`` incluye **parámetros** (``V_L``, ``R``, ``F_{\\mathrm{col}}``,
    ``\\phi_F``, ``\\kappa_F(\\theta_F)``, ``\\nu_F``, ``\\alpha_0``, ``\\gamma_0``)
    y a continuación las **tres** esperanzas anidadas de **Tabla F.1** (Mechanism.tex).
    La columna **Nivel** en ``df_cal`` es solo ``\"Calibrado\"`` o ``\"Calculado\"``.
    """
    if p_det_logit_tab10 is not None:
        p_det = float(max(0.0, min(0.99, float(p_det_logit_tab10))))
    else:
        p_det = min(0.99, p_det_base + p_det_alpha * alpha)
    h_mu = blend_hazards(
        modelo, mu, 1, presion_S, alpha=alpha, gamma=gamma, p_det=p_det
    )
    p_death = float(h_mu["Muerte"])
    p_surv_inst = max(0.0, min(1.0, 1.0 - p_death))
    if p_surv_mu_tab10 is not None:
        p_surv_inst = float(max(0.0, min(1.0, float(p_surv_mu_tab10))))
    p_surv_coop = float(0.5 * p_surv_inst + 0.5 * cmh_p_alive_closure)
    e_t = float(family_institutional_cost_e(gamma, phi_F, kappa_F, nu_F))
    p_rel_col = cmh_p_alive_closure
    _tab10 = p_surv_mu_tab10 is not None or p_det_logit_tab10 is not None
    if f1_nested_triple is not None:
        r_rel, r_det, r_s1 = (
            float(f1_nested_triple[0]),
            float(f1_nested_triple[1]),
            float(f1_nested_triple[2]),
        )
        param_rows: List[Dict[str, Any]] = [
            {
                "Parámetro": "V_L",
                "Valor": round(V_L, 2),
                "Nivel": "Calibrado",
            },
            {
                "Parámetro": "R",
                "Valor": round(R, 2),
                "Nivel": "Calibrado",
            },
            {
                "Parámetro": "F_col",
                "Valor": round(F_col, 2),
                "Nivel": "Calibrado",
            },
            {
                "Parámetro": "phi_F",
                "Valor": round(phi_F, 4),
                "Nivel": "Calibrado",
            },
            {
                "Parámetro": "kappa_F (θ_F actual)",
                "Valor": round(kappa_F, 4),
                "Nivel": "Calibrado",
            },
            {
                "Parámetro": "nu_F",
                "Valor": round(nu_F, 4),
                "Nivel": "Calibrado",
            },
            {
                "Parámetro": "alpha_0",
                "Valor": round(alpha, 2),
                "Nivel": "Calibrado",
            },
            {
                "Parámetro": "gamma_0",
                "Valor": round(gamma, 2),
                "Nivel": "Calibrado",
            },
        ]
        df_params = pd.DataFrame(param_rows)
        df_nested = pd.DataFrame(
            [
                {
                    "Parámetro": (
                        "E_{θ_K|I_t^F}[ E_{Ã_t|Q_t^Coop}[ P_E(s_t=1 | γ_t, Ã_t, θ_K) ] ]"
                    ),
                    "Valor": round(r_s1, 4),
                    "Nivel": "Calculado",
                },
                {
                    "Parámetro": (
                        "E_{θ_K|I_t^F}[ E_{Ã_t|Q_t^Col}[ P_E(m_t=rel | Ã_t, R, θ_K) ] ]"
                    ),
                    "Valor": round(r_rel, 4),
                    "Nivel": "Calculado",
                },
                {
                    "Parámetro": (
                        "E_{ã_t^F|Q_t^F}[ P_E(d_t=1 | α_t, ã_t^F, I_t^F) ]"
                    ),
                    "Valor": round(r_det, 4),
                    "Nivel": "Calculado",
                },
            ]
        )
        df_cal = pd.concat([df_params, df_nested], ignore_index=True)
    else:
        df_cal = pd.DataFrame(
            [
                {
                    "Parámetro": "V_L",
                    "Valor": round(V_L, 2),
                    "Nivel": "Calibrado",
                },
                {
                    "Parámetro": "R",
                    "Valor": round(R, 2),
                    "Nivel": "Calibrado",
                },
                {
                    "Parámetro": "F_col",
                    "Valor": round(F_col, 2),
                    "Nivel": "Calibrado",
                },
                {
                    "Parámetro": "phi_F",
                    "Valor": round(phi_F, 4),
                    "Nivel": "Calibrado",
                },
                {
                    "Parámetro": "kappa_F (θ_F actual)",
                    "Valor": round(kappa_F, 4),
                    "Nivel": "Calibrado",
                },
                {
                    "Parámetro": "nu_F",
                    "Valor": round(nu_F, 4),
                    "Nivel": "Calibrado",
                },
                {
                    "Parámetro": "alpha_0",
                    "Valor": round(alpha, 2),
                    "Nivel": "Calibrado",
                },
                {
                    "Parámetro": "gamma_0",
                    "Valor": round(gamma, 2),
                    "Nivel": "Calibrado",
                },
                {
                    "Parámetro": "E_thetaK_I_F_tilde_psurv",
                    "Valor": round(p_surv_coop, 4),
                    "Nivel": "Calculado",
                },
                {
                    "Parámetro": "E_thetaK_I_F_tilde_prel",
                    "Valor": round(p_rel_col, 4),
                    "Nivel": "Calculado",
                },
            ]
        )
    df_end = pd.DataFrame(
        [
            {
                "Objeto": "h_mean_kill (mu)",
                "Valor": round(p_death, 2),
                "Nivel": "Endógeno",
                "Nota": "Suma_theta mu(theta)*h_kill(theta); beta y mu",
            },
            {
                "Objeto": "P(surv|inst)",
                "Valor": round(p_surv_inst, 2),
                "Nivel": "Inducido",
                "Nota": "Tabla 10: E_mu[p_surv,0]" if _tab10 else "1 - h_mean_kill",
            },
            {
                "Objeto": "P_mixta(surv|coop)",
                "Valor": round(p_surv_coop, 2),
                "Nivel": "Endógeno / mix",
                "Nota": "0.5*E_mu[p_surv,0]+0.5*p_CMH" if _tab10 else "0.5*P(surv|inst)+0.5*p_CMH",
            },
            {
                "Objeto": "e(gamma)",
                "Valor": round(e_t, 2),
                "Nivel": "Inducido",
                "Nota": "phi_F, kappa_F, nu_F, gamma_0 (eq. family-et-exp-convex)",
            },
            {
                "Objeto": "p_det(alpha)",
                "Valor": round(p_det, 2),
                "Nivel": "Inducido",
                "Nota": "Tabla 10 logit (η, α₀, γ₀) si app pasa override; si no, afín rb",
            },
            {
                "Objeto": "P(rel|col) proxy",
                "Valor": round(p_rel_col, 2),
                "Nivel": "Endógeno (CMH)",
                "Nota": "Cierre con vida agregado",
            },
        ]
    )
    return df_cal, df_end


def compute_state_table(
    mu: Dict[str, float],
    modelo: ModeloSecuestro,
    presion_S: float,
    iota: float,
    omega_k: float,
    omega_p: float,
    omega_G: float,
    alpha: float,
    gamma: float,
    R: float,
    c_ops: Tuple[float, float, float],
    c_maint: Tuple[float, float, float],
    c_inst: Tuple[float, float, float],
    cmh_p_kill: float,
    cmh_p_surv_proxy: float,
    maturity_mult: float = 1.0,
) -> Tuple[pd.DataFrame, str]:
    """V^R vs V^N; eq. state-expected-loss, rescue-cost, negotiation-cost."""
    V_R, V_N, P_surv_rescue, p_kill_neg = compute_state_VR_VN(
        mu,
        modelo,
        presion_S,
        iota,
        omega_k,
        omega_p,
        omega_G,
        alpha,
        gamma,
        R,
        c_ops,
        c_maint,
        c_inst,
        cmh_p_kill,
        cmh_p_surv_proxy,
        maturity_mult=maturity_mult,
    )

    pref = "Rescate (a_res)" if V_R <= V_N else "Negociar (a_neg)"
    rows = [
        {"Rama": "Rescate", "Pérdida": round(V_R, 2), "Ref.": "V_t_R"},
        {"Rama": "Negociar", "Pérdida": round(V_N, 2), "Ref.": "V_t_N"},
    ]
    df = pd.DataFrame(rows)
    note = (
        f"Regla discreta: minimizar pérdida $\\to$ **{pref}**. "
        fr"$\iota={iota:.2f}$; supervivencia focal rescate $\approx$ **{P_surv_rescue:.2f}**. "
        fr"$P(\mathrm{{kill}})$ negociación $\approx$ **{p_kill_neg:.2f}**."
    )
    return df, note


def state_calibrated_vs_endogenous(
    mu: Dict[str, float],
    modelo: ModeloSecuestro,
    presion_S: float,
    iota: float,
    omega_k: float,
    omega_p: float,
    omega_G: float,
    alpha: float,
    gamma: float,
    R: float,
    c_ops: Tuple[float, float, float],
    c_maint: Tuple[float, float, float],
    c_inst: Tuple[float, float, float],
    cmh_p_kill: float,
    cmh_p_surv_proxy: float,
    maturity_mult: float = 1.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Pesos y costos calibrables vs probabilidades y objetivos inducidos (Estado)."""
    V_R, V_N, P_surv_rescue, p_kill_neg = compute_state_VR_VN(
        mu,
        modelo,
        presion_S,
        iota,
        omega_k,
        omega_p,
        omega_G,
        alpha,
        gamma,
        R,
        c_ops,
        c_maint,
        c_inst,
        cmh_p_kill,
        cmh_p_surv_proxy,
        maturity_mult=maturity_mult,
    )
    g_inst = c_inst[0] * alpha**2 + c_inst[1] * gamma**2 + c_inst[2] * alpha * gamma
    cops = quadratic_cost(gamma, *tuple(c_ops), alpha=alpha)
    cmaint = quadratic_cost(gamma, *tuple(c_maint), alpha=alpha)
    df_cal = pd.DataFrame(
        [
            {"Parámetro": "omega_k", "Valor": round(omega_k, 2), "Nivel": "Calibrado"},
            {"Parámetro": "omega_p", "Valor": round(omega_p, 2), "Nivel": "Calibrado"},
            {"Parámetro": "omega_G", "Valor": round(omega_G, 2), "Nivel": "Calibrado"},
            {"Parámetro": "c0 ops", "Valor": round(c_ops[0], 2), "Nivel": "Calibrado"},
            {"Parámetro": "c1 ops", "Valor": round(c_ops[1], 2), "Nivel": "Calibrado"},
            {"Parámetro": "c2 ops", "Valor": round(c_ops[2], 2), "Nivel": "Calibrado"},
            {"Parámetro": "m0 maint", "Valor": round(c_maint[0], 2), "Nivel": "Calibrado"},
            {"Parámetro": "m1 maint", "Valor": round(c_maint[1], 2), "Nivel": "Calibrado"},
            {"Parámetro": "m2 maint", "Valor": round(c_maint[2], 2), "Nivel": "Calibrado"},
            {"Parámetro": "c_alpha (G)", "Valor": round(c_inst[0], 2), "Nivel": "Calibrado"},
            {"Parámetro": "c_gamma (G)", "Valor": round(c_inst[1], 2), "Nivel": "Calibrado"},
            {"Parámetro": "c_alpha_gamma (G)", "Valor": round(c_inst[2], 2), "Nivel": "Calibrado"},
            {"Parámetro": "alpha (snapshot)", "Valor": round(alpha, 2), "Nivel": "Ilustrativo"},
            {"Parámetro": "gamma (snapshot)", "Valor": round(gamma, 2), "Nivel": "Ilustrativo"},
            {"Parámetro": "R (snapshot)", "Valor": round(R, 2), "Nivel": "Ilustrativo"},
            {"Parámetro": "iota", "Valor": round(iota, 2), "Nivel": "Panel config."},
        ]
    )
    df_end = pd.DataFrame(
        [
            {
                "Objeto": "P_surv rescate focal",
                "Valor": round(P_surv_rescue, 2),
                "Nivel": "Endógeno",
                "Nota": "iota y ancla CMH (ilustrativo)",
            },
            {
                "Objeto": "P_kill negociación",
                "Valor": round(p_kill_neg, 2),
                "Nivel": "Endógeno",
                "Nota": "Mezcla mu y CMH",
            },
            {"Objeto": "C_ops(gamma,alpha)", "Valor": round(cops, 2), "Nivel": "Inducido", "Nota": "c0..c5,gamma,alpha"},
            {"Objeto": "C_maint(gamma,alpha)", "Valor": round(cmaint, 2), "Nivel": "Inducido", "Nota": "m0..m5,gamma,alpha"},
            {"Objeto": "G(alpha,gamma)", "Valor": round(g_inst, 2), "Nivel": "Inducido", "Nota": "Costo institucional"},
            {"Objeto": "V_R", "Valor": round(V_R, 2), "Nivel": "Objetivo rama", "Nota": "Perdida rescate"},
            {"Objeto": "V_N", "Valor": round(V_N, 2), "Nivel": "Objetivo rama", "Nota": "Perdida negociar"},
        ]
    )
    return df_cal, df_end


def verify_ir_ic_snapshot(
    tipo_focal: str,
    df_util_k: pd.DataFrame,
    u_coop: float,
    u_col: float,
    V_R: float,
    V_N: float,
    reservation_k: float = -1e9,
    reservation_f: float = -1e9,
) -> pd.DataFrame:
    """
    Verificación de IR / IC conforme a Mechanism.tex (Def. implementable-mechanism, ec. ir-K, ir-K-def, ir-family).

    IR^K (eq. ir-K): DOS condiciones estrictas para θ_K focal:
        (1) U_rel(θ) > V_cont,t(θ, α*, γ*)
        (2) U_rel(θ) > U_kill(θ, γ*)
    IC^K: para cada par (θ_i, θ_j), V^K(a*(θ_i)|θ_i) ≥ V^K(a*(θ_j)|θ_i)
          — cada tipo prefiere su propia trayectoria a imitar la de otro.
    IR^F (eq. ir-family): U_coop ≥ U_col.
    IC^S: política del Estado elige argmin(V_R, V_N).
    """
    _util_key = {"Liberar (a_rel)": "U_rel", "Matar (a_kill)": "U_kill", "Continuar (a_cont)": "V_cont"}

    row = df_util_k[df_util_k["theta_K"] == tipo_focal]
    if row.empty:
        row = df_util_k.iloc[[0]]
    ur = float(row["U_rel"].iloc[0])
    uk = float(row["U_kill"].iloc[0])
    vc = float(row["V_cont"].iloc[0])
    rama_focal = str(row["rama_optima"].iloc[0])
    u_best_focal = max(ur, uk, vc)

    # IR^K — dos condiciones estrictas (Mechanism.tex eq. ir-K / ir-K-def)
    ir_k_cont = ur > vc        # U_rel > V_cont
    ir_k_kill = ur > uk        # U_rel > U_kill
    ir_k = ir_k_cont and ir_k_kill

    # IC^K — no-mimetismo: para cada θ_j, θ_focal prefiere su acción sobre a*(θ_j)
    # V^K(a*(θ_j)|θ_focal) = U_rel/U_kill/V_cont de θ_focal según la rama de θ_j
    util_map_focal = {"Liberar (a_rel)": ur, "Matar (a_kill)": uk, "Continuar (a_cont)": vc}
    ic_k_pairs: List[bool] = []
    for _, row_j in df_util_k.iterrows():
        tj = str(row_j["theta_K"])
        if tj == tipo_focal:
            continue
        rama_j = str(row_j["rama_optima"])
        v_mimic = util_map_focal.get(rama_j, vc)   # V^K(a*(θ_j) | θ_focal)
        ic_k_pairs.append(u_best_focal >= v_mimic - 1e-9)
    ic_k_all = all(ic_k_pairs) if ic_k_pairs else True

    # IR^F (eq. ir-family)
    ir_f = u_coop >= u_col

    # IC^F — familia elige su mejor respuesta
    pref_f = "Cooperar (a_coop)" if u_coop >= u_col else "Colusión (a_col)"
    ic_f = True  # por construcción: argmax es la mejor respuesta

    # IC^S — Estado elige argmin pérdida esperada
    ic_s = np.isfinite(V_R) and np.isfinite(V_N)
    s_rule = "Rescate" if V_R <= V_N else "Negociar"

    checks = [
        (
            "IR^K · cond.1 U_rel > V_cont",
            ir_k_cont,
            f"U_rel({tipo_focal}) = {ur:.3f} {'>' if ir_k_cont else '≤'} V_cont = {vc:.3f}  "
            "(Mechanism.tex eq. ir-K, primera línea)",
        ),
        (
            "IR^K · cond.2 U_rel > U_kill",
            ir_k_kill,
            f"U_rel({tipo_focal}) = {ur:.3f} {'>' if ir_k_kill else '≤'} U_kill = {uk:.3f}  "
            "(Mechanism.tex eq. ir-K, segunda línea)",
        ),
        (
            "IC^K · no-mimetismo ∀ θ_j",
            ic_k_all,
            f"V*({tipo_focal}) = {u_best_focal:.3f} ≥ V^K(a*(θ_j)|{tipo_focal}) para todo θ_j ≠ {tipo_focal}.  "
            "Rama focal: " + rama_focal + "  (Mechanism.tex, def. implementable-mechanism)",
        ),
        (
            "IR^F · cooperación preferida",
            ir_f,
            f"U_coop = {u_coop:.3f} {'≥' if ir_f else '<'} U_col = {u_col:.3f}  "
            "(Mechanism.tex eq. ir-family)",
        ),
        (
            "IC^F · argmax bien definido",
            ic_f,
            f"Familia elige {pref_f} (mejor respuesta dada la información I_t^F).",
        ),
        (
            "IC^S · política Estado = argmin",
            ic_s,
            f"V_R = {V_R:.3f}, V_N = {V_N:.3f} → a_S* = {s_rule}  "
            "(Mechanism.tex eq. state-discrete-rule).",
        ),
    ]
    return pd.DataFrame(
        [{"Restricción": a, "Cumple": "Sí ✅" if ok else "No ❌", "Nota": n} for a, ok, n in checks]
    )


# Mapeo desenlace observado → clave de ``DESENLACES`` / hazards.
_OUTCOME_TO_DESENLACE: Dict[str, str] = {
    "lib": "Liberación",
    "liberación": "Liberación",
    "liberacion": "Liberación",
    "liberar": "Liberación",
    "fuga o liberación": "Liberación",
    "res": "Rescate",
    "rescate": "Rescate",
    "pay": "Pago",
    "pago": "Pago",
    "kill": "Muerte",
    "muerte": "Muerte",
    "cont": "Continuar",
    "continuar": "Continuar",
}


def resolve_observed_desenlace(m_raw: str) -> str:
    """Normaliza etiqueta de $m_t$ (Pestaña 3 / historia $h_0$) a ``DESENLACES``."""
    s = str(m_raw or "").strip()
    if not s or s in ("—", "-", "nan"):
        return "Continuar"
    if s in DESENLACES:
        return s
    key = s.lower().replace("ó", "o").replace("í", "i")
    if key in _OUTCOME_TO_DESENLACE:
        return _OUTCOME_TO_DESENLACE[key]
    for alias, canon in _OUTCOME_TO_DESENLACE.items():
        if alias in key or key in alias:
            return canon
    return "Continuar"


_CAUSES_COMPETITIVE = ("Pago", "Muerte", "Rescate", "Exógeno")


def mechanism_competitive_hazards_at_t(
    modelo: ModeloSecuestro,
    theta: str,
    t: int,
    *,
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
    delta_t: float = 1.0,
    zeta_by_j: Optional[Dict[str, Dict[str, float]]] = None,
    atilde_F: Optional[str] = None,
    atilde_K: Optional[str] = None,
    atilde_S: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Factores de Tabla 10 / Mechanism.tex en el periodo ``t``:
    ``\\tilde{\\lambda}_j``, ``p_{Cont,t}=\\exp(-\\sum_j \\tilde{\\lambda}_j \\Delta t)``,
    ``h_j=q\\xi_j`` (eq. LH-cont, LH-out, LH-compacta).
    """
    t_mad_f = float(max(1e-9, t_mad))
    M_t = float(min(1.0, (float(t) / t_mad_f) ** 2)) if float(t) > 0 else 0.0
    h = modelo.calcular_hazards(
        int(t),
        str(theta),
        presion_S,
        maturity_mult=M_t,
        z_region=z_region,
        v_victim=v_victim,
        alpha=float(alpha),
        gamma=float(gamma),
        p_det=float(p_det),
        zeta_alpha=float(zeta_alpha),
        zeta_gamma=float(zeta_gamma),
        zeta_d=float(zeta_d),
        zeta_R=float(zeta_R),
        estado_rescata=bool(estado_rescata),
        zeta_by_j=zeta_by_j,
        atilde_F=atilde_F,
        atilde_K=atilde_K,
        atilde_S=atilde_S,
    )
    lam = {
        "Pago": float(max(0.0, h["Pago"])),
        "Muerte": float(max(0.0, h["Muerte"])),
        "Rescate": float(max(0.0, h["Rescate"])),
        "Exógeno": float(max(0.0, lambda4)),
    }
    lam_sum = float(sum(lam.values()))
    dt = float(max(0.0, delta_t))
    p_cont = float(np.exp(-lam_sum * dt))
    q = float(1.0 - p_cont)
    if lam_sum > 1e-12:
        xi = {j: float(lam[j] / lam_sum) for j in _CAUSES_COMPETITIVE}
    else:
        xi = {j: 0.25 for j in _CAUSES_COMPETITIVE}
    h_daily = {j: float(q * xi[j]) for j in _CAUSES_COMPETITIVE}
    # En Mechanism.tex la cuarta causa competitiva es el canal exógeno
    # (fatiga/fuga/liberación administrativa). En la app se observa como
    # "Liberación", por lo que su verosimilitud debe ser h_4=q*xi_4.
    h_daily["Liberación"] = float(h_daily.get("Exógeno", 0.0))
    h_daily["Continuar"] = p_cont
    return {
        "M_t": M_t,
        "lam": lam,
        "lam_sum": lam_sum,
        "p_cont": p_cont,
        "q": q,
        "xi": xi,
        "h_daily": h_daily,
        "h_raw": h,
    }


def mechanism_L_H_physical(m_obs: str, factors: Dict[str, Any]) -> float:
    """Bloque físico: eq. LH-cont si ``m=Continuar``; eq. LH-out si causa terminal."""
    m_canon = resolve_observed_desenlace(m_obs)
    if m_canon == "Continuar":
        return float(max(1e-15, factors["p_cont"]))
    if m_canon == "Liberación":
        return float(max(1e-15, factors["h_daily"].get("Liberación", 1e-15)))
    if m_canon in _CAUSES_COMPETITIVE:
        return float(max(1e-15, factors["h_daily"].get(m_canon, 1e-15)))
    return float(max(1e-15, factors["h_raw"].get(m_canon, 1e-15)))


def mechanism_L_F_joint(L_H: float, p_det: float, d_obs: int) -> Tuple[float, float]:
    """``\\mathcal{L}_F = \\mathcal{L}_H \\cdot \\mathcal{L}_d`` (eq. LH-joint)."""
    L_d = detection_likelihood_Ld(p_det, d_obs)
    return float(max(1e-300, L_H * L_d)), float(L_d)


def physical_likelihood_LF(
    modelo: ModeloSecuestro,
    theta: str,
    t: int,
    m_obs: str,
    *,
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
    t_mad: Optional[float] = None,
    lambda4: float = 0.002,
    delta_t: float = 1.0,
) -> Tuple[float, Dict[str, float]]:
    """Atajo: solo el factor físico ``\\mathcal{L}_H`` (sin ``\\mathcal{L}_d``)."""
    t_mad_v = float(t_mad if t_mad is not None else modelo.T_mad)
    factors = mechanism_competitive_hazards_at_t(
        modelo,
        theta,
        t,
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
        t_mad=t_mad_v,
        lambda4=lambda4,
        delta_t=delta_t,
    )
    L_H = mechanism_L_H_physical(m_obs, factors)
    h_out = dict(factors["h_daily"])
    h_out["Continuar"] = factors["p_cont"]
    return L_H, h_out


def detection_likelihood_Ld(p_det: float, d_obs: int) -> float:
    """``\\ProbE(d_t|α^*,γ^*)`` — eq. ``Ld-bernoulli`` (no depende de ``θ_K`` en la app)."""
    pd = float(max(1e-9, min(1.0 - 1e-9, float(p_det))))
    return float(pd if int(d_obs) == 1 else (1.0 - pd))


def communication_likelihood_LC(
    theta: str,
    *,
    V_t: Optional[int] = None,
    omega_voz: float = 0.0,
    pi_call: Any = 0.2,
    x_obs: Optional[np.ndarray] = None,
    voz_params_by_theta: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Tuple[float, float]:
    """
    ``\\mathcal{L}_{C,t}(\\theta_K\\mid V_t)`` (eq. LC) y factor ``\\mathcal{L}_{\\mathrm{voz},t}``
    (eq. Lvoz-diag si $V_t=1$). Si ``V_t`` no está informado y ``omega_voz>0``,
    se interpreta como silencio: ``V_t=0``.
    """
    if float(omega_voz) <= 1e-12:
        return 1.0, 1.0
    if V_t is None:
        V_t = 0
    pi = _pi_call_for_theta(theta, pi_call)
    w = float(np.clip(float(omega_voz), 0.0, 1.0))
    if int(V_t) == 0:
        return float((1.0 - pi) ** w), 1.0
    if x_obs is None or voz_params_by_theta is None:
        return float((pi) ** w), 1.0
    l_voz = Lvoz_diagonal_likelihood(x_obs, theta, voz_params_by_theta)
    l_c = float(max(1e-300, (l_voz * pi) ** w))
    return l_c, float(l_voz)


def bayesian_posterior_update(
    mu_t: Dict[str, float],
    likelihood_by_theta: Dict[str, float],
) -> Tuple[Dict[str, float], float]:
    """``μ_{t+1}(θ) ∝ μ_t(θ) · \\mathcal{L}_t(θ)`` con normalización (eq. ``bayes-unif``)."""
    numer: Dict[str, float] = {}
    denom = 0.0
    for theta in TIPOS_SECUESTRADOR:
        w = float(mu_t.get(theta, 0.0))
        lk = float(max(1e-300, likelihood_by_theta.get(theta, 1e-300)))
        v = w * lk
        numer[theta] = v
        denom += v
    if denom <= 1e-15:
        return {t: float(mu_t.get(t, 0.0)) for t in TIPOS_SECUESTRADOR}, 0.0
    return {t: float(numer[t] / denom) for t in TIPOS_SECUESTRADOR}, float(denom)


def build_t0_bayesian_posterior_report(
    modelo: ModeloSecuestro,
    mu_0: Dict[str, float],
    m_obs: str,
    d_obs: int,
    *,
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
    lambda4: float = 0.002,
    t_eval: int = 0,
    omega_voz: float = 0.0,
    V_t: Optional[int] = None,
    x_obs: Optional[np.ndarray] = None,
    pi_call_by_theta: Optional[Dict[str, float]] = None,
    voz_params_by_theta: Optional[Dict[str, Dict[str, Any]]] = None,
    delta_t: float = 1.0,
    tab2_bundle_by_theta: Optional[Dict[str, Dict[str, Any]]] = None,
    atilde_F: Optional[str] = None,
    atilde_K: Optional[str] = None,
    atilde_S: Optional[str] = None,
    implementation_likelihood_by_theta: Optional[Dict[str, float]] = None,
    aggregate_unknown_theta: bool = True,
    aggregate_lc_unknown_theta: Optional[bool] = None,
) -> Tuple[pd.DataFrame, Dict[str, float], Dict[str, Any]]:
    """
    Paso Bayes en ``t_eval`` (Mechanism.tex): ``\\mathcal{L}_H=\\mathbb{P}(m_t\\mid\\cdot)`` (LH-compacta),
    ``\\mathcal{L}_d=\\mathbb{P}(d_t\\mid\\alpha^\\ast,\\gamma^\\ast)`` (Ld-bernoulli),
    ``\\mathcal{L}_{F,t}=\\mathcal{L}_{I,t}\\mathcal{L}_H\\mathcal{L}_d`` (LH-joint), ``\\mathcal{L}_{C,t}`` (LC);
    numerador Bayes $=\\mu_t(\\theta)\\,\\mathcal{L}_{F,t}\\,\\mathcal{L}_{C,t}$.
    """
    m_canon = resolve_observed_desenlace(m_obs)
    aggregate_lc = bool(aggregate_unknown_theta if aggregate_lc_unknown_theta is None else aggregate_lc_unknown_theta)
    theta_hat = max(mu_0, key=lambda k: float(mu_0.get(k, 0.0)))
    iota_mode = float(max(mu_0.values()) if mu_0 else 0.0)

    factor_by_theta: Dict[str, Dict[str, Any]] = {}
    for theta in TIPOS_SECUESTRADOR:
        modelo_t = copy.deepcopy(modelo)
        _bundle = (tab2_bundle_by_theta or {}).get(str(theta)) or {}
        if isinstance(_bundle.get("betas"), dict):
            modelo_t.betas[str(theta)].update(_bundle["betas"])
        if isinstance(_bundle.get("lambdas_0"), dict):
            modelo_t.lambdas_0.update(_bundle["lambdas_0"])
        _zeta_bj = _bundle.get("zeta_by_j") if isinstance(_bundle.get("zeta_by_j"), dict) else None
        factors = mechanism_competitive_hazards_at_t(
            modelo_t,
            theta,
            int(t_eval),
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
            t_mad=float(t_mad),
            lambda4=float(lambda4),
            delta_t=float(delta_t),
            zeta_by_j=_zeta_bj,
            atilde_F=atilde_F,
            atilde_K=atilde_K,
            atilde_S=atilde_S,
        )
        factor_by_theta[theta] = factors

    mu0_sum = float(sum(max(0.0, float(mu_0.get(theta, 0.0))) for theta in TIPOS_SECUESTRADOR))
    mu0_norm = {
        theta: (
            float(max(0.0, float(mu_0.get(theta, 0.0)))) / mu0_sum
            if mu0_sum > 1e-12
            else 1.0 / len(TIPOS_SECUESTRADOR)
        )
        for theta in TIPOS_SECUESTRADOR
    }
    expected_p_cont_mu0 = float(
        sum(
            mu0_norm[theta] * float(factor_by_theta[theta]["p_cont"])
            for theta in TIPOS_SECUESTRADOR
        )
    )
    expected_silence_mu0 = float(
        sum(
            mu0_norm[theta]
            * (1.0 - _pi_call_for_theta(theta, pi_call_by_theta if pi_call_by_theta is not None else {}))
            for theta in TIPOS_SECUESTRADOR
        )
    )
    expected_lc_silence_mu0 = float(
        expected_silence_mu0 ** float(np.clip(float(omega_voz), 0.0, 1.0))
        if float(omega_voz) > 1e-12
        else 1.0
    )
    use_expected_silence = bool(
        aggregate_lc
        and float(omega_voz) > 1e-12
        and (V_t is None or int(V_t) == 0)
    )

    rows: List[Dict[str, Any]] = []
    lk_total: Dict[str, float] = {}
    for theta in TIPOS_SECUESTRADOR:
        factors = factor_by_theta[theta]
        L_H = mechanism_L_H_physical(m_canon, factors)
        if aggregate_unknown_theta and m_canon == "Continuar":
            L_H = float(max(1e-15, expected_p_cont_mu0))
        L_F_base, L_d = mechanism_L_F_joint(L_H, p_det, d_obs)
        L_I = float(
            max(
                1e-300,
                (
                    implementation_likelihood_by_theta or {}
                ).get(theta, 1.0),
            )
        )
        L_F = float(L_I * L_F_base)
        L_C, L_voz = communication_likelihood_LC(
            theta,
            V_t=V_t,
            omega_voz=omega_voz,
            pi_call=pi_call_by_theta if pi_call_by_theta is not None else {},
            x_obs=x_obs,
            voz_params_by_theta=voz_params_by_theta,
        )
        if use_expected_silence:
            L_C = float(max(1e-300, expected_lc_silence_mu0))
        lk = float(L_F * L_C)
        lk_total[theta] = lk
        hd = factors["h_daily"]
        lam = factors["lam"]
        rows.append(
            {
                "theta_K": theta,
                "M_t": round(float(factors["M_t"]), 4),
                "mu_0": round(float(mu_0.get(theta, 0.0)), 4),
                "lam_1": round(float(lam["Pago"]), 6),
                "lam_2": round(float(lam["Muerte"]), 6),
                "lam_3": round(float(lam["Rescate"]), 6),
                "lam_4": round(float(lam["Exógeno"]), 6),
                "sum_lam": round(float(factors["lam_sum"]), 6),
                "L_H_cont": round(float(factors["p_cont"]), 6),
                "L_H_out_blk": round(
                    float(sum(hd.get(j, 0.0) for j in _CAUSES_COMPETITIVE)),
                    6,
                ),
                "L_H": round(L_H, 6),
                "L_I": round(L_I, 6),
                "L_d": round(L_d, 4),
                "L_voz": round(float(L_voz), 6),
                "L_C": round(L_C, 6),
                "L_F": round(L_F, 6),
                "L_total": round(lk, 6),
                "numerador": round(float(mu_0.get(theta, 0.0)) * lk, 6),
                "p_cont": round(float(factors["p_cont"]), 6),
                "q_daily": round(float(factors["q"]), 6),
                "h_Pago": round(float(hd.get("Pago", 0.0)), 4),
                "h_Muerte": round(float(hd.get("Muerte", 0.0)), 4),
                "h_Rescate": round(float(hd.get("Rescate", 0.0)), 4),
                "h_Liberacion": round(float(hd.get("Liberación", 0.0)), 4),
                "h_Continuar": round(float(factors["p_cont"]), 4),
            }
        )

    mu_1, denom = bayesian_posterior_update(mu_0, lk_total)
    for row in rows:
        row["mu_1"] = round(float(mu_1.get(row["theta_K"], 0.0)), 4)
        row["denom_Z"] = round(float(denom), 6)

    meta = {
        "m_obs": m_canon,
        "d_obs": int(d_obs),
        "p_det": float(p_det),
        "theta_hat": str(theta_hat),
        "iota_mode": float(iota_mode),
        "denom": float(denom),
        "t_eval": int(t_eval),
        "L_d_common": float(detection_likelihood_Ld(p_det, d_obs)),
        "omega_voz": float(omega_voz),
        "alpha": float(alpha),
        "gamma": float(gamma),
        "lambda4": float(lambda4),
        "expected_p_cont_mu0": float(expected_p_cont_mu0),
        "LH_uses_expected_p_cont": bool(aggregate_unknown_theta and m_canon == "Continuar"),
        "expected_silence_mu0": float(expected_silence_mu0),
        "expected_lc_silence_mu0": float(expected_lc_silence_mu0),
        "LC_uses_expected_silence": bool(use_expected_silence),
        "uses_implementation_likelihood": bool(implementation_likelihood_by_theta),
        "t_mad": float(t_mad),
        "V_t": int(V_t) if V_t is not None else None,
        "omega_voz": float(omega_voz),
    }
    return pd.DataFrame(rows), mu_1, meta


def format_belief_update_display_df(
    df: pd.DataFrame,
    *,
    include_hazards: bool = True,
) -> pd.DataFrame:
    """Columnas legibles para la tabla «Actualización de creencias» (eq. bayes-unif)."""
    if df is None or df.empty:
        return pd.DataFrame()
    cols = ["theta_K"]
    if include_hazards:
        cols += [
            c
            for c in ("M_t", "lam_1", "lam_2", "lam_3", "lam_4", "sum_lam", "p_cont")
            if c in df.columns
        ]
    cols += [
        c
        for c in (
            "mu_0",
            "L_H",
            "L_I",
            "L_d",
            "L_F",
            "L_C",
            "L_total",
            "numerador",
            "mu_1",
        )
        if c in df.columns
    ]
    out = df[[c for c in cols if c in df.columns]].copy()
    rename = {
        "theta_K": "θ_K",
        "M_t": "M(t)",
        "mu_0": "μ_t(θ)",
        "lam_1": "λ̃₁ (Pago)",
        "lam_2": "λ̃₂ (Muerte)",
        "lam_3": "λ̃₃ (Rescate)",
        "lam_4": "λ̃₄ (Exog.)",
        "sum_lam": "Σ_j λ̃_j",
        "p_cont": "p_{Cont,t}",
        "L_H": "ℒ_H",
        "L_I": "ℒ_{I,t}",
        "L_d": "ℒ_d",
        "L_F": "ℒ_{F,t}",
        "L_C": "ℒ_{C,t}",
        "L_total": "ℒ_F·ℒ_C",
        "numerador": "μ_t·ℒ",
        "mu_1": "μ_{t+1}",
    }
    return out.rename(columns={k: v for k, v in rename.items() if k in out.columns})


def summarize_bayes_likelihood_step(
    df_step: pd.DataFrame,
    meta: Dict[str, Any],
    *,
    theta_focus: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Resumen por periodo de las verosimilitudes en bayes-unif:
    $\\mu_{t+1}(\\theta)\\propto\\mu_t(\\theta)\\,\\mathcal{L}_{F,t}(\\theta)\\,\\mathcal{L}_{C,t}(\\theta)$,
    con $\\mathcal{L}_{F,t}=\\mathcal{L}_{I,t}\\mathcal{L}_H\\mathcal{L}_d$.
    """
    if df_step is None or df_step.empty:
        return {}
    mu_prior = {str(r["theta_K"]): float(r["mu_0"]) for _, r in df_step.iterrows()}
    l_h = {str(r["theta_K"]): float(r["L_H"]) for _, r in df_step.iterrows()}
    l_i = {str(r["theta_K"]): float(r.get("L_I", 1.0)) for _, r in df_step.iterrows()}
    l_f = {str(r["theta_K"]): float(r["L_F"]) for _, r in df_step.iterrows()}
    l_c = {str(r["theta_K"]): float(r["L_C"]) for _, r in df_step.iterrows()}
    l_voz = {str(r["theta_K"]): float(r.get("L_voz", 1.0)) for _, r in df_step.iterrows()}
    l_bayes = {str(r["theta_K"]): float(r["L_total"]) for _, r in df_step.iterrows()}
    hat_prior = max(mu_prior, key=lambda k: float(mu_prior.get(k, 0.0)))
    focus = str(theta_focus) if theta_focus else str(hat_prior)
    if focus not in mu_prior:
        focus = str(hat_prior)
    l_d = float(meta.get("L_d_common", df_step["L_d"].iloc[0]))
    v_t = meta.get("V_t")
    sub_focus = df_step[df_step["theta_K"].astype(str) == str(focus)]
    p_cont_hat = (
        float(sub_focus["p_cont"].iloc[0])
        if not sub_focus.empty and "p_cont" in sub_focus.columns
        else None
    )
    lh_cont_hat = (
        float(sub_focus["L_H_cont"].iloc[0])
        if not sub_focus.empty and "L_H_cont" in sub_focus.columns
        else p_cont_hat
    )
    lh_out_hat = (
        float(sub_focus["L_H_out_blk"].iloc[0])
        if not sub_focus.empty and "L_H_out_blk" in sub_focus.columns
        else (1.0 - float(lh_cont_hat) if lh_cont_hat is not None else None)
    )
    l_voz_show: Any = "—"
    if v_t is not None and int(v_t) == 1:
        l_voz_show = round(float(l_voz.get(focus, 1.0)), 6)
    _m_fac: Any = "—"
    if "M_t" in df_step.columns and not df_step.empty:
        try:
            _m_fac = round(float(df_step["M_t"].iloc[0]), 6)
        except (TypeError, ValueError):
            _m_fac = "—"
    lh_report = (
        float(meta.get("expected_p_cont_mu0", 0.0))
        if bool(meta.get("LH_uses_expected_p_cont", False))
        else float(l_h.get(focus, 0.0))
    )
    lc_report = (
        float(meta.get("expected_lc_silence_mu0", 1.0))
        if bool(meta.get("LC_uses_expected_silence", False))
        else float(l_c.get(focus, 0.0))
    )
    li_report = float(l_i.get(focus, 1.0))
    lf_report = float(max(1e-300, li_report * lh_report * l_d))
    l_bayes_report = float(max(1e-300, lf_report * lc_report))
    if bool(meta.get("LH_uses_expected_p_cont", False)):
        lh_cont_hat = lh_report
        lh_out_hat = max(0.0, 1.0 - lh_report)
    out: Dict[str, Any] = {
        "V_t": int(v_t) if v_t is not None else "—",
        "M_t": _m_fac,
        "L_H_cont": round(float(lh_cont_hat), 6)
        if lh_cont_hat is not None
        else "—",
        "q_comp": round(float(lh_out_hat), 6)
        if lh_out_hat is not None
        else "—",
        "L_H": round(float(lh_report), 6),
        "L_I": round(float(li_report), 6),
        "L_d": round(l_d, 4),
        "L_voz": l_voz_show,
        "L_F": round(float(lf_report), 6),
        "L_C": round(float(lc_report), 6),
        "L_bayes": round(float(l_bayes_report), 6),
        "Z_t": round(float(meta.get("denom", 0.0)), 6),
        "L_H_mu": round(sum(float(mu_prior[th]) * float(l_h[th]) for th in mu_prior), 6),
        "L_F_mu": round(sum(float(mu_prior[th]) * float(l_f[th]) for th in mu_prior), 6),
        "L_bayes_mu": round(
            sum(float(mu_prior[th]) * float(l_bayes[th]) for th in mu_prior), 6
        ),
    }
    if p_cont_hat is not None:
        out["p_cont"] = round(p_cont_hat, 6)
    return out


def build_mechanism_mu_trajectory(
    modelo: ModeloSecuestro,
    mu_0: Dict[str, float],
    *,
    t_max: int = 10,
    m_obs: str = "Continuar",
    d_obs: int = 0,
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
    lambda4: float = 0.002,
    omega_voz: float = 0.0,
    V_t: Optional[int] = None,
    delta_t: float = 1.0,
    continuation_path: bool = False,
    alpha_by_t: Optional[Sequence[float]] = None,
    gamma_by_t: Optional[Sequence[float]] = None,
    p_det_by_t: Optional[Sequence[float]] = None,
    pi_call_by_theta: Optional[Dict[str, float]] = None,
    voz_params_by_theta: Optional[Dict[str, Dict[str, Any]]] = None,
    voice_seed: Optional[int] = None,
    tipo_emit_voz: Optional[str] = None,
    voice_emit_from_mu: bool = True,
    voice_path: Optional[Sequence[Dict[str, Any]]] = None,
    voice_theta_focus: Optional[str] = None,
    tab2_bundle_by_theta: Optional[Dict[str, Dict[str, Any]]] = None,
    atilde_F: Optional[str] = None,
    atilde_K: Optional[str] = None,
    atilde_S: Optional[str] = None,
    implementation_likelihood_by_theta: Optional[Dict[str, float]] = None,
) -> Tuple[pd.DataFrame, List[pd.DataFrame]]:
    """
    Trayectoria $\\mu_0,\\ldots,\\mu_{t_{\\max}}$ (eq. bayes-unif).

    Si ``continuation_path=True`` (Tabla 14 / eq. kidnapper-cont), cada paso usa
    $m_t=\\mathrm{Cont}$ y $\\mu_{t+1}(a_{\\mathrm{cont}})$ con $\\mathcal{L}_H^{\\mathrm{cont}}$
    (LH-cont), coherente con el término futuro de Bellman del secuestrador.

    Si ``voice_path`` (Simulación e Incidente), $(V_t,x_t^{obs})$ provienen del incidente;
    las verosimilitudes de voz en la tabla resumen usan ``voice_theta_focus``
    ($\\theta^\\ast$) si se indica (sin columna explícita de tipo).
    """
    t_max = int(max(0, t_max))
    mu = {k: float(mu_0.get(k, 0.0)) for k in TIPOS_SECUESTRADOR}
    m_canon = "Continuar" if continuation_path else resolve_observed_desenlace(m_obs)
    rng_voz = (
        np.random.default_rng(int(voice_seed))
        if voice_seed is not None
        else np.random.default_rng()
    )
    _use_voz = (
        float(omega_voz) > 1e-12
        and voz_params_by_theta is not None
        and pi_call_by_theta is not None
    )
    # ── τ=0: compute likelihoods from initial conditions (V_t=0) ────────────
    _alpha0_r = float(alpha_by_t[0]) if alpha_by_t is not None and len(alpha_by_t) > 0 else float(alpha)
    _gamma0_r = float(gamma_by_t[0]) if gamma_by_t is not None and len(gamma_by_t) > 0 else float(gamma)
    _pdet0_r = float(p_det_by_t[0]) if p_det_by_t is not None and len(p_det_by_t) > 0 else float(p_det)
    try:
        _df_s0, _, _meta_r0 = build_t0_bayesian_posterior_report(
            modelo,
            dict(mu),
            m_canon,
            int(d_obs),
            presion_S=presion_S,
            z_region=z_region,
            v_victim=v_victim,
            alpha=_alpha0_r,
            gamma=_gamma0_r,
            p_det=_pdet0_r,
            zeta_alpha=zeta_alpha,
            zeta_gamma=zeta_gamma,
            zeta_d=zeta_d,
            zeta_R=zeta_R,
            estado_rescata=estado_rescata,
            t_mad=t_mad,
            lambda4=lambda4,
            omega_voz=omega_voz,
            V_t=0,
            pi_call_by_theta=pi_call_by_theta,
            voz_params_by_theta=voz_params_by_theta,
            t_eval=0,
            tab2_bundle_by_theta=tab2_bundle_by_theta,
            atilde_F=atilde_F,
            atilde_K=atilde_K,
            atilde_S=atilde_S,
            implementation_likelihood_by_theta=implementation_likelihood_by_theta,
            aggregate_unknown_theta=False,
            aggregate_lc_unknown_theta=True,
        )
        _lik0: Dict[str, Any] = summarize_bayes_likelihood_step(
            _df_s0, _meta_r0, theta_focus=voice_theta_focus
        )
        # Solo p_cont se pondera por μ₀; las demás verosimilitudes son per-θ* (summarize)
        _lh_w = float(_lik0.get("L_H_mu", _lik0.get("L_H") or 0.0))
        _lik0["L_H_cont"] = round(_lh_w, 6)
        _lik0["p_cont"] = round(_lh_w, 6)
        _lik0["q_comp"] = round(max(0.0, 1.0 - _lh_w), 6)
        _lik0["V_t"] = 0
        _lik0.setdefault("emisor_voz", "—")
    except Exception:
        _lik0 = {
            "V_t": 0,
            "emisor_voz": "—",
            "M_t": None,
            "L_H_cont": None,
            "q_comp": None,
            "L_H": None,
            "L_d": None,
            "L_voz": None,
            "L_F": None,
            "L_C": None,
            "L_bayes": None,
            "Z_t": None,
            "p_cont": None,
        }
    rows: List[Dict[str, Any]] = [
        {
            "t": 0,
            **{f"mu_{th}": round(mu[th], 4) for th in TIPOS_SECUESTRADOR},
            "m_t": "Continuar",
            "d_t": int(d_obs),
            "omega_voz": round(float(omega_voz), 4),
            "alpha_t": round(_alpha0_r, 4),
            "gamma_t": round(_gamma0_r, 4),
            "p_det_t": round(_pdet0_r, 4),
            "iota": round(float(max(mu.values()) if mu else 0.0), 4),
            "hat_theta": max(mu, key=lambda k: float(mu.get(k, 0.0))),
            **_lik0,
        }
    ]
    step_dfs: List[pd.DataFrame] = []
    for t in range(t_max):
        mu_prior_t = dict(mu)
        alpha_t = float(alpha_by_t[t]) if alpha_by_t is not None and t < len(alpha_by_t) else float(alpha)
        gamma_t = float(gamma_by_t[t]) if gamma_by_t is not None and t < len(gamma_by_t) else float(gamma)
        p_det_t = (
            float(p_det_by_t[t])
            if p_det_by_t is not None and t < len(p_det_by_t)
            else float(p_det)
        )
        V_step: Optional[int] = None
        x_step: Optional[np.ndarray] = None
        emisor = "—"
        if voice_path is not None:
            V_step, x_step, emisor = voice_path_step_at_t(voice_path, int(t))
        elif _use_voz:
            if voice_emit_from_mu and tipo_emit_voz is None:
                w_mu = np.array([float(mu.get(th, 0.0)) for th in TIPOS_SECUESTRADOR], dtype=float)
                if w_mu.sum() > 1e-15:
                    w_mu = w_mu / w_mu.sum()
                else:
                    w_mu = np.ones(len(TIPOS_SECUESTRADOR)) / len(TIPOS_SECUESTRADOR)
                emisor = str(rng_voz.choice(TIPOS_SECUESTRADOR, p=w_mu))
            else:
                emisor = str(tipo_emit_voz or TIPOS_SECUESTRADOR[0])
            pi_emit = _pi_call_for_theta(emisor, pi_call_by_theta)
            V_step = draw_voice_indicator(pi_emit, rng_voz)
            if int(V_step) == 1:
                x_step = sample_voice_observation(emisor, voz_params_by_theta, rng_voz)

        df_step, mu, meta = build_t0_bayesian_posterior_report(
            modelo,
            mu_prior_t,
            m_canon,
            int(d_obs),
            presion_S=presion_S,
            z_region=z_region,
            v_victim=v_victim,
            alpha=alpha_t,
            gamma=gamma_t,
            p_det=p_det_t,
            zeta_alpha=zeta_alpha,
            zeta_gamma=zeta_gamma,
            zeta_d=zeta_d,
            zeta_R=zeta_R,
            estado_rescata=estado_rescata,
            t_mad=t_mad,
            lambda4=lambda4,
            omega_voz=omega_voz,
            V_t=V_step,
            x_obs=x_step,
            pi_call_by_theta=pi_call_by_theta,
            voz_params_by_theta=voz_params_by_theta,
            delta_t=delta_t,
            t_eval=int(t),
            tab2_bundle_by_theta=tab2_bundle_by_theta,
            atilde_F=atilde_F,
            atilde_K=atilde_K,
            atilde_S=atilde_S,
            implementation_likelihood_by_theta=implementation_likelihood_by_theta,
            aggregate_unknown_theta=False,
            aggregate_lc_unknown_theta=True,
        )
        lk_by_theta_t = {
            str(r["theta_K"]): float(r["L_total"])
            for _, r in df_step.iterrows()
        }
        mu, denom_t = bayesian_posterior_update(mu_prior_t, lk_by_theta_t)
        meta["denom"] = float(denom_t)
        meta["emisor_voz"] = emisor
        step_dfs.append(df_step)
        _focus = voice_theta_focus
        if _focus is None and emisor not in ("—", ""):
            _focus = emisor
        lik = summarize_bayes_likelihood_step(df_step, meta, theta_focus=_focus)
        lik["emisor_voz"] = emisor
        lik_by_type_cols = {
            f"L_bayes_{th}": round(float(lk_by_theta_t.get(th, 0.0)), 6)
            for th in TIPOS_SECUESTRADOR
        }
        rows[-1].update(
            {
                "m_t": str(meta.get("m_obs", m_canon)),
                "d_t": int(meta.get("d_obs", d_obs)),
                "omega_voz": round(float(omega_voz), 4),
                "alpha_t": round(alpha_t, 4),
                "gamma_t": round(gamma_t, 4),
                "p_det_t": round(p_det_t, 4),
                "iota": round(float(max(mu_prior_t.values()) if mu_prior_t else 0.0), 4),
                "hat_theta": max(mu_prior_t, key=lambda k: float(mu_prior_t.get(k, 0.0))),
                **lik,
                **lik_by_type_cols,
            }
        )
        next_t = int(t + 1)
        alpha_next = (
            float(alpha_by_t[next_t])
            if alpha_by_t is not None and next_t < len(alpha_by_t)
            else float(alpha)
        )
        gamma_next = (
            float(gamma_by_t[next_t])
            if gamma_by_t is not None and next_t < len(gamma_by_t)
            else float(gamma)
        )
        p_det_next = (
            float(p_det_by_t[next_t])
            if p_det_by_t is not None and next_t < len(p_det_by_t)
            else float(p_det)
        )
        rows.append(
            {
                "t": next_t,
                **{f"mu_{th}": round(mu[th], 4) for th in TIPOS_SECUESTRADOR},
                "m_t": str(m_canon),
                "d_t": int(d_obs),
                "omega_voz": round(float(omega_voz), 4),
                "alpha_t": round(alpha_next, 4),
                "gamma_t": round(gamma_next, 4),
                "p_det_t": round(p_det_next, 4),
                "iota": round(float(max(mu.values()) if mu else 0.0), 4),
                "hat_theta": max(mu, key=lambda k: float(mu.get(k, 0.0))),
                "V_t": "—",
                "emisor_voz": "—",
                "M_t": None,
                "L_H_cont": None,
                "q_comp": None,
                "L_H": None,
                "L_d": None,
                "L_voz": None,
                "L_F": None,
                "L_C": None,
                "L_bayes": None,
                "Z_t": None,
                "p_cont": None,
            }
        )
    if rows:
        t_final = int(t_max)
        mu_prior_t = dict(mu)
        alpha_t = (
            float(alpha_by_t[t_final])
            if alpha_by_t is not None and t_final < len(alpha_by_t)
            else float(alpha)
        )
        gamma_t = (
            float(gamma_by_t[t_final])
            if gamma_by_t is not None and t_final < len(gamma_by_t)
            else float(gamma)
        )
        p_det_t = (
            float(p_det_by_t[t_final])
            if p_det_by_t is not None and t_final < len(p_det_by_t)
            else float(p_det)
        )
        V_step = None
        x_step: Optional[np.ndarray] = None
        emisor = "—"
        if voice_path is not None:
            V_step, x_step, emisor = voice_path_step_at_t(voice_path, t_final)
        elif _use_voz:
            if voice_emit_from_mu and tipo_emit_voz is None:
                w_mu = np.array([float(mu_prior_t.get(th, 0.0)) for th in TIPOS_SECUESTRADOR], dtype=float)
                if w_mu.sum() > 1e-15:
                    w_mu = w_mu / w_mu.sum()
                else:
                    w_mu = np.ones(len(TIPOS_SECUESTRADOR)) / len(TIPOS_SECUESTRADOR)
                emisor = str(rng_voz.choice(TIPOS_SECUESTRADOR, p=w_mu))
            else:
                emisor = str(tipo_emit_voz or TIPOS_SECUESTRADOR[0])
            pi_emit = _pi_call_for_theta(emisor, pi_call_by_theta)
            V_step = draw_voice_indicator(pi_emit, rng_voz)
            if int(V_step) == 1:
                x_step = sample_voice_observation(emisor, voz_params_by_theta, rng_voz)
        df_step, _, meta = build_t0_bayesian_posterior_report(
            modelo,
            mu_prior_t,
            m_canon,
            int(d_obs),
            presion_S=presion_S,
            z_region=z_region,
            v_victim=v_victim,
            alpha=alpha_t,
            gamma=gamma_t,
            p_det=p_det_t,
            zeta_alpha=zeta_alpha,
            zeta_gamma=zeta_gamma,
            zeta_d=zeta_d,
            zeta_R=zeta_R,
            estado_rescata=estado_rescata,
            t_mad=t_mad,
            lambda4=lambda4,
            omega_voz=omega_voz,
            V_t=V_step,
            x_obs=x_step,
            pi_call_by_theta=pi_call_by_theta,
            voz_params_by_theta=voz_params_by_theta,
            delta_t=delta_t,
            t_eval=t_final,
            tab2_bundle_by_theta=tab2_bundle_by_theta,
            atilde_F=atilde_F,
            atilde_K=atilde_K,
            atilde_S=atilde_S,
            implementation_likelihood_by_theta=implementation_likelihood_by_theta,
            aggregate_unknown_theta=False,
            aggregate_lc_unknown_theta=True,
        )
        lk_by_theta_t = {
            str(r["theta_K"]): float(r["L_total"])
            for _, r in df_step.iterrows()
        }
        _, denom_t = bayesian_posterior_update(mu_prior_t, lk_by_theta_t)
        meta["denom"] = float(denom_t)
        meta["emisor_voz"] = emisor
        step_dfs.append(df_step)
        _focus = voice_theta_focus
        if _focus is None and emisor not in ("—", ""):
            _focus = emisor
        lik = summarize_bayes_likelihood_step(df_step, meta, theta_focus=_focus)
        lik["emisor_voz"] = emisor
        rows[-1].update(
            {
                "m_t": str(meta.get("m_obs", m_canon)),
                "d_t": int(meta.get("d_obs", d_obs)),
                "omega_voz": round(float(omega_voz), 4),
                "V_t": lik.get("V_t", "—"),
                "alpha_t": round(alpha_t, 4),
                "gamma_t": round(gamma_t, 4),
                "p_det_t": round(p_det_t, 4),
                "iota": round(float(max(mu_prior_t.values()) if mu_prior_t else 0.0), 4),
                "hat_theta": max(mu_prior_t, key=lambda k: float(mu_prior_t.get(k, 0.0))),
                **lik,
                **{
                    f"L_bayes_{th}": round(float(lk_by_theta_t.get(th, 0.0)), 6)
                    for th in TIPOS_SECUESTRADOR
                },
            }
        )
    return pd.DataFrame(rows), step_dfs


def enrich_mu_trajectory_kidnapper_bellman(
    df_traj: pd.DataFrame,
    df_k_params: pd.DataFrame,
    modelo: ModeloSecuestro,
    *,
    presion_S: float,
    alpha: float,
    gamma: float,
    R: float,
    tipo_incidente: str,
    beta_k: float = 0.92,
) -> pd.DataFrame:
    """
    Añade $\\mathbb{E}_{\\theta_K\\sim\\mu_t}[V^K_{\\mathrm{cont},t}(\\theta_K)]$ (eq. kidnapper-cont)
    fila a fila, usando la misma tabla de parámetros K que Tabla 12.
    """
    if df_traj is None or df_traj.empty or df_k_params is None or df_k_params.empty:
        return df_traj
    out = df_traj.copy()
    ev_vals: List[float] = []
    for _, row in out.iterrows():
        mu_row = {
            th: float(row.get(f"mu_{th}", 0.0))
            for th in TIPOS_SECUESTRADOR
        }
        a_t = float(row.get("alpha_t", alpha))
        g_t = float(row.get("gamma_t", gamma))
        df_k = refresh_kidnapper_endogenous_columns(
            df_k_params.copy(), modelo, presion_S, g_t, alpha=a_t
        )
        ev, _ = kidnapper_V_cont_expectation_over_posterior(
            df_k,
            mu_row,
            modelo,
            presion_S,
            a_t,
            g_t,
            R,
            tipo_incidente,
            beta_k,
        )
        ev_vals.append(round(float(ev), 3))
    out["Ev_V_cont"] = ev_vals
    return out


def build_kidnapper_continuation_posterior_report(
    modelo: ModeloSecuestro,
    mu_0: Dict[str, float],
    d_obs: int,
    **kwargs: Any,
) -> Tuple[pd.DataFrame, Dict[str, float], Dict[str, Any]]:
    """
    Posterior $\\mu_{t+1}(\\cdot\\mid m_t{=}\\mathrm{cont},d_t)$ en ``t=0`` (eq. LH-cont + LH-joint)
    para $\\mathbb{E}[V^K_{\\mathrm{cont}}]$ (kidnapper-cont).
    """
    kw = dict(kwargs)
    kw.setdefault("t_eval", 0)
    return build_t0_bayesian_posterior_report(
        modelo,
        mu_0,
        "Continuar",
        int(d_obs),
        **kw,
    )


def kidnapper_V_cont_expectation_over_posterior(
    df_params: pd.DataFrame,
    mu_post: Dict[str, float],
    modelo: ModeloSecuestro,
    presion_S: float,
    alpha: float,
    gamma: float,
    R: float,
    tipo_incidente: str,
    beta_k: float = 0.92,
) -> Tuple[float, pd.DataFrame]:
    """
    $\\mathbb{E}_{\\theta_K \\sim \\mu_{t+1}}[V^K_{\\mathrm{cont},t}(\\theta_K)]$ usando la
    tabla de parámetros por tipo (Tabla 9) y ``kidnapper_util_df_from_param_df``.
    """
    df_util = kidnapper_util_df_from_param_df(
        df_params,
        modelo,
        presion_S,
        alpha,
        gamma,
        R,
        tipo_incidente,
        beta_k,
    )
    exp_v = 0.0
    detail: List[Dict[str, Any]] = []
    for theta in TIPOS_SECUESTRADOR:
        sub = df_util[df_util["theta_K"].astype(str) == str(theta)]
        if sub.empty:
            continue
        row = sub.iloc[0]
        w = float(mu_post.get(theta, 0.0))
        vc = float(row["V_cont"])
        exp_v += w * vc
        detail.append(
            {
                "theta_K": str(theta),
                "mu_post": round(w, 4),
                "V_cont": round(vc, 3),
                "mu_post_x_V_cont": round(w * vc, 4),
            }
        )
    return float(exp_v), pd.DataFrame(detail)


def kidnapper_backward_induction_k_table(
    modelo: ModeloSecuestro,
    df_mu_traj: pd.DataFrame,
    df_k_params: pd.DataFrame,
    *,
    tipo_real: str,
    beta_k: float,
    R: float,
    t_mad: float,
    T: int = 500,
    alpha_fallback: float,
    gamma_fallback: float,
    alpha_tab12: float,
    ransom_tab12: Optional[float] = None,
    p_cap_expect_fn: Optional[Callable[[str, float, float], float]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Inducción hacia atrás con **creencias agregadas** $\\mu_t(\\theta)$ (columnas `mu_DC`, … de la Tabla 14).

    En $\\tau=T$ **no** entra valor futuro (no existe $W_{T+1}$ en la rama continuar): el oferente
    continuar es $\\sum_\\theta \\mu_\\tau(\\theta)\\,\\mathrm{flow}_\\tau(\\theta)$.

    Para $\\tau<T$: col.~13 y $V^K_{\\mathrm{cont},\\tau}(\\theta)$ usan
    $V^K_{\\mathrm{cont},\\tau+1}(\\theta)$ por tipo, no el escalar $W_{\\tau+1}$.
    ``ransom_tab12``: escala ``R`` del editor Tabla 12 (``tab3_R_override``) para **col. ``flow_rev``**.
    ``alpha_tab12`` queda como respaldo; la columna 9 usa el bloqueo financiero de la fila,
    $\\alpha_\\tau$, en $(1-\\alpha_\\tau)$.
    **Cols. 9–10** (``flow_cost``, ``flow_cap``): $-C_t(\\gamma_t,\\theta^\\ast)$ y
    $-\\tilde p_{cap,t}(\\theta^\\ast)F_{cap}(\\theta^\\ast)$ (sin $\\mu$), tras cols.~7–8.

    **Col. 13** (``flow_rev``): ``Epi_pay_Qcont_mu`` de la **Tabla 14** en la fila $t=\\tau$
    ($\\tilde{\\mathbb E}_{\\tilde A\\mid Q^{\\mathrm{Cont}}}[\\mathbb P(\\mathrm{pay}\\mid\\cdot,\\theta_K=\\theta^\\ast)]$, sin $\\mu$),
    multiplicada por ``ransom_tab12`` y por $(1-\\alpha_\\tau)$.
    Si falta ``Epi_pay_Qcont_mu`` o no es finita, se usa ``h_LibPago`` (Tabla 12, fila $\\theta^\\ast$) como respaldo,
    y si aún falla, ``calcular_hazards`` para $\\theta^\\ast$.

    **Cols. 7–8** (``U_kill``, ``U_rel``): solo $\\theta^\\ast$; $\\kappa_{rel},\\eta,F_{cap}$ de **Tabla 12**;
    $\\tilde p_{cap}$ de **Tabla 14** (``Epi_pcap_Qcap`` en $t=\\tau$), ec.~kidnapper-kill; si falta, ``p_cap_tilde`` Tab.~12.
    Esos valores alimentan también $pc_{\\theta^\\ast}$ en cols.~10 y~12 para $\\theta^\\ast$.

    **Col.~12** (``V_next``): $\\sum_\\theta \\mu_\\tau(\\theta)\\,\\beta\\,(1-\\tilde p_{cap,\\tau}(\\theta))\\,
    V^K_{\\mathrm{cont},\\tau+1}(\\theta)$, con $V^K_{\\mathrm{cont},\\tau+1}(\\theta)$ de la inducción en $\\tau{+}1$
    (no $W_{\\tau+1}$). $\\tilde p_{cap}$ por tipo vía Tab.~14 / ``p_cap_expect_fn`` / Tab.~12.

    **Col.~13** (``V_cont``): suma de cols.~9–12 ($\\tilde p_{\\mathrm{pay}}R(1-\\alpha) - C_t - \\tilde p_{\\mathrm{cap}}F_{\\mathrm{cap}}
    +$ col.~12).

    **Col.~14** (``opcion_BW``): $\\arg\\max\\{\\text{col.~7},\\,\\text{col.~8},\\,\\text{col.~13}\\}$
    ($\\bar V_{\\mathrm{cont}}$ = cols.~9–12).
    $C_t$ (col.~10) escala con $\\tau$ y la infraestructura del tipo (col.~13 varía por $\\theta^\\ast$).
    Tabla devuelta (orden $\\tau=T,\\ldots,1$): **t**, **U_kill**, …, **flow_rev**, …
    """
    meta: Dict[str, Any] = {
        "T": int(T),
        "theta_star": str(tipo_real),
        "primer_tau_backward": None,
        "primer_tau_stationary_below": None,
    }
    if (
        df_mu_traj is None
        or df_mu_traj.empty
        or df_k_params is None
        or df_k_params.empty
    ):
        return pd.DataFrame(), meta

    for _th in TIPOS_SECUESTRADOR:
        if df_k_params[df_k_params["theta_K"].astype(str) == str(_th)].empty:
            return pd.DataFrame(), meta

    t_mad_f = float(max(1e-9, t_mad))
    T_eff = int(max(1, T))
    b = float(np.clip(beta_k, 0.0, 0.9999))
    r_by_theta = kidnapper_r_escala_by_theta(
        df_k_params,
        R_fallback=float(R),
        tipo_real=str(tipo_real),
        ransom_tab12=ransom_tab12,
    )
    R_col9 = float(
        kidnapper_r_escala_tab12_for_type(df_k_params, str(tipo_real), float(R))
    )
    if ransom_tab12 is not None and np.isfinite(float(ransom_tab12)):
        R_col9 = float(ransom_tab12)
    r_by_theta[str(tipo_real)] = float(R_col9)
    meta["R_escala_by_theta"] = {k: round(float(v), 2) for k, v in r_by_theta.items()}
    meta["R_escala_theta_star"] = round(float(R_col9), 2)
    meta["R_col9_flow_rev"] = round(float(R_col9), 2)
    beta_by_theta: Dict[str, float] = {}
    for _th_b in TIPOS_SECUESTRADOR:
        _b_eff = b
        _sub_b = df_k_params[df_k_params["theta_K"].astype(str) == str(_th_b)]
        if not _sub_b.empty and "beta_k" in _sub_b.columns:
            try:
                _b_eff = float(_sub_b.iloc[0]["beta_k"])
            except (TypeError, ValueError, KeyError):
                _b_eff = b
        beta_by_theta[str(_th_b)] = float(np.clip(_b_eff, 0.0, 0.9999))
    meta["beta_by_theta"] = {
        str(k): round(float(v), 4) for k, v in beta_by_theta.items()
    }

    Wmap: Dict[int, float] = {}
    # V_cont(θ, τ) per tipo, guardado en cada paso para usar en τ-1
    v_cont_type_map: Dict[int, Dict[str, float]] = {}
    rev_rows: List[Dict[str, Any]] = []

    for tau in range(T_eff, 0, -1):
        tr_match = df_mu_traj.loc[df_mu_traj["t"].astype(int) == int(tau)]
        if tr_match.empty:
            alpha_t = float(alpha_fallback)
            gamma_t = float(gamma_fallback)
            mu_w = {th: 0.25 for th in TIPOS_SECUESTRADOR}
        else:
            rtr = tr_match.iloc[0]
            alpha_t = float(rtr.get("alpha_t", alpha_fallback))
            gamma_t = float(rtr.get("gamma_t", gamma_fallback))
            mu_raw = {
                th: float(rtr.get(f"mu_{th}", 0.0)) for th in TIPOS_SECUESTRADOR
            }
            s_mu = float(sum(mu_raw.values()))
            if s_mu > 1e-15:
                mu_w = {th: float(mu_raw[th] / s_mu) for th in TIPOS_SECUESTRADOR}
            else:
                mu_w = {th: 0.25 for th in TIPOS_SECUESTRADOR}

        t_eval = int(tau - 1)
        M_tm = (
            float(min(1.0, (float(t_eval) / t_mad_f) ** 2)) if t_eval > 0 else 0.0
        )

        u_kill_by: Dict[str, float] = {}
        u_rel_by: Dict[str, float] = {}
        flow_by: Dict[str, float] = {}
        pc_by: Dict[str, float] = {}
        v_stat_by: Dict[str, float] = {}
        rev_by: Dict[str, float] = {}   # p̃_pay·R·(1-α) (hazards / fila Tab.12)
        rev_col13_by: Dict[str, float] = {}  # col. 9 por θ: h_LibPago o Tab.14 × R(θ)·(1−α)
        cost_by: Dict[str, float] = {}  # φ·exp(κ_c·γ)+ν
        cap_by: Dict[str, float] = {}   # p̃_cap·F_cap
        _alpha_t12 = float(np.clip(alpha_t, 0.0, 1.0))
        _pp_t14 = float("nan")
        _tr_row = tr_match.iloc[0] if not tr_match.empty else None
        if not tr_match.empty and "Epi_pay_Qcont_mu" in df_mu_traj.columns:
            try:
                _pp_t14 = float(tr_match.iloc[0]["Epi_pay_Qcont_mu"])
            except (TypeError, ValueError, KeyError):
                _pp_t14 = float("nan")

        for th in TIPOS_SECUESTRADOR:
            sub = df_k_params[df_k_params["theta_K"].astype(str) == str(th)].iloc[
                [0]
            ]
            r0 = sub.iloc[0]
            u_rel_th, u_kill_th, flow_th = kidnapper_branch_payoffs_from_tab12_row(
                r0,
                modelo,
                t_hazard=int(t_eval),
                presion_S=float(gamma_t),
                alpha=float(alpha_t),
                gamma=float(gamma_t),
                maturity_mult=float(M_tm),
                R=float(r_by_theta[str(th)]),
            )
            pc = float(r0["p_cap_tilde"])
            if p_cap_expect_fn is not None:
                try:
                    pc = float(
                        p_cap_expect_fn(str(th), float(alpha_t), float(gamma_t))
                    )
                except (TypeError, ValueError):
                    pc = float(r0["p_cap_tilde"])
            _pc_tab14_th = _tab14_pcap_for_theta(_tr_row, str(th))
            if np.isfinite(_pc_tab14_th):
                pc = float(_pc_tab14_th)
            pc = float(np.clip(pc, 0.0, 1.0))
            cost_th = kidnapper_cost_c(
                float(gamma_t), float(r0["phi"]), float(r0["kappa_c"]), float(r0["nu"])
            )
            cost_th = float(cost_th)
            cap_th = pc * float(r0["F_cap"])
            rev_th = flow_th + cost_th + cap_th  # flow = rev - cost - cap
            if str(th) == str(tipo_real):
                _R_th = float(R_col9)
            else:
                _R_th = float(r_by_theta[str(th)])
                if "R_escala" in r0.index:
                    try:
                        _R_row = float(r0["R_escala"])
                        if np.isfinite(_R_row) and _R_row > 0.0:
                            _R_th = _R_row
                            r_by_theta[str(th)] = _R_row
                    except (TypeError, ValueError):
                        pass
            _pp_th = _tab14_pay_for_theta(_tr_row, str(th))
            if not np.isfinite(_pp_th) and str(th) == str(tipo_real):
                _pp_th = float(_pp_t14) if np.isfinite(_pp_t14) else float("nan")
            if not np.isfinite(_pp_th):
                try:
                    _pp_th = float(r0["h_LibPago"])
                except (KeyError, TypeError, ValueError):
                    _pp_th = float("nan")
            if not np.isfinite(_pp_th):
                _h_pp = modelo.calcular_hazards(
                    int(t_eval),
                    str(th),
                    float(gamma_t),
                    maturity_mult=float(M_tm),
                    alpha=float(alpha_t),
                    gamma=float(gamma_t),
                )
                _pp_th = float(_h_pp["Pago"])
            rev_col13_by[str(th)] = float(_pp_th) * _R_th * (1.0 - _alpha_t12)
            u_kill_by[str(th)] = u_kill_th
            u_rel_by[str(th)] = u_rel_th
            flow_by[str(th)] = flow_th
            pc_by[str(th)] = pc
            rev_by[str(th)] = rev_th
            cost_by[str(th)] = cost_th
            cap_by[str(th)] = cap_th
            v_stat_by[str(th)] = float(
                kidnapper_V_cont_branch(
                    u_rel_th, u_kill_th, flow_th, float(beta_by_theta[str(th)]), pc
                )
            )

        # Col. 9 (flow_rev): Epi_pay Tab.14 × R(θ*) Tabla 12 × (1 − α Tabla 12)
        _pp_col9 = _tab14_pay_for_theta(_tr_row, str(tipo_real))
        if not np.isfinite(_pp_col9):
            _pp_col9 = float(_pp_t14) if np.isfinite(_pp_t14) else float("nan")
        if not np.isfinite(_pp_col9):
            _th_panel = str(tipo_real)
            if _th_panel not in TIPOS_SECUESTRADOR:
                _th_panel = str(TIPOS_SECUESTRADOR[0])
            _kp_star = df_k_params[
                df_k_params["theta_K"].astype(str) == str(_th_panel)
            ]
            if not _kp_star.empty:
                _r0_star = _kp_star.iloc[0]
                try:
                    _pp_col9 = float(_r0_star["h_LibPago"])
                except (KeyError, TypeError, ValueError):
                    _pp_col9 = float("nan")
            if not np.isfinite(_pp_col9):
                _h_fallback = modelo.calcular_hazards(
                    int(t_eval),
                    _th_panel,
                    float(gamma_t),
                    maturity_mult=float(M_tm),
                    alpha=float(alpha_t),
                    gamma=float(gamma_t),
                )
                _pp_col9 = float(_h_fallback["Pago"])
        flow_rev_theta_star_tab12_R = kidnapper_tab15_flow_rev_col9(
            float(_pp_col9),
            float(R_col9),
            float(_alpha_t12),
        )

        # Cols. 7–8 (θ*): κ_rel, η, F_cap Tab.12; Ê[p_cap|Q^Cap] Tab.14 en t=τ
        _th_s = str(tipo_real) if str(tipo_real) in u_kill_by else TIPOS_SECUESTRADOR[0]
        _kp_s = df_k_params[df_k_params["theta_K"].astype(str) == str(_th_s)]
        if not _kp_s.empty:
            _r0_s = _kp_s.iloc[0]
            _pc_star = _tab14_pcap_for_theta(_tr_row, _th_s)
            if not tr_match.empty and "Epi_pcap_Qcap" in df_mu_traj.columns:
                if not np.isfinite(_pc_star):
                    try:
                        _pc_star = float(tr_match.iloc[0]["Epi_pcap_Qcap"])
                    except (TypeError, ValueError, KeyError):
                        _pc_star = float("nan")
            if not np.isfinite(_pc_star):
                try:
                    _pc_star = float(_r0_s["p_cap_tilde"])
                except (KeyError, TypeError, ValueError):
                    _pc_star = float("nan")
            if np.isfinite(_pc_star):
                u_rel_s, u_kill_s = kidnapper_u_kill_u_rel_from_tab12(_r0_s, _pc_star)
                u_kill_by[_th_s] = float(u_kill_s)
                u_rel_by[_th_s] = float(u_rel_s)
                pc_by[_th_s] = float(_pc_star)
                cap_by[_th_s] = float(_pc_star) * float(_r0_s["F_cap"])

        u_kill_bar = float(
            sum(mu_w[th] * u_kill_by[str(th)] for th in TIPOS_SECUESTRADOR)
        )
        u_rel_bar = float(
            sum(mu_w[th] * u_rel_by[str(th)] for th in TIPOS_SECUESTRADOR)
        )

        _cost_star = float(cost_by.get(_th_s, 0.0))
        _cap_star = float(cap_by.get(_th_s, 0.0))
        _flow_rev_r = round(float(flow_rev_theta_star_tab12_R), 4)
        _flow_cost_r = round(-_cost_star, 4)
        _flow_cap_r = round(-_cap_star, 4)

        v_cont_at_tau_plus_1 = v_cont_type_map.get(int(tau + 1), {})
        if int(tau) == int(T_eff):
            v_next_vcont = 0.0
        else:
            v_next_vcont = float(
                sum(
                    mu_w[th]
                    * float(beta_by_theta[str(th)])
                    * (1.0 - float(pc_by[str(th)]))
                    * float(v_cont_at_tau_plus_1.get(str(th), 0.0))
                    for th in TIPOS_SECUESTRADOR
                )
            )

        # V^K_cont(θ,τ): cols. 9–11 por tipo + β(1−p̃_cap)V^K_cont(θ,τ+1) (alineado con col. 13)
        if int(tau) == int(T_eff):
            v_cont_type_tau = {
                str(th): float(
                    rev_col13_by[str(th)] - cost_by[str(th)] - cap_by[str(th)]
                )
                for th in TIPOS_SECUESTRADOR
            }
        else:
            v_cont_type_tau = {
                str(th): float(
                    rev_col13_by[str(th)]
                    - cost_by[str(th)]
                    - cap_by[str(th)]
                    + float(beta_by_theta[str(th)])
                    * (1.0 - float(pc_by[str(th)]))
                    * float(v_cont_at_tau_plus_1.get(str(th), 0.0))
                )
                for th in TIPOS_SECUESTRADOR
            }
        v_cont_type_map[int(tau)] = v_cont_type_tau

        _mu_star = float(mu_w.get(str(tipo_real), 0.25))
        _u_kill_r = round(float(u_kill_by[_th_s]), 4)
        _u_rel_r = round(float(u_rel_by[_th_s]), 4)
        _v_next_r = round(float(v_next_vcont), 4)
        # Col. 13 = cols. 9 + 10 + 11 + 12 (flujo + continuación; sin cols. 7–8)
        v_cont_sum_9_12 = float(
            _flow_rev_r + _flow_cost_r + _flow_cap_r + _v_next_r
        )
        # Col. 14 = argmax{col. 7, col. 8, col. 13} (V_cont = cols. 9–12)
        opcion_bw, best_v = kidnapper_tab15_argmax_opcion_bw(
            _u_kill_r, _u_rel_r, v_cont_sum_9_12
        )
        Wmap[int(tau)] = float(best_v)

        v_stat_w = float(
            sum(mu_w[th] * v_stat_by[str(th)] for th in TIPOS_SECUESTRADOR)
        )
        um_w = float(max(u_rel_bar, u_kill_bar))

        rev_rows.append(
            {
                "t": int(tau),
                "mu_star": round(_mu_star, 4),
                **{f"mu_{th}": round(float(mu_w.get(th, 0.25)), 4) for th in TIPOS_SECUESTRADOR},
                "U_kill": _u_kill_r,
                "U_rel": _u_rel_r,
                "R_tab12_col9": round(float(R_col9), 2),
                "flow_rev": _flow_rev_r,
                "flow_cost": _flow_cost_r,
                "flow_cap": _flow_cap_r,
                "V_next": _v_next_r,
                "V_cont": round(v_cont_sum_9_12, 4),
                "W_tau": round(float(Wmap[int(tau)]), 4),
                "opcion_BW": str(opcion_bw),
                "_V_stationary_w": round(v_stat_w, 4),
                "_umbral_w": round(um_w, 4),
            }
        )

    # ── τ=0: estado inicial (usa V_cont(θ,τ=1) como valor futuro) ───────
    tr0 = df_mu_traj.loc[df_mu_traj["t"].astype(int) == 0]
    if tr0.empty:
        alpha_t0 = float(alpha_fallback)
        gamma_t0 = float(gamma_fallback)
        mu_w0 = {th: 0.25 for th in TIPOS_SECUESTRADOR}
    else:
        _r0_row = tr0.iloc[0]
        alpha_t0 = float(_r0_row.get("alpha_t", alpha_fallback))
        gamma_t0 = float(_r0_row.get("gamma_t", gamma_fallback))
        _mu_raw0 = {th: float(_r0_row.get(f"mu_{th}", 0.0)) for th in TIPOS_SECUESTRADOR}
        _s_mu0 = float(sum(_mu_raw0.values()))
        if _s_mu0 > 1e-15:
            mu_w0 = {th: float(_mu_raw0[th] / _s_mu0) for th in TIPOS_SECUESTRADOR}
        else:
            mu_w0 = {th: 0.25 for th in TIPOS_SECUESTRADOR}
    _alpha_t12_0 = float(np.clip(alpha_t0, 0.0, 1.0))
    _pp_t14_0 = float("nan")
    _tr0_row = tr0.iloc[0] if not tr0.empty else None
    if not tr0.empty and "Epi_pay_Qcont_mu" in df_mu_traj.columns:
        try:
            _pp_t14_0 = float(tr0.iloc[0]["Epi_pay_Qcont_mu"])
        except (TypeError, ValueError, KeyError):
            _pp_t14_0 = float("nan")
    _u_kill_by0: Dict[str, float] = {}
    _u_rel_by0: Dict[str, float] = {}
    _pc_by0: Dict[str, float] = {}
    _cost_by0: Dict[str, float] = {}
    _cap_by0: Dict[str, float] = {}
    _rev_col13_by0: Dict[str, float] = {}
    for _th0 in TIPOS_SECUESTRADOR:
        _sub0 = df_k_params[df_k_params["theta_K"].astype(str) == str(_th0)].iloc[[0]]
        _r0_th = _sub0.iloc[0]
        _u_rel_th0, _u_kill_th0, _ = kidnapper_branch_payoffs_from_tab12_row(
            _r0_th, modelo,
            t_hazard=-1, presion_S=float(gamma_t0),
            alpha=float(alpha_t0), gamma=float(gamma_t0),
            maturity_mult=0.0, R=float(r_by_theta[str(_th0)]),
        )
        _pc0 = float(_r0_th["p_cap_tilde"])
        if p_cap_expect_fn is not None:
            try:
                _pc0 = float(p_cap_expect_fn(str(_th0), float(alpha_t0), float(gamma_t0)))
            except (TypeError, ValueError):
                _pc0 = float(_r0_th["p_cap_tilde"])
        _pc0_tab14 = _tab14_pcap_for_theta(_tr0_row, str(_th0))
        if np.isfinite(_pc0_tab14):
            _pc0 = float(_pc0_tab14)
        _pc0 = float(np.clip(_pc0, 0.0, 1.0))
        _cost0 = float(kidnapper_cost_c(
            float(gamma_t0), float(_r0_th["phi"]), float(_r0_th["kappa_c"]), float(_r0_th["nu"])
        ))
        _cap0 = _pc0 * float(_r0_th["F_cap"])
        _pp_th0 = _tab14_pay_for_theta(_tr0_row, str(_th0))
        if not np.isfinite(_pp_th0) and str(_th0) == str(tipo_real):
            _pp_th0 = float(_pp_t14_0) if np.isfinite(_pp_t14_0) else float("nan")
        if not np.isfinite(_pp_th0):
            try:
                _pp_th0 = float(_r0_th["h_LibPago"])
            except (KeyError, TypeError, ValueError):
                _pp_th0 = float("nan")
        if not np.isfinite(_pp_th0):
            _h0 = modelo.calcular_hazards(
                -1, str(_th0), float(gamma_t0),
                maturity_mult=0.0, alpha=float(alpha_t0), gamma=float(gamma_t0),
            )
            _pp_th0 = float(_h0["Pago"])
        _R_th0 = float(R_col9) if str(_th0) == str(tipo_real) else float(r_by_theta[str(_th0)])
        _rev_col13_by0[str(_th0)] = float(_pp_th0) * _R_th0 * (1.0 - _alpha_t12_0)
        _u_kill_by0[str(_th0)] = _u_kill_th0
        _u_rel_by0[str(_th0)] = _u_rel_th0
        _pc_by0[str(_th0)] = _pc0
        _cost_by0[str(_th0)] = _cost0
        _cap_by0[str(_th0)] = _cap0
    _pp_col9_0 = _tab14_pay_for_theta(_tr0_row, str(tipo_real))
    if not np.isfinite(_pp_col9_0):
        _pp_col9_0 = float(_pp_t14_0) if np.isfinite(_pp_t14_0) else float("nan")
    if not np.isfinite(_pp_col9_0):
        _kp_star0 = df_k_params[df_k_params["theta_K"].astype(str) == str(tipo_real)]
        if not _kp_star0.empty:
            try:
                _pp_col9_0 = float(_kp_star0.iloc[0]["h_LibPago"])
            except (KeyError, TypeError, ValueError):
                _pp_col9_0 = float("nan")
        if not np.isfinite(_pp_col9_0):
            _h_fb0 = modelo.calcular_hazards(
                -1, str(tipo_real), float(gamma_t0),
                maturity_mult=0.0, alpha=float(alpha_t0), gamma=float(gamma_t0),
            )
            _pp_col9_0 = float(_h_fb0["Pago"])
    _flow_rev0 = round(kidnapper_tab15_flow_rev_col9(float(_pp_col9_0), float(R_col9), float(_alpha_t12_0)), 4)
    _th_s0 = str(tipo_real) if str(tipo_real) in _u_kill_by0 else TIPOS_SECUESTRADOR[0]
    _kp_s0 = df_k_params[df_k_params["theta_K"].astype(str) == str(_th_s0)]
    if not _kp_s0.empty:
        _r0_s0 = _kp_s0.iloc[0]
        _pc_star0 = _tab14_pcap_for_theta(_tr0_row, _th_s0)
        if not tr0.empty and "Epi_pcap_Qcap" in df_mu_traj.columns:
            if not np.isfinite(_pc_star0):
                try:
                    _pc_star0 = float(tr0.iloc[0]["Epi_pcap_Qcap"])
                except (TypeError, ValueError, KeyError):
                    _pc_star0 = float("nan")
        if not np.isfinite(_pc_star0):
            try:
                _pc_star0 = float(_r0_s0["p_cap_tilde"])
            except (KeyError, TypeError, ValueError):
                _pc_star0 = float("nan")
        if np.isfinite(_pc_star0):
            _u_rel_s0, _u_kill_s0 = kidnapper_u_kill_u_rel_from_tab12(_r0_s0, _pc_star0)
            _u_kill_by0[_th_s0] = float(_u_kill_s0)
            _u_rel_by0[_th_s0] = float(_u_rel_s0)
            _pc_by0[_th_s0] = float(_pc_star0)
            _cap_by0[_th_s0] = float(_pc_star0) * float(_r0_s0["F_cap"])
    _flow_cost0 = round(-float(_cost_by0.get(_th_s0, 0.0)), 4)
    _flow_cap0 = round(-float(_cap_by0.get(_th_s0, 0.0)), 4)
    _v_cont_at_1 = v_cont_type_map.get(1, {})
    _v_next0 = round(float(sum(
        mu_w0[_th0]
        * float(beta_by_theta[str(_th0)])
        * (1.0 - float(_pc_by0[str(_th0)]))
        * float(_v_cont_at_1.get(str(_th0), 0.0))
        for _th0 in TIPOS_SECUESTRADOR
    )), 4)
    _v_cont0 = round(_flow_rev0 + _flow_cost0 + _flow_cap0 + _v_next0, 4)
    _opcion0, _ = kidnapper_tab15_argmax_opcion_bw(
        round(float(_u_kill_by0.get(_th_s0, 0.0)), 4),
        round(float(_u_rel_by0.get(_th_s0, 0.0)), 4),
        _v_cont0,
    )
    rev_rows.append({
        "t": 0,
        "mu_star": round(float(mu_w0.get(str(tipo_real), 0.25)), 4),
        **{f"mu_{_th0}": round(float(mu_w0.get(_th0, 0.25)), 4) for _th0 in TIPOS_SECUESTRADOR},
        "U_kill": round(float(_u_kill_by0.get(_th_s0, 0.0)), 4),
        "U_rel": round(float(_u_rel_by0.get(_th_s0, 0.0)), 4),
        "R_tab12_col9": round(float(R_col9), 2),
        "flow_rev": _flow_rev0,
        "flow_cost": _flow_cost0,
        "flow_cap": _flow_cap0,
        "V_next": _v_next0,
        "V_cont": _v_cont0,
        "W_tau": round(float(max(
            float(_u_kill_by0.get(_th_s0, 0.0)),
            float(_u_rel_by0.get(_th_s0, 0.0)),
            _v_cont0,
        )), 4),
        "opcion_BW": str(_opcion0),
        "_V_stationary_w": 0.0,
        "_umbral_w": 0.0,
    })

    df_full = pd.DataFrame(rev_rows)
    # Orden pedido: desde t=T hacia adelante en la tabla = filas T, T-1, …, 1
    df_full = df_full.sort_values("t", ascending=False).reset_index(drop=True)

    primera_bw: Optional[int] = None
    for _, rr in df_full.sort_values("t", ascending=True).iterrows():
        if str(rr["opcion_BW"]) != "Continuar (a_cont)":
            primera_bw = int(rr["t"])
            break

    primera_st: Optional[int] = None
    seen_started_st = False
    for _, rr in df_full.sort_values("t", ascending=True).iterrows():
        vv = float(rr["_V_stationary_w"])
        uu = float(rr["_umbral_w"])
        ok = vv > uu
        if ok:
            seen_started_st = True
        if seen_started_st and not ok:
            primera_st = int(rr["t"])
            break

    _mu_cols = ["mu_star"] + [f"mu_{th}" for th in TIPOS_SECUESTRADOR]
    df_out = df_full[
        ["t"]
        + _mu_cols
        + [
            "U_kill",
            "U_rel",
            "R_tab12_col9",
            "flow_rev",
            "flow_cost",
            "flow_cap",
            "V_next",
            "V_cont",
            "opcion_BW",
        ]
    ].copy()

    meta["primer_tau_backward"] = primera_bw
    meta["primer_tau_stationary_below"] = primera_st
    return df_out, meta


def kidnapper_backward_tau1_switch_fast(
    modelo: ModeloSecuestro,
    df_mu_traj: pd.DataFrame,
    df_k_params: pd.DataFrame,
    *,
    tipo_real: str,
    beta_k: float,
    R: float,
    t_mad: float,
    T: int,
    alpha_fallback: float,
    gamma_fallback: float,
    alpha_tab12: float,
    ransom_tab12: Optional[float] = None,
    p_cap_expect_fn: Optional[Callable[[str, float, float], float]] = None,
) -> Dict[str, Any]:
    """
    Versión ligera de Tabla 15 para calibración.

    Calcula la misma inducción hacia atrás necesaria para la col. 14, pero no
    construye ni ordena la tabla completa. Devuelve solo τ=1 y el primer cambio.
    """
    meta: Dict[str, Any] = {
        "T": int(T),
        "theta_star": str(tipo_real),
        "primer_tau_backward": None,
        "opcion_tau1": "",
        "opcion_tau2": "",
        "ok_tau1": False,
        "ok_tau2": False,
    }
    if (
        df_mu_traj is None
        or df_mu_traj.empty
        or df_k_params is None
        or df_k_params.empty
    ):
        return meta

    param_by_th: Dict[str, pd.Series] = {}
    for th in TIPOS_SECUESTRADOR:
        sub = df_k_params[df_k_params["theta_K"].astype(str) == str(th)]
        if sub.empty:
            return meta
        param_by_th[str(th)] = sub.iloc[0]

    mu_rows: Dict[int, pd.Series] = {}
    if "t" in df_mu_traj.columns:
        for _, rtr in df_mu_traj.iterrows():
            try:
                mu_rows[int(rtr["t"])] = rtr
            except (TypeError, ValueError, KeyError):
                continue

    t_mad_f = float(max(1e-9, t_mad))
    T_eff = int(max(1, T))
    b = float(np.clip(beta_k, 0.0, 0.9999))
    r_by_theta = kidnapper_r_escala_by_theta(
        df_k_params,
        R_fallback=float(R),
        tipo_real=str(tipo_real),
        ransom_tab12=ransom_tab12,
    )
    R_col9 = float(
        kidnapper_r_escala_tab12_for_type(df_k_params, str(tipo_real), float(R))
    )
    if ransom_tab12 is not None and np.isfinite(float(ransom_tab12)):
        R_col9 = float(ransom_tab12)
    r_by_theta[str(tipo_real)] = float(R_col9)

    beta_by_theta: Dict[str, float] = {}
    for th in TIPOS_SECUESTRADOR:
        r0 = param_by_th[str(th)]
        _b_eff = b
        if "beta_k" in r0.index:
            try:
                _b_eff = float(r0["beta_k"])
            except (TypeError, ValueError, KeyError):
                _b_eff = b
        beta_by_theta[str(th)] = float(np.clip(_b_eff, 0.0, 0.9999))

    v_cont_next_by_th: Dict[str, float] = {}
    action_by_tau: Dict[int, str] = {}

    for tau in range(T_eff, 0, -1):
        rtr = mu_rows.get(int(tau))
        if rtr is None:
            alpha_t = float(alpha_fallback)
            gamma_t = float(gamma_fallback)
            mu_w = {th: 0.25 for th in TIPOS_SECUESTRADOR}
            pp_t14 = float("nan")
            pc_star_t14 = float("nan")
        else:
            alpha_t = float(rtr.get("alpha_t", alpha_fallback))
            gamma_t = float(rtr.get("gamma_t", gamma_fallback))
            mu_raw = {
                th: float(rtr.get(f"mu_{th}", 0.0)) for th in TIPOS_SECUESTRADOR
            }
            s_mu = float(sum(mu_raw.values()))
            mu_w = (
                {th: float(mu_raw[th] / s_mu) for th in TIPOS_SECUESTRADOR}
                if s_mu > 1e-15
                else {th: 0.25 for th in TIPOS_SECUESTRADOR}
            )
            try:
                pp_t14 = float(rtr.get("Epi_pay_Qcont_mu", float("nan")))
            except (TypeError, ValueError):
                pp_t14 = float("nan")
            try:
                pc_star_t14 = float(rtr.get("Epi_pcap_Qcap", float("nan")))
            except (TypeError, ValueError):
                pc_star_t14 = float("nan")
        th_star = str(tipo_real) if str(tipo_real) in param_by_th else TIPOS_SECUESTRADOR[0]
        pp_star_t14 = _tab14_pay_for_theta(rtr, th_star)
        if not np.isfinite(pp_star_t14):
            pp_star_t14 = float(pp_t14) if np.isfinite(pp_t14) else float("nan")
        pc_star_tab14 = _tab14_pcap_for_theta(rtr, th_star)
        if not np.isfinite(pc_star_tab14):
            pc_star_tab14 = float(pc_star_t14) if np.isfinite(pc_star_t14) else float("nan")

        t_eval = int(tau - 1)
        M_tm = (
            float(min(1.0, (float(t_eval) / t_mad_f) ** 2)) if t_eval > 0 else 0.0
        )
        alpha_t12 = float(np.clip(alpha_t, 0.0, 1.0))

        u_kill_by: Dict[str, float] = {}
        u_rel_by: Dict[str, float] = {}
        pc_by: Dict[str, float] = {}
        rev_col13_by: Dict[str, float] = {}
        cost_by: Dict[str, float] = {}
        cap_by: Dict[str, float] = {}

        for th in TIPOS_SECUESTRADOR:
            r0 = param_by_th[str(th)]
            u_rel_th, u_kill_th, flow_th = kidnapper_branch_payoffs_from_tab12_row(
                r0,
                modelo,
                t_hazard=int(t_eval),
                presion_S=float(gamma_t),
                alpha=float(alpha_t),
                gamma=float(gamma_t),
                maturity_mult=float(M_tm),
                R=float(r_by_theta[str(th)]),
            )
            pc = float(r0["p_cap_tilde"])
            if p_cap_expect_fn is not None:
                try:
                    pc = float(p_cap_expect_fn(str(th), float(alpha_t), float(gamma_t)))
                except (TypeError, ValueError):
                    pc = float(r0["p_cap_tilde"])
            pc_tab14_th = _tab14_pcap_for_theta(rtr, str(th))
            if np.isfinite(pc_tab14_th):
                pc = float(pc_tab14_th)
            pc = float(np.clip(pc, 0.0, 1.0))
            cost_th = kidnapper_cost_c(
                float(gamma_t), float(r0["phi"]), float(r0["kappa_c"]), float(r0["nu"])
            )
            cost_th = float(cost_th)
            cap_th = pc * float(r0["F_cap"])
            R_th = float(R_col9) if str(th) == str(tipo_real) else float(r_by_theta[str(th)])
            if str(th) != str(tipo_real) and "R_escala" in r0.index:
                try:
                    R_row = float(r0["R_escala"])
                    if np.isfinite(R_row) and R_row > 0.0:
                        R_th = R_row
                        r_by_theta[str(th)] = R_row
                except (TypeError, ValueError):
                    pass
            pp_th = _tab14_pay_for_theta(rtr, str(th))
            if not np.isfinite(pp_th) and str(th) == th_star:
                pp_th = float(pp_star_t14) if np.isfinite(pp_star_t14) else float("nan")
            if not np.isfinite(pp_th):
                try:
                    pp_th = float(r0["h_LibPago"])
                except (KeyError, TypeError, ValueError):
                    pp_th = float("nan")
            if not np.isfinite(pp_th):
                h_pp = modelo.calcular_hazards(
                    int(t_eval),
                    str(th),
                    float(gamma_t),
                    maturity_mult=float(M_tm),
                    alpha=float(alpha_t),
                    gamma=float(gamma_t),
                )
                pp_th = float(h_pp["Pago"])
            rev_col13_by[str(th)] = float(pp_th) * float(R_th) * (1.0 - alpha_t12)
            u_kill_by[str(th)] = float(u_kill_th)
            u_rel_by[str(th)] = float(u_rel_th)
            pc_by[str(th)] = float(pc)
            cost_by[str(th)] = float(cost_th)
            cap_by[str(th)] = float(cap_th)

        pp_col9 = float(pp_star_t14) if np.isfinite(pp_star_t14) else float("nan")
        r0_star = param_by_th[str(th_star)]
        if not np.isfinite(pp_col9):
            try:
                pp_col9 = float(r0_star["h_LibPago"])
            except (KeyError, TypeError, ValueError):
                pp_col9 = float("nan")
        if not np.isfinite(pp_col9):
            h_fallback = modelo.calcular_hazards(
                int(t_eval),
                th_star,
                float(gamma_t),
                maturity_mult=float(M_tm),
                alpha=float(alpha_t),
                gamma=float(gamma_t),
            )
            pp_col9 = float(h_fallback["Pago"])

        if np.isfinite(pc_star_tab14):
            u_rel_s, u_kill_s = kidnapper_u_kill_u_rel_from_tab12(r0_star, pc_star_tab14)
            u_kill_by[th_star] = float(u_kill_s)
            u_rel_by[th_star] = float(u_rel_s)
            pc_by[th_star] = float(pc_star_tab14)
            cap_by[th_star] = float(pc_star_tab14) * float(r0_star["F_cap"])

        if int(tau) == int(T_eff):
            v_next = 0.0
            v_cont_this_by_th = {
                str(th): float(rev_col13_by[str(th)] - cost_by[str(th)] - cap_by[str(th)])
                for th in TIPOS_SECUESTRADOR
            }
        else:
            v_next = float(
                sum(
                    mu_w[th]
                    * float(beta_by_theta[str(th)])
                    * (1.0 - float(pc_by[str(th)]))
                    * float(v_cont_next_by_th.get(str(th), 0.0))
                    for th in TIPOS_SECUESTRADOR
                )
            )
            v_cont_this_by_th = {
                str(th): float(
                    rev_col13_by[str(th)]
                    - cost_by[str(th)]
                    - cap_by[str(th)]
                    + float(beta_by_theta[str(th)])
                    * (1.0 - float(pc_by[str(th)]))
                    * float(v_cont_next_by_th.get(str(th), 0.0))
                )
                for th in TIPOS_SECUESTRADOR
            }
        v_cont_next_by_th = v_cont_this_by_th

        flow_rev = kidnapper_tab15_flow_rev_col9(
            float(pp_col9),
            float(R_col9),
            float(alpha_t12),
        )
        v_cont_star = float(
            round(float(flow_rev), 4)
            + round(-float(cost_by.get(th_star, 0.0)), 4)
            + round(-float(cap_by.get(th_star, 0.0)), 4)
            + round(float(v_next), 4)
        )
        opcion_tau, _ = kidnapper_tab15_argmax_opcion_bw(
            round(float(u_kill_by[th_star]), 4),
            round(float(u_rel_by[th_star]), 4),
            round(float(v_cont_star), 4),
        )
        action_by_tau[int(tau)] = str(opcion_tau)

    primer_tau: Optional[int] = None
    for tau in range(1, T_eff + 1):
        if action_by_tau.get(int(tau)) != "Continuar (a_cont)":
            primer_tau = int(tau)
            break
    opt1 = str(action_by_tau.get(1, ""))
    opt2 = str(action_by_tau.get(2, "")) if T_eff >= 2 else opt1
    meta["primer_tau_backward"] = primer_tau
    meta["opcion_tau1"] = opt1
    meta["opcion_tau2"] = opt2
    meta["ok_tau1"] = opt1 == "Continuar (a_cont)"
    meta["ok_tau2"] = opt2 == "Continuar (a_cont)"
    return meta


def trajectory_entropy_series(historia_mu: List[Dict[str, float]]) -> pd.DataFrame:
    """Serie $H(\\mu_t)$ para visualizar purga de tipos (Teorema 7.9, ilustrativo)."""
    rows = []
    for t, m in enumerate(historia_mu):
        rows.append({"t": t, "H_mu": round(shannon_entropy(m), 2)})
    return pd.DataFrame(rows)


def absorption_posterior_check(
    historia_mu: List[Dict[str, float]],
    tipo_verdadero: str,
    umbral: float = 0.85,
) -> Tuple[bool, float, str]:
    """Al parar, ¿la masa en $\\theta^*$ supera el umbral?"""
    if not historia_mu:
        return False, 0.0, "Sin trayectoria."
    mu_last = historia_mu[-1]
    p_star = float(mu_last.get(tipo_verdadero, 0.0))
    ok = p_star >= umbral
    argmax_t = max(mu_last, key=lambda k: mu_last[k])
    msg = (
        f"Masa en θ* = **{tipo_verdadero}**: **{p_star:.2f}**. "
        f"Máximo posterior: **{argmax_t}** ({mu_last[argmax_t]:.2f})."
    )
    return ok, p_star, msg


def cmh_alive_and_kill_shares() -> Tuple[float, float]:
    """Momentos descriptivos Y_Resultado (CMH)."""
    m = load_cmh_outcome_moments()
    if not m or not m.get("outcomes"):
        return 0.92, 0.02
    out = m["outcomes"]
    p_kill = float(out.get("Muerte", 0.0))
    p_pago = float(out.get("Pago", 0.0))
    p_rescate = float(out.get("Rescate", 0.0))
    p_fuga = float(out.get("Fuga o Liberación", 0.0))
    p_alive_close = p_pago + p_rescate + p_fuga
    return p_alive_close, p_kill
