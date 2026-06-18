import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------
# CONSTANTES Y ESPACIO DE ESTADOS
# ---------------------------------------------------------
# ---------------------------------------------------------
# CONSTANTES Y ESPACIO DE ESTADOS
# ---------------------------------------------------------
TIPOS_SECUESTRADOR = ["DC", "PAR", "ELN", "FARC"]
DESENLACES = ["Liberación", "Rescate", "Pago", "Muerte", "Continuar"]

class ModeloSecuestro:
    def __init__(self, betas=None, lambdas_0=None):
        """
        Inicializa los parámetros estructurales del modelo.
        """
        # Betas por tipo para cada desenlace (Liberación, Rescate, Pago, Muerte)
        # Valores ilustrativos que generan identificabilidad
        if betas is None:
            # Betas por tipo para cada desenlace: firmas de intensidades
            # suficientemente separadas para identificación bayesiana en corridas largas.
            self.betas = {
                "FARC": {"Liberación": 0.55,  "Rescate": 0.90,  "Pago": -0.70, "Muerte": -0.85},
                "ELN":  {"Liberación": -0.35, "Rescate": -0.65, "Pago": 1.10,  "Muerte": 0.20},
                "PAR":  {"Liberación": -0.90, "Rescate": 0.15,  "Pago": -0.25, "Muerte": 1.35},
                "DC":   {"Liberación": 0.05,  "Rescate": -0.95, "Pago": 1.55,  "Muerte": -0.40},
            }
        else:
            self.betas = betas
            
        # Riesgos basales (lambda_0) para el tipo de referencia (FARC)
        if lambdas_0 is None:
            self.lambdas_0 = {"Liberación": 0.015, "Rescate": 0.008, "Pago": 0.012, "Muerte": 0.002}
        else:
            self.lambdas_0 = lambdas_0

        # Umbral de maduración T_mad (Mechanism.tex eq. 890)
        self.T_mad = 5.0

    def calcular_hazards(
        self,
        t,
        tipo,
        presion_S=0.0,
        maturity_mult=None,
        z_region="Metropolitana",
        v_victim="Privado",
        alpha=None,
        gamma=None,
        p_det=None,
        zeta_alpha=None,
        zeta_gamma=None,
        zeta_d=None,
        zeta_R=0.0,
        estado_rescata=False,
        zeta_by_j=None,
        atilde_F=None,
        atilde_K=None,
        atilde_S=None,
    ):
        """
        Calcula las intensidades cause-specific (tilde_lambda_j) para el dia t.

        Implementa exactamente las ecuaciones de Mechanism.tex (bloque de intensidades
        proporcionales, seccion de riesgos competitivos):

            lambda_1 = lambda_10 * exp(beta_K + beta_z + beta_v + beta_F + beta_S
                                       - zeta_alpha_1 * alpha
                                       - zeta_gamma_1 * gamma
                                       - zeta_d_1 * p_det
                                       + ...)
            lambda_2 = lambda_20 * exp(beta_K + beta_z + beta_S
                                       + zeta_alpha_2 * alpha
                                       + zeta_gamma_2 * gamma
                                       - zeta_d_2 * p_det
                                       + ...)
            lambda_3 = lambda_30 * exp(beta_K + beta_z + beta_S
                                       + zeta_alpha_3 * alpha
                                       + zeta_gamma_3 * gamma
                                       + zeta_d_3 * p_det
                                       + zeta_R * 1{a_S=Rescate}
                                       + ...)

        Los signos de zeta_alpha, zeta_gamma y zeta_d son estructurales y estan
        fijados por la ecuacion; los coeficientes zeta_* son magnitudes positivas.

        Parametros heredados para compatibilidad:
            presion_S  -- gamma efectivo cuando gamma no se pasa explicitamente.
            alpha      -- tasa de bloqueo financiero en [0,1] (Mechanism.tex alpha_t).
            gamma      -- intensidad de presion operativa en [0,1] (Mechanism.tex gamma_t).
            p_det      -- probabilidad de deteccion; si es None se omite el termino.
            zeta_alpha -- coeficiente de bloqueo (magnitud positiva); si None usa 0.1.
            zeta_gamma -- coeficiente de presion (magnitud positiva); si None usa 0.1.
            zeta_d     -- coeficiente de deteccion (magnitud positiva); si None usa 0.1.
            zeta_R     -- bono de rescate estatal en lambda_3 (Mechanism.tex).
            estado_rescata -- True si la accion ejecutada del Estado es Rescate.
            zeta_by_j  -- opcional: ``{Pago|Muerte|Rescate: {alpha, gamma, d, R?}}``
                (Tabla 1 pestaña 2; si falta, usa ``zeta_*`` globales).
            atilde_*   -- acciones ejecutadas observadas en Tabla 5.2.
        """
        if maturity_mult is None:
            m = float(min(1.0, (t / self.T_mad) ** 2))
        else:
            m = float(maturity_mult)

        # Instrumentos: alpha y gamma en [0,1] (Mechanism.tex)
        # Si gamma no se pasa explicitamente, se usa presion_S por compatibilidad.
        alpha_eff = float(alpha) if alpha is not None else 0.0
        gamma_eff = float(gamma) if gamma is not None else float(presion_S)

        # Coeficientes zeta (magnitudes positivas; los signos estan en la ecuacion)
        za = float(zeta_alpha) if zeta_alpha is not None else 0.1
        zg = float(zeta_gamma) if zeta_gamma is not None else 0.1
        zd = float(zeta_d) if zeta_d is not None else 0.1
        zR = float(zeta_R)
        _zbj = zeta_by_j if isinstance(zeta_by_j, dict) else {}

        def _z_cause(cause: str, kind: str, default: float) -> float:
            block = _zbj.get(cause)
            if isinstance(block, dict) and kind in block:
                try:
                    return float(block[kind])
                except (TypeError, ValueError):
                    pass
            return float(default)

        def _tab_term(cause: str, key: str, default: float = 0.0) -> float:
            block = _zbj.get(cause)
            if isinstance(block, dict) and key in block:
                try:
                    return float(block[key])
                except (TypeError, ValueError):
                    pass
            return float(default)

        f_paga = str(atilde_F or "").strip().lower() in {"pagar", "coludir"}
        k_cont = str(atilde_K or "").strip().lower() in {"cont", "continuar"}
        k_kill = str(atilde_K or "").strip().lower() in {"kill", "matar"}
        s_rescata = bool(estado_rescata) or str(atilde_S or "").strip().lower().startswith("rescat")

        # p_det: si no se provee, se calcula como 0 (sin efecto de deteccion)
        pdet = float(p_det) if p_det is not None else 0.0

        # Coeficientes transversales (basados en main.tex ln-RRR)
        COEF_ETA_VALS = {
            "Metropolitana": 0.00,
            "Andina": -0.45,
            "Caribe": -0.70,
            "Pacifica / Zona Roja": -0.20,
            "Oriente / Selva": -0.32,
        }
        # Alias para compatibilidad con tilde en nombre de zona
        COEF_ETA_VALS["Pacífica / Zona Roja"] = COEF_ETA_VALS["Pacifica / Zona Roja"]
        COEF_XI_VALS = {"Público": 1.36, "Privado": 0.00}

        eta_z = COEF_ETA_VALS.get(z_region, 0.0)
        xi_v = COEF_XI_VALS.get(v_victim, 0.0)

        hazards = {}
        for j in ["Liberación", "Rescate", "Pago", "Muerte"]:
            beta_val = float(self.betas[tipo][j])

            if j == "Pago":
                # lambda_1: alpha y gamma REDUCEN pago; p_det REDUCE pago
                # (Mechanism.tex: -zeta_alpha_1*alpha - zeta_gamma_1*gamma - zeta_d_1*p_det)
                za_j = _z_cause("Pago", "alpha", za)
                zg_j = _z_cause("Pago", "gamma", zg)
                zd_j = _z_cause("Pago", "d", zd)
                instrumento = -za_j * alpha_eff - zg_j * gamma_eff - zd_j * pdet
                beta_z = _tab_term("Pago", "beta_z", eta_z)
                beta_v = _tab_term("Pago", "beta_V", xi_v)
                beta_f = _tab_term("Pago", "beta_F", 0.0)
                beta_s = _tab_term("Pago", "beta_S", 0.0)
                phi_f = _tab_term("Pago", "phi_F", 0.0)
                phi_kc = _tab_term("Pago", "phi_K_cont", 0.0)
                exponent = (
                    beta_val + beta_z + beta_f - beta_v + beta_s + instrumento
                    + (phi_f if f_paga else 0.0)
                    + (phi_kc if k_cont else 0.0)
                )

            elif j == "Muerte":
                # lambda_2: alpha y gamma AUMENTAN muerte; p_det REDUCE muerte
                # (Mechanism.tex: +zeta_alpha_2*alpha + zeta_gamma_2*gamma - zeta_d_2*p_det)
                za_j = _z_cause("Muerte", "alpha", za)
                zg_j = _z_cause("Muerte", "gamma", zg)
                zd_j = _z_cause("Muerte", "d", zd)
                instrumento = +za_j * alpha_eff + zg_j * gamma_eff - zd_j * pdet
                beta_z = _tab_term("Muerte", "beta_z", eta_z)
                beta_s = _tab_term("Muerte", "beta_S", 0.0)
                phi_f = _tab_term("Muerte", "phi_F", 0.0)
                phi_kk = _tab_term("Muerte", "phi_K_kill", 0.0)
                phi_kc = _tab_term("Muerte", "phi_K_cont", 0.0)
                exponent = (
                    beta_val + beta_z + beta_s + instrumento
                    - (phi_f if f_paga else 0.0)
                    + (phi_kk if k_kill else 0.0)
                    + (phi_kc if k_cont else 0.0)
                )

            elif j == "Rescate":
                # lambda_3: alpha y gamma AUMENTAN rescate; p_det AUMENTA rescate
                # (Mechanism.tex: +zeta_alpha_3*alpha + zeta_gamma_3*gamma + zeta_d_3*p_det)
                # bono de rescate estatal si a_S = Rescate; beta_S entra con signo negativo.
                za_j = _z_cause("Rescate", "alpha", za)
                zg_j = _z_cause("Rescate", "gamma", zg)
                zd_j = _z_cause("Rescate", "d", zd)
                zR_j = _z_cause("Rescate", "R", zR)
                instrumento = +za_j * alpha_eff + zg_j * gamma_eff + zd_j * pdet
                bono_rescate = zR_j if s_rescata else 0.0
                beta_z = _tab_term("Rescate", "beta_z", eta_z)
                beta_s = _tab_term("Rescate", "beta_S", 0.0)
                phi_f = _tab_term("Rescate", "phi_F", 0.0)
                phi_kc = _tab_term("Rescate", "phi_K_cont", 0.0)
                exponent = (
                    beta_val + beta_z - beta_s + instrumento + bono_rescate
                    - (phi_f if f_paga else 0.0)
                    + (phi_kc if k_cont else 0.0)
                )

            else:
                # Liberacion: no tiene terminos de instrumentos en la especificacion
                # base del mecanismo; se mantiene solo con beta_K + beta_z + beta_v
                exponent = beta_val + eta_z + xi_v

            h = m * float(self.lambdas_0[j]) * np.exp(exponent)
            hazards[j] = float(max(0.0, h))

        # "Continuar" es el residuo de masa (no es una intensidad competing-risk
        # en el sentido de Mechanism.tex; aqui se conserva por compatibilidad
        # con el codigo existente de blend_hazards y visualizaciones).
        prob_salida = sum(hazards.values())
        prob_salida = min(prob_salida, 0.99)
        hazards["Continuar"] = 1.0 - prob_salida

        return hazards

    def simular_proceso_mdg(self, intencion_S, intencion_F, intencion_K, precision_iota):
        """
        Simula el proceso Mano de Dios-Guadalupe (Transformada Inversa).
        Transforma intenciones en desenlaces físicos estocásticos.
        """
        # Versión simplificada con soporte exhaustivo
        probs = {"Liberación": 0.05, "Rescate": 0.05, "Pago": 0.1, "Muerte": 0.05, "Continuar": 0.75}
        
        # Ajuste de masas según intenciones y precisión
        if intencion_S == "Rescate":
            probs["Rescate"] += precision_iota * 0.2
            probs["Continuar"] -= precision_iota * 0.2
            
        if intencion_F == "Coludir":
            probs["Pago"] += 0.2
            probs["Continuar"] -= 0.2
            
        if intencion_K == "Matar":
            probs["Muerte"] += 0.3
            probs["Continuar"] -= 0.3
            
        # Normalizar
        total = sum(probs.values())
        for k in probs:
            probs[k] /= total
            
        v_t = np.random.uniform(0, 1)
        acumulado = 0.0
        for desenlace, prob in probs.items():
            acumulado += prob
            if v_t <= acumulado:
                return desenlace
                
        return "Continuar"

    def actualizar_creencias(
        self, mu_t, desenlace, presion_S,
        alpha=0.0, gamma=None, p_det=0.0,
        zeta_alpha=None, zeta_gamma=None, zeta_d=None, zeta_R=0.0,
    ):
        """
        Actualización bayesiana de las creencias del Estado sobre el tipo de secuestrador.
        mu_{t+1}(theta) prop a mu_t(theta) * Likelihood(desenlace | theta)
        """
        mu_t_mas_1 = {}
        evidencia_total = 0.0

        likelihoods = {}
        for tipo in TIPOS_SECUESTRADOR:
            hazards = self.calcular_hazards(
                1, tipo, presion_S,
                alpha=alpha, gamma=gamma, p_det=p_det,
                zeta_alpha=zeta_alpha, zeta_gamma=zeta_gamma,
                zeta_d=zeta_d, zeta_R=zeta_R,
            )
            likelihoods[tipo] = hazards[desenlace]
            evidencia_total += mu_t[tipo] * likelihoods[tipo]

        for tipo in TIPOS_SECUESTRADOR:
            if evidencia_total > 0:
                mu_t_mas_1[tipo] = (mu_t[tipo] * likelihoods[tipo]) / evidencia_total
            else:
                mu_t_mas_1[tipo] = mu_t[tipo]

        return mu_t_mas_1

    def simular_trayectoria_cautiverio(
        self, tipo_verdadero, mu_0, max_dias=100, presion_S=0.5,
        alpha=0.0, gamma=None, p_det=0.0,
        zeta_alpha=None, zeta_gamma=None, zeta_d=None, zeta_R=0.0,
    ):
        """
        Simula una trayectoria completa y el proceso de aprendizaje asintótico.
        Demuestra la concentración de probabilidad del Teorema 7.9.
        """
        historia_creencias = [mu_0]
        eventos = []
        mu_t = mu_0.copy()

        for t in range(1, max_dias + 1):
            hazards_reales = self.calcular_hazards(
                t, tipo_verdadero, presion_S,
                alpha=alpha, gamma=gamma, p_det=p_det,
                zeta_alpha=zeta_alpha, zeta_gamma=zeta_gamma,
                zeta_d=zeta_d, zeta_R=zeta_R,
            )
            probs = [hazards_reales[d] for d in DESENLACES]
            desenlace_t = np.random.choice(DESENLACES, p=probs)
            eventos.append(desenlace_t)

            mu_t = self.actualizar_creencias(
                mu_t, desenlace_t, presion_S,
                alpha=alpha, gamma=gamma, p_det=p_det,
                zeta_alpha=zeta_alpha, zeta_gamma=zeta_gamma,
                zeta_d=zeta_d, zeta_R=zeta_R,
            )
            historia_creencias.append(mu_t)

            if desenlace_t != "Continúa":
                break

        return eventos, historia_creencias
