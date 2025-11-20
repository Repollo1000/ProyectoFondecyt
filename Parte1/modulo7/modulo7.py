# -*- coding: utf-8 -*-
"""
modulo7.py — Modelo FV con 4 stocks: Population, Households,
Potential adopters of PV systems (W) y Adopters of PV systems (A).
Unidades: MESES. Librería silenciosa (sin prints).
"""
from __future__ import annotations
import numpy as np
from scipy.integrate import solve_ivp

# --- NUEVA FUNCIÓN DE UTILIDAD: INTERPOLACIÓN POR TIEMPO ---
def _get_time_dependent_dpf(t: float, dpf_series: np.ndarray) -> np.ndarray:
    """Obtiene el valor de la delayed_payback_fraction en el tiempo t.
    Asume que los pasos de tiempo son de 1 mes (t=0, 1, 2, ...).
    """
    if dpf_series.ndim <= 1:
        # Si es un vector (3,) o escalar, lo usa directamente
        return dpf_series
    
    # Asumimos que la serie es (Regiones, Tiempos) y los tiempos son enteros (meses)
    T = dpf_series.shape[1]
    
    # El tiempo t debe redondearse al mes más cercano. Clip para evitar IndexError.
    t_idx = np.clip(int(np.round(t)), 0, T - 1)
    
    # Retorna la columna de la serie correspondiente al tiempo t
    return dpf_series[:, t_idx]


def _safe_ratio(numer: np.ndarray, denom: np.ndarray) -> np.ndarray:
    out = np.zeros_like(numer, dtype=float)
    m = denom > 0
    out[m] = numer[m] / denom[m]
    return out

def ode_system(
    t: float,
    y: np.ndarray,
    pop_growth_rate_continuous_per_month: np.ndarray,  # r continuo
    r_hh_cont: np.ndarray,                             # r_H continuo
    delayed_payback_fraction_series: np.ndarray,       # AHORA ES LA SERIE COMPLETA
    initial_households: np.ndarray,
    market_fraction_existing: np.ndarray,
    market_fraction_new: np.ndarray,
    p_month: np.ndarray,
    q_month: np.ndarray,
    tau_months: float,
    use_exact_willing_inflow: bool,
) -> np.ndarray:
    n = r_hh_cont.shape[0]

    # ------------------------------
    # Desempaquetado de estados
    # ------------------------------
    population = y[0:n]
    households = y[n:2*n]
    potential_adopters_of_pv_systems = y[2*n:3*n]   # W(t)
    adopters_of_pv_systems           = y[3*n:4*n]   # A(t)

    # ------------------------------
    # OBTENER DPF EN TIEMPO t (Para la ODE)
    # ------------------------------
    dpf_vec_current = _get_time_dependent_dpf(t, delayed_payback_fraction_series)
    
    # ------------------------------------------------------------
    # Población — dP/dt = r * P
    # ------------------------------------------------------------
    d_population_dt = pop_growth_rate_continuous_per_month * population

    # ------------------------------------------------------------
    # Hogares — dH/dt = r_H * H
    # ------------------------------------------------------------
    d_households_dt = r_hh_cont * households

    # ------------------------------------------------------------
    # Ecuaciones algebraicas de mercado:
    # able = mf_existing * H0 + mf_new * new_households
    # willing = δ * able
    # ------------------------------------------------------------
    new_households = np.clip(households - initial_households, 0.0, None)
    able    = market_fraction_existing * initial_households + market_fraction_new * new_households
    willing = dpf_vec_current * able # Usa el valor interpolado/actual

    # ------------------------------------------------------------
    # Tasa de adopción:
    # ------------------------------------------------------------
    M    = adopters_of_pv_systems + potential_adopters_of_pv_systems
    frac = _safe_ratio(adopters_of_pv_systems, M)
    adoption_rate = (p_month + q_month * frac) * np.clip(potential_adopters_of_pv_systems, 0.0, None)
    adoption_rate = np.clip(adoption_rate, 0.0, None)

    # ------------------------------------------------------------
    # Inflow a W(t):
    # ------------------------------------------------------------
    if use_exact_willing_inflow:
        dWilling_dt = dpf_vec_current * market_fraction_new * d_households_dt
        growth_inflow = np.clip(dWilling_dt, 0.0, None)
    else:
        gap = willing - (potential_adopters_of_pv_systems + adopters_of_pv_systems)
        growth_inflow = np.clip(gap / tau_months, 0.0, None)

    # ------------------------------------------------------------
    # Estados W y A:
    # ------------------------------------------------------------
    d_potential_adopters_of_pv_systems_dt = growth_inflow - adoption_rate
    mask = (potential_adopters_of_pv_systems <= 0.0) & (d_potential_adopters_of_pv_systems_dt < 0.0)
    d_potential_adopters_of_pv_systems_dt[mask] = 0.0

    d_adopters_of_pv_systems_dt  = adoption_rate

    return np.concatenate([
        d_population_dt,
        d_households_dt,
        d_potential_adopters_of_pv_systems_dt,
        d_adopters_of_pv_systems_dt
    ], axis=0)

def simulate_system(
    # === Nombres "de paper" (preferidos) ===
    population_initial: np.ndarray = None,                     # P0
    fractional_growth_rate: np.ndarray = None,                 # r continuo/mes
    # === Resto de parámetros ===
    average_people_per_household: np.ndarray = None,
    market_fraction_existing: np.ndarray = None,
    market_fraction_new: np.ndarray = None,
    initial_installed_power_kW: np.ndarray = None,
    p_month: np.ndarray = None,
    q_month: np.ndarray = None,
    pvgp_kW_per_household: np.ndarray = None,
    t0: float = 0.0,
    tf: float = 300.0,
    dt: float = 1.0,
    rtol: float = 1e-6,
    atol: float = 1e-8,
    delayed_payback_fraction: float | np.ndarray = 0.5,
    households_fractional_growth_rate_month: np.ndarray | None = None,
    tau_months: float = 1.0,
    use_exact_willing_inflow: bool = False,
    growth_rate: np.ndarray | None = None,   # compatibilidad (no usado)
    **kwargs,  # compatibilidad con nombres antiguos del dict
):
    # ----------------------------
    # Compatibilidad y Normalización
    # ----------------------------
    if population_initial is None:
        population_initial = kwargs.get("initial_population")
    if fractional_growth_rate is None:
        fractional_growth_rate = kwargs.get("fractional_growth_rate_month")

    population_initial            = np.asarray(population_initial, dtype=float)
    fractional_growth_rate        = np.asarray(fractional_growth_rate, dtype=float)
    average_people_per_household  = np.asarray(average_people_per_household, dtype=float)
    market_fraction_existing      = np.asarray(market_fraction_existing, dtype=float)
    market_fraction_new           = np.asarray(market_fraction_new, dtype=float)
    initial_installed_power_kW    = np.asarray(initial_installed_power_kW, dtype=float)
    p_month                       = np.asarray(p_month, dtype=float)
    q_month                       = np.asarray(q_month, dtype=float)
    pvgp_kW_per_household         = np.asarray(pvgp_kW_per_household, dtype=float)

    if np.isscalar(delayed_payback_fraction):
        # Si es escalar, lo convertimos a vector (3,)
        delayed_payback_fraction_vec = np.full_like(population_initial, float(delayed_payback_fraction), dtype=float)
    else:
        # Si es un array, lo usamos. Puede ser (3,) o (3, T)
        delayed_payback_fraction_vec = np.asarray(delayed_payback_fraction, dtype=float)

    n = population_initial.shape[0]

    # ------------------------------------------------------------
    # Inicial (t = 0)
    # ------------------------------------------------------------
    initial_households = population_initial / average_people_per_household
    A0 = np.clip(initial_installed_power_kW / pvgp_kW_per_household, 0.0, None)

    market_existing_initial = market_fraction_existing * initial_households
    
    # --- CORRECCIÓN CLAVE: USAR SOLO EL VALOR INICIAL PARA T=0 ---
    if delayed_payback_fraction_vec.ndim == 2:
        # Si es una serie (3, T), tomamos solo el valor inicial (columna 0)
        dpf_initial = delayed_payback_fraction_vec[:, 0]
    else:
        # Si es un vector (3,) o escalar, se usa directamente
        dpf_initial = delayed_payback_fraction_vec
        
    willing0 = dpf_initial * market_existing_initial
    # -----------------------------------------------------------

    initial_adopters_of_pv_systems = A0
    initial_potential_adopters_of_pv_systems = np.clip(willing0 - A0, 0.0, None)

    # ------------------------------------------------------------
    # Tasas para ODE (CONTINUAS):
    # ------------------------------------------------------------
    pop_growth_rate_continuous_per_month = fractional_growth_rate
    if households_fractional_growth_rate_month is None:
        r_hh_cont = pop_growth_rate_continuous_per_month.copy()
    else:
        r_hh_cont = np.asarray(households_fractional_growth_rate_month, dtype=float)

    # ------------------------------------------------------------
    # Integración de ODEs (stocks: P, H, W, A)
    # ------------------------------------------------------------
    y0 = np.concatenate([
        population_initial,
        initial_households,
        initial_potential_adopters_of_pv_systems,
        initial_adopters_of_pv_systems
    ], axis=0)
    t_eval = np.arange(t0, tf + dt, dt, dtype=float)

    sol = solve_ivp(
        fun=lambda t, y: ode_system(
            t, y,
            pop_growth_rate_continuous_per_month=pop_growth_rate_continuous_per_month,
            r_hh_cont=r_hh_cont,
            delayed_payback_fraction_series=delayed_payback_fraction_vec, # Pasa la serie completa
            initial_households=initial_households,
            market_fraction_existing=market_fraction_existing,
            market_fraction_new=market_fraction_new,
            p_month=p_month,
            q_month=q_month,
            tau_months=tau_months,
            use_exact_willing_inflow=use_exact_willing_inflow,
        ),
        t_span=(t0, tf),
        y0=y0,
        t_eval=t_eval,
        rtol=rtol,
        atol=atol,
    )
    if not sol.success:
        raise RuntimeError(f"Error en la integración: {sol.message}")

    # ----------------------------
    # Reconstrucción de estados (T x 3)
    # ----------------------------
    population                          = sol.y[0:n, :].T
    households                          = sol.y[n:2*n, :].T
    potential_adopters_of_pv_systems    = sol.y[2*n:3*n, :].T   # W(t)
    adopters_of_pv_systems              = sol.y[3*n:4*n, :].T   # A(t)

    # ------------------------------------------------------------
    # Derivadas y métricas en el tiempo (algebraico sobre estados)
    # ------------------------------------------------------------
    initial_households = population_initial / average_people_per_household
    new_households = np.clip(households - initial_households, 0.0, None)
    household_able_to_adopt = (
        market_fraction_existing * initial_households
        + market_fraction_new * new_households
    )
    
    # --- CORRECCIÓN 2: USO DE LA SERIE COMPLETA EN LA RECONSTRUCCIÓN ---
    # Interpolamos los valores de la serie al tiempo de la solución (sol.t)
    dpf_series_interpolated = np.array([
        _get_time_dependent_dpf(t_val, delayed_payback_fraction_vec) 
        for t_val in sol.t
    ]) # (T_solucion, 3)

    households_willing_to_adopt_pv_system = dpf_series_interpolated * household_able_to_adopt
    # -------------------------------------------------------------------
    
    gap_time = households_willing_to_adopt_pv_system - potential_adopters_of_pv_systems - adopters_of_pv_systems

    if use_exact_willing_inflow:
        # dH/dt = r_H * H  (coherente con el ODE continuo)
        dHH_dt_time = (
            (fractional_growth_rate if households_fractional_growth_rate_month is None
             else np.asarray(households_fractional_growth_rate_month, dtype=float))[None, :] * households
        )
        
        # --- CORRECCIÓN 3: USO DE LA SERIE COMPLETA EN LA TASA DE CRECIMIENTO ---
        growth_rate_time = np.clip(
            dpf_series_interpolated * market_fraction_new[None, :] * dHH_dt_time, 0.0, None)
    else:
        growth_rate_time = np.clip(gap_time / tau_months, 0.0, None)

    households_remaining_willing_to_adopt = gap_time
    pv_power_kW = adopters_of_pv_systems * pvgp_kW_per_household

    adopters_flow_per_month = np.zeros_like(adopters_of_pv_systems)
    adopters_flow_per_month[1:, :] = adopters_of_pv_systems[1:, :] - adopters_of_pv_systems[:-1, :]
    adopters_flow_per_month = np.where(adopters_flow_per_month < 1e-9,
                                       np.maximum(adopters_flow_per_month, 0.0),
                                       adopters_flow_per_month)

    M_time = adopters_of_pv_systems + potential_adopters_of_pv_systems
    frac_t = np.where(M_time > 0.0, adopters_of_pv_systems / M_time, 0.0)
    adoption_rate_time = (p_month[None, :] + q_month[None, :] * frac_t) * np.clip(potential_adopters_of_pv_systems, 0.0, None)
    adoption_rate_time = np.clip(adoption_rate_time, 0.0, None)

    # Inicial: hogares “aptos” no ocupados por A0 (solo informativo)
    initial_able_to_adopt = np.clip(market_existing_initial - initial_adopters_of_pv_systems, 0.0, None)

    # Serie explícita de net growth rate de población (r * P(t))
    net_growth_rate_population_time = pop_growth_rate_continuous_per_month[None, :] * population

    return (
        sol.t,
        population,
        households,
        new_households,
        adopters_of_pv_systems,
        household_able_to_adopt,
        pv_power_kW,
        adopters_flow_per_month,
        initial_potential_adopters_of_pv_systems,
        initial_able_to_adopt,
        initial_adopters_of_pv_systems,
        households_willing_to_adopt_pv_system,
        potential_adopters_of_pv_systems,
        households_remaining_willing_to_adopt,
        adoption_rate_time,
        growth_rate_time,
        # ---- Nuevos retornos (al final para no romper orden previo) ----
        pop_growth_rate_continuous_per_month,   # r (3,)
        net_growth_rate_population_time         # (T x 3)
    )