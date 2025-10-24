# -*- coding: utf-8 -*-
"""
modulo7.py — Modelo FV con 4 stocks: Population, Households,
Potential adopters of PV systems (W) y Adopters of PV systems (A).
Unidades: MESES. Librería silenciosa (sin prints).

NOMBRES "DE PAPER" PARA POBLACIÓN:
- population_initial                        ≡ P0
- fractional_growth_rate                    ≡ r (continua/mes)
- pop_growth_rate_continuous_per_month      ≡ r  [1/mes]
- net_growth_rate_population_time           ≡ r * P(t) [personas/mes]
"""
from __future__ import annotations
import numpy as np
from scipy.integrate import solve_ivp

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
    delayed_payback_fraction_vec: np.ndarray,
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
    # new_households = max(H - H0, 0)
    # able = mf_existing * H0 + mf_new * new_households
    # willing = δ * able
    # ------------------------------------------------------------
    new_households = np.clip(households - initial_households, 0.0, None)
    able    = market_fraction_existing * initial_households + market_fraction_new * new_households
    willing = delayed_payback_fraction_vec * able

    # ------------------------------------------------------------
    # Tasa de adopción (Bass-like):
    # M = A + W;  frac = A / M  (con protección)
    # adoption_rate = (p + q * frac) * max(W, 0)
    # ------------------------------------------------------------
    M    = adopters_of_pv_systems + potential_adopters_of_pv_systems
    frac = _safe_ratio(adopters_of_pv_systems, M)
    adoption_rate = (p_month + q_month * frac) * np.clip(potential_adopters_of_pv_systems, 0.0, None)
    adoption_rate = np.clip(adoption_rate, 0.0, None)

    # ------------------------------------------------------------
    # Inflow a W(t):
    # Exacto: d(willing)/dt = δ * mf_new * dH/dt
    # Relajación: growth_inflow = max( (willing - (W + A)) / τ , 0 )
    # ------------------------------------------------------------
    if use_exact_willing_inflow:
        dWilling_dt = delayed_payback_fraction_vec * market_fraction_new * d_households_dt
        growth_inflow = np.clip(dWilling_dt, 0.0, None)
    else:
        gap = willing - (potential_adopters_of_pv_systems + adopters_of_pv_systems)
        growth_inflow = np.clip(gap / tau_months, 0.0, None)

    # ------------------------------------------------------------
    # Estados W y A:
    # dW/dt = growth_inflow - adoption_rate   (no-negatividad)
    # dA/dt = adoption_rate
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
    # Compatibilidad hacia atrás (si vinieran los nombres antiguos)
    # ----------------------------
    if population_initial is None:
        population_initial = kwargs.get("initial_population")
    if fractional_growth_rate is None:
        fractional_growth_rate = kwargs.get("fractional_growth_rate_month")

    # ----------------------------
    # Normalización y checks
    # ----------------------------
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
        delayed_payback_fraction_vec = np.full_like(population_initial, float(delayed_payback_fraction), dtype=float)
    else:
        delayed_payback_fraction_vec = np.asarray(delayed_payback_fraction, dtype=float)

    n = population_initial.shape[0]
    assert all(arr.shape == (n,) for arr in [
        fractional_growth_rate, average_people_per_household,
        market_fraction_existing, market_fraction_new,
        initial_installed_power_kW, p_month, q_month, pvgp_kW_per_household
    ]), "Entradas shape (3,) para [Norte, Centro, Sur]."

    # ------------------------------------------------------------
    # Inicial (t = 0)
    # ------------------------------------------------------------
    initial_households = population_initial / average_people_per_household
    A0 = np.clip(initial_installed_power_kW / pvgp_kW_per_household, 0.0, None)

    market_existing_initial = market_fraction_existing * initial_households
    willing0 = delayed_payback_fraction_vec * market_existing_initial

    initial_adopters_of_pv_systems = A0
    initial_potential_adopters_of_pv_systems = np.clip(willing0 - A0, 0.0, None)

    # ------------------------------------------------------------
    # Tasas para ODE (CONTINUAS): usamos la entrada tal cual como r
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
            delayed_payback_fraction_vec=delayed_payback_fraction_vec,
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
    households_willing_to_adopt_pv_system = delayed_payback_fraction_vec * household_able_to_adopt
    gap_time = households_willing_to_adopt_pv_system - potential_adopters_of_pv_systems - adopters_of_pv_systems

    if use_exact_willing_inflow:
        # dH/dt = r_H * H  (coherente con el ODE continuo)
        dHH_dt_time = (
            (fractional_growth_rate if households_fractional_growth_rate_month is None
             else np.asarray(households_fractional_growth_rate_month, dtype=float))[None, :] * households
        )
        growth_rate_time = np.clip(
            delayed_payback_fraction_vec[None, :] * market_fraction_new[None, :] * dHH_dt_time, 0.0, None)
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
