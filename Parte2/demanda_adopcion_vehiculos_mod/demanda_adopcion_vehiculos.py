# -*- coding: utf-8 -*-
import numpy as np

def simulate_system(
    population_initial, ev_stock_initial, icev_stock_initial,
    population_growth_rate, motorization_rate,
    ev_lifetime, icev_lifetime, time_delay,
    market_share_ev_series, market_share_icev_series,
    tf, dt, **kwargs
):
    # Preparar el tiempo
    t_eval = np.arange(0, tf + dt, dt)
    T = len(t_eval)
    
    # Crear las cajas (Stocks)
    pop = np.zeros((T, 3))
    ev_s = np.zeros((T, 3))
    ice_s = np.zeros((T, 3))

    # Poner valores iniciales (Mes 0)
    pop[0] = population_initial
    ev_s[0] = ev_stock_initial
    ice_s[0] = icev_stock_initial

    # Bucle mes a mes
    for t in range(T - 1):
        # 1. Cálculos de información (Círculos en Vensim)
        total_v = ev_s[t] + ice_s[t]
        target_v = pop[t] / motorization_rate
        gap = np.maximum(target_v - total_v, 0) / (time_delay * 12)
        
        # 2. Válvulas de entrada (Ventas)
        # Usamos el Market Share que le pasamos desde el main
        ev_sales = gap * market_share_ev_series[t]
        ice_sales = gap * market_share_icev_series[t]
        
        # 3. Válvulas de salida (Descartes)
        ev_disc = ev_s[t] / (ev_lifetime * 12)
        ice_disc = ice_s[t] / (icev_lifetime * 12)
        
        # 4. Crecimiento Población
        pop_growth = pop[t] * (population_growth_rate / 12)

        # 5. Actualizar cajas para el mes siguiente (Integración)
        ev_s[t+1] = ev_s[t] + (ev_sales - ev_disc) * dt
        ice_s[t+1] = ice_s[t] + (ice_sales - ice_disc) * dt
        pop[t+1] = pop[t] + (pop_growth) * dt

    return {"t": t_eval, "population": pop, "ev_stock": ev_s, "icev_stock": ice_s}