# -*- coding: utf-8 -*-
import numpy as np

def simulate_system(population_initial, ev_stock_initial, icev_stock_initial, 
                    population_growth_rate, motorization_rate, 
                    ev_lifetime, icev_lifetime, time_delay, 
                    u_ev_serie, u_icev_serie, tf, dt):
    
    num_regiones = len(population_initial)
    steps = int(tf / dt) + 1
    
    # Inicialización de matrices
    pop = np.zeros((steps, num_regiones)); ev_s = np.zeros((steps, num_regiones))
    icev_s = np.zeros((steps, num_regiones)); v_demand_auditoria = np.zeros((steps, num_regiones))
    
    pop[0] = population_initial
    ev_s[0] = ev_stock_initial
    icev_s[0] = icev_stock_initial
    
    for t in range(steps - 1):
        # 1. Market Share (Lógica literal de tu fórmula)
        u_max = np.maximum(u_ev_serie[t], u_icev_serie[t])
        exp_ev = np.exp(u_ev_serie[t] - u_max)
        exp_icev = np.exp(u_icev_serie[t] - u_max)
        ms_ev = exp_ev / (exp_ev + exp_icev)
        ms_icev = exp_icev / (exp_ev + exp_icev)
        
        # 2. Vehicle Demand (Literal: Vehicle Demand = Vehicle Gap)
        total_vehicles = ev_s[t] + icev_s[t]
        # Total Vehicles to Satisfy = Population / Motorization rate
        total_to_satisfy = pop[t] / motorization_rate
        # Vehicle Gap = MAX(Satisfy - Total, 0) / Time Delay
        vehicle_gap = np.maximum(total_to_satisfy - total_vehicles, 0) / time_delay
        
        v_demand_total = vehicle_gap # <--- AQUÍ ESTÁ LA SINCRONIZACIÓN CON VENSIM
        v_demand_auditoria[t] = v_demand_total
        
        # 3. Sales
        ev_sales = ms_ev * v_demand_total
        icev_sales = ms_icev * v_demand_total
        
        # 4. Discards (Stock / Lifetime)
        ev_discards = ev_s[t] / ev_lifetime
        icev_discards = icev_s[t] / icev_lifetime
        
        # 5. Integ (Actualización de Stock y Población)
        ev_s[t+1] = ev_s[t] + (ev_sales - ev_discards) * dt
        icev_s[t+1] = icev_s[t] + (icev_sales - icev_discards) * dt
        pop[t+1] = pop[t] + (pop[t] * population_growth_rate) * dt
        
    return {
        "ev_stock": ev_s, 
        "icev_stock": icev_s, 
        "population": pop, 
        "vehicle_demand": v_demand_auditoria
    }