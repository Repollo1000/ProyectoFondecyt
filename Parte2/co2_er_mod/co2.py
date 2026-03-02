# -*- coding: utf-8 -*-

def calcular_reporte_sostenibilidad_completo(stock_ice, stock_ev, ventas_ev_anio):
    """
    Cálculo exclusivo para el reporte de impacto ambiental.
    Mantiene la lógica de Vensim separada de la decisión de compra.
    """
    # --- LÓGICA ICEV ---
    yearly_km = 41 * 12
    perf_ice = 11
    ind_co2_ice = 0.00016893
    emi_lt = 2.74 * 0.01
    
    emi_unit_ice = (ind_co2_ice * yearly_km) + ((emi_lt / perf_ice) * yearly_km)
    total_ice_emi = emi_unit_ice * stock_ice
    
    # --- LÓGICA EV ---
    grid_intensity = 0.0003006
    monthly_consume = 101.753
    
    emi_unit_ev = grid_intensity * monthly_consume * 12
    total_ev_emi = emi_unit_ev * stock_ev
    
    # --- EMISIONES EVITADAS (INDICADOR CLAVE) ---
    # Diferencia de lo que se ahorra por cada auto nuevo que NO es ICEV
    evitadas_rate = (emi_unit_ice - emi_unit_ev) * ventas_ev_anio
    
    return {
        "total_ice_ton": total_ice_emi,
        "total_ev_ton": total_ev_emi,
        "emi_unitaria_ice": emi_unit_ice,
        "emi_unitaria_ev": emi_unit_ev,
        "co2_evitado_anio": evitadas_rate
    }