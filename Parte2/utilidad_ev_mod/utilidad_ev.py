import numpy as np

def calcular_utilidad_ev_completa(params, stocks_ev, region_idx):
    """
    Calcula la utilidad y todos sus componentes internos siguiendo 
    fielmente el módulo de Vensim.
    """
    
    # --- 1. CÁLCULOS ECONÓMICOS (TCO) ---
    
    # Precio de compra (Ecuación 89)
    # Purchase price cost EV = EV Base price - (EV Base price * EV Subsidies percent)
    purchase_price = stocks_ev['ev_base_price'] - (stocks_ev['ev_base_price'] * params['ev_subsidies_percent'])
    
    # Costos de Operación Anuales (Ecuaciones 94, 95, 96)
    anual_maint = params['maintance_cost_per_km'] * params['yearly_km_traveled']
    # Annual Charge Cost Ev = Charging price * Monthly Electricity Consume per EV * 12
    [cite_start],anual_charge = params['charging_price'] * 101.753 * 12 # 101.753 es el consumo mensual del doc [cite: 28]
    
    anual_op_cost = (params['anual_insurance_cost_EV'] + 
                     anual_maint + 
                     params['anual_technical_revision_cost_ev'] + 
                     anual_charge)
    
    # Operation LCC of EV (Ecuación 90) - Valor presente de costos futuros
    r = params['discount_rate']
    [cite_start],n = 8.0 # EV Lifetime [cite: 6]
    pv_factor = (1 / r) * (1 - (1 / (1 + r)**n))
    # [cite_start]Por ahora Registration Fee EV es 0 según la nota "REVISAR" [cite: 93]
    operation_lcc = 0 + (anual_op_cost * pv_factor)
    
    # TCO Factor EV (Ecuación 88)
    suma_ev = operation_lcc + purchase_price
    tco_factor = max(suma_ev, 5000) * params['rel_importance_tco'][region_idx]
    
    # --- 2. FACTOR DE INFRAESTRUCTURA (Ecuación 76) ---
    infra_factor = stocks_ev['num_chargers_ev'] * params['rel_importance_infra'][region_idx]
    
    # --- 3. FACTOR DE TIEMPO DE CARGA (Ecuación 78) ---
    charging_time_factor = max(5, stocks_ev['ev_charging_time']) * params['rel_importance_charging_time'][region_idx]
    
    # --- 4. FACTOR DE AUTONOMÍA (Ecuación 82) ---
    range_factor = stocks_ev['ev_driving_range'] * params['rel_importance_driving_range'][region_idx]
    
    # --- RESULTADO FINAL: UTILIDAD PERCIBIDA (Ecuación 72) ---
    ev_perceived_utility = charging_time_factor + range_factor + tco_factor + infra_factor
    
    return ev_perceived_utility

def flujos_ev_utility(params, stocks_ev, region_idx):
    """
    Calcula los cambios (válvulas) que ocurren cada mes.
    """
    # Crecimiento de cargadores (Ecuación 74, 75)
    optimal_chargers = params['charging_station_per_ev'] * stocks_ev['ev_stock_region']
    chargers_growth = max(optimal_chargers - stocks_ev['num_chargers_ev'], 0) * params['time_delay_cg']
    
    # Reducción de tiempo de carga (Ecuación 81)
    ct_reduction = params['ev_charge_time_improve_rate'] * stocks_ev['ev_charging_time']
    
    # Crecimiento de autonomía (Ecuación 84)
    dr_net_growth = stocks_ev['ev_driving_range'] * params['dr_growth_rate_ev']
    
    return {
        'chargers_growth': chargers_growth,
        'ct_reduction': ct_reduction,
        'dr_net_growth': dr_net_growth
    }