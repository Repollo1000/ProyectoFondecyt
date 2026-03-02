# -*- coding: utf-8 -*-

def calcular_utilidad_icev_completa(params, datos_icev, region_idx, anio_actual):
    # Ya no bloqueamos el retorno en 2023. El orquestador controla el reporte.
    
    # --- 1. TCO (Precio + Operativos LCC) ---
    p_base = datos_icev.get('icev_base_price_purchase', 10240.1)
    purchase_price_cost = p_base * 1.015 # Registration Fee
    
    yearly_km = 41 * 12
    fuel_cost = (1.28 / 11) * yearly_km
    maint_cost = 0.05 * yearly_km
    annual_op_cost = 6.7 + maint_cost + 208.79 + 31.31 + fuel_cost
    
    annuity_factor = (1 - (1.06**-8)) / 0.06
    operation_lcc = annual_op_cost * annuity_factor
    
    tco_total = max(operation_lcc + purchase_price_cost, 5000)
    tco_factor = tco_total * params['rel_importance_tco'][region_idx]

    # --- 2. REFUELING ---
    imp_refuel = params.get('rel_importance_refuelling', params.get('rel_importance_charging_time'))
    refuel_factor = 5.0 * imp_refuel[region_idx]

    # --- 3. RANGE ---
    range_val = datos_icev.get('icev_driving_range_purchase', 400.0)
    range_factor = range_val * params['rel_importance_driving_range'][region_idx]

    # --- 4. INFRAESTRUCTURA ---
    infra_val = datos_icev.get('value_fs', 0.0)
    infra_factor = infra_val * params['rel_importance_infra'][region_idx]

    return {
        "utilidad_total": tco_factor + refuel_factor + range_factor + infra_factor,
        "tco": tco_factor, "refuel": refuel_factor, "range": range_factor, "infra": infra_factor
    }