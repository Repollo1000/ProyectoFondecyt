# -*- coding: utf-8 -*-

def calcular_utilidad_ev_completa(params, stocks_ev, region_idx, anio_actual):
    # --- 1. INFRAESTRUCTURA (Sincronización Directa Vensim) ---
    # Recibe el factor ya escalado (Cargadores * Importancia) desde el orquestador
    infra_factor = stocks_ev.get('num_chargers_ev', 0.0)
    
    # --- DATOS BASE ---
    Discount_rate = 0.06
    EV_Lifetime = 8
    Ev_appraisal_SII = 25960.5
    
    # --- 2. TCO (REGISTRATION FEE SII) ---
    def calc_tramo(appraisal):
        t1 = 60 * 0.01 if appraisal > 60 else appraisal * 0.01
        t2 = 60 * 0.02 if appraisal > 120 else max((appraisal - 60) * 0.02, 0)
        t3 = 130 * 0.03 if appraisal > 250 else max((appraisal - 120) * 0.03, 0)
        t4 = (150 * 0.04 + 0.045 * (appraisal - 400)) if appraisal > 400 else max((appraisal - 250) * 0.04, 0)
        return t1 + t2 + t3 + t4

    base_fee = calc_tramo(Ev_appraisal_SII)
    Registration_Fee_EV = (
        (base_fee * 0.25 * (1 / (1 + Discount_rate)**3)) +
        (base_fee * 0.25 * (1 / (1 + Discount_rate)**4)) +
        (base_fee * 0.50 * (1 / (1 + Discount_rate)**5)) +
        (base_fee * 0.50 * (1 / (1 + Discount_rate)**6)) +
        (base_fee * 0.75 * (1 / (1 + Discount_rate)**7)) +
        (base_fee * 0.75 * (1 / (1 + Discount_rate)**8))
    )

    # --- 3. OPERATING COST & LCC ---
    Annual_maint = (0.06 / 1.60934) * (41 * 12)
    Annual_charge = 0.026 * 101.753 * 12
    Anual_Op_Cost = 31.31 + 6.7 + Annual_maint + Annual_charge
    Op_LCC = Registration_Fee_EV + Anual_Op_Cost * (1 / Discount_rate) * (1 - (1 / (1 + Discount_rate)**EV_Lifetime))

    # TCO FACTOR
    base_price = stocks_ev.get('ev_base_price_purchase', 19475.0)
    subsidies = params.get('ev_subsidies_percent', 0.0)
    purchase_cost = base_price - (base_price * subsidies)
    tco_total = max(purchase_cost + Op_LCC, 5000)
    tco_factor = tco_total * params['rel_importance_tco'][region_idx]

    # --- 4. RANGE Y CHARGING ---
    range_factor = stocks_ev.get('ev_driving_range_purchase', 300.0) * params['rel_importance_driving_range'][region_idx]
    charging_time = stocks_ev.get('ev_charging_time_purchase', 70.308)
    charging_factor = max(5, charging_time) * params['rel_importance_charging_time'][region_idx]

    utilidad_total = tco_factor + range_factor + charging_factor + infra_factor

    return {
        "utilidad_total": utilidad_total,
        "tco": tco_factor, "range": range_factor, "charging": charging_factor, "infra": infra_factor
    }