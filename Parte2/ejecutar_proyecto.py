# -*- coding: utf-8 -*-
import sys, os, numpy as np, importlib.util

def importar_modulo(nombre_modulo, ruta_relativa):
    ruta_abs = os.path.join(os.path.dirname(os.path.abspath(__file__)), ruta_relativa)
    spec = importlib.util.spec_from_file_location(nombre_modulo, ruta_abs)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo

# --- CARGA DE MÓDULOS EXTERNOS ---
try:
    util_ev = importar_modulo("utilidad_ev", "utilidad_ev_mod/utilidad_ev.py")
    util_icev = importar_modulo("utilidad_icev", "utilidad_icev_mod/utilidad_icev.py")
    demanda = importar_modulo("demanda_adopcion_vehiculos", "demanda_adopcion_vehiculos_mod/demanda_adopcion_vehiculos.py")
    politicas = importar_modulo("policy_cost", "policy_cost_mod/policy_cost.py")
    import parametros_globales_parte2 as params_global
except Exception as e:
    print(f"\n❌ Error: {e}"); sys.exit(1)

def calcular_reporte_sostenibilidad_completo(stock_ice, stock_ev, ventas_acumuladas_ev):
    yearly_km = 41 * 12
    perf_ice = 11
    ind_co2_ice = 0.00016893
    emi_lt = 2.74 * 0.01
    emi_unit_ice = (ind_co2_ice * yearly_km) + ((emi_lt / perf_ice) * yearly_km)
    total_ice_emi = emi_unit_ice * stock_ice
    grid_intensity = 0.0003006
    monthly_consume = 101.753
    emi_unit_ev = grid_intensity * monthly_consume * 12
    total_ev_emi = emi_unit_ev * stock_ev
    evitadas_rate = (emi_unit_ice - emi_unit_ev) * ventas_acumuladas_ev
    return {
        "total_ice_ton": total_ice_emi,
        "total_ev_ton": total_ev_emi,
        "co2_evitado_anio": evitadas_rate
    }

def mostrar_menu():
    print("\n" + "="*120)
    print("      SISTEMA DE AUDITORÍA TOTAL - FONDECYT (Módulo Multirregional)")
    print("="*120)
    print("1. Módulo Demanda y Stocks   (Población, Stocks, Demand, MS EV/ICE, Sales EV/ICE)")
    print("2. Módulo Utilidad EV        (Utilidad Total, TCO, Range, Charging, Infraestructura)")
    print("3. Módulo Utilidad ICEV      (Utilidad Total, TCO, Range, Refuelling, FS)")
    print("4. Módulo Emisiones de CO2   (Emisiones ICEV, EV, Totales, CO2 Evitado)")
    print("5. Módulo Políticas y Costos (Base CS, Taxes, Subsidios Cum, Cost Pol CS)")
    print("6. Salir")
    return input("\nSeleccione opción: ")

def ejecutar_motor(opcion):
    # --- ESTADO INICIAL (2023) ---
    pop = np.array([2.56368e+06, 1.30784e+07, 4.13657e+06])
    ev_s = np.array([828.0, 10999.0, 1517.0])
    ice_s = np.array([745762.0, 3686350.0, 1420100.0])
    p_growth_rates = np.array([0.0170807, 0.0143146, 0.00595781])
    m_rates = np.array([3.168, 2.8525, 2.9216])
    
    p_ev_act, t_ev, r_ev_act = 19475.0, 70.308, 300.0
    p_ice_act, r_ice_act = 10240.1, 400.0
    
    cargadores_reales = np.array([39.0, 340.0, 84.0]).astype(float)
    base_cargadores_ev = np.array([39.0, 340.0, 84.0]).astype(float)
    rel_importance_infra = [0.000266909, 0.000314744, 0.564]
    fs_stat = np.array([216.0, 1000.0, 551.0]).astype(float)
    
    cum_ev_sales = np.zeros(3)
    
    # --- VARIABLES INICIALES MÓDULO 5 (POLÍTICAS) ---
    cum_subsidy_cost = np.zeros(3)
    cum_tax_income = np.zeros(3)
    cum_cs_policy_cost = np.zeros(3)
    cost_per_cs = 12605.0 # Valor original (Positivo, la resta lo hará negativo)
    
    # PARÁMETROS MÓDULO 5
    ev_subsidies_percent = 0.0 
    icev_additional_taxes = 0.015
    base_cs_per_ev_rate = 0.033
    time_delay_cg = 0.532634
    price_reduction_per_year = 0.145
    
    regiones = ['Norte', 'Centro', 'Sur']

    if opcion == '1':
        h = f"{'Año':<5} | {'Región':<6} | {'Población':>10} | {'Stock EV':>9} | {'Stock ICE':>10} | {'Demand':>9} | {'MS EV':>9} | {'MS ICE':>9} | {'Sales EV':>9} | {'Sales ICE':>9}"
    elif opcion == '2':
        h = f"{'Año':<5} | {'Región':<6} | {'U. EV':>7} | {'TCO':>7} | {'RNG':>6} | {'CHG':>6} | {'INF':>7}"
    elif opcion == '3':
        h = f"{'Año':<5} | {'Región':<6} | {'U. ICE':>7} | {'TCO':>7} | {'RNG':>6} | {'REF':>6} | {'FS':>7}"
    elif opcion == '4':
        h = f"{'Año':<5} | {'Región':<6} | {'Emi ICE (t)':>12} | {'Emi EV (t)':>11} | {'Emi Total (t)':>13} | {'CO2 Evitado (t)':>15}"
    elif opcion == '5':
        h = f"{'Año':<5} | {'Región':<6} | {'Base CS EV':>12} | {'Tax ICEV ($)':>14} | {'Subsidy Cum ($)':>15} | {'Cost Pol CS ($)':>16}"

    ancho_linea = len(h)
    print("\n" + h)
    print("=" * ancho_linea)

    for i in range(3):
        if opcion == '1':
            print(f"{2023:<5} | {regiones[i]:<6} | {pop[i]:>10.0f} | {ev_s[i]:>9.0f} | {ice_s[i]:>10.0f} | {'---':>9} | {'---':>9} | {'---':>9} | {'---':>9} | {'---':>9}")
        elif opcion in ['2', '3', '4']:
            print(f"{2023:<5} | {regiones[i]:<6} | " + " | ".join(["---"] * (len(h.split('|')) - 2)))
        elif opcion == '5':
            print(f"{2023:<5} | {regiones[i]:<6} | {base_cargadores_ev[i]:>12.1f} | {'0':>14} | {'0':>15} | {'0':>16}")
            
    print("-" * ancho_linea)

    for anio in range(2024, 2051):
        
        stock_ev_actual = ev_s.copy()
        
        if anio >= 2025:
            p_ev_act *= (1 - 0.136186)
            r_ev_act *= (1 + 0.0986)
            t_ev *= (1 - 0.0670057)
            p_ice_act *= (1 - 0.07)
            r_ice_act *= (1 + 0.0064)
            cost_per_cs *= (1 - price_reduction_per_year)

        next_cargadores = cargadores_reales.copy()
        next_base_cargadores = base_cargadores_ev.copy()
        actual_chargers_growth = np.zeros(3)
        base_chargers_growth = np.zeros(3)
        
        for i in range(3):
            opt_c = 0.04 * ev_s[i]
            if opt_c > cargadores_reales[i]:
                actual_chargers_growth[i] = (opt_c - cargadores_reales[i]) * time_delay_cg
                next_cargadores[i] += actual_chargers_growth[i]
            else:
                actual_chargers_growth[i] = 0.0
                
            opt_base = base_cs_per_ev_rate * stock_ev_actual[i]
            base_chargers_growth[i] = opt_base * time_delay_cg
            next_base_cargadores[i] += base_chargers_growth[i]
                
            calc_fs = (1013.0 / 3.68635e+06) * ice_s[i]
            if calc_fs > fs_stat[i]:
                fs_stat[i] = calc_fs

        u_ev_list, u_ice_list = [], []
        for reg_idx in range(3):
            f_infra = cargadores_reales[reg_idx] * rel_importance_infra[reg_idx]
            res_ev = util_ev.calcular_utilidad_ev_completa(params_global.UTILIDAD_EV_PARAMETROS, 
                {'ev_base_price_purchase': p_ev_act, 'ev_charging_time_purchase': t_ev, 'ev_driving_range_purchase': r_ev_act, 'num_chargers_ev': f_infra}, reg_idx, anio)
            u_ev_list.append(res_ev)
            res_ice = util_icev.calcular_utilidad_icev_completa(params_global.UTILIDAD_EV_PARAMETROS, 
                {'icev_base_price_purchase': p_ice_act, 'icev_driving_range_purchase': r_ice_act, 'value_fs': fs_stat[reg_idx]}, reg_idx, anio)
            u_ice_list.append(res_ice)

        u_ev_tot = [u['utilidad_total'] for u in u_ev_list]
        u_ice_tot = [ui['utilidad_total'] for ui in u_ice_list]

        cargadores_reales = next_cargadores

        res = demanda.simulate_system(pop, ev_s, ice_s, p_growth_rates, m_rates, 8, 8, 1, [u_ev_tot], [u_ice_tot], 1, 1)
        
        v_demand_all = res["vehicle_demand"][0] 
        u_diff_all = np.array(u_ev_tot) - np.array(u_ice_tot)
        m_share_ev_all = 1 / (1 + np.exp(-u_diff_all))
        m_share_ice_all = 1 - m_share_ev_all
        sales_ev_all = v_demand_all * m_share_ev_all
        sales_ice_all = v_demand_all * m_share_ice_all
        
        cum_ev_sales += sales_ev_all
        
        for i in range(3):
            res_pol = politicas.calcular_politicas_y_costos(
                ev_sales=sales_ev_all[i], 
                icev_sales=sales_ice_all[i], 
                ev_base_price=p_ev_act, 
                icev_base_price=p_ice_act,
                crecimiento_real_cs=actual_chargers_growth[i], 
                crecimiento_base_cs=base_chargers_growth[i], 
                costo_unitario_cs=cost_per_cs,
                pct_subsidio_ev=ev_subsidies_percent
            )
            
            cum_subsidy_cost[i] += res_pol["annual_subsidy"]
            cum_tax_income[i] += res_pol["annual_tax"]
            cum_cs_policy_cost[i] += res_pol["annual_cs_policy"]
        
        next_pop = res["population"][1]
        next_ev_s = res["ev_stock"][1]
        next_ice_s = res["icev_stock"][1]
        base_cargadores_ev = next_base_cargadores

        for i in range(3):
            if opcion == '1':
                print(f"{anio:<5} | {regiones[i]:<6} | {next_pop[i]:>10.0f} | {next_ev_s[i]:>9.0f} | {next_ice_s[i]:>10.0f} | {v_demand_all[i]:>9.0f} | {m_share_ev_all[i]:>9.2e} | {m_share_ice_all[i]:>9.2e} | {sales_ev_all[i]:>9.0f} | {sales_ice_all[i]:>9.0f}")
            elif opcion == '2':
                p = u_ev_list[i]
                print(f"{anio:<5} | {regiones[i]:<6} | {u_ev_tot[i]:>7.1f} | {p.get('tco', 0):>7.1f} | {p.get('range', 0):>6.1f} | {p.get('charging', 0):>6.1f} | {p.get('infra', 0):>7.4f}")
            elif opcion == '3':
                p_ice = u_ice_list[i]
                val_refuel = p_ice.get('refuel', 0)
                val_fs = fs_stat[i] * rel_importance_infra[i]
                print(f"{anio:<5} | {regiones[i]:<6} | {u_ice_tot[i]:>7.1f} | {p_ice.get('tco', 0):>7.1f} | {p_ice.get('range', 0):>6.1f} | {val_refuel:>6.1f} | {val_fs:>7.4f}")
            elif opcion == '4':
                rep = calcular_reporte_sostenibilidad_completo(ice_s[i], ev_s[i], cum_ev_sales[i])
                emi_tot = rep['total_ice_ton'] + rep['total_ev_ton']
                print(f"{anio:<5} | {regiones[i]:<6} | {rep['total_ice_ton']:>12.1f} | {rep['total_ev_ton']:>11.1f} | {emi_tot:>13.1f} | {rep['co2_evitado_anio']:>15.1f}")
            elif opcion == '5':
                print(f"{anio:<5} | {regiones[i]:<6} | {next_base_cargadores[i]:>12.1f} | {cum_tax_income[i]:>14.0f} | {cum_subsidy_cost[i]:>15.0f} | {cum_cs_policy_cost[i]:>16.0f}")

        print("-" * ancho_linea)

        pop, ev_s, ice_s = next_pop, next_ev_s, next_ice_s

if __name__ == "__main__":
    while True:
        opt = mostrar_menu()
        if opt == '6': break
        if opt in ['1', '2', '3', '4', '5']:
            ejecutar_motor(opt)
        else:
            print("Opción no válida.")