# -*- coding: utf-8 -*-
import sys, os, numpy as np, importlib.util

def importar_modulo(nombre_modulo, ruta_relativa):
    ruta_abs = os.path.join(os.path.dirname(os.path.abspath(__file__)), ruta_relativa)
    spec = importlib.util.spec_from_file_location(nombre_modulo, ruta_abs)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo

# --- CARGA DE MÓDULOS ---
try:
    util_ev = importar_modulo("utilidad_ev", "utilidad_ev_mod/utilidad_ev.py")
    util_icev = importar_modulo("utilidad_icev", "utilidad_icev_mod/utilidad_icev.py")
    demanda = importar_modulo("demanda_adopcion_vehiculos", "demanda_adopcion_vehiculos_mod/demanda_adopcion_vehiculos.py")
    import parametros_globales_parte2 as params_global
    p_ev_dict = params_global.UTILIDAD_EV_PARAMETROS
except Exception as e:
    print(f"\n❌ Error cargando archivos: {e}"); sys.exit(1)

# --- MÓDULO DE EMISIONES (NOMBRES VENSIM) ---
def calcular_emisiones_vensim(stock_ice, stock_ev, ev_sales_region):
    # Variables base
    yearly_km = 41 * 12
    perf_ice = 11
    ind_co2_ice = 0.00016893
    emi_lt = 2.74 * 0.01
    
    # 1. Annual emissions per ICEV
    ann_emi_icev = (ind_co2_ice * yearly_km) + ((emi_lt / perf_ice) * yearly_km)
    
    # 2. Total emission per year ICEV
    tot_emi_icev = ann_emi_icev * stock_ice
    
    # 3. Annual emissions per EV
    grid_intensity = 0.0003006
    monthly_consume = 101.753
    ann_emi_ev = grid_intensity * monthly_consume * 12
    
    # 4. Total emission per year EV
    tot_emi_ev = ann_emi_ev * stock_ev
    
    # 5. Avoided CO2 Emissions Rate (Flujo)
    avoided_rate = (ann_emi_icev - ann_emi_ev) * ev_sales_region * 1
    
    return {
        "ann_emi_icev": ann_emi_icev,
        "tot_emi_icev": tot_emi_icev,
        "ann_emi_ev": ann_emi_ev,
        "tot_emi_ev": tot_emi_ev,
        "avoided_rate": avoided_rate
    }

def mostrar_menu():
    print("\n" + "="*145)
    print("      SISTEMA DE AUDITORÍA MODULAR - FONDECYT (Sincronización Vensim)")
    print("="*145)
    print("1. [Auditoría] Módulo EV")
    print("2. [Auditoría] Módulo ICEV")
    print("3. [Reporte]   Módulo Emisiones (Desglose Completo Vensim)")
    print("4. Salir")
    return input("\nSeleccione opción: ")

def ejecutar_motor(opcion):
    pop = np.array([2.56368e+06, 1.30784e+07, 4.13657e+06])
    ev_s = np.array([828.0, 10999.0, 1517.0])
    ice_s = np.array([745762.0, 3686350.0, 1420100.0])
    p_growth_rates = np.array([0.0170807, 0.0143146, 0.00595781])
    m_rates = np.array([3.168, 2.8525, 2.9216])
    p_ev_act, t_ev, r_ev_act = 19475.0, 70.308, 300.0
    p_ice_act, r_ice_act = 10240.1, 400.0
    cargadores_reales = np.array([39.0, 340.0, 84.0]).astype(float) 
    cargadores_percibidos = np.array([0.0, 0.0, 0.0]).astype(float) 
    fs_stat = np.array([216.0, 1000.0, 551.0]).astype(float)
    
    stock_inicial_ev_foto = ev_s.copy() 
    params = p_ev_dict.copy()

    # --- INTEG: Avoided CO2 Emissions (Stock) ---
    avoided_co2_emissions_stock = 0.0

    for anio in range(2023, 2051):
        u_ev_list, u_ice_list = [], []
        for reg_idx in range(3):
            res_ev = util_ev.calcular_utilidad_ev_completa(params, {
                'ev_base_price_purchase': p_ev_act, 'ev_charging_time_purchase': t_ev, 
                'ev_driving_range_purchase': r_ev_act, 'num_chargers_ev': cargadores_percibidos[reg_idx]
            }, reg_idx, anio)
            u_ev_list.append(res_ev)
            
            res_ice = util_icev.calcular_utilidad_icev_completa(params, {
                'icev_base_price_purchase': p_ice_act, 'icev_driving_range_purchase': r_ice_act, 
                'value_fs': fs_stat[reg_idx]
            }, reg_idx, anio)
            u_ice_list.append(res_ice)

        # CÁLCULO EMISIONES (Para Región Centro [1])
        ev_sales_hoy = ev_s[1] - stock_inicial_ev_foto[1] if anio > 2023 else 0
        emi = calcular_emisiones_vensim(ice_s[1], ev_s[1], ev_sales_hoy)
        
        # INTEG
        avoided_co2_emissions_stock += emi['avoided_rate']

        if opcion == '3':
            if anio == 2023:
                h = (f"{'Año':<5} | {'Ann.Emi ICE':>11} | {'Tot.Emi ICE':>12} | {'Ann.Emi EV':>11} | "
                     f"{'Tot.Emi EV':>11} | {'EV Sales':>10} | {'Avoid.Rate':>11} | {'Avoid.Stock':>12}")
                print("\n" + h + "\n" + "-" * len(h))
            print(f"{anio:<5} | {emi['ann_emi_icev']:>11.4f} | {emi['tot_emi_icev']:>12.1f} | {emi['ann_emi_ev']:>11.4f} | "
                  f"{emi['tot_emi_ev']:>11.1f} | {ev_sales_hoy:>10.1f} | {emi['avoided_rate']:>11.2f} | {avoided_co2_emissions_stock:>12.2f}")

        if anio < 2050:
            stock_inicial_ev_foto = ev_s.copy()
            cargadores_percibidos = cargadores_reales.copy()
            if anio >= 2024:
                p_ev_act *= (1 - 0.136186); r_ev_act *= (1 + 0.0986); t_ev *= (1 - 0.0670057)
                p_ice_act *= (1 - 0.07); r_ice_act *= (1 + 0.0064)

            u_ev_tot = [u_item['utilidad_total'] for u_item in u_ev_list]
            u_ice_tot = [ui_item['utilidad_total'] for ui_item in u_ice_list]
            res = demanda.simulate_system(pop, ev_s, ice_s, p_growth_rates, m_rates, 8, 8, 1, [u_ev_tot], [u_ice_tot], 1, 1)
            pop, ev_s, ice_s = res["population"][1], res["ev_stock"][1], res["icev_stock"][1]
            for i in range(3):
                opt_c = 0.04 * ev_s[i]
                if opt_c > cargadores_reales[i]:
                    cargadores_reales[i] += (opt_c - cargadores_reales[i]) * 0.532634
                fs_stat[i] = (1013.0 / 3.68635e+06) * ice_s[i]

if __name__ == "__main__":
    while True:
        opt = mostrar_menu()
        if opt == '4': break
        if opt in ['1', '2', '3']: ejecutar_motor(opt)