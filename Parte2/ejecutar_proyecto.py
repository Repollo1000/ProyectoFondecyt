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

def mostrar_menu():
    print("\n" + "="*110)
    print("      SISTEMA DE AUDITORÍA MODULAR - FONDECYT (Sincronización Vensim)")
    print("="*110)
    print("1. [Auditoría] Módulo EV (Tabla Completa)")
    print("2. [Auditoría] Módulo ICEV (Desglose)")
    print("4. Salir")
    return input("\nSeleccione opción: ")

def ejecutar_motor(opcion):
    # --- 1. ESTADO INICIAL 2023 ---
    pop = np.array([2.56368e+06, 1.30784e+07, 4.13657e+06])
    ev_s = np.array([828.0, 10999.0, 1517.0])
    ice_s = np.array([745762.0, 3686350.0, 1420100.0])
    
    p_growth_rates = np.array([0.0170807, 0.0143146, 0.00595781])
    m_rates = np.array([3.168, 2.8525, 2.9216])
    
    # Parámetros Tecnológicos (Valores Base T=0)
    p_ev_act, t_ev, r_ev_act = 19475.0, 70.308, 300.0
    p_ice_act, r_ice_act = 10240.1, 400.0
    
    cargadores_reales = np.array([39.0, 340.0, 84.0]).astype(float) 
    cargadores_percibidos = np.array([0.0, 0.0, 0.0]).astype(float) 
    fs_stat = np.array([216.0, 1000.0, 551.0]).astype(float)
    
    # Foto inicial para el reporte de Optimal (Delay de un año)
    stock_inicial_ev_foto = ev_s.copy() 
    params = p_ev_dict.copy()

    for anio in range(2023, 2051):
        # A. CÁLCULO DE UTILIDADES (Con el estado con el que NACE el año)
        u_ev_list, u_ice_list = [], []
        for reg_idx in range(3):
            res_ev = util_ev.calcular_utilidad_ev_completa(params, {
                'ev_base_price_purchase': p_ev_act, 
                'ev_charging_time_purchase': t_ev, 
                'ev_driving_range_purchase': r_ev_act, 
                'num_chargers_ev': cargadores_percibidos[reg_idx]
            }, reg_idx, anio)
            u_ev_list.append(res_ev)
            
            res_ice = util_icev.calcular_utilidad_icev_completa(params, {
                'icev_base_price_purchase': p_ice_act, 
                'icev_driving_range_purchase': r_ice_act, 
                'value_fs': fs_stat[reg_idx]
            }, reg_idx, anio)
            u_ice_list.append(res_ice)

        # B. REPORTES (Captura el estado ANTES de evolucionar)
        if opcion == '1': # REPORTE EV
            u = u_ev_list[1]
            opt_reporte = 0.0 if anio == 2023 else 0.04 * stock_inicial_ev_foto[1]
            if anio == 2023:
                header = (f"{'Año':<5} | {'Stock EV':>10} | {'Optimal':>9} | {'Cargad.':>9} | "
                          f"{'U. Tot':>9} | {'TCO':>9} | {'Range':>8} | {'Charg':>8} | {'Infra':>8}")
                print("\n" + header + "\n" + "-" * len(header))
                print(f"{anio:<5} | {ev_s[1]:>10.1f} | {0.0:>9.1f} | {0.00:>9.2f} | {0.00:>9.2f} | {0.00:>9.2f} | {0.00:>8.2f} | {0.00:>8.2f} | {0.0000:>8.4f}")
            else:
                print(f"{anio:<5} | {ev_s[1]:>10.1f} | {opt_reporte:>9.1f} | {cargadores_percibidos[1]:>9.2f} | {u['utilidad_total']:>9.2f} | {u['tco']:>9.2f} | {u['range']:>8.2f} | {u['charging']:>8.2f} | {u['infra']:>8.4f}")

        elif opcion == '2': # REPORTE ICEV
            ui = u_ice_list[1]
            if anio == 2023:
                header_ice = (f"{'Año':<5} | {'Stock ICE':>12} | {'U. Tot ICE':>11} | "
                              f"{'TCO':>10} | {'Range':>10} | {'Fuel Stat':>10}")
                print("\n" + header_ice + "\n" + "-" * len(header_ice))
                print(f"{anio:<5} | {ice_s[1]:>12.1f} | {0.00:>11.2f} | {0.00:>10.2f} | {0.00:>10.2f} | {0.0000:>10.4f}")
            else:
                print(f"{anio:<5} | {ice_s[1]:>12.1f} | {ui['utilidad_total']:>11.2f} | "
                      f"{ui['tco']:>10.2f} | {ui['range']:>10.2f} | {ui['infra']:>10.4f}")

        # C. EVOLUCIÓN (DINÁMICA DE SISTEMAS - EL ORDEN ES CRÍTICO)
        if anio < 2050:
            # 1. Foto para el reporte del Optimal de mañana (Antes de la demanda)
            stock_inicial_ev_foto = ev_s.copy()

            # 2. Shift de Percepción: Mañana se ve lo que hoy era real
            cargadores_percibidos = cargadores_reales.copy()

            # 3. TECNOLOGÍA: BLOQUE DE PROTECCIÓN
            # Solo mejoramos si ya reportamos 2024. Así 2024 mantiene valores base.
            if anio >= 2024:
                p_ev_act *= (1 - 0.136186)
                r_ev_act *= (1 + 0.0986)
                t_ev *= (1 - 0.0670057)
                p_ice_act *= (1 - 0.07)
                r_ice_act *= (1 + 0.0064)

            # 4. DEMANDA: El flujo de hoy calcula el stock de MAÑANA
            u_ev_tot = [u_item['utilidad_total'] for u_item in u_ev_list]
            u_ice_tot = [ui_item['utilidad_total'] for ui_item in u_ice_list]
            res = demanda.simulate_system(pop, ev_s, ice_s, p_growth_rates, m_rates, 8, 8, 1, [u_ev_tot], [u_ice_tot], 1, 1)
            pop, ev_s, ice_s = res["population"][1], res["ev_stock"][1], res["icev_stock"][1]

            # 5. INFRAESTRUCTURA REAL: Crece usando el stock actualizado por demanda
            for i in range(3):
                opt_c = 0.04 * ev_s[i]
                if opt_c > cargadores_reales[i]:
                    cargadores_reales[i] += (opt_c - cargadores_reales[i]) * 0.532634
                fs_stat[i] = (1013.0 / 3.68635e+06) * ice_s[i]

if __name__ == "__main__":
    while True:
        opt = mostrar_menu()
        if opt == '4': break
        if opt in ['1', '2']: ejecutar_motor(opt)