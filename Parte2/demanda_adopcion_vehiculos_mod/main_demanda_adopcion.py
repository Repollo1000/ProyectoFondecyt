# -*- coding: utf-8 -*-
import sys, os, numpy as np

# Configurar rutas
ruta_script = os.path.dirname(os.path.abspath(__file__))
ruta_raiz = os.path.abspath(os.path.join(ruta_script, '..'))
if ruta_raiz not in sys.path: sys.path.insert(0, ruta_raiz)

from utilidad_ev_mod.utilidad_ev import calcular_utilidad_ev_completa
from utilidad_icev_mod.utilidad_icev import calcular_utilidad_icev_completa
from demanda_adopcion_vehiculos_mod.demanda_adopcion_vehiculos import simulate_system
from parametros_globales_parte2 import UTILIDAD_EV_PARAMETROS as p_ev_global, DEMANDA_ADOPCION_INICIALES as p_a

def ejecutar_simulacion_total():
    # DATOS INICIALES 2023
    pop = p_a["population_initial"].copy().astype(float)
    ev_stock = p_a["ev_stock_initial"].copy().astype(float)
    icev_stock = p_a["icev_stock_initial"].copy().astype(float)
    motorization_rate = p_a["motorization_rate"]
    pop_growth = p_a["population_growth_rate"]
    
    # Tecnología e Infraestructura 2023
    p_ev, r_ev = p_ev_global["initial_base_price_ev"], p_ev_global["initial_driving_range_ev"]
    t_ev = (30.8 * 60 * (1 - 0.163)) / 22 
    p_icev, r_icev = 10240.1, 400.0
    cargadores = p_ev_global["inital_quantity_public_charges"].copy().astype(float)
    fs_stations = np.array([216.0, 1000.0, 551.0])

    # Sincronización de parámetros
    params_sync = p_ev_global.copy()
    params_sync['rel_importance_refuelling'] = params_sync.get('rel_importance_charging_time', np.array([-4.956, -5.12323, -4.852]))

    u_ev_disp, u_icev_disp = None, None

    print(f"\nAUDITORÍA CIRCULAR SINCRONIZADA (REGIÓN CENTRO)")
    print("=" * 135)
    print(f"{'Año':<5} | {'Stock EV':>10} | {'Stock ICEV':>11} | {'U. EV':>12} | {'U. ICEV':>11} | {'Cargad.':>9} | {'FS Stat':>9}")
    print("-" * 135)

    for t in range(28):
        anio = 2023 + t
        
        u_ev_s = f"{u_ev_disp[1]:>12.2f}" if u_ev_disp is not None else "---"
        u_icev_s = f"{u_icev_disp[1]:>11.2f}" if u_icev_disp is not None else "---"
        c_s = f"{cargadores[1]:>9.1f}" if anio > 2023 else "---"
        f_s = f"{fs_stations[1]:>9.1f}" if anio > 2023 else "---"
        print(f"{anio:<5} | {ev_stock[1]:>10.0f} | {icev_stock[1]:>11.0f} | {u_ev_s} | {u_icev_s} | {c_s} | {f_s}")

        if anio < 2050:
            # 1. ACTUALIZAR INFRAESTRUCTURA (Para el año próximo)
            next_carg = cargadores.copy(); next_fs = fs_stations.copy()
            for i in range(3):
                req_c = 0.04 * ev_stock[i]
                if req_c > next_carg[i]: next_carg[i] += (req_c - next_carg[i]) * 0.53
                req_f = (1013.0 / 3686350.0) * icev_stock[i]
                if req_f > next_fs[i]: next_fs[i] = req_f

            # 2. CALCULAR UTILIDADES (Percepción para el salto)
            u_ev_now, u_icev_now = np.zeros(3), np.zeros(3)
            for i in range(3):
                u_ev_now[i] = calcular_utilidad_ev_completa(params_sync, {'ev_base_price_purchase': p_ev, 'ev_driving_range_purchase': r_ev, 'ev_charging_time_purchase': t_ev, 'num_chargers_ev': next_carg[i]}, i)
                u_icev_now[i] = calcular_utilidad_icev_completa(params_sync, {'icev_base_price_purchase': p_icev, 'icev_driving_range_purchase': r_icev, 'value_fs': next_fs[i]}, i, anio + 1)["utilidad_total"]

            # 3. SIMULAR ADOPCIÓN
            res = simulate_system(pop, ev_stock, icev_stock, pop_growth, motorization_rate, 8.0, 8.0, 1.0, [u_ev_now, u_ev_now], [u_icev_now, u_icev_now], 1, 1)

            # 4. EVOLUCIÓN TECNOLOGÍA
            p_ev *= (1 + params_sync["price_reduction_rate_ev"])
            r_ev *= (1 + params_sync["dr_growth_rate_ev"])
            t_ev *= (1 - params_sync["ev_charge_time_improve_rate"])
            p_icev *= (1 - 0.07); r_icev *= (1 + 0.0064)

            # Sincronización
            u_ev_disp, u_icev_disp = u_ev_now, u_icev_now
            cargadores, fs_stations = next_carg, next_fs
            pop, ev_stock, icev_stock = res["population"][1], res["ev_stock"][1], res["icev_stock"][1]

if __name__ == "__main__":
    ejecutar_simulacion_total()