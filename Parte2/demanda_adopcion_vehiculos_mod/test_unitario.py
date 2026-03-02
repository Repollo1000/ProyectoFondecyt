# -*- coding: utf-8 -*-
import sys
import os
import numpy as np
import pandas as pd

# --- CONEXIÓN DE RUTAS (Estilo Seba) ---
# Subimos a la carpeta 'Parte2' para poder importar todo
ruta_raiz = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ruta_raiz not in sys.path:
    sys.path.append(ruta_raiz)

try:
    # Importamos los parámetros de la raíz
    from parametros_globales_parte2 import UTILIDAD_EV_PARAMETROS, DEMANDA_ADOPCION_INICIALES
    # Importamos la lógica de los otros módulos
    from utilidad_ev_mod.utilidad_ev import calcular_utilidad_ev_completa, flujos_ev_utility
    from demanda_adopcion_vehiculos import simulate_system
    print("✅ Conexiones exitosas desde el Main de Adopción.")
except ImportError as e:
    print(f"❌ Error al conectar módulos: {e}")
    sys.exit()

def ejecutar_simulacion_completa():
    p_u = UTILIDAD_EV_PARAMETROS
    p_a = DEMANDA_ADOPCION_INICIALES
    
    # Stocks iniciales
    stock_ev = p_a["ev_stock_initial"].copy().astype(float)
    stock_icev = p_a["icev_stock_initial"].copy().astype(float)
    pop = p_a["population_initial"].copy().astype(float)
    cargadores = p_u["inital_quantity_public_charges"].copy().astype(float)
    
    # Variables tecnológicas
    precio_ev = p_u["ev_appraisal_SII"]
    rango_ev = p_u["initial_driving_range_ev"]

    print(f"\n{'AÑO':<6} | {'UTILIDAD CENTRO':<18} | {'STOCK EV CENTRO':<15}")
    print("-" * 45)

    for anio_idx in range(27): # 2023 a 2049
        anio_actual = 2023 + anio_idx
        
        # 1. LLAMADA AL MÓDULO DE UTILIDAD (Conexión)
        u_ev_anio = np.zeros(3)
        for i in range(3):
            datos_u = {
                'ev_base_price': precio_ev,
                'num_chargers_ev': cargadores[i],
                'ev_driving_range': rango_ev,
                'ev_charging_time': 82.8, # Valor base
                'ev_stock_region': stock_ev[i]
            }
            u_ev_anio[i] = calcular_utilidad_ev_completa(p_u, datos_u, i)

        print(f"{anio_actual:<6} | {u_ev_anio[1]:<18.2f} | {stock_ev[1]:<15.0f}")

        # 2. LLAMADA AL MÓDULO DE ADOPCIÓN
        if anio_idx < 26:
            u_ev_in = np.array([u_ev_anio, u_ev_anio])
            u_icev_in = np.array([[-550.0]*3, [-550.0]*3]) # Utilidad fija ICEV

            res = simulate_system(
                pop, stock_ev, stock_icev,
                p_a["population_growth_rate"], p_a["motorization_rate"],
                8.0, 8.0, 1.0, u_ev_in, u_icev_in, 1, 1
            )
            
            # Actualizamos stocks para el siguiente año
            stock_ev = res["ev_stock"][1]
            stock_icev = res["icev_stock"][1]
            pop = res["population"][1]

            # 3. ACTUALIZAR CARGADORES (Flujos de Utilidad)
            for i in range(3):
                f = flujos_ev_utility(p_u, {'num_chargers_ev': cargadores[i], 'ev_stock_region': stock_ev[i]}, i)
                cargadores[i] += f['chargers_growth']

            # Evolución tecnológica
            precio_ev *= (1 - 0.07)
            rango_ev *= (1 + p_u["dr_growth_rate_ev"])

if __name__ == "__main__":
    ejecutar_simulacion_completa()