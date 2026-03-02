# -*- coding: utf-8 -*-
import sys
import os

def ejecutar_auditoria_total():
    ruta_padre = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if ruta_padre not in sys.path: sys.path.insert(0, ruta_padre)

    from parametros_globales_parte2 import UTILIDAD_EV_PARAMETROS as p
    from utilidad_ev import calcular_utilidad_ev_completa

    # --- ESTADOS INICIALES 2023 ---
    precio_stock = p["initial_base_price_ev"]
    rango_stock = p["initial_driving_range_ev"]
    tiempo_stock = (30.8 * 60 * (1 - 0.163)) / 22
    
    # INFRAESTRUCTURA REAL
    chargers_stock = p["inital_quantity_public_charges"][1] # 340
    ev_stock_real = 10999.0 # <--- DATO QUE ME PASASTE
    
    # Lags
    precio_ant, rango_ant, tiempo_ant = 0.0, 0.0, 0.0

    print(f"\nAUDITORÍA FINAL: INFRAESTRUCTURA Y UTILIDAD (CENTRO)")
    print("=" * 80)
    print(f"{'Año':<5} | {'EV Stock':>10} | {'Chargers':>10} | {'Infra Fact':>12} | {'Total U':>10}")
    print("-" * 80)

    for anio in range(2023, 2051):
        datos = {
            'ev_base_price': precio_stock,
            'ev_base_price_purchase': precio_ant,
            'ev_driving_range': rango_stock,
            'ev_driving_range_purchase': rango_ant,
            'ev_charging_time_purchase': tiempo_ant,
            'num_chargers_ev': chargers_stock
        }
        
        print(f"{anio:<5} | {ev_stock_real:>10.0f} | ", end="")
        calcular_utilidad_ev_completa(p, datos, 1, anio)
        
        # --- EVOLUCIÓN PARA EL PRÓXIMO AÑO ---
        precio_ant, rango_ant, tiempo_ant = precio_stock, rango_stock, tiempo_stock
        precio_stock *= (1 + p["price_reduction_rate_ev"])
        rango_stock *= (1 + p["dr_growth_rate_ev"])
        tiempo_stock *= (1 - 0.0670057)
        
        # LOGICA DE INFRAESTRUCTURA (VENSIM)
        optimal_chargers = 0.04 * ev_stock_real
        # Chargers growth = MAX(Optimal - Actual, 0) * 0.532634
        growth = max(optimal_chargers - chargers_stock, 0) * 0.532634
        chargers_stock += growth
        
        # NOTA: En una simulación real, ev_stock_real debería venir 
        # de los resultados de adopción. Por ahora lo dejamos fijo o 
        # le puedes poner un crecimiento estimado para probar.
        ev_stock_real *= 1.10 # (Crecimiento estimado del 10% para la prueba)

if __name__ == "__main__":
    ejecutar_auditoria_total()