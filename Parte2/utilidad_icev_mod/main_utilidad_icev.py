# -*- coding: utf-8 -*-
import sys
import os
import numpy as np

# Configurar ruta para que Python encuentre los módulos
ruta_actual = os.path.dirname(os.path.abspath(__file__))
ruta_raiz = os.path.abspath(os.path.join(ruta_actual, '..'))
if ruta_raiz not in sys.path:
    sys.path.insert(0, ruta_raiz)

# Importamos la función desde el archivo que acabamos de limpiar
from utilidad_icev_mod.utilidad_icev import calcular_utilidad_icev_completa

def auditar_icev_total():
    params_icev = {
        "rel_importance_tco": np.array([-0.011259, -0.0236662, -0.0168225]),
        "rel_importance_refuelling": np.array([-2.84744, -5.12323, -2.91483]),
        "rel_importance_driving_range": np.array([0.0942897, 0.0974696, 0.00795017]),
        "rel_importance_infra": np.array([0.000266909, 0.000314744, 0.564])
    }
    
    # Datos iniciales Región Centro
    p_act, r_act = 10240.1, 400.0
    fs_act = 1000.0 
    p_lag, r_lag = 0.0, 0.0
    u_display = None

    print(f"\nAUDITORÍA COMPLETA ICEV: TODAS LAS RAMAS (CENTRO)")
    print("=" * 125)
    print(f"{'Año':<5} | {'U. Total':>10} | {'FS Stat':>8} | {'TCO F':>9} | {'Refuel':>8} | {'Range':>8} | {'Infra':>8}")
    print("-" * 125)

    for anio in range(2023, 2031):
        # Mostrar resultados calculados el año pasado
        u_str = f"{u_display['utilidad_total']:>10.2f}" if u_display else "---"
        tco_s = f"{u_display['tco_factor']:>9.1f}" if u_display else "---"
        ref_s = f"{u_display['refuelling_factor']:>8.1f}" if u_display else "---"
        ran_s = f"{u_display['range_factor']:>8.1f}" if u_display else "---"
        inf_s = f"{u_display['infra_factor']:>8.1f}" if u_display else "---"
        
        print(f"{anio:<5} | {u_str} | {fs_act:>8.1f} | {tco_s} | {ref_s} | {ran_s} | {inf_s}")

        # Cálculos para la utilidad que se percibirá el próximo año
        datos = {
            'icev_base_price_purchase': p_lag, 
            'icev_driving_range_purchase': r_lag, 
            'value_fs': fs_act
        }
        u_display = calcular_utilidad_icev_completa(params_icev, datos, 1, anio)

        # Actualizar Lags y Evolución
        p_lag, r_lag = p_act, r_act
        p_act *= (1 - 0.07)
        r_act *= (1 + 0.0064)

if __name__ == "__main__":
    auditar_icev_total()