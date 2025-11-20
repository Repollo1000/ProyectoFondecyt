# -*- coding: utf-8 -*-
from __future__ import annotations
import numpy as np
from prettytable import PrettyTable, SINGLE_BORDER
from . import modulo5 as m5
import os 

# --- IMPORTACIONES CLAVE ---
# Importamos el nuevo archivo de parámetros
from .. import parametros_globales as p_g 
# -------------------------

# ==================================
# UTILIDADES DE IMPRESIÓN
# ==================================

def tabla_subsidio(info_subsidio, regiones):
    """Muestra tabla con información de subsidios aplicados."""
    t = PrettyTable()
    t.set_style(SINGLE_BORDER)
    t.field_names = ["Región", "Difference usado", "Subsidio aplicado"]
    
    diff = info_subsidio["difference"]
    subs = info_subsidio["subsidio"]
    
    for i, reg in enumerate(regiones):
        d = float(diff[i]) if diff is not None and getattr(diff, "ndim", 0) > 0 else (
            float(diff) if diff is not None else float('nan'))
        s = float(subs[i])
        t.add_row([reg, f"{d:.3f}" if diff is not None else "-", f"{s:.3f}"])
    
    return t

def tabla_sstc_mensual(sstc, regiones, meses_mostrar=12, mes_inicio=0):
    """
    Tabla compacta de SSTC mensual.
    """
    N = sstc.shape[1]
    mes_fin = min(mes_inicio + meses_mostrar, N)
    
    t = PrettyTable()
    t.set_style(SINGLE_BORDER)
    t.field_names = ["Mes"] + list(regiones)
    
    for m in range(mes_inicio, mes_fin):
        fila = [m] + [f"{sstc[i, m]:,.0f}" for i in range(len(regiones))]
        t.add_row(fila)
    
    return t

def tabla_energia_mensual(energia, regiones, meses_mostrar=12, mes_inicio=0):
    """
    Tabla compacta de Energía mensual.
    """
    N = energia.shape[1]
    mes_fin = min(mes_inicio + meses_mostrar, N)
    
    t = PrettyTable()
    t.set_style(SINGLE_BORDER)
    t.field_names = ["Mes"] + list(regiones)
    
    for m in range(mes_inicio, mes_fin):
        fila = [m] + [f"{energia[i, m]:,.1f}" for i in range(len(regiones))]
        t.add_row(fila)
    
    return t

def tabla_lcoe_mensual(lcoe, regiones, meses_mostrar=12, mes_inicio=0):
    """
    Tabla compacta de LCOE mensual.
    """
    N = lcoe.shape[1]
    mes_fin = min(mes_inicio + meses_mostrar, N)
    
    t = PrettyTable()
    t.set_style(SINGLE_BORDER)
    t.field_names = ["Mes"] + list(regiones)
    
    for m in range(mes_inicio, mes_fin):
        # LCOE se muestra con más decimales
        fila = [m] + [f"{lcoe[i, m]:.4f}" for i in range(len(regiones))]
        t.add_row(fila)
    
    return t


# ==================================
# FUNCIÓN PRINCIPAL
# ==================================
def main():
    print("=" * 80)
    print("MÓDULO 5: Costo Nivelado de la Electricidad (LCOE) Mensual")
    print("=" * 80)
    
    # 1. Definición de rutas
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')) 
    DATOS_DIR = os.path.join(BASE_DIR, 'Parte1', 'Datos')

    RUTA_COSTO_EXCEL = os.path.join(DATOS_DIR, "costoAño.xlsx")

    # 2. Correr el modelo
    variables = m5.default_variables()
    
    # Usamos la vida útil del proyecto definida globalmente
    variables["project_lifetime_months"] = p_g.PROJECT_LIFETIME_MONTHS

    print(f"\n[1/2] Calculando LCOE para {p_g.PROJECT_LIFETIME_MONTHS} meses ({p_g.PROJECT_LIFETIME_MONTHS // 12} años)...")
    
    try:
        resultados = m5.correr_modelo(RUTA_COSTO_EXCEL, variables)
    except FileNotFoundError:
        print(f"\n✗ ERROR: Archivo de costos no encontrado en: {RUTA_COSTO_EXCEL}")
        return
    
    # 3. Extraer resultados
    regiones = resultados["regiones"] 
    sstc = resultados["sstc_mensual"]
    
    # --- CORRECCIÓN: EXTRAER ENERGÍA Y LCOE ---
    energia = resultados["energia_mensual"] 
    lcoe = resultados["lcoe_mensual"]
    # ------------------------------------------
    
    info_subsidio = resultados["info_subsidio"]
    
    # 4. Imprimir resultados
    print("\n[2/2] Resultados Completados.")
    
    # Subsidio
    print("\n" + "=" * 70)
    print("INFORMACIÓN DE SUBSIDIO APLICADO")
    print("=" * 70)
    print(f"Fuente: {info_subsidio['fuente']}")
    print(tabla_subsidio(info_subsidio, regiones))
    
    # Tablas de resultados mensuales (primeros 12 meses)
    print("\n" + "=" * 70)
    print("SSTC MENSUAL [USD] - Primeros 12 meses")
    print("=" * 70)
    print(tabla_sstc_mensual(sstc, regiones, meses_mostrar=12))
    
    # --- CORRECCIÓN: IMPRIMIR ENERGÍA ---
    print("\n" + "=" * 70)
    print("ENERGÍA MENSUAL [kWh/mes por hogar] - Primeros 12 meses")
    print("=" * 70)
    print(tabla_energia_mensual(energia, regiones, meses_mostrar=12))
    
    # --- CORRECCIÓN: IMPRIMIR LCOE ---
    print("\n" + "=" * 70)
    print("LCOE MENSUAL [USD/kWh] - Primeros 12 meses")
    print("=" * 70)
    print(tabla_lcoe_mensual(lcoe, regiones, meses_mostrar=12))
    # ----------------------------------

    # 6) Estadísticas (Ejemplo conceptual, adapte a su código original)
    print("\n" + "=" * 70)
    print("ESTADÍSTICAS DEL MODELO")
    print("=" * 70)
    # ... (Aquí iría su lógica de impresión de estadísticas) ...
    
    print("\n" + "=" * 80)
    print("MÓDULO 5 COMPLETADO ✓")
    print("=" * 80)

if __name__ == "__main__":
    main()