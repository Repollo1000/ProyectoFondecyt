# -*- coding: utf-8 -*-
from __future__ import annotations
import os
import numpy as np
from prettytable import PrettyTable, SINGLE_BORDER

# --- Importaciones de módulos
from ..modulo5 import modulo5 as m5
from . import modulo8 as m8
from .. import parametros_globales as p_g
# ----------------------------


# ----------------------------
# Configuración rápida
# ----------------------------
PRINT_ROWS = 8      # filas a mostrar en tablas
EVAL_YEARS = None   # ej. 5 para evaluar solo primeros N años; None = todos
TAU_YEARS = 1       # ventana de suavizado en años (1 = sin suavizado)


def tabla_por_año(nombre: str, years, arr_3xT, max_years: int = 8, fmt="{:.4f}"):
    """
    Imprime tabla con encabezado [t, Norte, Centro, Sur] para series (3,T).
    Ahora usa las REGIONES de parametros_globales.
    """
    print(f"\n=== {nombre}: ===")
    t = PrettyTable()
    t.set_style(SINGLE_BORDER)
    t.field_names = ["t"] + list(p_g.REGIONES)
    t.align["t"] = "r"
    
    limite = min(len(years), max_years)
    for k in range(limite):
        fila = [int(years[k])] + [fmt.format(float(arr_3xT[i, k])) for i in range(len(p_g.REGIONES))]
        t.add_row(fila)
    print(t)


def main():
    print("=" * 80)
    print("MÓDULO 8: Payback Analysis & Delayed Payback Fraction")
    print("=" * 80)
    
    # --- Definición de Rutas (Centralizadas) ---
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) # Carpeta Parte1
    DATOS_DIR = os.path.join(BASE_DIR, 'Datos')

    RUTA_COSTOS_M5 = os.path.join(DATOS_DIR, "costoAño.xlsx")
    RUTA_AHORROS_M8 = os.path.join(DATOS_DIR, "annual_savings.xlsx")
    
    # ----------------------------------------------------------------------
    # PASO 1: CARGAR COSTOS (SSTC) y AHORROS (S)
    # ----------------------------------------------------------------------
    
    # 1. Cargar costos SSTC (Módulo 5)
    print("\n[1/2] Calculando SSTC (C) con Módulo 5...")
    variables_m5 = m5.default_variables()
    
    # Sincronizar vida útil con el valor global para M5
    variables_m5["project_lifetime_months"] = p_g.PROJECT_LIFETIME_MONTHS
    
    try:
        C_series = m8.get_subsidized_costs_series(RUTA_COSTOS_M5, variables_m5)  # (3, T_años)
    except Exception as e:
        print(f"✗ ERROR: No se pudo obtener SSTC de Módulo 5: {e}")
        return

    # 2. Cargar Ahorros (S)
    print("[2/2] Cargando Ahorros Anuales (S)...")
    try:
        years, S_series = m8.load_annual_savings_series(RUTA_AHORROS_M8) # (T_años,), (3, T_años)
    except Exception as e:
        print(f"✗ ERROR: No se pudo cargar la serie de ahorros anuales: {e}")
        return


    # ----------------------------------------------------------------------
    # PASO 2: CÁLCULOS DE PAYBACK Y FRACCIÓN
    # ----------------------------------------------------------------------
    
    # Sincronizar longitudes (T) de las series de costos y ahorros
    assert C_series.shape[0] == 3 and S_series.shape[0] == 3, "Se esperan 3 regiones."
    T = min(C_series.shape[1], S_series.shape[1])
    C_series = C_series[:, :T]
    S_series = S_series[:, :T]
    years    = years[:T]
    
    # --- 3) Payback(t) por región y año
    PB = m8.payback_years_series(C_series, S_series)  # (3, T)

    # --- 4) Payback Fraction(t) (modo exponencial)
    inputs = m8.ExpoInputs(beta=-0.3)                   
    frac   = m8.exponential_probability(inputs.beta, PB)  # (3, T)

    # --- 5) Delayed Payback Fraction (ANUAL)
    delayed_frac = m8.delayed_payback_fraction_series(
        frac,
        tau_years=TAU_YEARS,
        data_frequency="annual"
    )

    # ----------------------------------------------------------------------
    # PASO 3: IMPRIMIR RESULTADOS
    # ----------------------------------------------------------------------

    tabla_por_año("Payback [años] (C/S)", years, PB, max_years=PRINT_ROWS, fmt="{:,.3f}")
    tabla_por_año("Payback Fraction (e^(beta*PB))", years, frac, max_years=PRINT_ROWS, fmt="{:.4f}")
    tabla_por_año(f"Delayed Payback Fraction (MA {TAU_YEARS} año/s)", years, delayed_frac,
                  max_years=PRINT_ROWS, fmt="{:.4f}")

    # Impresión "estilo ecuación" con números (solo primer año)
    print("\n" + "=" * 80)
    print("VERIFICACIÓN DE ECUACIONES (AÑO 1)")
    print("=" * 80)
    
    # Primer año (índice 0)
    k = 0
    print(f"Año {years[k]}:")
    
    print(f"\nCostos SSTC (C): {C_series[:, k]}")
    print(f"Ahorros Anuales (S): {S_series[:, k]}")
    
    print("\nEcuación 1 (Payback):")
    print(f"PB = C / S = {PB[:, k]}")
    
    print("\nEcuación 2 (Fracción Payback):")
    print(f"Fracción = exp(-0.3 * PB) = {frac[:, k]}")
    
    print("\n" + "=" * 80)
    print("MÓDULO 8 COMPLETADO ✓")
    print("=" * 80)


if __name__ == "__main__":
    main()