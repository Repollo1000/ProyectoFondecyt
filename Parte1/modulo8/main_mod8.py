# -*- coding: utf-8 -*-
from __future__ import annotations
import os
import numpy as np
from prettytable import PrettyTable

from Parte1.modulo5 import modulo5 as m5     # usa sstc_temporal (3, T)
from Parte1.modulo8 import modulo8 as m8     # actualizado: trabaja en años


# ----------------------------
# Configuración rápida
# ----------------------------
PRINT_ROWS = 8      # filas a mostrar en tablas
EVAL_YEARS = None   # ej. 5 para evaluar solo primeros N años; None = todos
TAU_YEARS = 1       # ventana de suavizado en años (1 = sin suavizado)


def tabla_por_año(nombre: str, years, arr_3xT, max_years: int = 8, fmt="{:.4f}"):
    """
    Imprime tabla con encabezado [t, Norte, Centro, Sur] para series (3,T).
    years: (T,)
    arr_3xT: (3, T)
    """
    print(f"\n{nombre}:")
    t = PrettyTable()
    t.field_names = ["t", "Norte", "Centro", "Sur"]
    limite = min(len(years), max_years)
    for k in range(limite):
        fila = [int(years[k])] + [fmt.format(float(arr_3xT[i, k])) for i in range(3)]
        t.add_row(fila)
    print(t)


def imprimir_ecuaciones_payback(years, C_series, S_series, PB_series, filas: int = 3):
    """
    Imprime las ecuaciones de Payback con números (primeras 'filas'):
      Payback(t) = SubsidizedCost(t) / AnnualSavings(t)
    Usa C_series (3,T), S_series (3,T), PB_series (3,T).
    """
    limite = min(S_series.shape[1], filas)
    print("\n=== Detalle Payback por región (ecuación con números) ===")
    for k in range(limite):
        t_val = int(years[k])
        print(f"\n t = {t_val}")
        for i, reg in enumerate(m8.REGIONES):
            num = C_series[i, k]   # costo subsidiado en año t
            den = S_series[i, k]   # ahorro anual en año t
            val = PB_series[i, k]  # payback en año t
            print(f"  {reg}: Payback = {num:,.2f} / {den:,.2f} = {val:,.3f}")


def imprimir_ecuaciones_fraction_exponential(years, beta, PB_series, frac_series, filas: int = 3):
    """
    Imprime las ecuaciones de Payback Fraction (modo exponencial):
      Fraction(t) = exp(beta * Payback(t))
    """
    limite = min(PB_series.shape[1], filas)
    print("\n=== Detalle Payback Fraction (exponencial) con números ===")
    for k in range(limite):
        t_val = int(years[k])
        print(f"\n t = {t_val}")
        for i, reg in enumerate(m8.REGIONES):
            pb_val = PB_series[i, k]
            frac_val = frac_series[i, k]
            print(f"  {reg}: Fraction = exp({beta:.3f} * {pb_val:.3f}) = {frac_val:.4f}")


def main():
    # --- Rutas robustas
    base_dir = os.path.dirname(os.path.dirname(__file__))  # .../Parte1
    ruta_costos  = os.path.join(base_dir, "Datos", "costoAño.xlsx")
    ruta_savings = os.path.join(base_dir, "Datos", "annual_savings.xlsx")  # serie anual

    # --- Variables del módulo 5 (ajustables sin tocar modulo5.py)
    variables = m5.default_variables()

    # --- 1) Costos subsidiados SERIE (USD) por región y año: (3, T)
    C_series = m8.get_subsidized_costs_series(ruta_costos, variables)  # (3, T)

    # --- 2) Annual savings por año (3, T)
    years, S_series = m8.load_annual_savings_series(
        ruta_excel=ruta_savings,
        sheet=0,
        cols=("North", "Center", "South"),
        year_col="Year"
    )

    # --- Recorte de horizonte (opcional: primeros N años)
    if EVAL_YEARS is not None:
        years = years[:EVAL_YEARS]
        C_series = C_series[:, :EVAL_YEARS]
        S_series = S_series[:, :EVAL_YEARS]

    # --- Sanidad mínima antes de dividir
    assert C_series.shape[0] == 3 and S_series.shape[0] == 3, "Se esperan 3 regiones."
    T = min(C_series.shape[1], S_series.shape[1])
    C_series = C_series[:, :T]
    S_series = S_series[:, :T]
    years    = years[:T]

    # --- 3) Payback(t) por región y año
    PB = m8.payback_years_series(C_series, S_series)  # (3, T)

    # --- 4) Payback Fraction(t) (modo exponencial)
    inputs = m8.ExpoInputs(beta=-0.3)                   # parámetro del modelo
    frac   = m8.exponential_probability(inputs.beta, PB)  # (3, T)

    # --- 5) Delayed Payback Fraction (ANUAL)
    delayed_frac = m8.delayed_payback_fraction_series(
        frac,
        tau_years=TAU_YEARS,
        data_frequency="annual"
    )

    # --- 6) Tablas compactas (máx PRINT_ROWS filas)
    tabla_por_año("Payback [años]", years, PB, max_years=PRINT_ROWS, fmt="{:.3f}")
    tabla_por_año("Payback Fraction", years, frac, max_years=PRINT_ROWS, fmt="{:.4f}")
    tabla_por_año(f"Delayed Payback Fraction (MA {TAU_YEARS} año/s)", years, delayed_frac,
                  max_years=PRINT_ROWS, fmt="{:.4f}")

    # --- 7) Impresión "estilo ecuación" con números (primeras 3 filas por claridad)
    imprimir_ecuaciones_payback(years, C_series, S_series, PB, filas=3)
    imprimir_ecuaciones_fraction_exponential(years, inputs.beta, PB, frac, filas=3)

    # --- Asserts de sanidad mínimos
    assert C_series.shape == S_series.shape == PB.shape == frac.shape == delayed_frac.shape
    assert (C_series >= 0).all()
    assert (S_series >= 0).all()


if __name__ == "__main__":
    main()
