# -*- coding: utf-8 -*-
"""
main_mod9.py — Prueba básica del Módulo 9.

Objetivo en esta etapa:
- Mostrar las variables iniciales del módulo 9.
- Verificar que se carga correctamente factor_emisionv2.csv.
- Verificar que se carga correctamente curva_de_carga.xlsx.
- Obtener y mostrar la serie de households desde el Módulo 7.
NO se calculan todavía emisiones, solo se revisan inputs.
"""

from __future__ import annotations

import numpy as np
from prettytable import PrettyTable, SINGLE_BORDER

try:
    # Ejecución como paquete: python -m Parte1.modulo9.main_mod9
    from .. import parametros_globales as p_g
    from . import modulo9 as m9
    from ..modulo7 import modulo7 as m7
except ImportError:
    # Ejecución directa: python modulo9/main_mod9.py
    import parametros_globales as p_g
    import modulo9 as m9
    from modulo7 import modulo7 as m7


# ------------ Helpers con PrettyTable ------------

def mostrar_dict_en_tabla(titulo: str, d: dict):
    table = PrettyTable()
    table.set_style(SINGLE_BORDER)
    table.field_names = ["Clave", "Valor"]
    for k, v in d.items():
        table.add_row([k, str(v)])
    print(f"\n=== {titulo} ===")
    print(table)


def mostrar_factores(tiempos: np.ndarray, factors: np.ndarray, titulo: str):
    table = PrettyTable()
    table.set_style(SINGLE_BORDER)
    table.field_names = ["idx", "t (mes)", "factor (tCO2/MWh)"]

    n = min(10, len(factors))
    for i in range(n):
        table.add_row([i, f"{tiempos[i]:.0f}", f"{factors[i]:.4f}"])

    print(f"\n=== {titulo} (primeros {n}) ===")
    print(table)


def mostrar_consumo_mensual(consumo_df):
    table = PrettyTable()
    table.set_style(SINGLE_BORDER)
    table.field_names = ["Mes"] + list(consumo_df.columns)

    for mes, row in consumo_df.iterrows():
        fila = [mes] + [f"{v:.2f}" for v in row.values]
        table.add_row(fila)

    print("\n=== Consumo mensual por hogar (kWh/mes·hogar) ===")
    print(table)


def mostrar_households(t: np.ndarray, households: np.ndarray, regiones, n_meses: int = 6):
    table = PrettyTable()
    table.set_style(SINGLE_BORDER)
    table.field_names = ["Mes"] + list(regiones)

    for i in range(min(n_meses, len(t))):
        fila = [int(t[i])] + [f"{v:.1f}" for v in households[i, :]]
        table.add_row(fila)

    print("\n=== Households por región (primeros meses) ===")
    print(table)


# ------------ main ------------

def main():
    # ------------------------------------------------------
    # 0) Variables iniciales y rutas
    # ------------------------------------------------------
    print("=== PRUEBA MÓDULO 9: VARIABLES DE ENTRADA ===")
    print("REGIONES:", p_g.REGIONES)

    mostrar_dict_en_tabla("MOD9_VARIABLES_INICIALES", p_g.MOD9_VARIABLES_INICIALES)
    mostrar_dict_en_tabla("MOD9_RUTAS", p_g.MOD9_RUTAS)

    # ------------------------------------------------------
    # 1) Factores de emisión (CSV)
    # ------------------------------------------------------
    escenario = p_g.MOD9_VARIABLES_INICIALES["default_emission_scenario"]
    tiempos_em, factors = m9.cargar_factor_emision(escenario)

    print("\nEscenario de emisión seleccionado:", escenario)
    print("N° puntos en el vector de factores:", len(factors))
    mostrar_factores(tiempos_em, factors, "Factores de emisión (factor_emisionv2.csv)")

    # ------------------------------------------------------
    # 2) Curva de carga y consumo mensual por hogar
    # ------------------------------------------------------
    df_curva = m9.cargar_curva_de_carga()

    print("\n=== Curva de carga (primeras 5 filas) ===")
    print(df_curva.head())

    consumo_mensual = m9.consumo_mensual_por_hogar_desde_curva(df_curva)
    mostrar_consumo_mensual(consumo_mensual)

    # ------------------------------------------------------
    # 3) Households desde el Módulo 7
    # ------------------------------------------------------
    print("\n=== HOUSEHOLDS DESDE MÓDULO 7 (simulate_system) ===")

    variables_m7 = p_g.MOD7_VARIABLES_INICIALES.copy()

    (
        t,
        population,
        households,
        new_households,
        adopters,
        *_
    ) = m7.simulate_system(**variables_m7)

    print(f"Longitud de la simulación de M7 (t): {len(t)} meses")
    mostrar_households(t, households, p_g.REGIONES, n_meses=6)

    # Chequeo rápido de coherencia de largos
    print("\n=== CHEQUEO DE CONSISTENCIA DE LONGITUDES ===")
    print(f"  len(tiempos_em) (factores emisión) = {len(tiempos_em)}")
    print(f"  len(t)           (Módulo 7)        = {len(t)}")
    if len(tiempos_em) == len(t):
        print("  → OK: mismo número de pasos de tiempo.")
    else:
        print("  → OJO: no coinciden las longitudes; habrá que alinear series antes de calcular emisiones.")

    print("\nFIN de prueba de inputs del Módulo 9 (sin cálculos de emisiones).")


if __name__ == "__main__":
    main()
