# -*- coding: utf-8 -*-
"""
modulo9.py — Utilidades de entrada de datos para el Módulo 9 (emisiones).

Por AHORA este módulo SOLO se preocupa de:
- Cargar el vector de factores de emisión desde factor_emisionv2.csv
- Cargar la curva de carga desde curva_de_carga.xlsx
- Calcular un consumo mensual por hogar a partir de la curva de carga

No realiza aún cálculos de emisiones; la idea es verificar que
todas las variables de entrada se leen y tienen sentido.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    # Import relativo cuando se usa como paquete: python -m Parte1.modulo9.main_mod9
    from .. import parametros_globales as p_g
except ImportError:
    # Import directo para ejecuciones sueltas
    import parametros_globales as p_g


# --------------------------------------------------------------------
# 1. FACTOR DE EMISIÓN
# --------------------------------------------------------------------


def cargar_factor_emision(scenario: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Lee el archivo factor_emisionv2.csv y devuelve:

    - tiempos: vector 1D con la columna "tiempo" (en meses, o como esté definido)
    - factors: vector 1D con los factores de emisión del escenario elegido

    Parameters
    ----------
    scenario : str, opcional
        Escenario a usar: "CN", "SR" o "AT".
        Si es None, usa p_g.MOD9_VARIABLES_INICIALES["default_emission_scenario"].

    Returns
    -------
    tiempos : np.ndarray
        Vector 1D (float) con los tiempos.
    factors : np.ndarray
        Vector 1D (float) con los factores de emisión (tCO2/MWh).
    """
    if scenario is None:
        scenario = p_g.MOD9_VARIABLES_INICIALES["default_emission_scenario"]

    scenario = scenario.upper().strip()
    if scenario not in ("CN", "SR", "AT"):
        raise ValueError(f"Escenario de emisión no válido: {scenario}. Use 'CN', 'SR' o 'AT'.")

    csv_path = p_g.MOD9_RUTAS["emission_factor_file"]

    # El archivo está separado por ; y la primera fila es de etiquetas/unidades,
    # así que usaremos header=0 y luego drop de la fila 0.
    df_raw = pd.read_csv(csv_path, sep=";")

    # Columnas esperadas:
    #   'Unnamed: 0' → tiempo
    #   'CN scenario', 'SR scenario', 'AT scenario'
    # La fila 0 tiene las unidades, la descartamos.
    df = df_raw.drop(index=0).copy()

    # Renombramos columnas para algo más manejable
    df = df.rename(
        columns={
            "Unnamed: 0": "tiempo",
            "CN scenario": "CN",
            "SR scenario": "SR",
            "AT scenario": "AT",
        }
    )

    # Convertimos a numérico (por si vinieron como strings)
    df["tiempo"] = pd.to_numeric(df["tiempo"], errors="coerce")
    df["CN"] = pd.to_numeric(df["CN"], errors="coerce")
    df["SR"] = pd.to_numeric(df["SR"], errors="coerce")
    df["AT"] = pd.to_numeric(df["AT"], errors="coerce")

    # Eliminamos filas que quedaron con NaN
    df = df.dropna(subset=["tiempo", scenario])

    tiempos = df["tiempo"].to_numpy(dtype=float)
    factors = df[scenario].to_numpy(dtype=float)

    return tiempos, factors


# --------------------------------------------------------------------
# 2. CURVA DE CARGA
# --------------------------------------------------------------------


def cargar_curva_de_carga() -> pd.DataFrame:
    """
    Lee el archivo curva_de_carga.xlsx y devuelve un DataFrame con las columnas:

        'anio', 'mes', 'hora', 'demanda_norte', 'demanda_centro', 'demanda_sur'

    (Normalizamos nombres para usarlos de forma consistente.)
    """
    xls_path = p_g.MOD9_RUTAS["curva_carga_file"]

    df = pd.read_excel(xls_path, sheet_name="curvas de carga")

    df = df.rename(
        columns={
            "Año": "anio",
            "mes": "mes",
            "hora": "hora",
            "demanda_norte": "demanda_norte",
            "demanda_centro": "demanda_centro",
            "demanda_sur": "demanda_sur",
        }
    )

    return df


def consumo_mensual_por_hogar_desde_curva(
    df_curva: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calcula el consumo mensual por hogar (kWh/mes·hogar) para cada región
    a partir de la curva de carga horaria.

    Suposición:
    - Cada fila corresponde a una hora.
    - Las columnas demanda_* están en kWh/h por hogar (equivalente a kW promedio en esa hora).
    - Sumando sobre todas las horas del mes obtenemos kWh/mes·hogar.

    El DataFrame retornado tiene:
        índice: mes (1..12)
        columnas: 'Norte', 'Centro', 'Sur'
        valores: kWh/mes·hogar aproximados.
    """
    # Agrupamos por mes (si después tienes varios años, se puede agrupar por ['anio','mes'])
    grp = (
        df_curva.groupby("mes")[["demanda_norte", "demanda_centro", "demanda_sur"]]
        .sum()
        .rename(
            columns={
                "demanda_norte": "Norte",
                "demanda_centro": "Centro",
                "demanda_sur": "Sur",
            }
        )
    )

    # Cada entrada del grp es el consumo mensual por hogar (kWh/mes·hogar) por región.
    return grp


def calcular_emisiones_consumo(factores,households,consumo):
    households = m7.calcular_households_totales()
    consumo=0#convertir a  MWh
    factor=0

    emisiones = households*consumo_hogar*factor
    for region in 100:




    return 0
    

