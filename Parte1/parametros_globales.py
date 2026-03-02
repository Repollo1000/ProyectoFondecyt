# -*- coding: utf-8 -*-
"""
parametros_globales.py — Define constantes y variables usadas por múltiples módulos.
"""

from __future__ import annotations

import os
import numpy as np

# --- 0. RUTAS BASE ---

# Directorio base de la Parte 1 (donde está este archivo)
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# Directorio donde están los datos (Excels, CSV, etc.)
DATOS_DIR = os.path.join(BASE_DIR, "Datos")


# --- 1. DEFINICIONES GENERALES ---

# Orden consistente con el resto del modelo
REGIONES = ("Norte", "Centro", "Sur")

# Vida útil del proyecto en meses (27 años * 12 meses)
PROJECT_LIFETIME_MONTHS = 324


# --- 2. PARÁMETROS DEL MÓDULO 7 (Adopción) ---

# Estos valores vienen de tu configuración original
MOD7_VARIABLES_INICIALES = {
    # P0 (Población inicial por región)
    "population_initial": np.array([2356180, 11839100, 3885100], dtype=float),

    # r_m (Tasa de crecimiento fraccional continua / mes)
    "fractional_growth_rate": np.array(
        [0.000903063, 0.000563083, 0.000575048], dtype=float
    ),

    # Resto de parámetros de mercado y Bass
    "average_people_per_household": np.array([2.89, 2.77, 2.74], dtype=float),
    "market_fraction_existing": np.array([0.36, 0.38, 0.59], dtype=float),
    "market_fraction_new": np.array([0.48, 0.44, 0.61], dtype=float),
    "initial_installed_power_kW": np.array(
        [5261.5, 46468.8, 8669.35], dtype=float
    ),
    "pvgp_kW_per_household": np.array([3.3, 3.85, 4.95], dtype=float),
    "p_month": np.array([0.000167, 0.000167, 0.000167], dtype=float),
    "q_month": np.array([0.0333333, 0.0333333, 0.0333333], dtype=float),

    "t0": 0.0,
    "tf": PROJECT_LIFETIME_MONTHS,
    "dt": 1.0,
}


# --- 3. RUTAS DE ARCHIVOS ESPECÍFICOS PARA MÓDULO 9 ---

# Archivo de factores de emisión
FACTOR_EMISION_FILE = os.path.join(DATOS_DIR, "factor_emisionv2.csv")

# Archivo de curva de carga
CURVA_CARGA_FILE = os.path.join(DATOS_DIR, "curva_de_carga.xlsx")

MOD9_RUTAS = {
    "emission_factor_file": FACTOR_EMISION_FILE,
    "curva_carga_file": CURVA_CARGA_FILE,
}


# --- 4. PARÁMETROS DEL MÓDULO 9 (placeholders) ---

# OJO: estos valores son "de mentira" por ahora, solo para probar el flujo.
# Después se pueden reemplazar usando resultados reales de otros módulos.
MOD9_VARIABLES_INICIALES = {
    # Número de hogares por región (placeholder, será reemplazado por Módulo 7)
    "households_por_region": np.array(
        [1_000_000.0, 2_000_000.0, 1_500_000.0], dtype=float
    ),

    # Consumo eléctrico típico por hogar (kWh/mes) por región (placeholder).
    # Luego lo reemplazaremos usando la curva de carga.
    "consumo_elec_hogar_kWh_mes": np.array(
        [180.0, 200.0, 220.0], dtype=float
    ),

    # Escenario de factor de emisión por defecto:
    #   "CN", "SR" o "AT" según factor_emisionv2.csv
    "default_emission_scenario": "CN",
}
