# -*- coding: utf-8 -*-
"""
parametros_globales.py — Define constantes y variables usadas por múltiples módulos.
"""
import numpy as np

# --- 1. DEFINICIONES GENERALES ---
REGIONES = ("Norte", "Centro", "Sur")

# Vida útil del proyecto en meses (27 años * 12 meses)
PROJECT_LIFETIME_MONTHS = 312


# --- 2. PARÁMETROS DEL MÓDULO 7 (Adopción) ---

# Parámetros que eran hardcodeados en main_mod7.py
MOD7_VARIABLES_INICIALES = {
    # P0 (Población inicial por región)
    "population_initial": np.array([2356180, 11839100, 3885100], dtype=float),       
    # r_m (Tasa de crecimiento fraccional continua / mes)
    "fractional_growth_rate": np.array([0.000903063, 0.000563083, 0.000575048], dtype=float), 
    
    # Resto de parámetros de mercado y Bass
    "average_people_per_household": np.array([2.89, 2.77, 2.74], dtype=float),
    "market_fraction_existing": np.array([0.36, 0.38, 0.59], dtype=float),
    "market_fraction_new":       np.array([0.48, 0.44, 0.61], dtype=float),
    "initial_installed_power_kW": np.array([5261.5, 46468.8, 8669.35], dtype=float),
    "pvgp_kW_per_household": np.array([3.3, 3.85, 4.95], dtype=float),
    "p_month": np.array([0.000167, 0.000167, 0.000167], dtype=float),
    "q_month": np.array([0.0333333, 0.0333333, 0.0333333], dtype=float),
    
    "t0": 0.0,
    "tf": PROJECT_LIFETIME_MONTHS, # Usa la constante global
    "dt": 1.0,
}