# -*- coding: utf-8 -*-
"""
main_mod7.py — Simulación y revisión de ecuaciones + tablas estilo CSV.
Imprime t=0..5 y guarda CSV/plots (PNG).
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from prettytable import PrettyTable, SINGLE_BORDER
# Importar módulos
from . import modulo7 as m7 
from .. import parametros_globales as p_g 
from ..modulo5 import modulo5 as m5
from ..modulo8 import modulo8 as m8 


# --- DEFINICIÓN DE RUTAS DE DATOS ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) # .../Parte1
DATOS_DIR = os.path.join(BASE_DIR, 'Datos')

RUTA_COSTOS_M5 = os.path.join(DATOS_DIR, "costoAño.xlsx") 
RUTA_AHORROS_M8 = os.path.join(DATOS_DIR, "annual_savings.xlsx")
# ------------------------------------


# ----------------------------
# Parámetros del modelo (Iniciales de M7)
# --- CARGADOS DESDE EL ARCHIVO CENTRAL ---
variables = p_g.MOD7_VARIABLES_INICIALES.copy()
# Parámetro inicial (W0) que se usará si falla el cálculo dinámico
variables["delayed_payback_fraction"] = np.array([0.386097, 0.282533, 0.151619], dtype=float)
# -----------------------------------------

REGIONES = list(p_g.REGIONES)
SHOW_MONTHS = list(range(0, 6))


# ==========================================================
# CÓDIGO DE CONEXIÓN DINÁMICA (Módulo 8 <--> Módulo 7)
# ==========================================================
print("\n[M7-INTEGRACIÓN] Ejecutando Módulo 5 y 8 para obtener la serie dinámica...")

# PASO 1: Ejecutar Módulo 5 (Obtener SSTC)
variables_m5 = m5.default_variables()
variables_m5["project_lifetime_months"] = p_g.PROJECT_LIFETIME_MONTHS
variables_m5["pvgp_kW_per_household"] = variables["pvgp_kW_per_household"]

try:
    resultados_m5 = m5.correr_modelo(RUTA_COSTOS_M5, variables_m5)
except Exception as e:
    print(f"✗ ERROR M5: No se pudo correr el Módulo 5. {e}")
    # Si falla, se usan los valores iniciales por defecto de DPF
    pass
else:
    # Si M5 es exitoso, procedemos a calcular DPF
    
    # PASO 2: Preparar datos ANUALES para Módulo 8
    sstc_mensual = resultados_m5["sstc_mensual"] 
    n_meses = sstc_mensual.shape[1]
    n_anios = n_meses // 12
    indices_anuales = np.arange(1, n_anios + 1) * 12 - 1 
    sstc_anual = sstc_mensual[:, indices_anuales] # (3, T_años)

    # Cargar Ahorro Anual (S)
    try:
        _, serie_ahorro_anual = m8.load_annual_savings_series(
            RUTA_AHORROS_M8, 
            sheet=0, 
            cols=("North", "Center", "South")
        ) # (3, T_años)
    except Exception as e:
        print(f"✗ ERROR: No se pudo cargar el archivo de ahorros anuales ({RUTA_AHORROS_M8}). Usando DPF fijo.")
        # Usamos los valores fijos, terminamos la integración dinámica aquí.
    else:
        # Asegurar que las series sean del mismo largo (T_años)
        T = min(sstc_anual.shape[1], serie_ahorro_anual.shape[1])
        sstc_anual = sstc_anual[:, :T]
        serie_ahorro_anual = serie_ahorro_anual[:, :T]


        # PASO 3: Calcular delayed_payback_fraction (Módulo 8)
        payback = m8.payback_years_series(sstc_anual, serie_ahorro_anual) 
        prob_exp = m8.exponential_probability(beta=-0.3, payback=payback)
        fraccion_payback = m8.payback_fraction(prob_exp)
        serie_delayed_payback_anual = m8.delayed_payback_fraction_series(
            fraccion_payback, 
            tau_years=1
        ) # (3, T_años)

        # Convertir la serie ANUAL a MENSUAL (n_meses es la longitud real del SSTC, ej. 312)
        serie_delayed_payback_mensual = np.repeat(serie_delayed_payback_anual, 12, axis=1)[:, :n_meses]


        # PASO 4: Inyectar la serie dinámica y sincronizar el tiempo de M7
        variables["delayed_payback_fraction"] = serie_delayed_payback_mensual

        # --- SINCRONIZACIÓN CLAVE ---
        n_meses_serie = serie_delayed_payback_mensual.shape[1]
        variables["tf"] = n_meses_serie - 1 
        # ----------------------------

        print(f"✓ La variable 'delayed_payback_fraction' ha sido actualizada con una serie dinámica de dimensión {serie_delayed_payback_mensual.shape}.")
        print(f"✓ El tiempo final de simulación (tf) se ha ajustado a {variables['tf']} meses.")

print("==========================================================")
# ==========================================================

# ----------------------------
# Simulación (USA LAS VARIABLES ACTUALIZADAS)
# ----------------------------
(
    t,
    population,
    households,
    new_households,
    adopters,
    household_able_to_adopt,
    pv_power_kW,
    adopters_flow,
    initial_potential_adopter_pv,
    initial_able_to_adopt,
    initial_adopter_of_pv,
    households_willing_to_adopt_pv_system,
    potential_adopters_pv,                  # W(t)
    households_remaining_willing_to_adopt,
    adoption_rate,                          # hogares/mes
    growth_rate,                            # inflow a W
    pop_growth_rate_continuous_per_month,   # r (vector 3,)
    net_growth_rate_population_time         # r*P(t) (T x 3)
) = m7.simulate_system(**variables)

# ----------------------------
# Utilidades de impresión
# ----------------------------
def _nearest_idx(x: np.ndarray, val: float) -> int:
    return int(np.argmin(np.abs(x - val)))

def print_series(name: str, t: np.ndarray, Y: np.ndarray, months, decimals: int = 2):
    """Imprime como tabla PrettyTable: columnas = Mes, Norte, Centro, Sur."""
    table = PrettyTable()
    table.set_style(SINGLE_BORDER)          
    table.field_names = ["Mes"] + REGIONES 

    table.align["Mes"] = "r"
    for r in REGIONES:
        table.align[r] = "r"

    for m in months:
        i = _nearest_idx(t, m)
        fila = [int(t[i])] + [f"{v:.{decimals}f}" for v in Y[i, :]]
        table.add_row(fila)

    print(f"\n=== {name} ===")
    print(table)

def print_series_1d(name: str, t: np.ndarray, y: np.ndarray, months, as_pct=False, decimals=2):
    """Imprime una serie 1D como PrettyTable: columnas = Mes, Valor."""
    table = PrettyTable()
    table.set_style(SINGLE_BORDER)
    table.field_names = ["Mes", "Valor"]
    table.align["Mes"] = "r"
    table.align["Valor"] = "r"

    for m in months:
        i = _nearest_idx(t, m)
        val = y[i]
        s = f"{val*100:.{decimals}f}%" if as_pct else f"{val:.{decimals}f}"
        table.add_row([int(t[i]), s])

    print(f"\n=== {name} ===")
    print(table)

# Constantes replicadas para poder imprimir con la misma función
initial_potential_series = np.tile(initial_potential_adopter_pv, (t.shape[0], 1))
initial_adopter_series   = np.tile(initial_adopter_of_pv,        (t.shape[0], 1))
W_from_initial = households_willing_to_adopt_pv_system - initial_adopter_series

# ----------------------------
# Revisión de ecuaciones base (solo t=0..5)
# ----------------------------
print_series("population P(t) [personas]",                      t, population,          SHOW_MONTHS, 1)
print_series("net_growth_rate_population r*P(t) [personas/mes]", t, net_growth_rate_population_time,  SHOW_MONTHS, 1)

print_series("households [hogares]",                         t, households,          SHOW_MONTHS, 1)
print_series("new_households [hogares]",                     t, new_households,           SHOW_MONTHS, 1)

print_series("household_able_to_adopt [hogares]",            t, household_able_to_adopt,         SHOW_MONTHS, 1)
print_series("households_willing_to_adopt_pv_system [hogares]", t, households_willing_to_adopt_pv_system, SHOW_MONTHS, 1)

print_series("initial_adopter_of_pv (A0) [hogares]",         t, initial_adopter_series,           SHOW_MONTHS, 1)
print_series("initial_potential_adopter_pv (W0) [hogares]",  t, initial_potential_series,         SHOW_MONTHS, 1)

print_series("potential_adopters_pv W(t) (estado) [hogares]", t, potential_adopters_pv,          SHOW_MONTHS, 1)
print_series("W_from_initial(t) = willing(t) - A0 [hogares]",  t, W_from_initial,                 SHOW_MONTHS, 1)
print_series("gap = willing - (W + A) [hogares]",             t, households_remaining_willing_to_adopt, SHOW_MONTHS, 1)

print_series("adopters A(t) [hogares]",                       t, adopters,             SHOW_MONTHS, 1)
print_series("adopters_flow (ΔA por mes) [hogares/mes]",      t, adopters_flow,                SHOW_MONTHS, 3)
print_series("adoption_rate (p + q*A/M)*W [hogares/mes]",     t, adoption_rate,                SHOW_MONTHS, 3)

print_series("growth_rate (inflow a W) [hogares/mes]",        t, growth_rate,                  SHOW_MONTHS, 3)
print_series("pv_power_kW = A*PV_size [kW]",                  t, pv_power_kW,                  SHOW_MONTHS, 1)

# ----------------------------
# Métricas solicitadas
# ----------------------------
available_market = household_able_to_adopt 

national_available_market = np.sum(available_market, axis=1)   
national_adopters         = np.sum(adopters, axis=1)          

def _safe_ratio_1d(numer, denom):
    out = np.zeros_like(numer, dtype=float)
    m = denom > 0
    out[m] = numer[m] / denom[m]
    return out

national_pct_adoption = _safe_ratio_1d(national_adopters, national_available_market)

pct_adoption_region = np.zeros_like(adopters)
mask = available_market > 0
pct_adoption_region[mask] = adopters[mask] / available_market[mask]

# --------- Impresiones rápidas (csv-style) ----------
print_series("Available market [hogares]",           t, available_market,     SHOW_MONTHS, 1)
print_series("Adopters [hogares]",                   t, adopters,             SHOW_MONTHS, 1)
print_series("Percentage of adoption por región [%]", t, pct_adoption_region*100, SHOW_MONTHS, 2)
print_series_1d("National Available Market [hogares]", t, national_available_market, SHOW_MONTHS, as_pct=False, decimals=1)
print_series_1d("National Adopters [hogares]",        t, national_adopters,           SHOW_MONTHS, as_pct=False, decimals=1)
print_series_1d("National Percentage of Adoption [%]", t, national_pct_adoption,       SHOW_MONTHS, as_pct=True,  decimals=2)

# ----------------------------
# CSV y gráficos (rutas robustas)
# ----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
SAVE_DIR = Path(os.path.join(BASE_DIR, "Resultados", "Modulo7"))
PLOT_DIR = SAVE_DIR / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.DataFrame({"Mes": t.astype(int)})
for i, r in enumerate(REGIONES):
    df[f"W_potential_{r}"]              = potential_adopters_pv[:, i]
    df[f"adoption_rate_{r}"]            = adoption_rate[:, i]
    df[f"adopters_{r}"]                 = adopters[:, i]
    df[f"willing_{r}"]                  = households_willing_to_adopt_pv_system[:, i]
    df[f"available_market_{r}"]         = available_market[:, i]
    df[f"Pct_adoption_{r}"]             = pct_adoption_region[:, i]
    df[f"initial_adopters0_{r}"]        = np.full_like(t, initial_adopter_of_pv[i], dtype=float)
    df[f"initial_W0_{r}"]               = np.full_like(t, initial_potential_adopter_pv[i], dtype=float)

df["National_Available_Market"]       = national_available_market
df["National_Adopters"]               = national_adopters
df["National_Percentage_of_Adoption"] = national_pct_adoption

csv_path = SAVE_DIR / "pv_series.csv"
df.to_csv(csv_path, index=False)

def plot_lines(x, Y, title, ylabel, fname):
    """Guarda un gráfico de líneas (PNG) para las tres regiones."""
    plt.figure()
    for i, r in enumerate(REGIONES):
        plt.plot(x, Y[:, i], label=r, linewidth=0.6)
    plt.title(title)
    plt.xlabel("Mes")
    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", linewidth=0.4)
    plt.legend()
    out_path = PLOT_DIR / f"{fname}.png"
    plt.savefig(out_path, bbox_inches="tight")
    return out_path

# Gráficos
plot_lines(t, potential_adopters_pv, "Potential adopters of PV systems (W)", "Hogares", "plot_potential_adopters")
plot_lines(t, adoption_rate,        "Adoption rate (hogares/mes)", "Hogares por mes", "plot_adoption_rate")
plot_lines(t, pct_adoption_region,  "Percentage of adoption por región (fracción)", "Fracción (0..1)", "plot_pct_adoption_region")

plt.figure()
plt.plot(t, national_pct_adoption, linewidth=0.8)
plt.title("National Percentage of Adoption")
plt.xlabel("Mes")
plt.ylabel("Fracción (0..1)")
plt.grid(True, linestyle="--", linewidth=0.4)
plt.savefig(PLOT_DIR / "plot_national_pct_adoption.png", bbox_inches="tight")

print("CSV:", csv_path.as_posix())
print("Plots en:", PLOT_DIR.as_posix())