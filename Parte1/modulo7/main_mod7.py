# -*- coding: utf-8 -*-
"""
main_mod7.py — Simulación y revisión de ecuaciones + tablas estilo CSV.
Imprime t=0..5 y guarda CSV/plots (PNG). No genera PDF.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from prettytable import PrettyTable, SINGLE_BORDER
# Importar el módulo como paquete
from . import modulo7 as m7  # <-- 1. CORRECCIÓN: Añadido un punto "."

# ----------------------------
# Parámetros del modelo
# ----------------------------
variables = {
    # === NOMBRES "DE PAPER" PARA POBLACIÓN ===
    "population_initial": np.array([2356180, 11839100, 3885100], dtype=float),               # P0
    "fractional_growth_rate": np.array([0.000903063, 0.000563083, 0.000575048], dtype=float),  # g_m

    # === Resto igual que antes ===
    "average_people_per_household": np.array([2.89, 2.77, 2.74], dtype=float),
    "market_fraction_existing": np.array([0.36, 0.38, 0.59], dtype=float),
    "market_fraction_new":      np.array([0.48, 0.44, 0.61], dtype=float),
    "initial_installed_power_kW": np.array([5261.5, 46468.8, 8669.35], dtype=float),
    "pvgp_kW_per_household": np.array([3.3, 3.85, 4.95], dtype=float),
    "p_month": np.array([0.000167, 0.000167, 0.000167], dtype=float),
    "q_month": np.array([0.0333333, 0.0333333, 0.0333333], dtype=float),
    "t0": 0.0,
    "tf": 324,
    "dt": 1.0,
    "delayed_payback_fraction": np.array([0.386097, 0.282533, 0.151619], dtype=float),
}

REGIONES = ["Norte", "Centro", "Sur"]
SHOW_MONTHS = list(range(0, 6))  # 0..5

# ----------------------------
# Simulación
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
    potential_adopters_pv,              # W(t)
    households_remaining_willing_to_adopt,
    adoption_rate,                      # hogares/mes
    growth_rate,                        # inflow a W
    pop_growth_rate_continuous_per_month,   # r (vector 3,)
    net_growth_rate_population_time          # r*P(t) (T x 3)
) = m7.simulate_system(**variables)

# ----------------------------
# Utilidades de impresión
# ----------------------------
def _nearest_idx(x: np.ndarray, val: float) -> int:
    return int(np.argmin(np.abs(x - val)))

def print_series(name: str, t: np.ndarray, Y: np.ndarray, months, decimals: int = 2):
    """Imprime como tabla PrettyTable: columnas = Mes, Norte, Centro, Sur."""
    table = PrettyTable()
    table.set_style(SINGLE_BORDER)          # bordes compactos, bonito en consola
    table.field_names = ["Mes"] + REGIONES  # encabezados

    # Alineaciones
    table.align["Mes"] = "r"
    for r in REGIONES:
        table.align[r] = "r"

    # Filas
    for m in months:
        i = _nearest_idx(t, m)
        fila = [int(t[i])] + [f"{v:.{decimals}f}" for v in Y[i, :]]
        table.add_row(fila)

    # Título arriba y tabla debajo
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
print_series("population P(t) [personas]",                        t, population,                         SHOW_MONTHS, 1)
print_series("net_growth_rate_population r*P(t) [personas/mes]",  t, net_growth_rate_population_time,    SHOW_MONTHS, 1)

print_series("households [hogares]",                         t, households,                         SHOW_MONTHS, 1)
print_series("new_households [hogares]",                     t, new_households,                     SHOW_MONTHS, 1)

print_series("household_able_to_adopt [hogares]",            t, household_able_to_adopt,            SHOW_MONTHS, 1)
print_series("households_willing_to_adopt_pv_system [hogares]", t, households_willing_to_adopt_pv_system, SHOW_MONTHS, 1)

print_series("initial_adopter_of_pv (A0) [hogares]",         t, initial_adopter_series,             SHOW_MONTHS, 1)
print_series("initial_potential_adopter_pv (W0) [hogares]",  t, initial_potential_series,           SHOW_MONTHS, 1)

print_series("potential_adopters_pv W(t) (estado) [hogares]", t, potential_adopters_pv,             SHOW_MONTHS, 1)
print_series("W_from_initial(t) = willing(t) - A0 [hogares]",  t, W_from_initial,                    SHOW_MONTHS, 1)
print_series("gap = willing - (W + A) [hogares]",             t, households_remaining_willing_to_adopt, SHOW_MONTHS, 1)

print_series("adopters A(t) [hogares]",                       t, adopters,                           SHOW_MONTHS, 1)
print_series("adopters_flow (ΔA por mes) [hogares/mes]",      t, adopters_flow,                      SHOW_MONTHS, 3)
print_series("adoption_rate (p + q*A/M)*W [hogares/mes]",     t, adoption_rate,                      SHOW_MONTHS, 3)

print_series("growth_rate (inflow a W) [hogares/mes]",        t, growth_rate,                        SHOW_MONTHS, 3)
print_series("pv_power_kW = A*PV_size [kW]",                  t, pv_power_kW,                        SHOW_MONTHS, 1)

# ----------------------------
# Métricas solicitadas
# ----------------------------
available_market = household_able_to_adopt  # alias semántico

# Nacionales (sumas por columnas)
national_available_market = np.sum(available_market, axis=1)   # (T,)
national_adopters         = np.sum(adopters, axis=1)           # (T,)

def _safe_ratio_1d(numer, denom):
    out = np.zeros_like(numer, dtype=float)
    m = denom > 0
    out[m] = numer[m] / denom[m]
    return out

national_pct_adoption = _safe_ratio_1d(national_adopters, national_available_market)  # [0..1]

# Porcentaje de adopción por región (T x 3)
pct_adoption_region = np.zeros_like(adopters)
mask = available_market > 0
pct_adoption_region[mask] = adopters[mask] / available_market[mask]

# --------- Impresiones rápidas (csv-style) ----------
print_series("Available market [hogares]",                t, available_market,        SHOW_MONTHS, 1)
print_series("Adopters [hogares]",                        t, adopters,                SHOW_MONTHS, 1)
print_series("Percentage of adoption por región [%]",     t, pct_adoption_region*100, SHOW_MONTHS, 2)
print_series_1d("National Available Market [hogares]",    t, national_available_market, SHOW_MONTHS, as_pct=False, decimals=1)
print_series_1d("National Adopters [hogares]",            t, national_adopters,         SHOW_MONTHS, as_pct=False, decimals=1)
print_series_1d("National Percentage of Adoption [%]",    t, national_pct_adoption,     SHOW_MONTHS, as_pct=True,  decimals=2)

# ----------------------------
# CSV y gráficos (rutas robustas)
# ----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # .../Parte1
SAVE_DIR = Path(os.path.join(BASE_DIR, "Resultados", "Modulo7"))
PLOT_DIR = SAVE_DIR / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.DataFrame({"Mes": t.astype(int)})
for i, r in enumerate(REGIONES):
    df[f"W_potential_{r}"]            = potential_adopters_pv[:, i]
    df[f"adoption_rate_{r}"]          = adoption_rate[:, i]
    df[f"adopters_{r}"]               = adopters[:, i]
    df[f"willing_{r}"]                = households_willing_to_adopt_pv_system[:, i]
    df[f"available_market_{r}"]       = available_market[:, i]
    df[f"Pct_adoption_{r}"]           = pct_adoption_region[:, i]  # fracción [0..1]
    df[f"initial_adopters0_{r}"]      = np.full_like(t, initial_adopter_of_pv[i], dtype=float)
    df[f"initial_W0_{r}"]             = np.full_like(t, initial_potential_adopter_pv[i], dtype=float)

# Nacionales
df["National_Available_Market"]       = national_available_market
df["National_Adopters"]               = national_adopters
df["National_Percentage_of_Adoption"] = national_pct_adoption  # fracción [0..1]

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
#plot_lines(t, adopters,             "Adopters of PV systems (hogares)", "Hogares", "plot_adopters")
plot_lines(t, pct_adoption_region,  "Percentage of adoption por región (fracción)", "Fracción (0..1)", "plot_pct_adoption_region")

plt.figure()
plt.plot(t, national_pct_adoption, linewidth=0.8)
plt.title("National Percentage of Adoption")
plt.xlabel("Mes")
plt.ylabel("Fracción (0..1)")
plt.grid(True, linestyle="--", linewidth=0.4)
plt.savefig(PLOT_DIR / "plot_national_pct_adoption.png", bbox_inches="tight")

# plt.show()  # descomenta si quieres ver ventanas

print("CSV:", csv_path.as_posix())
print("Plots en:", PLOT_DIR.as_posix())
