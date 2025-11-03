# -*- coding: utf-8 -*-
# ==============================================
# MÓDULO 8 — Payback Analysis (solo modelo exponencial)
# Flujo: Payback → Exponential probability → Payback fraction → Delayed payback fraction
# Todo en AÑOS (coherente con modulo5 oficial que retorna sstc_temporal).
# ==============================================
from __future__ import annotations
from typing import Dict, Optional, Tuple, Literal
from dataclasses import dataclass
import numpy as np
import pandas as pd

# Importa modulo5 oficial (retorna sstc_temporal)
# --- INICIO DE CORRECCIÓN ---
# 1. Se usa '..' para subir un nivel (de 'modulo8' a 'Parte1') y luego entrar a 'modulo5'
from ..modulo5 import modulo5 as m5
# --- FIN DE CORRECCIÓN ---

# Regiones del modelo
REGIONES = m5.REGIONES  # ("Norte", "Centro", "Sur")


# ----------------------------
# Lecturas desde Excel (AÑOS)
# ----------------------------
def load_annual_savings(ruta_excel: str,
                        sheet: str | int = 0,
                        columns: Tuple[str, str, str] = ("North", "Center", "South"),
                        row: int | None = None) -> np.ndarray:
    """
    Lee Annual savings como un vector (3,) [USD/año] para N, C, S de una fila.
    Útil cuando quieres un único año.
    """
    df = pd.read_excel(ruta_excel, sheet_name=sheet)
    if row is None:
        row = 0
    vec = df.loc[row, list(columns)].to_numpy(dtype=float)
    if vec.shape != (3,):
        raise ValueError("Annual savings: la fila seleccionada no produce vector (3,).")
    return vec


def load_annual_savings_series(ruta_excel: str,
                               sheet: str | int = 0,
                               cols: Tuple[str, str, str] = ("North", "Center", "South"),
                               year_col: str = "Year") -> tuple[np.ndarray, np.ndarray]:
    """
    Serie anual: years (T,), S (3,T) en USD/año por región.
    """
    df = pd.read_excel(ruta_excel, sheet_name=sheet)
    years = df[year_col].to_numpy()
    S = np.vstack([
        df[cols[0]].to_numpy(dtype=float),
        df[cols[1]].to_numpy(dtype=float),
        df[cols[2]].to_numpy(dtype=float),
    ])
    return years, S


# ----------------------------
# Costos desde Módulo 5 (AÑOS)
# ----------------------------
def get_subsidized_costs_vector(ruta_excel_costos: str,
                                variables: Optional[Dict] = None,
                                t_index: int = 0) -> np.ndarray:
    """
    Retorna SSTC de UN AÑO específico como vector (3,) [N, C, S].
    - t_index = 0 => primer año desde anio_inicio del Excel.
    - Usa modulo5.correr_modelo (oficial), que devuelve 'sstc_temporal' (3, T).
    """
    res = m5.correr_modelo(ruta_excel_costos, variables)
    if "sstc_temporal" not in res:
        # CORRECCIÓN: El 'correr_modelo' original (en modulo5.py) devuelve 'sstc_mensual'.
        # Si este código espera 'sstc_temporal', puede que estés usando una versión
        # de modulo5.py que no es la que me has pasado. 
        # Voy a asumir que la clave es 'sstc_mensual' como en el modulo5.py que vi.
        if "sstc_mensual" not in res:
             raise KeyError("El resultado de correr_modelo no contiene 'sstc_mensual' ni 'sstc_temporal'.")
        # Si la clave es 'sstc_mensual', la usamos
        sstc_temporal = np.asarray(res["sstc_mensual"], dtype=float) # (3, T_meses)
        
        # --- Lógica para convertir SSTC mensual a anual ---
        # Asumimos que quieres el SSTC al final del año (índice 11, 23, 35...)
        # t_index aquí significa AÑO, no mes.
        
        mes_index = (t_index + 1) * 12 - 1 # t_index=0 -> mes 11
        
        if not (0 <= mes_index < sstc_temporal.shape[1]):
             raise IndexError(f"Índice de año t_index={t_index} (mes {mes_index}) fuera de rango [0, {sstc_temporal.shape[1]-1}]")
        return sstc_temporal[:, mes_index] # (3,)
        
    else:
        # Si 'sstc_temporal' SÍ existe, se usa la lógica original
        sstc_temporal = np.asarray(res["sstc_temporal"], dtype=float)  # (3, T_años)
        if not (0 <= t_index < sstc_temporal.shape[1]):
            raise IndexError(f"t_index={t_index} fuera de rango [0, {sstc_temporal.shape[1]-1}]")
        return sstc_temporal[:, t_index]  # (3,)


def get_subsidized_costs_series(ruta_excel_costos: str,
                                variables: Optional[Dict] = None) -> np.ndarray:
    """
    Retorna la serie completa SSTC como matriz (3, T) [N, C, S] x años.
    """
    res = m5.correr_modelo(ruta_excel_costos, variables)
    if "sstc_temporal" not in res:
        # Misma corrección/lógica que en la función anterior
        if "sstc_mensual" not in res:
             raise KeyError("El resultado de correr_modelo no contiene 'sstc_mensual' ni 'sstc_temporal'.")
        
        sstc_mensual = np.asarray(res["sstc_mensual"], dtype=float) # (3, T_meses)
        n_meses = sstc_mensual.shape[1]
        n_anios = n_meses // 12
        
        # Tomar el valor del SSTC del último mes de cada año
        indices_anuales = np.arange(1, n_anios + 1) * 12 - 1
        return sstc_mensual[:, indices_anuales] # (3, T_años)
    else:
        # Si 'sstc_temporal' SÍ existe, se usa la lógica original
        return np.asarray(res["sstc_temporal"], dtype=float)  # (3, T_años)


# ----------------------------
# Ecuación 1: Payback (años)
# Payback = Cost / AnnualSavings
# ----------------------------
def payback_years(subsidized_costs: np.ndarray,
                  annual_savings: np.ndarray,
                  zero_mode: Literal["inf", "nan", "zero"] = "inf") -> np.ndarray:
    """
    Vector a vector: (3,) / (3,) → (3,)
    """
    C = np.asarray(subsidized_costs, dtype=float)
    S = np.asarray(annual_savings, dtype=float)
    out = np.full_like(C, np.nan, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        np.divide(C, S, out=out, where=(S > 0))
    mask = ~(S > 0)
    if zero_mode == "inf":
        out[mask] = np.inf
    elif zero_mode == "zero":
        out[mask] = 0.0
    else:
        out[mask] = np.nan
    return out


def payback_years_series(subsidized_costs: np.ndarray,
                         annual_savings_series: np.ndarray,
                         zero_mode: Literal["inf", "nan", "zero"] = "inf") -> np.ndarray:
    """
    Series:
    - Si subsidized_costs es (3,), se difunde a todas las columnas de S (3,T).
    - Si subsidized_costs es (3,T), se hace división elemento a elemento.
    Retorna (3, T).
    """
    S = np.asarray(annual_savings_series, dtype=float)  # (3,T)
    C = np.asarray(subsidized_costs, dtype=float)
    if C.ndim == 1:
        C = C[:, None]  # (3,1) → broadcast a (3,T)
    
    # Asegurarse de que las formas sean compatibles para la división
    # Si S es (3, T_s) y C es (3, T_c), tomar el mínimo de T_s y T_c
    if C.shape[1] != S.shape[1] and C.shape[1] > 1:
        T = min(C.shape[1], S.shape[1])
        C = C[:, :T]
        S = S[:, :T]

    out = np.full_like(S, np.nan, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        np.divide(C, S, out=out, where=(S > 0))
    
    mask = ~(S > 0)
    if zero_mode == "inf":
        out[mask] = np.inf
    elif zero_mode == "zero":
        out[mask] = 0.0
    else:
        out[mask] = np.nan
    return out


# ----------------------------
# Ecuación 2: Exponential probability
# p(t) = exp(beta * Payback(t))  (beta < 0) → clip [0,1]
# ----------------------------
@dataclass
class ExpoInputs:
    beta: float = -0.3   # negativo para que p decrezca con PB


def exponential_probability(beta: float, payback: np.ndarray) -> np.ndarray:
    pb = np.asarray(payback, dtype=float)
    p  = np.exp(beta * pb)
    return np.clip(p, 0.0, 1.0)


# ----------------------------
# Ecuación 3: Payback fraction
# f(t) = p(t)
# ----------------------------
def payback_fraction(prob_exponential: np.ndarray) -> np.ndarray:
    return np.asarray(prob_exponential, dtype=float)


# ----------------------------
# Ecuación 4: Delayed payback fraction
# f_delay(t) ≈ promedio móvil (o smoothing exponencial si quieres)
# - anual: ventana = 1 por defecto
# ----------------------------
def _moving_average_1d(x: np.ndarray, window: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if window is None or window <= 1:
        return x.copy()
    k = np.ones(window, dtype=float) / float(window)
    left = np.repeat(x[0], window // 2)
    right = np.repeat(x[-1], window - 1 - window // 2)
    xpad = np.concatenate([left, x, right], axis=0)
    y = np.convolve(xpad, k, mode="valid")
    return y


def moving_average_series(x: np.ndarray, window_steps: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        return _moving_average_1d(x, window_steps)
    elif x.ndim == 2:
        out = np.zeros_like(x, dtype=float)
        for i in range(x.shape[0]):
            out[i, :] = _moving_average_1d(x[i, :], window_steps)
        return out
    else:
        raise ValueError("x debe ser (T,) o (3,T).")


def delayed_payback_fraction_series(payback_fraction_series: np.ndarray,
                                    tau_years: int | None = 1,
                                    data_frequency: Literal["annual", "monthly"] = "annual") -> np.ndarray:
    """
    Devuelve fracción retrasada usando promedio móvil simple.
    En este módulo trabajamos en AÑOS → window = tau_years (≥1).
    """
    if data_frequency != "annual":
        raise ValueError("Este módulo opera en años. Usa data_frequency='annual'.")
    window = int(tau_years) if tau_years is not None else 1
    window = max(window, 1)
    return moving_average_series(payback_fraction_series, window)


__all__ = [
    "REGIONES",
    "load_annual_savings",
    "load_annual_savings_series",
    "get_subsidized_costs_vector",   # NUEVO: (3,)
    "get_subsidized_costs_series",   # NUEVO: (3,T)
    "payback_years",                  # Eq.1 (vector)
    "payback_years_series",           # Eq.1 (serie)
    "ExpoInputs",
    "exponential_probability",        # Eq.2
    "payback_fraction",               # Eq.3
    "moving_average_series",
    "delayed_payback_fraction_series" # Eq.4
]