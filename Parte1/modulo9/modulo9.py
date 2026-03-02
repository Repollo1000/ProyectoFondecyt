# -*- coding: utf-8 -*-
"""
modulo9.py — Módulo de Emisiones (Solo Sección 9.2: Consumo de Electricidad).

Flujo:
1. Recolección de datos (Solo Perfil de Consumo; Emisiones vienen de fuera).
2. Agrupación de consumo horario a mensual.
3. Cálculo de emisiones por consumo (Hogares * Consumo * Factor).
4. Acumulación anual.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

try:
    from .. import parametros_globales as p_g
except ImportError:
    import parametros_globales as p_g


# =============================================================================
# 1. RECOLECCIÓN DE DATOS (Perfil de Consumo)
# =============================================================================

def cargar_perfil_consumo_mensual() -> pd.DataFrame:
    """
    Lee curva_de_carga.xlsx (Horaria) y la agrupa por mes.
    Retorna: DataFrame (12 filas, 3 columnas) [kWh/mes/hogar].
    """
    xls_path = p_g.MOD9_RUTAS["curva_carga_file"]
    df = pd.read_excel(xls_path, sheet_name="curvas de carga")
    
    # Normalizar nombres
    df = df.rename(columns={
        "mes": "mes",
        "demanda_norte": "Norte",
        "demanda_centro": "Centro",
        "demanda_sur": "Sur",
    })
    
    # Groupby por mes (sumando las horas del mes)
    # Esto nos da el consumo total de un hogar en ese mes (kWh/mes)
    perfil_mensual = df.groupby("mes")[["Norte", "Centro", "Sur"]].sum()
    
    return perfil_mensual


# =============================================================================
# 2. CÁLCULO DE EMISIONES (SECCIÓN 9.2)
# =============================================================================

def calcular_emisiones_consumo(
    factores_emision: np.ndarray, 
    households: np.ndarray, 
    perfil_consumo_12_meses: pd.DataFrame
) -> np.ndarray:
    """
    Calcula emisiones = Factor * Consumo * Hogares.
    
    Args:
        factores_emision: Vector (T,) [tCO2/MWh].
        households: Matriz (T, 3) [Cantidad de Hogares].
        perfil_consumo_12_meses: DF 12 filas [kWh/mes].
        
    Returns:
        emisiones: Matriz (T, 3) [tCO2/mes].
    """
    # A. Preparar Consumo (Expandir 12 meses a todo el periodo T)
    consumo_base = perfil_consumo_12_meses[["Norte", "Centro", "Sur"]].to_numpy(dtype=float) # (12, 3)
    
    # Calculamos cuántas veces repetir el año para cubrir la simulación
    n_meses_necesarios = households.shape[0]
    n_repeticiones = int(np.ceil(n_meses_necesarios / 12))
    
    # Repetimos el año las veces necesarias (Tile)
    consumo_t_completo = np.tile(consumo_base, (n_repeticiones, 1))
    
    # B. Definir T (Mínimo largo común)
    T = min(len(factores_emision), households.shape[0], consumo_t_completo.shape[0])
    
    
    F = factores_emision[:T]       # (T,)
    H = households[:T, :]          # (T, 3)
    C_kwh = consumo_t_completo[:T, :]  # (T, 3) en kWh/mes
    
    # D. Cálculo Vectorial
    # 1. Convertir Consumo unitario de kWh a MWh (dividir por 1000)
    C_mwh = C_kwh / 1000.0
    
    # 2. Calcular Emisiones: Factor (tCO2/MWh) * Consumo (MWh) * Hogares
    emisiones = F[:, None] * C_mwh * H
    
    return emisiones


# =============================================================================
# 3. ACUMULACIÓN ANUAL
# =============================================================================

def acumular_anualmente(datos_mensuales: np.ndarray) -> np.ndarray:
    """
    Suma los datos mensuales en bloques de 12 meses (Años).
    Input: (T_meses, 3) -> Output: (T_años, 3).
    """
    T, regiones = datos_mensuales.shape
    n_anios = T // 12
    
    # Recortar sobrantes
    datos_limpios = datos_mensuales[:n_anios*12, :]
    
    # Reshape: (Años, 12 meses, 3 regiones)
    datos_reshaped = datos_limpios.reshape(n_anios, 12, regiones)
    
    # Sumar eje 1 (los 12 meses)
    anual = np.sum(datos_reshaped, axis=1)
    
    return anual

__all__ = ["cargar_perfil_consumo_mensual", 
           "calcular_emisiones_consumo", "acumular_anualmente"]