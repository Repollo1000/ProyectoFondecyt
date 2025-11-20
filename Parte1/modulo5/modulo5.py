# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd

# --- IMPORTACIÓN CLAVE ---
# Importamos el nuevo archivo de parámetros
from .. import parametros_globales as p_g 
# ------------------------

# ============================
# CONSTANTES
# ============================
DIFF_TABLE = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0], dtype=float)
SUB_TABLE  = np.array([0.30, 0.29, 0.25, 0.18, 0.10, 0.04, 0.02, 0.00, 0.00], dtype=float)


# ============================
# VARIABLES POR DEFECTO
# ============================
def default_variables() -> Dict:
    """Variables con valores por defecto para el modelo."""
    return {
        # Finanzas
        "rtasa": np.array([0.02, 0.02, 0.02], dtype=float),        # tasa anual
        # USAMOS LA CONSTANTE GLOBAL
        "project_lifetime_months": p_g.PROJECT_LIFETIME_MONTHS, 
        "Percentage_capital_subsidy": np.array([0.3, 0.3, 0.3], dtype=float),
        
        # Energía
        # USAMOS EL VALOR DE PVGP DE LOS PARÁMETROS GLOBALES
        "pvgp_kW_per_household": p_g.MOD7_VARIABLES_INICIALES["pvgp_kW_per_household"],
        "capacity_factor_avg": np.array([0.35, 0.25, 0.25], dtype=float),
        "performance_ratio": np.array([0.75, 0.75, 0.75], dtype=float),
        "degradation_factor": np.array([0.6, 0.6, 0.6], dtype=float),  # por región
        "hours_per_month": 720.0,
        
        # Subsidio dinámico (opcional)
        "use_dynamic_subsidy": False,
        "difference_init": np.array([0.45, 0.72, 0.88], dtype=float),
    }


# ============================
# UTILIDADES
# ============================
def tasa_mensual(r_anual: np.ndarray) -> np.ndarray:
    """Convierte tasa anual efectiva a mensual: r_m = (1 + r_anual)^(1/12) - 1"""
    return np.power(1.0 + r_anual, 1.0/12.0) - 1.0


def costos_anuales_a_mensuales(mat_años: np.ndarray, meses_totales: int) -> np.ndarray:
    """
    Convierte matriz de costos anuales a mensuales:
    - Divide cada valor anual por 12
    - Repite 12 veces por año
    - Recorta a meses_totales
    """
    mensual = np.repeat(mat_años / 12.0, 12, axis=1)
    return mensual[:, :meses_totales]


def subsidy_from_difference(difference: np.ndarray) -> np.ndarray:
    """Interpola subsidio según tabla de referencia."""
    diff = np.clip(difference, DIFF_TABLE.min(), DIFF_TABLE.max())
    return np.interp(diff, DIFF_TABLE, SUB_TABLE)


def safe_ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """División segura evitando división por cero."""
    out = np.zeros_like(num, dtype=float)
    mask = den > 0
    out[mask] = num[mask] / den[mask]
    return out


# ============================
# CÁLCULO SSTC MENSUAL
# ============================
def calcular_sstc_mensual(
    tabla_costos: pd.DataFrame,
    anio_inicio: int,
    variables: Dict
) -> np.ndarray:
    """
    Calcula el Subsidized System Total Cost mensual.
    
    Returns:
        np.ndarray (3, N_meses): SSTC por región y mes
    """
    N = int(variables["project_lifetime_months"])
    r_m = tasa_mensual(np.asarray(variables["rtasa"], dtype=float))
    subs = np.asarray(variables["Percentage_capital_subsidy"], dtype=float)
    
    # Nombres de columnas (Normalizados)
    inv_cols = ["Costo _inversion_norte", "Costo _inversion_centro", "Costo _inversion_sur"]
    aoc_cols = ["Costo _operacion_norte", "Costo _operacion_centro", "Costo _operacion_sur"]
    
    # Filtrar años desde inicio
    datos = tabla_costos.loc[tabla_costos["Año"] >= anio_inicio]
    años_disponibles = len(datos)
    
    if años_disponibles * 12 < N:
        # El código recorta la simulación al máximo disponible si los datos son insuficientes
        N_max = años_disponibles * 12
        print(f"ADVERTENCIA [MODULO5]: Datos insuficientes. Se necesitan {N} meses, pero solo hay {N_max}. Recortando simulación.")
        variables["project_lifetime_months"] = N_max # Recortar N para que no haya crash en otros cálculos
        N = N_max 
    
    # Extraer costos anuales (3, T_años)
    # Aseguramos que las columnas existan tras la limpieza
    try:
        inv_años = datos[inv_cols].to_numpy(dtype=float).T
        aoc_años = datos[aoc_cols].to_numpy(dtype=float).T
    except KeyError as e:
        print(f"ERROR [MODULO5]: No se encontraron las columnas de costos esperadas. Columnas disponibles: {list(datos.columns)}")
        raise e
    
    # Convertir a mensual (3, N)
    inv_m = costos_anuales_a_mensuales(inv_años, N)
    aoc_m = costos_anuales_a_mensuales(aoc_años, N)
    
    # Calcular SSTC para cada mes
    sstc = np.zeros((3, N), dtype=float)
    
    for m in range(1, N + 1):
        inv_t = inv_m[:, m-1]
        aoc_t = aoc_m[:, m-1]
        
        # Factor de anualidad mensual
        factor = (1.0 - np.power(1.0 + r_m, -m)) / r_m
        pv_aoc = aoc_t * factor
        
        sstc[:, m-1] = (inv_t + pv_aoc) * (1.0 - subs)
    
    return sstc


# ============================
# ENERGÍA MENSUAL
# ============================
def calcular_energia_mensual(variables: Dict) -> np.ndarray:
    """
    Calcula energía mensual por hogar con degradación.
    
    Returns:
        np.ndarray (3, N_meses): Energía en kWh/mes por hogar
    """
    N = int(variables["project_lifetime_months"])
    kW = np.asarray(variables["pvgp_kW_per_household"], dtype=float)
    CF = np.asarray(variables["capacity_factor_avg"], dtype=float)
    PR = np.asarray(variables["performance_ratio"], dtype=float)
    
    # Degradación por región (puede ser escalar o array)
    d = variables["degradation_factor"]
    if np.isscalar(d):
        d = np.full(3, d, dtype=float)
    else:
        d = np.asarray(d, dtype=float)
    
    hpm = float(variables["hours_per_month"])
    
    # Energía base sin degradación
    base = kW * hpm * CF * PR
    
    # Degradación aplicada mensualmente
    m = np.arange(0, N, dtype=float)
    degradacion = np.power(1.0 - d[:, None], m[None, :] / 12.0)
    
    return base[:, None] * degradacion


# ============================
# LCOE MENSUAL
# ============================
def calcular_lcoe_mensual(sstc: np.ndarray, variables: Dict) -> np.ndarray:
    """
    Calcula LCOE mensual.
    
    Returns:
        np.ndarray (3, N_meses): LCOE por región y mes
    """
    N = int(variables["project_lifetime_months"])
    r_m = tasa_mensual(np.asarray(variables["rtasa"], dtype=float))
    E = calcular_energia_mensual(variables)
    
    lcoe = np.zeros((3, N), dtype=float)
    
    for m in range(1, N + 1):
        # Energía acumulada descontada
        i = np.arange(1, m + 1, dtype=float)
        descuento = 1.0 / np.power(1.0 + r_m[:, None], i[None, :])
        energia_pv = np.sum(E[:, :m] * descuento, axis=1)
        
        # División segura
        lcoe[:, m-1] = safe_ratio(sstc[:, m-1], energia_pv)
    
    return lcoe


# ============================
# SUBSIDIO DINÁMICO
# ============================
def aplicar_subsidio_dinamico(variables: Dict) -> Tuple[Dict, Dict]:
    """Actualiza subsidio si use_dynamic_subsidy es True."""
    info = {
        "fuente": "subsidio fijo",
        "difference": None,
        "subsidio": np.asarray(variables["Percentage_capital_subsidy"], dtype=float)
    }
    
    if not variables.get("use_dynamic_subsidy", False):
        return variables, info
    
    # Determinar diferencia
    if "difference_init" in variables:
        diff = np.asarray(variables["difference_init"], dtype=float)
        fuente = "difference_init (t0)"
    elif ("adopters_series" in variables) and ("able_series" in variables):
        adop = np.asarray(variables["adopters_series"], dtype=float)
        able = np.asarray(variables["able_series"], dtype=float)
        diff = safe_ratio(adop[:, 0], able[:, 0])
        fuente = "series adopters/able (t0)"
    else:
        return variables, info
    
    # Actualizar subsidio
    variables = dict(variables)
    variables["Percentage_capital_subsidy"] = subsidy_from_difference(diff)
    
    info["fuente"] = fuente
    info["difference"] = diff
    info["subsidio"] = np.asarray(variables["Percentage_capital_subsidy"], dtype=float)
    
    return variables, info


# ============================
# FUNCIÓN PRINCIPAL
# ============================
def correr_modelo(
    ruta_excel_costos: str,
    variables: Optional[Dict] = None
) -> Dict:
    """
    Ejecuta el modelo completo en granularidad mensual.
    """
    if variables is None:
        variables = default_variables()
    
    # =========================================================================
    # CORRECCIÓN DE LECTURA DE EXCEL
    # =========================================================================
    try:
        # 1. Intentamos leer 'Hoja2' explícitamente (donde están los datos en costos.xlsx)
        tabla_costos = pd.read_excel(ruta_excel_costos, sheet_name="Hoja2")
    except (ValueError, IndexError):
        # 2. Si falla (ej. el archivo no tiene Hoja2), leemos la primera hoja por defecto
        # Esto mantiene compatibilidad si usas costoAño.xlsx en el futuro
        print("Advertencia [MODULO5]: No se encontró 'Hoja2', leyendo la hoja por defecto.")
        tabla_costos = pd.read_excel(ruta_excel_costos)
    
    # LIMPIEZA DE COLUMNAS: Quitar espacios en blanco al inicio/final (ej: "Costo _inversion")
    tabla_costos.columns = tabla_costos.columns.str.strip()
    
    # Validar que exista la columna Año
    if "Año" not in tabla_costos.columns:
         raise ValueError(f"El archivo de costos no tiene la columna 'Año'. Columnas encontradas: {list(tabla_costos.columns)}")

    anio_inicio = int(tabla_costos["Año"].min())
    
    # Aplicar subsidio dinámico si corresponde
    variables, info_sub = aplicar_subsidio_dinamico(variables)
    
    # Calcular métricas mensuales
    sstc = calcular_sstc_mensual(tabla_costos, anio_inicio, variables)
    
    # Las siguientes funciones usan el nuevo 'project_lifetime_months' que puede haber cambiado en calcular_sstc_mensual
    energia = calcular_energia_mensual(variables)
    lcoe = calcular_lcoe_mensual(sstc, variables)
    
    return {
        "regiones": p_g.REGIONES, # USAMOS LAS REGIONES GLOBALES
        "variables": variables,
        "info_subsidio": info_sub,
        "tabla_costos": tabla_costos,
        "anio_inicio": anio_inicio,
        "sstc_mensual": sstc,              # (3, N_meses)
        "energia_mensual": energia,         # (3, N_meses)
        "lcoe_mensual": lcoe                # (3, N_meses)
    }


# ============================
# EXPORTS
# ============================
__all__ = [
    "default_variables",
    "correr_modelo",
    "calcular_sstc_mensual",
    "calcular_energia_mensual",
    "calcular_lcoe_mensual",
    "subsidy_from_difference",
    "aplicar_subsidio_dinamico",
]