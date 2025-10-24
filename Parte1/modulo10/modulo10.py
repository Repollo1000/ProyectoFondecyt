# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd
import os

# ============================
# CONSTANTES
# ============================
REGIONES = ("Norte", "Centro", "Sur")

# Mapeo de archivos CSV por región
CSV_FILES = {
    "Norte": "Antofagasta.csv",
    "Centro": "Santiago.csv",
    "Sur": "Puerto montt.csv"
}


# ============================
# VARIABLES POR DEFECTO
# ============================
def default_variables_balance() -> Dict:
    """Variables por defecto para el balance energético."""
    return {
        # Consumo mensual promedio por hogar [kWh/mes]
        "consumo_mensual_hogar": np.array([250.0, 240.0, 230.0], dtype=float),  # Norte, Centro, Sur
        
        # Tarifas [$/kWh]
        "tarifa_netbilling": np.array([0.08, 0.09, 0.085], dtype=float),  # Precio inyección a red
        "precio_electricidad": np.array([0.12, 0.13, 0.125], dtype=float),  # Precio compra red
        
        # Opciones de cálculo
        "usar_perfil_horario": False,  # True: hora a hora, False: simplificación 60/40
        "porcentaje_autoconsumo_simple": 0.60,  # Solo si usar_perfil_horario=False
    }


# ============================
# CARGA DE DATOS HORARIOS
# ============================
def cargar_datos_horarios(ruta_csv: str, region: str) -> pd.DataFrame:
    """
    Carga los datos horarios desde CSV.
    
    Returns:
        DataFrame con datos horarios
    """
    try:
        df = pd.read_csv(ruta_csv)
        print(f"✓ Cargado {os.path.basename(ruta_csv)} para región {region}: {len(df)} registros")
        print(f"  Columnas: {list(df.columns)}")
        return df
    except Exception as e:
        print(f"✗ Error cargando {ruta_csv}: {e}")
        return pd.DataFrame()


def calcular_perfil_horario_consumo(df: pd.DataFrame) -> np.ndarray:
    """
    Calcula perfil horario normalizado de consumo (%).
    
    Returns:
        np.ndarray (12, 24): [mes, hora] con porcentajes que suman 1.0 por mes
    """
    # TODO: Adaptar según columnas reales del CSV
    # Por ahora retorna distribución uniforme
    perfil = np.ones((12, 24), dtype=float) / 24.0
    return perfil


def calcular_capacity_factor_horario(df: pd.DataFrame) -> np.ndarray:
    """
    Calcula capacity factor horario desde datos CSV.
    
    Returns:
        np.ndarray (12, 24): [mes, hora] con CF horario normalizado
    """
    # TODO: Adaptar según columnas reales del CSV
    # Por ahora retorna perfil solar típico
    perfil = np.zeros((12, 24), dtype=float)
    for mes in range(12):
        for hora in range(24):
            if 6 <= hora <= 18:  # Horas de sol
                # Curva gaussiana centrada en hora 12
                perfil[mes, hora] = np.exp(-((hora - 12) ** 2) / 18.0)
    
    # Normalizar para que la suma diaria sea 1.0
    for mes in range(12):
        suma = np.sum(perfil[mes, :])
        if suma > 0:
            perfil[mes, :] /= suma
    
    return perfil


# ============================
# BALANCE ENERGÉTICO
# ============================
def balance_energetico_horario(
    consumo_mensual: float,
    generacion_mensual: float,
    perfil_consumo: np.ndarray,  # (12, 24)
    cf_horario: np.ndarray,      # (12, 24)
    mes: int
) -> Tuple[float, float]:
    """
    Calcula balance energético hora a hora para un mes específico.
    
    Args:
        consumo_mensual: Consumo total del mes [kWh]
        generacion_mensual: Generación total del mes [kWh]
        perfil_consumo: Perfil horario de consumo normalizado (12, 24)
        cf_horario: Capacity factor horario normalizado (12, 24)
        mes: Mes (0-11)
    
    Returns:
        (autoconsumo_mes, inyeccion_mes) en kWh
    """
    autoconsumo_total = 0.0
    inyeccion_total = 0.0
    
    # Días del mes (simplificado: 30 días)
    dias_mes = 30
    
    for hora in range(24):
        # Consumo horario promedio por hora [kWh]
        consumo_hora = consumo_mensual * perfil_consumo[mes, hora]
        
        # Generación horaria promedio por hora [kWh]
        generacion_hora = generacion_mensual * cf_horario[mes, hora]
        
        # Balance
        if generacion_hora >= consumo_hora:
            # Autogenero todo mi consumo
            autoconsumo_total += consumo_hora
            inyeccion_total += (generacion_hora - consumo_hora)
        else:
            # Autogenero solo lo que puedo
            autoconsumo_total += generacion_hora
            # No hay inyección
    
    return autoconsumo_total, inyeccion_total


def balance_energetico_simple(
    generacion_mensual: float,
    porcentaje_autoconsumo: float = 0.60
) -> Tuple[float, float]:
    """
    Balance simplificado (método Felipe: 60% autoconsumo, 40% inyección).
    
    Args:
        generacion_mensual: Generación total del mes [kWh]
        porcentaje_autoconsumo: % que se autoconsumo (default 60%)
    
    Returns:
        (autoconsumo_mes, inyeccion_mes) en kWh
    """
    autoconsumo = generacion_mensual * porcentaje_autoconsumo
    inyeccion = generacion_mensual * (1.0 - porcentaje_autoconsumo)
    return autoconsumo, inyeccion


# ============================
# CÁLCULO DE AHORRO
# ============================
def calcular_ahorro_mensual(
    autoconsumo: float,
    inyeccion: float,
    lcoe: float,
    precio_electricidad: float,
    tarifa_netbilling: float
) -> Dict[str, float]:
    """
    Calcula el ahorro mensual por balance energético.
    
    Ahorro = Inyección × Tarifa_NetBilling 
           + Autoconsumo × (Precio_Electricidad - LCOE)
    
    Args:
        autoconsumo: Energía autoconsumida [kWh]
        inyeccion: Energía inyectada a red [kWh]
        lcoe: Costo nivelado de energía [$/kWh]
        precio_electricidad: Precio de compra de red [$/kWh]
        tarifa_netbilling: Precio de venta a red [$/kWh]
    
    Returns:
        Dict con desglose de ahorro
    """
    ahorro_inyeccion = inyeccion * tarifa_netbilling
    ahorro_autoconsumo = autoconsumo * (precio_electricidad - lcoe)
    ahorro_total = ahorro_inyeccion + ahorro_autoconsumo
    
    return {
        "ahorro_inyeccion": ahorro_inyeccion,
        "ahorro_autoconsumo": ahorro_autoconsumo,
        "ahorro_total": ahorro_total,
        "autoconsumo_kWh": autoconsumo,
        "inyeccion_kWh": inyeccion
    }


# ============================
# BALANCE COMPLETO
# ============================
def calcular_balance_energetico(
    energia_mensual: np.ndarray,  # (3, N_meses) del módulo 5
    lcoe_mensual: np.ndarray,     # (3, N_meses) del módulo 5
    variables_balance: Dict,
    ruta_csvs: Optional[Dict[str, str]] = None
) -> Dict:
    """
    Calcula el balance energético completo para todas las regiones.
    
    Args:
        energia_mensual: Generación solar por región y mes (3, N_meses)
        lcoe_mensual: LCOE por región y mes (3, N_meses)
        variables_balance: Variables de configuración
        ruta_csvs: Dict con rutas a CSVs por región (opcional)
    
    Returns:
        Dict con resultados del balance energético
    """
    N_meses = energia_mensual.shape[1]
    
    # Variables
    consumo_mensual = np.asarray(variables_balance["consumo_mensual_hogar"], dtype=float)
    tarifa_nb = np.asarray(variables_balance["tarifa_netbilling"], dtype=float)
    precio_elec = np.asarray(variables_balance["precio_electricidad"], dtype=float)
    usar_horario = variables_balance.get("usar_perfil_horario", False)
    
    # Cargar datos horarios si están disponibles
    perfiles_consumo = {}
    cf_horarios = {}
    
    if usar_horario and ruta_csvs is not None:
        for i, region in enumerate(REGIONES):
            if region in ruta_csvs:
                df = cargar_datos_horarios(ruta_csvs[region], region)
                if not df.empty:
                    perfiles_consumo[region] = calcular_perfil_horario_consumo(df)
                    cf_horarios[region] = calcular_capacity_factor_horario(df)
    
    # Inicializar matrices de resultados
    autoconsumo_mat = np.zeros((3, N_meses), dtype=float)
    inyeccion_mat = np.zeros((3, N_meses), dtype=float)
    ahorro_mat = np.zeros((3, N_meses), dtype=float)
    ahorro_inyeccion_mat = np.zeros((3, N_meses), dtype=float)
    ahorro_autoconsumo_mat = np.zeros((3, N_meses), dtype=float)
    
    # Calcular balance para cada región y mes
    for i, region in enumerate(REGIONES):
        for m in range(N_meses):
            generacion = energia_mensual[i, m]
            lcoe = lcoe_mensual[i, m]
            consumo = consumo_mensual[i]
            
            # Balance (horario o simple)
            if usar_horario and region in perfiles_consumo:
                mes_año = m % 12  # Mes del año (0-11)
                autoconsumo, inyeccion = balance_energetico_horario(
                    consumo, generacion,
                    perfiles_consumo[region],
                    cf_horarios[region],
                    mes_año
                )
            else:
                # Método simple
                pct = variables_balance.get("porcentaje_autoconsumo_simple", 0.60)
                autoconsumo, inyeccion = balance_energetico_simple(generacion, pct)
            
            # Calcular ahorro
            resultado_ahorro = calcular_ahorro_mensual(
                autoconsumo, inyeccion,
                lcoe, precio_elec[i], tarifa_nb[i]
            )
            
            autoconsumo_mat[i, m] = autoconsumo
            inyeccion_mat[i, m] = inyeccion
            ahorro_mat[i, m] = resultado_ahorro["ahorro_total"]
            ahorro_inyeccion_mat[i, m] = resultado_ahorro["ahorro_inyeccion"]
            ahorro_autoconsumo_mat[i, m] = resultado_ahorro["ahorro_autoconsumo"]
    
    return {
        "autoconsumo_mensual": autoconsumo_mat,  # (3, N_meses) kWh
        "inyeccion_mensual": inyeccion_mat,      # (3, N_meses) kWh
        "ahorro_mensual": ahorro_mat,            # (3, N_meses) USD
        "ahorro_inyeccion": ahorro_inyeccion_mat,
        "ahorro_autoconsumo": ahorro_autoconsumo_mat,
        "variables": variables_balance
    }


# ============================
# INTEGRACIÓN CON MÓDULO 5
# ============================
def correr_modelo_completo(
    resultados_m5: Dict,
    variables_balance: Optional[Dict] = None,
    ruta_csvs: Optional[Dict[str, str]] = None
) -> Dict:
    """
    Integra el balance energético con los resultados del módulo 5.
    
    Args:
        resultados_m5: Resultados de modulo5.correr_modelo()
        variables_balance: Variables de balance (usa defaults si None)
        ruta_csvs: Rutas a CSVs por región
    
    Returns:
        Dict con resultados completos (módulo 5 + balance energético)
    """
    if variables_balance is None:
        variables_balance = default_variables_balance()
    
    # Extraer datos del módulo 5
    energia_mensual = resultados_m5["energia_mensual"]
    lcoe_mensual = resultados_m5["lcoe_mensual"]
    
    # Calcular balance
    balance = calcular_balance_energetico(
        energia_mensual,
        lcoe_mensual,
        variables_balance,
        ruta_csvs
    )
    
    # Combinar resultados
    return {
        **resultados_m5,  # Todos los resultados del módulo 5
        "balance_energetico": balance
    }


# ============================
# EXPORTS
# ============================
__all__ = [
    "default_variables_balance",
    "calcular_balance_energetico",
    "calcular_ahorro_mensual",
    "balance_energetico_horario",
    "balance_energetico_simple",
    "correr_modelo_completo",
    "cargar_datos_horarios",
]