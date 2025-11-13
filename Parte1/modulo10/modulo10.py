# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd
import os

# --- IMPORTACIÓN CLAVE ---
from .. import parametros_globales as p_g 
# ------------------------

# ============================
# CONSTANTES
# ============================
REGIONES = p_g.REGIONES # USAMOS LAS REGIONES GLOBALES

# Columnas esperadas en los archivos Excel de perfiles
COLUMNAS_CONSUMO = {
    "Norte": "consumo_norte",
    "Centro": "consumo_centro",
    "Sur": "consumo_sur"
}

# Apuntamos a las columnas del archivo Factor_capacidad_solar.csv
COLUMNAS_GENERACION = {
    "Norte": "factor_antofagasta",
    "Centro": "factor_santiago",
    "Sur": "factor_puertomontt"
}
# ==========================================

# ============================
# VARIABLES POR DEFECTO
# ============================
def default_variables_balance() -> Dict:
    """
    Variables por defecto para el balance energético.
    Estos valores se usan si main_mod10.py no los sobrescribe.
    (Se asume que los precios se proporcionarán en CLP en main_mod10.py)
    """
    return {
        # Consumo mensual promedio por hogar [kWh/mes]
        "consumo_mensual_hogar": np.array([250.0, 240.0, 230.0], dtype=float),  # Norte, Centro, Sur
        
        # Tarifas [CLP/kWh] (Valores por defecto, serán sobrescritos por el Excel)
        # Estos son solo ejemplos si no se carga el Excel
        "tarifa_netbilling": np.array([120.0, 130.0, 125.0], dtype=float),  
        "precio_electricidad": np.array([160.0, 170.0, 165.0], dtype=float), 
        
        # Opciones de cálculo
        "usar_perfil_horario": False,  
        "porcentaje_autoconsumo_simple": 0.60, 
    }


# ============================
# PROCESAMIENTO DE PERFILES (SIN CAMBIOS)
# ============================
def _procesar_perfil(df: pd.DataFrame, columnas_regiones: Dict[str, str]) -> Dict[str, np.ndarray]:
    """
    Función interna para procesar DataFrames de perfiles horarios.
    Convierte el DF (mes, hora, region1, region2, ...) 
    en un dict de {region: np.array[12, 24]}
    """
    perfiles_por_region = {}
    
    df.columns = df.columns.str.lower().str.strip()
    
    for region, nombre_columna in columnas_regiones.items():
        nombre_columna_norm = nombre_columna.lower().strip()
        
        if nombre_columna_norm not in df.columns:
            print(f"✗ ADVERTENCIA: No se encontró la columna '{nombre_columna_norm}' para {region}.")
            perfiles_por_region[region] = np.ones((12, 24), dtype=float) / 24.0
            continue
        
        try:
            perfil = df.groupby(['mes', 'hora'])[nombre_columna_norm].mean().unstack(fill_value=0)
            
            perfil = perfil.reindex(index=range(1, 13), columns=range(0, 24), fill_value=0)

        except KeyError:
             print(f"✗ ERROR: Faltan columnas 'mes' o 'hora' en el archivo de perfil.")
             perfiles_por_region[region] = np.ones((12, 24), dtype=float) / 24.0
             continue

        suma_mensual = perfil.sum(axis=1)
        suma_mensual[suma_mensual == 0] = 1.0 
        
        perfil_normalizado = perfil.div(suma_mensual, axis=0)
        
        perfiles_por_region[region] = perfil_normalizado.values 
    
    return perfiles_por_region


# ============================
# BALANCE ENERGÉTICO (SIN CAMBIOS)
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
    """
    autoconsumo_total = 0.0
    inyeccion_total = 0.0
    
    for hora in range(24): 
        
        consumo_hora = consumo_mensual * perfil_consumo[mes, hora]
        generacion_hora = generacion_mensual * cf_horario[mes, hora]
        
        if generacion_hora >= consumo_hora:
            autoconsumo_total += consumo_hora
            inyeccion_total += (generacion_hora - consumo_hora)
        else:
            autoconsumo_total += generacion_hora
    
    return autoconsumo_total, inyeccion_total


def balance_energetico_simple(
    generacion_mensual: float,
    porcentaje_autoconsumo: float = 0.60
) -> Tuple[float, float]:
    """
    Balance simplificado (método Felipe: 60% autoconsumo, 40% inyección).
    """
    autoconsumo = generacion_mensual * porcentaje_autoconsumo
    inyeccion = generacion_mensual * (1.0 - porcentaje_autoconsumo)
    return autoconsumo, inyeccion


# ============================
# CÁLCULO DE AHORRO (SIN CAMBIOS)
# ============================
def calcular_ahorro_mensual(
    autoconsumo: float,
    inyeccion: float,
    lcoe: float,
    precio_electricidad: float,
    tarifa_inyeccion: float 
) -> Dict[str, float]:
    """
    Calcula el ahorro mensual por balance energético.
    """
    ahorro_inyeccion = inyeccion * tarifa_inyeccion 
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
# BALANCE COMPLETO (CORRECCIÓN: Lógica de políticas al 50/100/80%)
# ============================
def calcular_balance_energetico(
    energia_mensual: np.ndarray, 
    lcoe_mensual: np.ndarray,     
    variables_balance: Dict,
    df_consumo_horario: Optional[pd.DataFrame] = None,
    df_generacion_horario: Optional[pd.DataFrame] = None,
    politica: str = "Net Billing" 
) -> Dict:
    """
    Calcula el balance energético completo para todas las regiones
    basado en la política de inyección especificada.
    """
    N_meses = energia_mensual.shape[1]
    
    consumo_mensual_hogar = np.asarray(variables_balance["consumo_mensual_hogar"], dtype=float)
    tarifa_netbilling_base = np.asarray(variables_balance["tarifa_netbilling"], dtype=float) 
    precio_elec = np.asarray(variables_balance["precio_electricidad"], dtype=float)
    
    usar_horario = variables_balance.get("usar_perfil_horario", False)
    
    perfiles_consumo = {}
    cf_horarios = {}
    
    if usar_horario:
        if df_consumo_horario is not None and df_generacion_horario is not None:
            print("Modo Balance Horario activado. Procesando perfiles...")
            
            perfiles_consumo = _procesar_perfil(df_consumo_horario, COLUMNAS_CONSUMO)
            print("✓ Perfiles de consumo procesados.")
            
            cf_horarios = _procesar_perfil(df_generacion_horario, COLUMNAS_GENERACION)
            print("✓ Perfiles de generación procesados.")
            
        else:
            print("ADVERTENCIA: 'usar_perfil_horario' es True, pero no se proveyeron DataFrames de perfiles.")
            usar_horario = False 
    
    if not usar_horario:
        print("Usando Modo Balance Simple (60/40).")

    
    autoconsumo_mat = np.zeros((3, N_meses), dtype=float)
    inyeccion_mat = np.zeros((3, N_meses), dtype=float)
    ahorro_mat = np.zeros((3, N_meses), dtype=float)
    ahorro_inyeccion_mat = np.zeros((3, N_meses), dtype=float)
    ahorro_autoconsumo_mat = np.zeros((3, N_meses), dtype=float)
    
    for i, region in enumerate(REGIONES):
        for m in range(N_meses):
            generacion = energia_mensual[i, m]
            lcoe = lcoe_mensual[i, m]
            consumo = consumo_mensual_hogar[i] 
            
            # 1. CALCULAR BALANCE (Autoconsumo vs Inyección)
            if usar_horario and region in perfiles_consumo and region in cf_horarios:
                mes_año = m % 12  
                
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
            
            
            # 2. DEFINIR TARIFA DE INYECCIÓN SEGÚN POLÍTICA (50/100/80%)
            precio_elec_regional = precio_elec[i]
            
            if politica == "Net Billing":
                # Tarifa = 50% del precio de electricidad (REGLA SOLICITADA)
                tarifa_inyeccion = precio_elec_regional * 0.5 
            elif politica == "Net Metering":
                # Tarifa = 100% del precio de electricidad (REGLA SOLICITADA)
                tarifa_inyeccion = precio_elec_regional
            elif politica == "Feed-in Tariff":
                # Tarifa = 80% del precio de electricidad (REGLA SOLICITADA)
                tarifa_inyeccion = precio_elec_regional * 0.8
            else:
                # Caso por defecto: Usar Net Billing 50%
                tarifa_inyeccion = precio_elec_regional * 0.5 
            
            
            # 3. CALCULAR AHORRO
            resultado_ahorro = calcular_ahorro_mensual(
                autoconsumo,
                inyeccion,
                lcoe,
                precio_elec_regional,
                tarifa_inyeccion
            )
            
            autoconsumo_mat[i, m] = autoconsumo
            inyeccion_mat[i, m] = inyeccion
            ahorro_mat[i, m] = resultado_ahorro["ahorro_total"]
            ahorro_inyeccion_mat[i, m] = resultado_ahorro["ahorro_inyeccion"]
            ahorro_autoconsumo_mat[i, m] = resultado_ahorro["ahorro_autoconsumo"]
    
    return {
        "autoconsumo_mensual": autoconsumo_mat,
        "inyeccion_mensual": inyeccion_mat,
        "ahorro_mensual": ahorro_mat, 
        "ahorro_inyeccion": ahorro_inyeccion_mat,
        "ahorro_autoconsumo": ahorro_autoconsumo_mat,
        "variables": variables_balance,
        "politica_ejecutada": politica
    }


# ============================
# INTEGRACIÓN CON MÓDULO 5 (SIN CAMBIOS)
# ============================
def correr_modelo_completo(
    resultados_m5: Dict,
    variables_balance: Optional[Dict] = None,
    df_consumo_horario: Optional[pd.DataFrame] = None,
    df_generacion_horario: Optional[pd.DataFrame] = None,
    politica: str = "Net Billing" 
) -> Dict:
    """
    Integra el balance energético con los resultados del módulo 5.
    """
    if variables_balance is None:
        variables_balance = default_variables_balance()
    
    energia_mensual = resultados_m5["energia_mensual"]
    lcoe_mensual = resultados_m5["lcoe_mensual"]
    
    balance = calcular_balance_energetico(
        energia_mensual,
        lcoe_mensual,
        variables_balance,
        df_consumo_horario,
        df_generacion_horario,
        politica 
    )
    
    return {
        **resultados_m5,  
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
]