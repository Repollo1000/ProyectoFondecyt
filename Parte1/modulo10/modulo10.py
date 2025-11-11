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

# Columnas esperadas en los archivos Excel de perfiles
# (basado en tus imágenes: image_a34e54.png, image_a3bad8.png)
COLUMNAS_CONSUMO = {
    "Norte": "consumo_norte",
    "Centro": "consumo_centro",
    "Sur": "consumo_sur"
}

# --- ¡¡AQUÍ ESTÁ LA CORRECCIÓN DE NOMBRES DE COLUMNA!! ---
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
    ¡¡OJO: Estos precios están en USD!!
    """
    return {
        # Consumo mensual promedio por hogar [kWh/mes]
        "consumo_mensual_hogar": np.array([250.0, 240.0, 230.0], dtype=float),  # Norte, Centro, Sur
        
        # Tarifas [$/kWh] (Valores en USD por defecto)
        # Este 'tarifa_netbilling' es el default si NO se carga el Excel
        "tarifa_netbilling": np.array([0.08, 0.09, 0.085], dtype=float),  # Precio inyección a red
        "precio_electricidad": np.array([0.12, 0.13, 0.125], dtype=float),  # Precio compra red
        
        # Opciones de cálculo
        "usar_perfil_horario": False,  # True: hora a hora, False: simplificación 60/40
        "porcentaje_autoconsumo_simple": 0.60,  # Solo si usar_perfil_horario=False
    }


# ============================
# PROCESAMIENTO DE PERFILES
# ============================

def _procesar_perfil(df: pd.DataFrame, columnas_regiones: Dict[str, str]) -> Dict[str, np.ndarray]:
    """
    Función interna para procesar DataFrames de perfiles horarios.
    Convierte el DF (mes, hora, region1, region2, ...) 
    en un dict de {region: np.array[12, 24]}
    """
    perfiles_por_region = {}
    
    # Normalizar nombres de columnas (minúsculas, sin espacios)
    df.columns = df.columns.str.lower().str.strip()
    
    # Asumimos que las columnas 'mes' y 'hora' existen
    
    for region, nombre_columna in columnas_regiones.items():
        nombre_columna_norm = nombre_columna.lower().strip()
        
        if nombre_columna_norm not in df.columns:
            print(f"✗ ADVERTENCIA: No se encontró la columna '{nombre_columna_norm}' para {region}.")
            # Retornar un perfil uniforme si falla
            perfiles_por_region[region] = np.ones((12, 24), dtype=float) / 24.0
            continue
        
        # Agrupar por mes y hora, promediar el valor
        # Asumimos horas 1-24 y mes 1-12
        try:
            # Los CSV pueden tener hora 0-23 o 1-24. 
            # El archivo Factor_capacidad_solar.csv tiene hora 0-23
            perfil = df.groupby(['mes', 'hora'])[nombre_columna_norm].mean().unstack(fill_value=0)
            
            # Reindexar para asegurar 12 meses (1-12) y 24 horas (0-23)
            perfil = perfil.reindex(index=range(1, 13), columns=range(0, 24), fill_value=0)

        except KeyError:
             print(f"✗ ERROR: Faltan columnas 'mes' o 'hora' en el archivo de perfil.")
             perfiles_por_region[region] = np.ones((12, 24), dtype=float) / 24.0
             continue

        # Normalizar por fila (mes) para que sume 1.0
        suma_mensual = perfil.sum(axis=1)
        suma_mensual[suma_mensual == 0] = 1.0 # Evitar división por cero
        
        perfil_normalizado = perfil.div(suma_mensual, axis=0)
        
        # Guardar como array numpy (se accede por [0-11, 0-23])
        perfiles_por_region[region] = perfil_normalizado.values 
    
    return perfiles_por_region


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
    """
    autoconsumo_total = 0.0
    inyeccion_total = 0.0
    
    for hora in range(24): # Horas 0-23
        
        # Consumo horario [kWh] = (Total mes) * (Porcentaje de esa hora)
        consumo_hora = consumo_mensual * perfil_consumo[mes, hora]
        
        # Generación horaria [kWh] = (Total mes) * (Porcentaje de esa hora)
        generacion_hora = generacion_mensual * cf_horario[mes, hora]
        
        # Balance
        if generacion_hora >= consumo_hora:
            autoconsumo_total += consumo_hora
            inyeccion_total += (generacion_hora - consumo_hora)
        else:
            autoconsumo_total += generacion_hora
            # No hay inyección
    
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
# CÁLCULO DE AHORRO
# ============================
def calcular_ahorro_mensual(
    autoconsumo: float,
    inyeccion: float,
    lcoe: float,
    precio_electricidad: float,
    tarifa_inyeccion: float # <-- Parámetro clave que cambiará
) -> Dict[str, float]:
    """
    Calcula el ahorro mensual por balance energético.
    Todas las monedas deben ser consistentes (ej. USD).
    
    Ahorro = Ahorro por Inyección + Ahorro por Autoconsumo
    
    Ahorro por Inyección = Inyección × Tarifa_Inyeccion
    Ahorro por Autoconsumo = Autoconsumo × (Precio_Electricidad - LCOE)
    """
    # Esta es la tarifa que cambia según la política
    ahorro_inyeccion = inyeccion * tarifa_inyeccion 
    
    # El ahorro por autoconsumo es lo que evito comprar menos lo que me costó generarlo
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
    df_consumo_horario: Optional[pd.DataFrame] = None,
    df_generacion_horario: Optional[pd.DataFrame] = None,
    politica: str = "Net Billing" # <--- NUEVO PARÁMETRO
) -> Dict:
    """
    Calcula el balance energético completo para todas las regiones
    basado en la política de inyección especificada.
    """
    N_meses = energia_mensual.shape[1]
    
    # Variables
    consumo_mensual_hogar = np.asarray(variables_balance["consumo_mensual_hogar"], dtype=float)
    # Tarifa de inyección para Net Billing (cargada del Excel)
    tarifa_netbilling_base = np.asarray(variables_balance["tarifa_netbilling"], dtype=float) 
    # Precio de compra de electricidad
    precio_elec = np.asarray(variables_balance["precio_electricidad"], dtype=float)
    
    usar_horario = variables_balance.get("usar_perfil_horario", False)
    
    # Cargar perfiles horarios
    perfiles_consumo = {}
    cf_horarios = {}
    
    if usar_horario:
        if df_consumo_horario is not None and df_generacion_horario is not None:
            print("Modo Balance Horario activado. Procesando perfiles...")
            
            # Procesar DF de Consumo -> {region: array[12,24]}
            perfiles_consumo = _procesar_perfil(df_consumo_horario, COLUMNAS_CONSUMO)
            print("✓ Perfiles de consumo procesados.")
            
            # Procesar DF de Generación -> {region: array[12,24]}
            cf_horarios = _procesar_perfil(df_generacion_horario, COLUMNAS_GENERACION)
            print("✓ Perfiles de generación procesados.")
            
        else:
            print("ADVERTENCIA: 'usar_perfil_horario' es True, pero no se proveyeron DataFrames de perfiles.")
            usar_horario = False # Forzar modo simple
    
    if not usar_horario:
        print("Usando Modo Balance Simple (60/40).")

    
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
            consumo = consumo_mensual_hogar[i] 
            
            # 1. CALCULAR BALANCE (Autoconsumo vs Inyección)
            if usar_horario and region in perfiles_consumo and region in cf_horarios:
                mes_año = m % 12  # Mes del año (0-11)
                
                autoconsumo, inyeccion = balance_energetico_horario(
                    consumo, generacion,
                    perfiles_consumo[region], # Array (12, 24)
                    cf_horarios[region],     # Array (12, 24)
                    mes_año
                )
            else:
                # Método simple
                pct = variables_balance.get("porcentaje_autoconsumo_simple", 0.60)
                autoconsumo, inyeccion = balance_energetico_simple(generacion, pct)
            
            
            # 2. DEFINIR TARIFA DE INYECCIÓN SEGÚN POLÍTICA
            precio_elec_regional = precio_elec[i]
            
            if politica == "Net Billing":
                # Usa la tarifa de inyección cargada del Excel
                tarifa_inyeccion = tarifa_netbilling_base[i] 
            elif politica == "Net Metering":
                # Tarifa = 100% del precio de electricidad
                tarifa_inyeccion = precio_elec_regional
            elif politica == "Feed-in Tariff":
                # Tarifa = 80% del precio de electricidad (supuesto del doc)
                tarifa_inyeccion = precio_elec_regional * 0.8
            else:
                # Caso por defecto: Net Billing
                tarifa_inyeccion = tarifa_netbilling_base[i]
            
            
            # 3. CALCULAR AHORRO
            resultado_ahorro = calcular_ahorro_mensual(
                autoconsumo,
                inyeccion,
                lcoe,
                precio_elec_regional,
                tarifa_inyeccion # <-- Tarifa variable según política
            )
            
            # Guardar resultados
            autoconsumo_mat[i, m] = autoconsumo
            inyeccion_mat[i, m] = inyeccion
            ahorro_mat[i, m] = resultado_ahorro["ahorro_total"]
            ahorro_inyeccion_mat[i, m] = resultado_ahorro["ahorro_inyeccion"]
            ahorro_autoconsumo_mat[i, m] = resultado_ahorro["ahorro_autoconsumo"]
    
    return {
        "autoconsumo_mensual": autoconsumo_mat,
        "inyeccion_mensual": inyeccion_mat,
        "ahorro_mensual": ahorro_mat, # <--- Este es el resultado que cambia
        "ahorro_inyeccion": ahorro_inyeccion_mat,
        "ahorro_autoconsumo": ahorro_autoconsumo_mat,
        "variables": variables_balance,
        "politica_ejecutada": politica
    }


# ============================
# INTEGRACIÓN CON MÓDULO 5
# ============================
def correr_modelo_completo(
    resultados_m5: Dict,
    variables_balance: Optional[Dict] = None,
    df_consumo_horario: Optional[pd.DataFrame] = None,
    df_generacion_horario: Optional[pd.DataFrame] = None,
    politica: str = "Net Billing" # <--- NUEVO PARÁMETRO
) -> Dict:
    """
    Integra el balance energético con los resultados del módulo 5.
    """
    if variables_balance is None:
        variables_balance = default_variables_balance()
    
    # Extraer datos del módulo 5
    energia_mensual = resultados_m5["energia_mensual"]
    lcoe_mensual = resultados_m5["lcoe_mensual"]
    
    # Calcular balance, pasando la política
    balance = calcular_balance_energetico(
        energia_mensual,
        lcoe_mensual,
        variables_balance,
        df_consumo_horario,
        df_generacion_horario,
        politica # <--- Pasa la política
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
    "calcular_ahorro_mensual", # <--- Mantenemos esta función
    "balance_energetico_horario",
    "balance_energetico_simple",
    "correr_modelo_completo",
]