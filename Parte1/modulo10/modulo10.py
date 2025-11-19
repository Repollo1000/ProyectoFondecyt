# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict
import numpy as np
import pandas as pd

# Importamos parámetros globales si existen, si no, definimos regiones por defecto
try:
    from .. import parametros_globales as p_g
    REGIONES = list(p_g.REGIONES)
except (ImportError, AttributeError):
    REGIONES = ["Norte", "Centro", "Sur"]

# ============================
# CONSTANTES BÁSICAS
# ============================

# Columnas de factor de capacidad en Factor_capacidad_solar.csv
COLUMNAS_FACTOR = {
    "Norte": "factor_antofagasta",
    "Centro": "factor_santiago",
    "Sur": "factor_puertomontt",
}

# Columnas de demanda en curva_de_carga.xlsx
COLUMNAS_DEMANDA = {
    "Norte": "demanda_norte",
    "Centro": "demanda_centro",
    "Sur": "demanda_sur",
}

# Prefijos para nombres de columnas resultantes
PREFIJOS_REGION = {
    "Norte": "norte",
    "Centro": "centro",
    "Sur": "sur",
}


# ============================
# CONSTRUCCIÓN DF HORARIO COMBINADO
# ============================
def construir_dataframe_horario_combinado(
    df_generacion: pd.DataFrame,
    df_consumo: pd.DataFrame,
    pvgp_kW_per_household: np.ndarray,
) -> pd.DataFrame:
    """
    Une Factor_capacidad_solar y curva_de_carga en un solo DataFrame horario.
    CORRECCIÓN: Se asume correspondencia fila a fila (pd.concat) para evitar
    producto cartesiano si se usara merge por mes/hora.
    """

    # 1. Copias de seguridad y normalización de columnas
    df_g = df_generacion.copy()
    df_c = df_consumo.copy()

    df_g.columns = df_g.columns.str.lower().str.strip()
    df_c.columns = df_c.columns.str.lower().str.strip()

    # 2. Reset de índices para asegurar pegado correcto
    df_g = df_g.reset_index(drop=True)
    df_c = df_c.reset_index(drop=True)

    # Verificación de longitud
    if len(df_g) != len(df_c):
        print(f"⚠️ ADVERTENCIA: Los archivos tienen distinta longitud ({len(df_g)} vs {len(df_c)}).")
        print("Se unirán por posición. Las filas sobrantes quedarán con NaN.")

    # 3. Eliminamos columnas repetidas del segundo DF (mes, hora, año)
    cols_to_drop = [col for col in ["año", "mes", "hora"] if col in df_c.columns]
    df_c_limpio = df_c.drop(columns=cols_to_drop)

    # 4. Unimos (Concatenación horizontal)
    df = pd.concat([df_g, df_c_limpio], axis=1)

    # 5. Calculamos generación (Factor * Potencia Instalada)
    df_result = df.copy()

    for idx, region in enumerate(REGIONES):
        pref = PREFIJOS_REGION[region]
        col_factor = COLUMNAS_FACTOR[region].lower()
        col_gen = f"gen_{pref}"

        if col_factor not in df_result.columns:
            # Intentamos buscarla sin lower por si acaso
            raise KeyError(f"No se encontró la columna '{col_factor}' en el archivo de generación.")

        pvgp = float(pvgp_kW_per_household[idx])  # kW por hogar
        df_result[col_gen] = df_result[col_factor] * pvgp

    return df_result


# ============================
# BALANCE HORA A HORA (DF)
# ============================
def calcular_balance_horario_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula: Autoconsumo, Inyección y Demanda de Red para cada hora.
    """
    df_result = df.copy()

    for region in REGIONES:
        pref = PREFIJOS_REGION[region]
        col_dem = COLUMNAS_DEMANDA[region].lower()
        col_gen = f"gen_{pref}"

        col_diff = f"diff_{pref}"
        col_autoc = f"autoc_{pref}"
        col_inyec = f"inyeccion_{pref}"
        col_red = f"demanda_red_{pref}"

        # Diferencia neta (Positivo = Sobra energía, Negativo = Falta)
        diff = df_result[col_gen] - df_result[col_dem]
        df_result[col_diff] = diff
        
        # Lógica vectorizada (numpy where):
        # Si diff >= 0 (Sobra): Autoconsumo = Demanda, Inyección = diff, Red = 0
        # Si diff < 0 (Falta): Autoconsumo = Generación, Inyección = 0, Red = -diff
        
        df_result[col_autoc] = np.where(diff >= 0, df_result[col_dem], df_result[col_gen])
        df_result[col_inyec] = np.where(diff >= 0, diff, 0.0)
        df_result[col_red]   = np.where(diff < 0, -diff, 0.0)

    return df_result


# ============================
# RESUMEN MENSUAL
# ============================
def resumir_balance_mensual_df(df_balance: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Suma las 720/744 horas de cada mes para obtener el balance mensual.
    """
    resultados = {}

    for region in REGIONES:
        pref = PREFIJOS_REGION[region]
        col_dem = COLUMNAS_DEMANDA[region].lower()
        col_gen = f"gen_{pref}"
        col_diff = f"diff_{pref}"
        col_autoc = f"autoc_{pref}"
        col_inyec = f"inyeccion_{pref}"
        col_red = f"demanda_red_{pref}"

        df_reg = df_balance[["mes", col_gen, col_dem, col_autoc, col_inyec, col_red, col_diff]].copy()

        # Agrupar por mes y sumar
        df_mensual = (
            df_reg
            .groupby("mes")
            .sum(numeric_only=True)
            .rename(columns={
                col_gen: "gen",
                col_dem: "demanda",
                col_autoc: "autoconsumo",
                col_inyec: "inyeccion",
                col_red: "demanda_red",
                col_diff: "dif_total",
            })
            .sort_index()
        )
        resultados[region] = df_mensual

    return resultados