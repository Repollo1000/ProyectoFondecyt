# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict
import numpy as np
import pandas as pd

# Importamos parámetros globales si existen
try:
    from .. import parametros_globales as p_g
    REGIONES = list(p_g.REGIONES)
except (ImportError, AttributeError):
    REGIONES = ["Norte", "Centro", "Sur"]

# ============================
# CONSTANTES BÁSICAS
# ============================

COLUMNAS_FACTOR = {
    "Norte": "factor_antofagasta",
    "Centro": "factor_santiago",
    "Sur": "factor_puertomontt",
}

COLUMNAS_DEMANDA = {
    "Norte": "demanda_norte",
    "Centro": "demanda_centro",
    "Sur": "demanda_sur",
}

PREFIJOS_REGION = {
    "Norte": "norte",
    "Centro": "centro",
    "Sur": "sur",
}


# ============================
# 1. CONSTRUCCIÓN DF HORARIO
# ============================
def construir_dataframe_horario_combinado(
    df_generacion: pd.DataFrame,
    df_consumo: pd.DataFrame,
    pvgp_kW_per_household: np.ndarray,
) -> pd.DataFrame:
    """
    Une Factor Solar y Curva de Carga fila a fila (concat).
    Se asume que ambos dataframes tienen columnas compatibles en año/mes/hora.
    """
    # Copias y limpieza
    df_g = df_generacion.copy()
    df_c = df_consumo.copy()

    df_g.columns = df_g.columns.str.lower().str.strip()
    df_c.columns = df_c.columns.str.lower().str.strip()

    df_g = df_g.reset_index(drop=True)
    df_c = df_c.reset_index(drop=True)

    # Eliminar columnas repetidas en consumo
    cols_to_drop = [col for col in ["año", "mes", "hora"] if col in df_c.columns]
    df_c_limpio = df_c.drop(columns=cols_to_drop, errors="ignore")

    # Unión horizontal
    df = pd.concat([df_g, df_c_limpio], axis=1)

    # Cálculo de Generación (Factor * Potencia instalada)
    df_result = df.copy()
    for idx, region in enumerate(REGIONES):
        col_factor = COLUMNAS_FACTOR[region].lower()
        col_gen = f"gen_{PREFIJOS_REGION[region]}"

        if col_factor not in df_result.columns:
            raise KeyError(f"No se encontró la columna '{col_factor}' en df_generacion")

        pvgp = float(pvgp_kW_per_household[idx])
        df_result[col_gen] = df_result[col_factor] * pvgp

    return df_result


# ============================
# 2. BALANCE FÍSICO HORA A HORA
# ============================
def calcular_balance_horario_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula: Autoconsumo, Inyección y Demanda desde la red por región, hora a hora.
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

        # Diferencia (Gen - Demanda)
        diff = df_result[col_gen] - df_result[col_dem]
        df_result[col_diff] = diff
        
        # Lógica:
        # diff >= 0 -> Sobra energía (Inyección)
        # diff < 0  -> Falta energía (Compra Red)
        df_result[col_autoc] = np.where(diff >= 0, df_result[col_dem], df_result[col_gen])
        df_result[col_inyec] = np.where(diff >= 0, diff, 0.0)
        df_result[col_red]   = np.where(diff < 0, -diff, 0.0)

    return df_result


# ============================
# 3. RESUMEN MENSUAL FÍSICO
# ============================
def resumir_balance_mensual_df(df_balance: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    A partir de un DF horario con columna 'mes', suma las horas
    para obtener totales mensuales (kWh) por región.

    Retorna un diccionario:
        { "Norte": df_mensual_norte, "Centro": df_mensual_centro, "Sur": df_mensual_sur }
    """
    resultados: Dict[str, pd.DataFrame] = {}

    for region in REGIONES:
        pref = PREFIJOS_REGION[region]
        col_gen = f"gen_{pref}"
        col_dem = COLUMNAS_DEMANDA[region].lower()
        col_autoc = f"autoc_{pref}"
        col_inyec = f"inyeccion_{pref}"
        col_red = f"demanda_red_{pref}"
        col_diff = f"diff_{pref}"

        columnas = ["mes", col_gen, col_dem, col_autoc, col_inyec, col_red, col_diff]
        faltantes = [c for c in columnas if c not in df_balance.columns]
        if faltantes:
            raise KeyError(f"Faltan columnas {faltantes} en df_balance para región {region}")

        df_reg = df_balance[columnas].copy()

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


# ============================
# 4. CÁLCULO ECONÓMICO COMPARATIVO (UTILIDAD)
# ============================
def calcular_flujo_economico_completo(
    df_balance_mensual: pd.DataFrame,
    df_precios: pd.DataFrame,
    col_precio_usd: str = "mediumUSD"
) -> pd.DataFrame:
    """
    Calcula la utilidad (Cash Flow mensual) bajo 3 políticas:
    1. Net Billing (Inyección al 50%)
    2. Net Metering (Inyección al 100%)
    3. Feed-in Tariff (Inyección al 80% - Supuesto)

    df_balance_mensual debe tener índice = 'periodo' compatible con df_precios['periodo']
    y columnas: ['inyeccion', 'demanda_red'].
    """
    df = df_balance_mensual.copy()

    if "periodo" in df_precios.columns:
        dict_precios = df_precios.set_index("periodo")[col_precio_usd].to_dict()
    else:
        # Asume orden 1..N si no hay columna explícita
        df_precios = df_precios.reset_index(drop=True)
        df_precios["periodo"] = np.arange(1, len(df_precios) + 1)
        dict_precios = df_precios.set_index("periodo")[col_precio_usd].to_dict()

    # Índice de df se usa como "periodo"
    df.index.name = "periodo"
    df["precio_usd_kwh"] = df.index.map(dict_precios)

    # --- COSTOS (Igual para todos) ---
    df["costo_compra_usd"] = df["demanda_red"] * df["precio_usd_kwh"]

    # 1. Net Billing (50%)
    df["ingreso_nb"] = df["inyeccion"] * df["precio_usd_kwh"] * 0.5
    df["utilidad_nb"] = df["ingreso_nb"] - df["costo_compra_usd"]

    # 2. Net Metering (100%)
    df["ingreso_nm"] = df["inyeccion"] * df["precio_usd_kwh"] * 1.0
    df["utilidad_nm"] = df["ingreso_nm"] - df["costo_compra_usd"]

    # 3. Feed-in Tariff (80% - Supuesto)
    df["ingreso_fit"] = df["inyeccion"] * df["precio_usd_kwh"] * 0.8
    df["utilidad_fit"] = df["ingreso_fit"] - df["costo_compra_usd"]

    return df


def calcular_flujo_economico_por_politica(
    df_balance_mensual: pd.DataFrame,
    df_precios: pd.DataFrame,
    col_precio_usd: str,
    politica: str
) -> pd.DataFrame:
    """
    Devuelve solo las columnas económicas relevantes para la política elegida.
    """
    df = calcular_flujo_economico_completo(df_balance_mensual, df_precios, col_precio_usd)

    if politica == "net_billing":
        cols = ["precio_usd_kwh", "inyeccion", "demanda_red", "ingreso_nb", "costo_compra_usd", "utilidad_nb"]
    elif politica == "net_metering":
        cols = ["precio_usd_kwh", "inyeccion", "demanda_red", "ingreso_nm", "costo_compra_usd", "utilidad_nm"]
    else:  # feed_in_tariff
        cols = ["precio_usd_kwh", "inyeccion", "demanda_red", "ingreso_fit", "costo_compra_usd", "utilidad_fit"]

    return df[cols].copy()


__all__ = [
    "REGIONES", "COLUMNAS_FACTOR", "COLUMNAS_DEMANDA", "PREFIJOS_REGION",
    "construir_dataframe_horario_combinado", "calcular_balance_horario_df",
    "resumir_balance_mensual_df", "calcular_flujo_economico_completo",
    "calcular_flujo_economico_por_politica"
]
