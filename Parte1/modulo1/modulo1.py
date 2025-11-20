# -*- coding: utf-8 -*-
import os
import pandas as pd

# Mapeo: Nombre del escenario -> Nombre de la hoja en Excel (o sufijo del CSV)
CONFIG_ESCENARIOS = {
    "low":    "low",
    "medium": "medium",
    "high":   "high"
}

def obtener_precios(nombre_escenario: str, ruta_archivo_precios: str) -> pd.DataFrame:
    """
    Carga el DataFrame de precios de electricidad según el escenario elegido.
    
    Args:
        nombre_escenario (str): "low", "medium" o "high".
        ruta_archivo_precios (str): Ruta al archivo 'precio_electricidad_vf.xlsx'.
        
    Returns:
        pd.DataFrame: DataFrame con columnas ['periodo', '...CLP', '...USD'].
    """
    key = nombre_escenario.lower()
    
    # 1. Validar que el escenario existe
    if key not in CONFIG_ESCENARIOS:
        raise ValueError(f"Escenario '{nombre_escenario}' no válido. Usa: {list(CONFIG_ESCENARIOS.keys())}")
    
    hoja_objetivo = CONFIG_ESCENARIOS[key]
    print(f"   ℹ️  [Módulo 1] Buscando precios para escenario: '{key.upper()}' (Hoja: {hoja_objetivo})...")
    
    df_precios = None
    
    # ---------------------------------------------------------
    # ESTRATEGIA A: Intentar leer directamente el Excel (.xlsx)
    # ---------------------------------------------------------
    try:
        # Intentamos leer la hoja específica
        df_precios = pd.read_excel(ruta_archivo_precios, sheet_name=hoja_objetivo)
    except Exception:
        # Si falla (ej: el archivo no existe, es un CSV, o no tiene la hoja), pasamos a la Estrategia B
        pass

    # ---------------------------------------------------------
    # ESTRATEGIA B: Intentar leer archivo CSV separado
    # (Útil si convertiste el Excel a varios CSVs: "archivo - hoja.csv")
    # ---------------------------------------------------------
    if df_precios is None:
        try:
            dir_name = os.path.dirname(ruta_archivo_precios)
            base_name = os.path.basename(ruta_archivo_precios)
            # Construimos nombre tipo: "precio_electricidad_vf.xlsx - medium.csv"
            nombre_csv = f"{base_name} - {hoja_objetivo}.csv"
            ruta_csv = os.path.join(dir_name, nombre_csv)
            
            if os.path.exists(ruta_csv):
                # Probamos separador coma (estándar)
                df_temp = pd.read_csv(ruta_csv)
                # Si detectamos que leyó todo en una sola columna, probamos punto y coma
                if df_temp.shape[1] < 2:
                    df_temp = pd.read_csv(ruta_csv, sep=";")
                
                df_precios = df_temp
                print(f"   ✓ [Módulo 1] Precios cargados desde CSV auxiliar: {nombre_csv}")
        except Exception as e:
            print(f"   ⚠️ [Módulo 1] Falló lectura de CSV auxiliar: {e}")

    # ---------------------------------------------------------
    # VERIFICACIÓN FINAL
    # ---------------------------------------------------------
    if df_precios is None:
        raise FileNotFoundError(
            f"❌ ERROR FATAL: No se pudieron cargar los precios para '{key}'.\n"
            f"   Verifique que '{ruta_archivo_precios}' exista y tenga la hoja '{hoja_objetivo}',\n"
            f"   o que exista el CSV correspondiente en la misma carpeta."
        )

    # Limpieza básica de nombres de columnas (quitar espacios extra)
    df_precios.columns = df_precios.columns.str.strip()
    
    return df_precios