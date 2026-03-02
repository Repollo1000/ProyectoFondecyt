# -*- coding: utf-8 -*-
"""
modulo1_3.py — Gestor Unificado de Datos (Precios y Emisiones).
Reemplaza a los antiguos modulo1 y modulo3.
"""
import os
import pandas as pd
import numpy as np

try:
    from prettytable import PrettyTable
except ImportError:
    PrettyTable = None

# =============================================================================
# CONFIGURACIÓN INTERNA
# =============================================================================
CONFIG_PRECIOS = {
    "low":    "low",
    "medium": "medium",
    "high":   "high"
}

CONFIG_EMISIONES = {
    "cn": "CN", "alto":  "CN", 
    "sr": "SR", "medio": "SR", 
    "at": "AT", "bajo":  "AT"
}

# =============================================================================
# 1. FUNCIÓN: OBTENER PRECIOS (Retorna DataFrame)
# =============================================================================
def obtener_precios(nombre_escenario: str, ruta_archivo: str) -> pd.DataFrame:
    """
    Carga precios de electricidad (Excel con fallback a CSV).
    Retorna DataFrame con columnas de precios limpias.
    """
    key = nombre_escenario.lower().strip()
    
    if key not in CONFIG_PRECIOS:
        raise ValueError(f"Escenario '{key}' no válido. Opciones: {list(CONFIG_PRECIOS.keys())}")
    
    hoja_objetivo = CONFIG_PRECIOS[key]
    df_precios = None
    
    # Intento 1: Leer Excel
    try:
        df_precios = pd.read_excel(ruta_archivo, sheet_name=hoja_objetivo)
    except Exception:
        pass # Falló, probar CSV

    # Intento 2: Leer CSV auxiliar
    if df_precios is None:
        try:
            dir_name = os.path.dirname(ruta_archivo)
            base_name = os.path.basename(ruta_archivo)
            nombre_csv = f"{base_name} - {hoja_objetivo}.csv"
            ruta_csv = os.path.join(dir_name, nombre_csv)
            
            if os.path.exists(ruta_csv):
                df_temp = pd.read_csv(ruta_csv)
                if df_temp.shape[1] < 2:
                    df_temp = pd.read_csv(ruta_csv, sep=";")
                df_precios = df_temp
        except Exception:
            pass

    if df_precios is None:
        raise FileNotFoundError(f"❌ No se pudieron cargar precios para '{key}' en {ruta_archivo}")

    df_precios.columns = df_precios.columns.str.strip()
    return df_precios


# =============================================================================
# 2. FUNCIÓN: OBTENER EMISIONES (Retorna DataFrame)
# =============================================================================
def obtener_emisiones(nombre_escenario: str, ruta_archivo: str) -> pd.DataFrame:
    """
    Carga datos de emisiones y devuelve un DataFrame limpio.
    Retorna: DataFrame con columnas ['tiempo', 'factor_emision']
    """
    key = nombre_escenario.lower().strip()
    col_objetivo = CONFIG_EMISIONES.get(key, "CN") # Default a CN
    
    try:
        df_raw = pd.read_csv(ruta_archivo, sep=";")
    except Exception:
        df_raw = pd.read_csv(ruta_archivo, sep=",")

    df = df_raw.copy()
    
    cols = df.columns
    if len(cols) >= 4:
        mapeo = {cols[0]: "tiempo", cols[1]: "CN", cols[2]: "SR", cols[3]: "AT"}
        df = df.rename(columns=mapeo)
    
    if col_objetivo not in df.columns:
         raise KeyError(f"Columna '{col_objetivo}' no encontrada. Disponibles: {list(df.columns)}")

    df["tiempo"] = pd.to_numeric(df["tiempo"], errors="coerce")
    df[col_objetivo] = pd.to_numeric(df[col_objetivo], errors="coerce")
    
    df_clean = df.dropna(subset=["tiempo", col_objetivo]).copy()
    
    df_final = df_clean[["tiempo", col_objetivo]].rename(columns={col_objetivo: "factor_emision"})
    
    return df_final


# =============================================================================
# BLOQUE DE VERIFICACIÓN (PRUEBA COMPLETA CON PRETTYTABLE)
# =============================================================================
if __name__ == "__main__":
    def imprimir_bonito(df, titulo, filas=10):
        print(f"\n>>> {titulo} (Primeras {filas} filas)")
        if df is None or df.empty:
            print("    [Vacío o Error]")
            return

        df_head = df.head(filas)
        
        if PrettyTable:
            t = PrettyTable()
            t.field_names = list(df_head.columns)
            # Alinear a la derecha los números
            for col in t.field_names:
                t.align[col] = "r"
            
            for _, row in df_head.iterrows():
                vals = []
                for v in row:
                    # Formatear si es float para que no salga con mil decimales
                    if isinstance(v, float):
                        vals.append(f"{v:.4f}")
                    else:
                        vals.append(str(v))
                t.add_row(vals)
            print(t)
        else:
            # Fallback si no hay PrettyTable
            print(df_head)

    print("="*80)
    print("   PRUEBA: MODULO 1_3 (GESTOR DE DATOS) - VISUALIZACIÓN MEJORADA")
    print("="*80)
    
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    DATOS_DIR = os.path.join(BASE_DIR, 'Datos')
    
    FILE_PRECIOS = os.path.join(DATOS_DIR, "precio_electricidad_vf.xlsx")
    FILE_EMISION = os.path.join(DATOS_DIR, "factor_emisionv2.csv")

    print(f"📂 Directorio: {DATOS_DIR}\n")

    # --- 1. PRECIOS ---
    print("-" * 60)
    print(" 1. DATASET PRECIOS")
    print("-" * 60)
    for esc in ["low", "medium", "high"]:
        try:
            df = obtener_precios(esc, FILE_PRECIOS)
            imprimir_bonito(df, f"ESCENARIO: {esc.upper()}", filas=10)
        except Exception as e:
            print(f"    ❌ Error cargando '{esc}': {e}")

    # --- 2. EMISIONES ---
    print("\n" + "-" * 60)
    print(" 2. DATASET EMISIONES")
    print("-" * 60)
    
    lista_emisiones = [("BAJO (AT)", "at"), ("MEDIO (SR)", "sr"), ("ALTO (CN)", "cn")]
    for nombre, key in lista_emisiones:
        try:
            df_em = obtener_emisiones(key, FILE_EMISION)
            imprimir_bonito(df_em, f"ESCENARIO: {nombre}", filas=10)
        except Exception as e:
            print(f"    ❌ Error cargando '{key}': {e}")
        
    print("\n" + "="*80)