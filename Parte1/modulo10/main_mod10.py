# -*- coding: utf-8 -*-
from __future__ import annotations
import os
import pandas as pd
import numpy as np
from prettytable import PrettyTable

# Importaciones locales
# Ajusta '..' según tu estructura real de carpetas
try:
    from .. import parametros_globales as p_g
    from . import modulo10 as m10
except ImportError:
    # Fallback por si se ejecuta en la misma carpeta para pruebas
    import parametros_globales as p_g
    import modulo10 as m10

# Rutas base
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATOS_DIR = os.path.join(BASE_DIR, 'Datos')

NOMBRE_ARCHIVO_CONSUMO_HORARIO = "curva_de_carga.xlsx"
NOMBRE_ARCHIVO_GENERACION_HORARIO = "Factor_capacidad_solar.csv"


def main():
    print("=" * 80)
    print("LECTURA + BALANCE HORARIO (SOLUCIÓN CORREGIDA)")
    print("=" * 80)

    # 1) Leer archivos
    ruta_consumo = os.path.join(DATOS_DIR, NOMBRE_ARCHIVO_CONSUMO_HORARIO)
    ruta_generacion = os.path.join(DATOS_DIR, NOMBRE_ARCHIVO_GENERACION_HORARIO)

    print(f"Cargando datos...")
    try:
        df_curva = pd.read_excel(ruta_consumo)
        # Lectura robusta de CSV (intenta ; o ,)
        try:
            df_generacion = pd.read_csv(ruta_generacion, sep=";", encoding="latin-1")
        except:
            df_generacion = pd.read_csv(ruta_generacion)
            
    except Exception as e:
        print(f"✗ Error leyendo archivos: {e}")
        return

    # 2) Obtener parámetros (PVGP)
    # Si falla p_g, usamos un valor dummy para probar
    try:
        pvgp_kW_per_household = p_g.MOD7_VARIABLES_INICIALES["pvgp_kW_per_household"]
    except AttributeError:
        print("⚠️ No se pudo leer pvgp de parametros_globales. Usando valores de prueba [1.0, 1.0, 1.0]")
        pvgp_kW_per_household = np.array([3.3, 3.85, 4.95])

    # 3) Procesamiento (Llamadas a modulo10)
    print("Calculando balance horario...")
    
    # A. Unir archivos (CORREGIDO con concat)
    df_comb = m10.construir_dataframe_horario_combinado(
        df_generacion=df_generacion,
        df_consumo=df_curva,
        pvgp_kW_per_household=pvgp_kW_per_household,
    )

    # B. Calcular balance (Inyección vs Red)
    df_balance = m10.calcular_balance_horario_df(df_comb)

    # =========================================================================
    # VERIFICACIÓN DE RESULTADOS (LAS 3 PRUEBAS)
    # =========================================================================
    print("\n" + "="*30)
    print(" VERIFICACIÓN DEL MODELO")
    print("="*30)
    
    region_test = "norte" # Región para probar
    cols_check = [f"gen_{region_test}", f"inyeccion_{region_test}", f"demanda_red_{region_test}"]

    # --- PRUEBA 1: VISUAL (MEDIANOCHE VS MEDIODÍA) ---
    print("\n[PRUEBA 1] Sentido Común (Día 1)")
    row_midnight = df_balance.iloc[0] # Hora 0
    row_noon = df_balance.iloc[12]    # Hora 12
    
    print(f"  Medianoche (Gen debería ser 0): Gen={row_midnight[cols_check[0]]:.4f}, Red={row_midnight[cols_check[2]]:.4f}")
    print(f"  Mediodía   (Gen debería ser alta): Gen={row_noon[cols_check[0]]:.4f}, Inyección={row_noon[cols_check[1]]:.4f}")

    # --- PRUEBA 2: MATEMÁTICA (BALANCE DE ENERGÍA) ---
    # Gen = Autoconsumo + Inyección
    # Demanda = Autoconsumo + Red
    print("\n[PRUEBA 2] Balance Matemático (Suma de errores anuales)")
    col_gen = f"gen_{region_test}"
    col_dem = f"demanda_{region_test}"
    col_autoc = f"autoc_{region_test}"
    col_iny = f"inyeccion_{region_test}"
    col_red = f"demanda_red_{region_test}"

    err_gen = (df_balance[col_autoc] + df_balance[col_iny] - df_balance[col_gen]).abs().sum()
    err_dem = (df_balance[col_autoc] + df_balance[col_red] - df_balance[col_dem]).abs().sum()
    
    print(f"  Error Ecuación Generación: {err_gen:.10f}")
    print(f"  Error Ecuación Demanda:    {err_dem:.10f}")
    if err_gen < 1e-5 and err_dem < 1e-5:
        print("  ✅ BALANCE PERFECTO")
    else:
        print("  ❌ ERROR NUMÉRICO DETECTADO")

    # --- PRUEBA 3: LÓGICA EXCLUSIVA ---
    print("\n[PRUEBA 3] Lógica Exclusiva (No Inyectar y Comprar a la vez)")
    errores = df_balance[
        (df_balance[col_iny] > 0.001) & (df_balance[col_red] > 0.001)
    ]
    if len(errores) == 0:
        print("  ✅ Lógica correcta: Nunca hay Inyección y Compra simultáneas.")
    else:
        print(f"  ❌ Error: Hay {len(errores)} horas con flujo simultáneo.")


    # =========================================================================
    # RESUMEN FINAL
    # =========================================================================
    resumen_mensual = m10.resumir_balance_mensual_df(df_balance)

    print("\n" + "="*80)
    print(" RESUMEN MENSUAL POR REGIÓN")
    print("="*80)

    for region in m10.REGIONES:
        if region not in resumen_mensual: continue
        
        print(f"\n>>> REGIÓN: {region}")
        tabla = PrettyTable()
        tabla.field_names = ["Mes", "Gen (kWh)", "Demanda", "Autoconsumo", "Inyección", "Compra Red"]
        
        df_reg = resumen_mensual[region]
        for mes, row in df_reg.iterrows():
            tabla.add_row([
                int(mes),
                f"{row['gen']:.1f}",
                f"{row['demanda']:.1f}",
                f"{row['autoconsumo']:.1f}",
                f"{row['inyeccion']:.1f}",
                f"{row['demanda_red']:.1f}"
            ])
        print(tabla)

if __name__ == "__main__":
    main()