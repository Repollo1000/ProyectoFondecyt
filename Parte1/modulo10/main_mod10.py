# -*- coding: utf-8 -*-

from __future__ import annotations
import os
import sys
import numpy as np
import pandas as pd
from prettytable import PrettyTable

# ============================================================================
# 1. IMPORTS y RUTAS
# ============================================================================
try:
    from .. import parametros_globales as p_g
    # --- CAMBIO IMPORTANTE: Usamos el nuevo gestor modulo1_3 ---
    from ..modulo1_3 import modulo1_3 as m_datos 
    from . import modulo10 as m10
except ImportError:
    # Fallback si se ejecuta directamente o rutas alternativas
    import parametros_globales as p_g
    from Parte1.modulo1_3 import modulo1_3 as m_datos
    import modulo10 as m10

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATOS_DIR = os.path.join(BASE_DIR, 'Datos')

FILE_CONSUMO = "curva_de_carga.xlsx"
FILE_GENERACION = "Factor_capacidad_solar.csv"
FILE_PRECIOS = "precio_electricidad_vf.xlsx"
FILE_ANNUAL_SAVINGS = "annual_savings.xlsx"


# ============================================================================
# 2. UTILIDADES
# ============================================================================
def input_seguro(prompt, opciones_validas):
    while True:
        val = input(prompt).strip().lower()
        if val in opciones_validas:
            return val
        print(f"Opción no válida. Debe ser una de: {opciones_validas}")


# ============================================================================
# 3. FLUJO PRINCIPAL
# ============================================================================
def main():
    print("=" * 100)
    print("   MÓDULO 10 - REPORTE FÍSICO + ECONÓMICO + GENERACIÓN DE ANNUAL_SAVINGS")
    print("=" * 100)

    # Vida útil desde parametros_globales
    PROJECT_LIFETIME_MONTHS = getattr(p_g, "PROJECT_LIFETIME_MONTHS", 324)
    print(f"--> Configurado para simular: {PROJECT_LIFETIME_MONTHS} meses")

    # ------------------------------------------------------------------------
    # 1) CARGA DE DATOS
    # ------------------------------------------------------------------------
    path_consumo = os.path.join(DATOS_DIR, FILE_CONSUMO)
    path_gen = os.path.join(DATOS_DIR, FILE_GENERACION)
    path_precios = os.path.join(DATOS_DIR, FILE_PRECIOS)

    try:
        df_curva = pd.read_excel(path_consumo)
        try:
            df_generacion = pd.read_csv(path_gen, sep=";", encoding="latin-1")
        except Exception:
            df_generacion = pd.read_csv(path_gen)
    except Exception as e:
        print(f"✗ Error leyendo curva o generación: {e}")
        return

    # --- Menú de escenario de precio ---
    print("\n[1] Selección de escenario de precio de electricidad")
    print("    a) Low")
    print("    b) Medium")
    print("    c) High")

    op_precio = input_seguro(">> Opción (a/b/c): ", ["a", "b", "c", "low", "medium", "high"])
    mapa_esc = {
        "a": "low", "low": "low",
        "b": "medium", "medium": "medium",
        "c": "high", "high": "high",
    }
    escenario = mapa_esc[op_precio]

    try:
        # --- CAMBIO: Usamos m_datos (el nuevo gestor) en vez de modulo1 ---
        print(f"   -> Cargando precios desde: {path_precios} (Escenario: {escenario})")
        df_precios = m_datos.obtener_precios(escenario, path_precios)
        
        # Buscar una columna en USD
        col_usd = next((c for c in df_precios.columns if "USD" in c.upper()), None)
        if not col_usd:
            col_usd = df_precios.columns[-1]
            print(f"   ⚠ No se encontró columna 'USD' explícita. Usando: {col_usd}")
        else:
            print(f"   → Usando columna de precio: {col_usd}")
    except Exception as e:
        print(f"✗ Error leyendo precios: {e}")
        return

    # ------------------------------------------------------------------------
    # 2) MENÚ DE POLÍTICA
    # ------------------------------------------------------------------------
    print("\n[2] Selección de política de red")
    print("    a) Net Billing")
    print("    b) Net Metering")
    print("    c) Feed-in Tariff")

    op_pol = input_seguro(">> Opción (a/b/c): ", ["a", "b", "c"])
    mapa_pol = {
        "a": "net_billing",
        "b": "net_metering",
        "c": "feed_in_tariff",
    }
    politica = mapa_pol[op_pol]

    # Etiquetas visuales
    if politica == "net_billing":
        col_ingreso, col_utilidad, nombre_politica_pretty = "ingreso_nb", "utilidad_nb", "Net Billing"
    elif politica == "net_metering":
        col_ingreso, col_utilidad, nombre_politica_pretty = "ingreso_nm", "utilidad_nm", "Net Metering"
    else:
        col_ingreso, col_utilidad, nombre_politica_pretty = "ingreso_fit", "utilidad_fit", "Feed-in Tariff"

    # ------------------------------------------------------------------------
    # 3) POTENCIAS INSTALADAS (PVGP)
    # ------------------------------------------------------------------------
    try:
        pvgp = p_g.MOD7_VARIABLES_INICIALES["pvgp_kW_per_household"]
    except Exception:
        pvgp = np.array([3.3, 3.85, 4.95])

    print("\n[3] Potencias instaladas PVGP (kW/hogar):")
    print(f"    Norte={pvgp[0]} kW, Centro={pvgp[1]} kW, Sur={pvgp[2]} kW")

    # ------------------------------------------------------------------------
    # 4) BALANCE FÍSICO (MÓDULO 10)
    # ------------------------------------------------------------------------
    print("\n[4] Cálculo físico hora a hora y resumen mensual...")

    # 4.1 Generación y Balance Hora a Hora
    df_comb = m10.construir_dataframe_horario_combinado(df_generacion, df_curva, pvgp)
    df_balance_horario = m10.calcular_balance_horario_df(df_comb)

    # 4.2 Resumen mensual por región (año típico)
    resumen_fisico = m10.resumir_balance_mensual_df(df_balance_horario)

    resultados_excel = {}
    ahorros_anuales_dict = {}

    # ------------------------------------------------------------------------
    # 5) BALANCE ECONÓMICO Y TABLA POR REGIÓN
    # ------------------------------------------------------------------------
    print("\n" + "=" * 100)
    print(f"REPORTE FINAL: RESUMEN MENSUAL ({nombre_politica_pretty}) - Escenario: {escenario.upper()}")
    print("=" * 100)

    # Extender precios si es necesario
    if len(df_precios) < PROJECT_LIFETIME_MONTHS:
        print(f"\n⚠ Aviso: df_precios ({len(df_precios)}) es más corto que la vida útil ({PROJECT_LIFETIME_MONTHS}). Se rellenará con el último valor.")
        ultimo_precio = df_precios.iloc[-1]
        filas_faltantes = PROJECT_LIFETIME_MONTHS - len(df_precios)
        
        # Relleno simple
        df_relleno = pd.DataFrame([ultimo_precio] * filas_faltantes)
        df_relleno.index = range(len(df_precios) + 1, PROJECT_LIFETIME_MONTHS + 1)
        
        # Concatenar
        df_precios_ext = pd.concat([df_precios, df_relleno], ignore_index=True)
        if "periodo" in df_precios_ext.columns:
             df_precios_ext["periodo"] = np.arange(1, len(df_precios_ext) + 1)
        df_precios = df_precios_ext

    for region in m10.REGIONES:
        if region not in resumen_fisico:
            continue

        print(f"\n>>> REGIÓN: {region}")

        # df_fisico_reg: año típico por mes
        df_fisico_reg = resumen_fisico[region].copy()
        df_fisico_reg = df_fisico_reg.reset_index()     # columna 'mes'
        n_base = len(df_fisico_reg)                     # 12

        # ------------------------ PATRÓN FÍSICO COMPLETO ------------------------
        repeticiones = int(np.ceil(PROJECT_LIFETIME_MONTHS / n_base))
        pattern_rep = pd.concat([df_fisico_reg] * repeticiones, ignore_index=True).iloc[:PROJECT_LIFETIME_MONTHS]

        pattern_rep["periodo"] = np.arange(1, PROJECT_LIFETIME_MONTHS + 1)
        pattern_rep["anio"] = (pattern_rep["periodo"] - 1) // n_base + 1
        pattern_rep = pattern_rep.set_index("periodo")

        # df_balance_mensual para el cálculo económico
        df_balance_mensual = pattern_rep[["inyeccion", "demanda_red"]].copy()

        # ------------------------ ECONOMÍA COMPLETA ------------------------
        df_econ_reg = m10.calcular_flujo_economico_por_politica(
            df_balance_mensual=df_balance_mensual,
            df_precios=df_precios,
            col_precio_usd=col_usd,
            politica=politica,
        )

        # Selección de columnas
        df_econ_selected = df_econ_reg[
            ["precio_usd_kwh", "inyeccion", "demanda_red", col_ingreso, "costo_compra_usd", col_utilidad]
        ].rename(columns={
            "precio_usd_kwh": "precio_usd_kwh",
            "inyeccion": "inyeccion_kwh",
            "demanda_red": "compra_red_kwh",
            col_ingreso: "ingreso_usd",
            "costo_compra_usd": "costo_compra_usd",
            col_utilidad: "utilidad_usd",
        })

        # Unir físico y económico
        df_salida_reg = pattern_rep.join(df_econ_selected)
        
        # --------------------------------------------------------------------
        # NUEVO: CÁLCULO DE AHORRO REAL PARA ANNUAL_SAVINGS
        # Ahorro = (Autoconsumo * Precio) + Ingreso_Venta
        # --------------------------------------------------------------------
        df_salida_reg["ahorro_total_usd"] = (
            df_salida_reg["autoconsumo"] * df_salida_reg["precio_usd_kwh"]
        ) + df_salida_reg["ingreso_usd"]
        
        # Agrupar por año
        ahorro_anual_serie = df_salida_reg.groupby("anio")["ahorro_total_usd"].sum()
        ahorros_anuales_dict[region] = ahorro_anual_serie

        # Reordenamos columnas
        cols_orden = [
            "anio", "mes",
            "gen", "demanda", "autoconsumo", "inyeccion", "demanda_red", "dif_total",
            "precio_usd_kwh", "inyeccion_kwh", "compra_red_kwh",
            "ingreso_usd", "costo_compra_usd", "utilidad_usd", "ahorro_total_usd"
        ]
        cols_orden = [c for c in cols_orden if c in df_salida_reg.columns]
        df_salida_reg = df_salida_reg[cols_orden]
        
        resultados_excel[region] = df_salida_reg

        # ------------------ TABLA PRETTY (AÑO 1) ------------------
        df_view = df_salida_reg.iloc[:12]
        t = PrettyTable()
        t.field_names = ["Mes", "Precio", "Autocon($)", "Ingreso($)", "AhorroTotal($)"]
        for _, r in df_view.iterrows():
            val_autoc = r["autoconsumo"] * r["precio_usd_kwh"]
            t.add_row([
                int(r["mes"]), 
                f"{r['precio_usd_kwh']:.3f}", 
                f"{val_autoc:.2f}", 
                f"{r['ingreso_usd']:.2f}", 
                f"{r['ahorro_total_usd']:.2f}"
            ])
        print(t)

    # ------------------------------------------------------------------------
    # 6) GUARDAR RESULTADOS DETALLADOS (REPORTE M10)
    # ------------------------------------------------------------------------
    if resultados_excel:
        nombre_archivo = f"reporte_mod10_{escenario}_{politica}.xlsx"
        ruta_salida = os.path.join(DATOS_DIR, nombre_archivo)
        with pd.ExcelWriter(ruta_salida) as writer:
            for region, df_out in resultados_excel.items():
                df_out.to_excel(writer, sheet_name=region[:31], index=True)
        print(f"\n[OK] Reporte detallado guardado en: {nombre_archivo}")

    # ------------------------------------------------------------------------
    # 7) GENERAR ANNUAL_SAVINGS.XLSX AUTOMÁTICO
    # ------------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("   GENERANDO ARCHIVO MAESTRO: ANNUAL_SAVINGS.XLSX")
    print("=" * 100)

    if ahorros_anuales_dict:
        # Crear DataFrame consolidado
        df_savings = pd.DataFrame(ahorros_anuales_dict)
        
        # Mapeo a Inglés
        mapa_regiones = { "Norte": "North", "Centro": "Center", "Sur": "South" }
        df_savings.rename(columns=mapa_regiones, inplace=True)
        
        # Ajustar índice Year (base 0)
        df_savings.index = df_savings.index - 1
        df_savings.index.name = "Year"
        df_savings.reset_index(inplace=True)
        
        # Ordenar columnas
        cols_final = ["Year", "North", "Center", "South"]
        cols_existentes = [c for c in cols_final if c in df_savings.columns]
        df_savings = df_savings[cols_existentes]

        ruta_savings = os.path.join(DATOS_DIR, FILE_ANNUAL_SAVINGS)
        
        try:
            df_savings.to_excel(ruta_savings, index=False)
            print(f"✓ Archivo generado exitosamente en:\n  {ruta_savings}")
            print(f"✓ Contiene datos para {len(df_savings)} años (Year 0 a {len(df_savings)-1}).")
            print("✓ Ahora puedes correr el Módulo 7 y usará estos nuevos datos.")
        except Exception as e:
            print(f"✗ Error guardando annual_savings.xlsx: {e}")
            print("  (Verifica que el archivo no esté abierto en Excel)")
    else:
        print("✗ No se calcularon ahorros. Revisa los pasos anteriores.")

    print("\n" + "=" * 100)
    print("FIN DE EJECUCIÓN")
    print("=" * 100 + "\n")


if __name__ == "__main__":
    main()