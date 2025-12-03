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
    from ..modulo1 import modulo1
    from . import modulo10 as m10
except ImportError:
    # Fallback si se ejecuta directamente dentro de Parte1/modulo10
    import parametros_globales as p_g
    from Parte1.modulo1 import modulo1
    import modulo10 as m10

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATOS_DIR = os.path.join(BASE_DIR, 'Datos')

FILE_CONSUMO = "curva_de_carga.xlsx"
FILE_GENERACION = "Factor_capacidad_solar.csv"
FILE_PRECIOS = "precio_electricidad_vf.xlsx"


# ============================================================================
# 2. UTILIDADES
# ============================================================================
def input_seguro(prompt, opciones_validas):
    while True:
        val = input(prompt).strip().lower()
        if val in opciones_validas:
            return val
        print(f"Opción no válida. Debe ser una de: {opciones_validas}")


def imprimir_tabla_bonita(df_input, titulo):
    """Convierte un DataFrame (o parte de él) en una PrettyTable formateada a 4 decimales."""
    t = PrettyTable()
    t.field_names = df_input.columns.tolist()

    for _, row in df_input.iterrows():
        fila_formateada = []
        for val in row:
            if isinstance(val, (float, np.floating)):
                fila_formateada.append(f"{val:.4f}")
            else:
                fila_formateada.append(str(val))
        t.add_row(fila_formateada)

    print(f"\n>>> {titulo}")
    print(t)


# ============================================================================
# 3. FLUJO PRINCIPAL
# ============================================================================
def main():
    print("=" * 100)
    print("   MÓDULO 10 - REPORTE FÍSICO + ECONÓMICO POR POLÍTICA Y REGIÓN")
    print("=" * 100)

    # Vida útil desde parametros_globales
    PROJECT_LIFETIME_MONTHS = getattr(p_g, "PROJECT_LIFETIME_MONTHS", 12)

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
        df_precios = modulo1.obtener_precios(escenario, path_precios)
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

    # Mapeo nombres columnas de ingreso/utilidad
    if politica == "net_billing":
        col_ingreso = "ingreso_nb"
        col_utilidad = "utilidad_nb"
        nombre_politica_pretty = "Net Billing"
    elif politica == "net_metering":
        col_ingreso = "ingreso_nm"
        col_utilidad = "utilidad_nm"
        nombre_politica_pretty = "Net Metering"
    else:
        col_ingreso = "ingreso_fit"
        col_utilidad = "utilidad_fit"
        nombre_politica_pretty = "Feed-in Tariff"

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

    # Diccionario para acumular resultados por región y guardarlos en Excel
    resultados_excel = {}

    # ------------------------------------------------------------------------
    # 5) BALANCE ECONÓMICO Y TABLA POR REGIÓN
    # ------------------------------------------------------------------------
    print("\n" + "=" * 100)
    print(f"REPORTE FINAL: RESUMEN MENSUAL ({nombre_politica_pretty}) - Escenario: {escenario.upper()}")
    print("=" * 100)

    # Advertencia si la tabla de precios es más corta que la vida útil
    if len(df_precios) < PROJECT_LIFETIME_MONTHS:
        print(f"\n⚠ OJO: df_precios solo tiene {len(df_precios)} periodos, "
              f"pero PROJECT_LIFETIME_MONTHS = {PROJECT_LIFETIME_MONTHS}. "
              "Los meses sin precio quedarán con NaN en el Excel.")

    for region in m10.REGIONES:
        if region not in resumen_fisico:
            continue

        print(f"\n>>> REGIÓN: {region}")

        # df_fisico_reg: año típico por mes
        # columnas: gen, demanda, autoconsumo, inyeccion, demanda_red, dif_total
        df_fisico_reg = resumen_fisico[region].copy()   # index = mes
        df_fisico_reg = df_fisico_reg.reset_index()     # columna 'mes'
        n_base = len(df_fisico_reg)                     # normalmente 12

        # ------------------------ PATRÓN FÍSICO COMPLETO ------------------------
        # Repetimos el año típico hasta cubrir PROJECT_LIFETIME_MONTHS
        repeticiones = int(np.ceil(PROJECT_LIFETIME_MONTHS / n_base))
        pattern_rep = pd.concat([df_fisico_reg] * repeticiones, ignore_index=True).iloc[:PROJECT_LIFETIME_MONTHS]

        # Agregamos columna periodo (1..N) y año
        pattern_rep["periodo"] = np.arange(1, PROJECT_LIFETIME_MONTHS + 1)
        pattern_rep["anio"] = (pattern_rep["periodo"] - 1) // n_base + 1
        # 'mes' ya viene de df_fisico_reg (1..12) y se repite

        pattern_rep = pattern_rep.set_index("periodo")

        # df_balance_mensual para el cálculo económico: solo inyección + demanda_red
        df_balance_mensual = pattern_rep[["inyeccion", "demanda_red"]].copy()

        # ------------------------ ECONOMÍA COMPLETA ------------------------
        df_econ_reg = m10.calcular_flujo_economico_por_politica(
            df_balance_mensual=df_balance_mensual,
            df_precios=df_precios,
            col_precio_usd=col_usd,
            politica=politica,
        )

        # Seleccionamos columnas económicas y renombramos para Excel
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

        # DataFrame final para esta región: físico (repetido) + económico, para TODOS los meses del proyecto
        df_salida_reg = pattern_rep.join(df_econ_selected)

        # Reordenamos un poco las columnas
        cols_orden = [
            "anio", "mes",
            "gen", "demanda", "autoconsumo", "inyeccion", "demanda_red", "dif_total",
            "precio_usd_kwh", "inyeccion_kwh", "compra_red_kwh",
            "ingreso_usd", "costo_compra_usd", "utilidad_usd",
        ]
        cols_orden = [c for c in cols_orden if c in df_salida_reg.columns]
        df_salida_reg = df_salida_reg[cols_orden]
        df_salida_reg.index.name = "periodo"

        # Guardamos en diccionario para escribir después en Excel
        resultados_excel[region] = df_salida_reg

        # ------------------ TABLA PRETTY (solo PRIMER AÑO) ------------------
        print("  (Mostrando solo el AÑO 1 en pantalla; el Excel contiene toda la vida útil)")

        # Primer año = primeros n_base periodos
        df_econ_ano1 = df_econ_reg.iloc[:n_base]

        tabla = PrettyTable()
        tabla.field_names = [
            "Mes",
            "Precio (USD/kWh)",
            "Inyección (kWh)",
            "Compra Red (kWh)",
            "Ingreso ($)",
            "Costo Compra ($)",
            "Utilidad ($)",
        ]

        for i in range(1, n_base + 1):
            row_fis = df_fisico_reg.loc[i - 1] 
            row_econ = df_econ_ano1.loc[i]

            mes = int(row_fis["mes"])
            precio = row_econ["precio_usd_kwh"]
            inyec = row_econ["inyeccion"]
            compra = row_econ["demanda_red"]
            ingreso = row_econ[col_ingreso]
            costo = row_econ["costo_compra_usd"]
            utilidad = row_econ[col_utilidad]

            tabla.add_row([
                mes,
                f"{precio:.4f}",
                f"{inyec:.4f}",
                f"{compra:.4f}",
                f"{ingreso:.4f}",
                f"{costo:.4f}",
                f"{utilidad:.4f}",
            ])

        print(tabla)
        print("(Utilidad: ingreso por inyección - costo de compra a la red)")

    # ------------------------------------------------------------------------
    # 6) GUARDAR RESULTADOS EN EXCEL
    # ------------------------------------------------------------------------
    if resultados_excel:
        nombre_archivo = f"reporte_mod10_{escenario}_{politica}.xlsx"
        ruta_salida = os.path.join(DATOS_DIR, nombre_archivo)

        with pd.ExcelWriter(ruta_salida) as writer:
            for region, df_out in resultados_excel.items():
                # sheet_name máx 31 caracteres
                sheet_name = region[:31]
                df_out.to_excel(writer, sheet_name=sheet_name, index=True)

        print(f"\nArchivo Excel guardado en:\n  {ruta_salida}")
        print(f"Cada hoja tiene {PROJECT_LIFETIME_MONTHS} filas (toda la vida útil del proyecto).")

    print("\n" + "=" * 100)
    print("FIN DE REPORTE MÓDULO 10")
    print("=" * 100 + "\n")


if __name__ == "__main__":
    main()