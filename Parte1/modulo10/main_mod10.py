# -*- coding: utf-8 -*-
from __future__ import annotations
import os
import pandas as pd
import numpy as np
from prettytable import PrettyTable

try:
    from .. import parametros_globales as p_g
    from . import modulo10 as m10
except ImportError:
    import parametros_globales as p_g
    import modulo10 as m10

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATOS_DIR = os.path.join(BASE_DIR, 'Datos')

FILE_CONSUMO = "curva_de_carga.xlsx"
FILE_GENERACION = "Factor_capacidad_solar.csv"
FILE_PRECIOS = "precio_electricidad_vf.xlsx"

# --- FUNCIÓN AUXILIAR PARA IMPRIMIR TABLAS CON 4 DECIMALES ---
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

def main():
    print("=" * 100)
    print("   REPORTE DE AUDITORÍA Y BALANCE INTEGRAL: FÍSICO + ECONÓMICO")
    print("=" * 100)

    # =========================================================================
    # 1. CARGA DE DATOS
    # =========================================================================
    path_consumo = os.path.join(DATOS_DIR, FILE_CONSUMO)
    path_gen = os.path.join(DATOS_DIR, FILE_GENERACION)
    path_precios = os.path.join(DATOS_DIR, FILE_PRECIOS)

    try:
        df_curva = pd.read_excel(path_consumo)
        try:
            df_generacion = pd.read_csv(path_gen, sep=";", encoding="latin-1")
        except:
            df_generacion = pd.read_csv(path_gen)
        try:
            df_precios = pd.read_excel(path_precios, sheet_name="medium")
        except:
            df_precios = pd.read_excel(path_precios)
    except Exception as e:
        print(f"✗ Error leyendo archivos básicos: {e}")
        return

    # --- AUDITORÍA 1: DATOS CRUDOS ---
    print("\n" + "-"*40)
    print("AUDITORÍA 1: DATOS DE ENTRADA (TOP 25)")
    print("-"*40)
    
    cols_ver_cons = [c for c in df_curva.columns if 'demanda' in c.lower() or 'mes' in c.lower() or 'hora' in c.lower()]
    imprimir_tabla_bonita(
        df_curva[cols_ver_cons].head(25), 
        "A) CURVA DE CARGA (CONSUMO)"
    )

    cols_ver_gen = [c for c in df_generacion.columns if 'factor' in c.lower() or 'mes' in c.lower() or 'hora' in c.lower()]
    imprimir_tabla_bonita(
        df_generacion[cols_ver_gen].head(25), 
        "B) FACTOR DE PLANTA SOLAR"
    )

    # =========================================================================
    # 2. BALANCE FÍSICO (MÓDULO 10)
    # =========================================================================
    print("\n" + "-"*80)
    print("2. CÁLCULO FÍSICO DETALLADO (Módulo 10)")
    
    try:
        pvgp = p_g.MOD7_VARIABLES_INICIALES["pvgp_kW_per_household"]
    except:
        pvgp = np.array([3.3, 3.85, 4.95])

    print(f"   > Potencias Instaladas (PVGP) usadas: Norte={pvgp[0]} kW, Centro={pvgp[1]} kW, Sur={pvgp[2]} kW")

    # 2.1 Generación y Balance Hora a Hora
    df_comb = m10.construir_dataframe_horario_combinado(df_generacion, df_curva, pvgp)
    
    cols_audit_gen = ['mes', 'hora', 'factor_antofagasta', 'gen_norte']
    imprimir_tabla_bonita(
        df_comb[cols_audit_gen].head(25),
        f"AUDITORÍA 2: CÁLCULO GENERACIÓN (Factor * PVGP={pvgp[0]})"
    )

    df_balance_horario = m10.calcular_balance_horario_df(df_comb)
    
    cols_audit_bal = ['mes', 'hora', 'gen_norte', 'demanda_norte', 'diff_norte', 'inyeccion_norte', 'demanda_red_norte']
    imprimir_tabla_bonita(
        df_balance_horario[cols_audit_bal].head(25),
        "AUDITORÍA 3: BALANCE HORARIO (Gen - Demanda = Diferencia)"
    )

    # =========================================================================
    # 3. BALANCE ECONÓMICO Y REPORTE FINAL
    # =========================================================================
    # Resumen mensual final
    resumen_fisico = m10.resumir_balance_mensual_df(df_balance_horario)
    
    resultados_eco = {}
    for region in m10.REGIONES:
        if region in resumen_fisico:
             resultados_eco[region] = m10.calcular_flujo_economico_completo(
                resumen_fisico[region], df_precios, col_precio_usd="mediumUSD"
            )

    print("\n" + "="*100)
    print("REPORTE FINAL: RESUMEN ANUAL (NET BILLING)")
    print("="*100)

    for region in m10.REGIONES:
        if region not in resultados_eco: continue
        
        print(f"\n>>> REGIÓN: {region}")
        df = resultados_eco[region]
        
        t = PrettyTable()
        # Se eliminó la columna LCOE
        t.field_names = ["Mes", "Precio Red ($)", "Total Inyec (kWh)", "Total Compra (kWh)", "Utilidad NB ($)"]
        
        for mes, row in df.iterrows():
            p_red = row['precio_usd_kwh']
            inyec = row['inyeccion']
            compra = row['demanda_red']
            unb = row['utilidad_nb']
            
            # Todo formateado a .4f
            t.add_row([
                int(mes), 
                f"{p_red:.4f}", 
                f"{inyec:.4f}", 
                f"{compra:.4f}", 
                f"{unb:.4f}"
            ])
        
        print(t)
        print("(Utilidad NB: Saldo a favor o en contra en la boleta bajo Net Billing)")

if __name__ == "__main__":
    main()