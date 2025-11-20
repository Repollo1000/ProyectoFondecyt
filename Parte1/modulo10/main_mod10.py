# -*- coding: utf-8 -*-
from __future__ import annotations
import os
import pandas as pd
import numpy as np
from prettytable import PrettyTable

# Intentamos importar los módulos locales
try:
    from .. import parametros_globales as p_g
    from . import modulo10 as m10
    # IMPORTANTE: Importamos el módulo 5 para el LCOE
    from ..modulo5 import modulo5 as m5
except ImportError:
    # Fallback para pruebas locales
    import parametros_globales as p_g
    import modulo10 as m10
    try:
        import modulo5 as m5
    except ImportError:
        m5 = None
        print("⚠️ Advertencia: modulo5 no encontrado. Se omitirá el LCOE.")

# Rutas
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATOS_DIR = os.path.join(BASE_DIR, 'Datos')

FILE_CONSUMO = "curva_de_carga.xlsx"
FILE_GENERACION = "Factor_capacidad_solar.csv"
FILE_PRECIOS = "precio_electricidad_vf.xlsx"
FILE_COSTOS = "costos.xlsx"

def main():
    print("=" * 80)
    print("   MÓDULO 10: BALANCE INTEGRAL (FÍSICO + ECONÓMICO + LCOE)")
    print("=" * 80)

    # 1. CARGA DE DATOS
    path_consumo = os.path.join(DATOS_DIR, FILE_CONSUMO)
    path_gen = os.path.join(DATOS_DIR, FILE_GENERACION)
    path_precios = os.path.join(DATOS_DIR, FILE_PRECIOS)
    path_costos = os.path.join(DATOS_DIR, FILE_COSTOS)

    print("1. Cargando datos...")
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

    # 2. OBTENER LCOE (Llamando al Módulo 5)
    print("2. Ejecutando Módulo 5 (Financiero)...")
    lcoe_data = {}
    
    if m5 is not None:
        try:
            # Ejecutamos el modelo 5 pasándole la ruta de costos
            # IMPORTANTE: Asegúrate de que modulo5.py esté corregido (leyendo Hoja2)
            resultado_m5 = m5.correr_modelo(ruta_excel_costos=path_costos)
            lcoe_matriz = resultado_m5["lcoe_mensual"]
            
            # Guardamos LCOE por región (solo primeros 12 meses para la tabla anual)
            lcoe_data = {
                "Norte":  lcoe_matriz[0, :12],
                "Centro": lcoe_matriz[1, :12],
                "Sur":    lcoe_matriz[2, :12]
            }
            print("   ✓ LCOE calculado exitosamente.")
        except Exception as e:
            print(f"   ⚠️ Error en Módulo 5: {e}")
            print("   (La columna LCOE saldrá vacía)")
    else:
        print("   (Omisión: Módulo 5 no importado)")


    # 3. BALANCE FÍSICO (kWh)
    print("3. Calculando Balance Físico (Módulo 10)...")
    try:
        pvgp = p_g.MOD7_VARIABLES_INICIALES["pvgp_kW_per_household"]
    except:
        pvgp = np.array([1.0, 1.0, 1.0])

    df_comb = m10.construir_dataframe_horario_combinado(df_generacion, df_curva, pvgp)
    df_balance_horario = m10.calcular_balance_horario_df(df_comb)
    resumen_fisico = m10.resumir_balance_mensual_df(df_balance_horario)

    # 4. BALANCE ECONÓMICO (USD)
    print("4. Calculando Utilidades por Política...")
    resultados_eco = {}
    col_precio = "mediumUSD"

    for region, df_fisico in resumen_fisico.items():
        if col_precio not in df_precios.columns:
            print(f"✗ Error: No existe columna {col_precio} en precios.")
            return
        
        # Calcula NB, NM, FiT
        df_eco = m10.calcular_flujo_economico_completo(
            df_fisico, df_precios, col_precio_usd=col_precio
        )
        resultados_eco[region] = df_eco

    # 5. REPORTE FINAL
    print("\n" + "="*80)
    print("RESULTADOS CONSOLIDADOS (AÑO 1)")
    print("="*80)
    
    for region in m10.REGIONES:
        if region not in resultados_eco: continue
        
        print(f"\n>>> REGIÓN: {region}")
        df = resultados_eco[region]
        lcoe_region = lcoe_data.get(region, np.zeros(12))
        
        t = PrettyTable()
        t.field_names = [
            "Mes", 
            "Precio Red ($)", 
            "LCOE Solar ($)", 
            "¿Conviene?",
            "Util. NB ($)", 
            "Util. FiT ($)",
            "Util. NM ($)"
        ]
        
        for mes, row in df.iterrows():
            idx = int(mes) - 1
            p_red = row['precio_usd_kwh']
            
            # LCOE (Manejo seguro de índices)
            lcoe_val = lcoe_region[idx] if idx < len(lcoe_region) else 0.0
            
            # Comparación
            if lcoe_val > 0:
                status = "✅ SÍ" if lcoe_val < p_red else "⚠️ NO"
            else:
                status = "-"

            t.add_row([
                int(mes),
                f"{p_red:.3f}",
                f"{lcoe_val:.3f}",
                status,
                f"{row['utilidad_nb']:.1f}",
                f"{row['utilidad_fit']:.1f}",
                f"{row['utilidad_nm']:.1f}"
            ])
        
        print(t)

if __name__ == "__main__":
    main()