# -*- coding: utf-8 -*-
from __future__ import annotations
import numpy as np
import pandas as pd  # <--- IMPORTANTE: Asegúrate de tener pandas (pip install pandas openpyxl)
from prettytable import PrettyTable
import sys
import os

# --- INICIO DE CORRECCIÓN DE IMPORTACIÓN ---
from ..modulo5 import modulo5 as m5
from . import modulo10 as m10
# --- FIN DE CORRECCIÓN ---

# ==========================================
# CONFIGURACIÓN DE PRECIOS Y MONEDA
# ==========================================
# ¡¡IMPORTANTE!! Define la tasa de cambio correcta
TASA_CAMBIO_CLP_A_USD = 900.0 

# ¡¡IMPORTANTE!! Nombres de tus archivos de datos (deben estar en la carpeta 'Datos')
NOMBRE_ARCHIVO_PRECIOS = "precio_electricidad_vf.xlsx"       
NOMBRE_ARCHIVO_CONSUMO_HORARIO = "curva_de_carga.xlsx"        
NOMBRE_ARCHIVO_GENERACION_HORARIO = "Factor_capacidad_solar.csv" 

# Nombres de columnas para el Excel de precios (según tu último error)
COLUMNA_PRECIO_COMPRA = "low1"        
COLUMNA_PRECIO_INYECCION = "low2"     
# ==========================================


def main():
    print("=" * 80)
    print("MODELO INTEGRADO: LCOE (Módulo 5) + BALANCE ENERGÉTICO (Módulo 10)")
    print("=" * 80)
    
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    DATOS_DIR = os.path.join(BASE_DIR, 'Datos')

    # ==========================================
    # PASO 1: EJECUTAR MÓDULO 5 (LCOE) - Se hace UNA VEZ
    # ==========================================
    print("\n[1/2] Ejecutando Módulo 5 - LCOE...")
    
    variables_m5 = m5.default_variables()
    variables_m5["use_dynamic_subsidy"] = False
    
    ruta_costo_excel = os.path.join(DATOS_DIR, "costoAño.xlsx")
    resultados_m5 = m5.correr_modelo(ruta_costo_excel, variables_m5)
    
    regiones = resultados_m5["regiones"]
    lcoe = resultados_m5["lcoe_mensual"] # LCOE ya está en USD
    
    N_meses = lcoe.shape[1]
    N_años = N_meses // 12
    
    print(f"✓ Módulo 5 completado: {N_meses} meses ({N_años} años)")
    
    # ==========================================
    # PASO 2: CARGAR DATOS DE BALANCE - Se hace UNA VEZ
    # ==========================================
    print("\n[2/2] Cargando datos para Módulo 10 - Balance Energético...")
    
    # --- Carga de Archivos de Datos (Precios y Perfiles) ---
    try:
        # 1. Cargar Precios
        print(f"Cargando precios desde {NOMBRE_ARCHIVO_PRECIOS}...")
        ruta_precios = os.path.join(DATOS_DIR, NOMBRE_ARCHIVO_PRECIOS)
        
        # Leemos el Excel de precios
        df_precios = pd.read_excel(ruta_precios)
        
        # Normalizar TODOS los nombres de columnas (minúsculas, sin espacios)
        print(f"   Columnas originales leídas: {list(df_precios.columns)}")
        df_precios.columns = df_precios.columns.str.lower().str.strip()
        print(f"   Columnas normalizadas: {list(df_precios.columns)}")
        
        # Nuestros nombres de columna objetivo (ahora 'low1' y 'low2')
        target_compra = COLUMNA_PRECIO_COMPRA.lower().strip()
        target_inyeccion = COLUMNA_PRECIO_INYECCION.lower().strip()
        
        # Extraer usando los nombres normalizados
        precios_compra_clp = df_precios[target_compra].values
        precios_inyeccion_clp = df_precios[target_inyeccion].values
        
        precios_compra_usd = precios_compra_clp / TASA_CAMBIO_CLP_A_USD
        precios_inyeccion_usd = precios_inyeccion_clp / TASA_CAMBIO_CLP_A_USD
        print(f"✓ Precios cargados y convertidos a USD (Tasa: {TASA_CAMBIO_CLP_A_USD})")

        # 2. Cargar Perfil de Consumo Horario
        print(f"Cargando perfiles de consumo desde {NOMBRE_ARCHIVO_CONSUMO_HORARIO}...")
        ruta_consumo = os.path.join(DATOS_DIR, NOMBRE_ARCHIVO_CONSUMO_HORARIO)
        df_consumo_horario = pd.read_excel(ruta_consumo) 
        print(f"✓ Perfil de consumo cargado.")

        # 3. Cargar Perfil de Generación Horario
        print(f"Cargando perfiles de generación desde {NOMBRE_ARCHIVO_GENERACION_HORARIO}...")
        ruta_generacion = os.path.join(DATOS_DIR, NOMBRE_ARCHIVO_GENERACION_HORARIO)
        
        # --- ¡¡AQUÍ ESTÁ LA CORRECCIÓN DE ENCODING!! ---
        # Leemos el CSV usando 'latin-1' (o 'iso-8859-1') por el caracter 'ñ' en 'año'
        df_generacion_horario = pd.read_csv(ruta_generacion, sep=';', encoding='latin-1') 
        print(f"✓ Perfil de generación cargado.")

    except Exception as e:
        print(f"✗ ERROR: No se pudo cargar un archivo de datos.")
        print(f"   Detalle: {e}") 
        print("   Asegúrate de que los nombres de archivo y columnas en main_mod10.py sean correctos.")
        print("   Abortando ejecución.")
        return # Salir del script

    # --- Configuración de Variables de Balance ---
    variables_balance = m10.default_variables_balance()
    
    # Sobrescribir precios con los valores en USD
    variables_balance["precio_electricidad"] = precios_compra_usd
    variables_balance["tarifa_netbilling"] = precios_inyeccion_usd # Esta es la tarifa base de Net Billing

    # *** ACTIVAR BALANCE HORARIO ***
    variables_balance["usar_perfil_horario"] = True 
    
    # ==========================================
    # PASO 3: SELECCIÓN DE POLÍTICA (INTERACTIVO)
    # ==========================================
    
    politica_seleccionada = None
    while politica_seleccionada is None:
        print("\n" + "=" * 80)
        print("PASO 3: SELECCIONE LA POLÍTICA A CALCULAR")
        print("=" * 80)
        print("1. Net Billing (Tarifa de inyección según Excel)")
        print("2. Net Metering (Tarifa = 100% precio de compra)")
        print("3. Feed-in Tariff (Tarifa = 80% precio de compra)")
        print("\n" + "=" * 80)
        
        choice = input("Ingrese el número (1, 2, o 3) de la política a ejecutar: ")

        if choice == "1":
            politica_seleccionada = "Net Billing"
        elif choice == "2":
            politica_seleccionada = "Net Metering"
        elif choice == "3":
            politica_seleccionada = "Feed-in Tariff"
        else:
            print(f"\n✗ Opción '{choice}' no válida. Por favor, intente de nuevo.")

    
    # ==========================================
    # PASO 4: EJECUTAR SIMULACIÓN (para la política elegida)
    # ==========================================
    
    print("\n" + "=" * 80)
    print(f"EJECUTANDO SIMULACIÓN PARA POLÍTICA: {politica_seleccionada.upper()}")
    print("=" * 80)

    # --- Ejecución del Modelo Completo ---
    resultados_completos = m10.correr_modelo_completo(
        resultados_m5,
        variables_balance,
        df_consumo_horario,
        df_generacion_horario,
        politica=politica_seleccionada  # <--- Pasa la política elegida
    )
    
    # Extraer resultados
    energia = resultados_completos["energia_mensual"]
    lcoe = resultados_completos["lcoe_mensual"]
    
    balance = resultados_completos["balance_energetico"]
    autoconsumo = balance["autoconsumo_mensual"]
    inyeccion = balance["inyeccion_mensual"]
    ahorro = balance["ahorro_mensual"] # El módulo 10 calcula el 'ahorro'
    
    print(f"✓ Módulo 10 completado para {politica_seleccionada}")
    
    # ==========================================
    # PASO 5: MOSTRAR RESULTADOS
    # ==========================================
    
    # 1) LCOE (primeros 12 meses)
    print("\n" + "=" * 80)
    print(f"LCOE MENSUAL [USD/kWh] - Primeros 12 meses (Tasa USD: {TASA_CAMBIO_CLP_A_USD})")
    print("=" * 80)
    
    t_lcoe = PrettyTable()
    t_lcoe.field_names = ["Mes"] + list(regiones)
    
    for m in range(min(12, lcoe.shape[1])):
        fila = [m] + [f"{lcoe[i, m]:.4f}" for i in range(len(regiones))]
        t_lcoe.add_row(fila)
    
    print(t_lcoe)
    
    # 2) Balance Energético (primeros 12 meses)
    print("\n" + "=" * 80)
    print(f"BALANCE ENERGÉTICO ({politica_seleccionada}) - Primeros 12 meses")
    print("=" * 80)
    
    t_balance = PrettyTable()
    t_balance.field_names = ["Mes", "Región", "Gen [kWh]", "Autoc [kWh]", "Inyec [kWh]", "Ahorro [USD]"]
    
    for m in range(min(12, autoconsumo.shape[1])):
        for i, reg in enumerate(regiones):
            t_balance.add_row([
                m,
                reg,
                f"{energia[i, m]:,.1f}",
                f"{autoconsumo[i, m]:,.1f}",
                f"{inyeccion[i, m]:,.1f}",
                f"${ahorro[i, m]:,.2f}"
            ])
    
    print(t_balance)
    
    # 3) Resumen Anual (Año 1)
    print("\n" + "=" * 80)
    print(f"RESUMEN ANUAL ({politica_seleccionada}) - Año 1")
    print("=" * 80)
    
    t_anual = PrettyTable()
    t_anual.field_names = ["Región", "Gen Total", "Autoc Total", "Inyec Total", "Ahorro Total [USD]", "LCOE Prom [USD]"]
    
    for i, reg in enumerate(regiones):
        gen_anual = np.sum(energia[i, :12])
        autoc_anual = np.sum(autoconsumo[i, :12])
        inyec_anual = np.sum(inyeccion[i, :12])
        ahorro_anual = np.sum(ahorro[i, :12])
        lcoe_prom = np.mean(lcoe[i, :12])
        
        t_anual.add_row([
            reg,
            f"{gen_anual:,.1f} kWh",
            f"{autoc_anual:,.1f} kWh",
            f"{inyec_anual:,.1f} kWh",
            f"${ahorro_anual:,.2f}",
            f"{lcoe_prom:.4f}"
        ])
    
    print(t_anual)
    
    # 4) Estadísticas por región (todos los años)
    print("\n" + "=" * 80)
    print(f"ESTADÍSTICAS COMPLETAS ({politica_seleccionada}) - (Todos los años)")
    print("=" * 80)
    
    t_stats = PrettyTable()
    t_stats.field_names = ["Región", "Ahorro Total [USD]", "Ahorro Prom/mes [USD]", "LCOE Min", "LCOE Max", "LCOE Prom"]
    
    for i, reg in enumerate(regiones):
        ahorro_total = np.sum(ahorro[i, :])
        ahorro_prom = np.mean(ahorro[i, :])
        lcoe_min = np.min(lcoe[i, :])
        lcoe_max = np.max(lcoe[i, :])
        lcoe_prom = np.mean(lcoe[i, :])
        
        t_stats.add_row([
            reg,
            f"${ahorro_total:,.2f}",
            f"${ahorro_prom:,.2f}",
            f"{lcoe_min:.4f}",
            f"{lcoe_max:.4f}",
            f"{lcoe_prom:.4f}"
        ])
    
    print(t_stats)
    
    print("\n" + "=" * 80)
    print("MODELO COMPLETADO ✓")
    print("=" * 80)


if __name__ == "__main__":
    main()