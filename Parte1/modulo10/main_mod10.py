# -*- coding: utf-8 -*
from __future__ import annotations
import numpy as np
import pandas as pd
from prettytable import PrettyTable, SINGLE_BORDER
import sys
import os

# --- INICIO DE CORRECCIÓN DE IMPORTACIÓN ---
from ..modulo5 import modulo5 as m5
from . import modulo10 as m10
from .. import parametros_globales as p_g # Importamos parámetros globales
# --- FIN DE CORRECCIÓN ---

# ==========================================
# CONFIGURACIÓN DE PRECIOS Y MONEDA
# ==========================================
# Se mantiene la tasa de cambio como referencia (aunque LCOE ya está en CLP)
TASA_CAMBIO_CLP_A_USD = 900.0 

# ¡¡IMPORTANTE!! Nombres de tus archivos de datos (deben estar en la carpeta 'Datos')
NOMBRE_ARCHIVO_PRECIOS = "precio_electricidad_vf.xlsx"       
NOMBRE_ARCHIVO_CONSUMO_HORARIO = "curva_de_carga.xlsx"        
NOMBRE_ARCHIVO_GENERACION_HORARIO = "Factor_capacidad_solar.csv" 

# Nombres de columnas para el Excel de precios
COLUMNA_PRECIO_COMPRA = "low1"        
COLUMNA_PRECIO_INYECCION = "low2"     
# ==========================================

# --- CONSTANTE PARA CONTROLAR LA SALIDA ANUAL ---
N_YEARS_TO_PRINT = 5 
# ------------------------------------------------

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
    
    # --- CORRECCIÓN: USAMOS LAS VARIABLES Y CONSTANTES GLOBALES ---
    variables_m5 = m5.default_variables()
    variables_m5["use_dynamic_subsidy"] = False
    
    # Sincronizamos la vida útil y PVGP con los valores globales
    variables_m5["project_lifetime_months"] = p_g.PROJECT_LIFETIME_MONTHS 
    variables_m5["pvgp_kW_per_household"] = p_g.MOD7_VARIABLES_INICIALES["pvgp_kW_per_household"]

    ruta_costo_excel = os.path.join(DATOS_DIR, "costoAño.xlsx")
    resultados_m5 = m5.correr_modelo(ruta_costo_excel, variables_m5)
    
    regiones = resultados_m5["regiones"]
    
    # AJUSTE: El LCOE se usa directamente en CLP (asumido)
    lcoe = resultados_m5["lcoe_mensual"] 
    
    N_meses = lcoe.shape[1]
    N_años = N_meses // 12
    
    print(f"✓ Módulo 5 completado: {N_meses} meses ({N_años} años). LCOE asumido en CLP.")
    
    # ==========================================
    # PASO 2: CARGAR DATOS DE BALANCE - Se hace UNA VEZ
    # ==========================================
    print("\n[2/2] Cargando datos para Módulo 10 - Balance Energético...")
    
    # --- Carga de Archivos de Datos (Precios y Perfiles) ---
    try:
        # 1. Cargar Precios
        print(f"Cargando precios desde {NOMBRE_ARCHIVO_PRECIOS}...")
        ruta_precios = os.path.join(DATOS_DIR, NOMBRE_ARCHIVO_PRECIOS)
        
        df_precios = pd.read_excel(ruta_precios)
        
        df_precios.columns = df_precios.columns.str.lower().str.strip()
        
        target_compra = COLUMNA_PRECIO_COMPRA.lower().strip()
        target_inyeccion = COLUMNA_PRECIO_INYECCION.lower().strip()
        
        precios_compra_clp = df_precios[target_compra].values
        precios_inyeccion_clp = df_precios[target_inyeccion].values
        
        print(f"✓ Precios cargados directamente en CLP.")

        # 2. Cargar Perfil de Consumo Horario
        print(f"Cargando perfiles de consumo desde {NOMBRE_ARCHIVO_CONSUMO_HORARIO}...")
        ruta_consumo = os.path.join(DATOS_DIR, NOMBRE_ARCHIVO_CONSUMO_HORARIO)
        df_consumo_horario = pd.read_excel(ruta_consumo) 
        print(f"✓ Perfil de consumo cargado.")

        # 3. Cargar Perfil de Generación Horario
        print(f"Cargando perfiles de generación desde {NOMBRE_ARCHIVO_GENERACION_HORARIO}...")
        ruta_generacion = os.path.join(DATOS_DIR, NOMBRE_ARCHIVO_GENERACION_HORARIO)
        
        df_generacion_horario = pd.read_csv(ruta_generacion, sep=';', encoding='latin-1') 
        print(f"✓ Perfil de generación cargado.")

    except Exception as e:
        print(f"✗ ERROR: No se pudo cargar un archivo de datos. Detalle: {e}") 
        print("   Abortando ejecución.")
        return 

    # --- Configuración de Variables de Balance ---
    variables_balance = m10.default_variables_balance()
    
    # Sobrescribir precios con los valores en CLP
    variables_balance["precio_electricidad"] = precios_compra_clp
    variables_balance["tarifa_netbilling"] = precios_inyeccion_clp 

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
        print("1. Net Billing (50% precio de compra)")
        print("2. Net Metering (100% precio de compra)")
        print("3. Feed-in Tariff (80% precio de compra)")
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
    # PASO 4: EJECUTAR SIMULACIÓN
    # ==========================================
    
    print("\n" + "=" * 80)
    print(f"EJECUTANDO SIMULACIÓN PARA POLÍTICA: {politica_seleccionada.upper()}")
    print("=" * 80)

    resultados_completos = m10.correr_modelo_completo(
        resultados_m5,
        variables_balance,
        df_consumo_horario,
        df_generacion_horario,
        politica=politica_seleccionada
    )
    
    energia = resultados_completos["energia_mensual"]
    lcoe = resultados_completos["lcoe_mensual"]
    balance = resultados_completos["balance_energetico"]
    autoconsumo = balance["autoconsumo_mensual"]
    inyeccion = balance["inyeccion_mensual"]
    ahorro = balance["ahorro_mensual"]
    
    print(f"✓ Módulo 10 completado para {politica_seleccionada}")
    
    # ==========================================
    # PASO 5: MOSTRAR RESULTADOS
    # ==========================================
    
    # 1) LCOE (primeros 12 meses)
    print("\n" + "=" * 80)
    print(f"LCOE MENSUAL [CLP/kWh] - Primeros 12 meses")
    print("=" * 80)
    
    t_lcoe = PrettyTable()
    t_lcoe.set_style(SINGLE_BORDER)
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
    t_balance.set_style(SINGLE_BORDER)
    t_balance.field_names = ["Mes", "Región", "Gen [kWh]", "Autoc [kWh]", "Inyec [kWh]", "Ahorro [CLP]"]
    
    for m in range(min(12, autoconsumo.shape[1])):
        for i, reg in enumerate(regiones):
            t_balance.add_row([
                m,
                reg,
                f"{energia[i, m]:,.1f}",
                f"{autoconsumo[i, m]:,.1f}",
                f"{inyeccion[i, m]:,.1f}",
                f"${ahorro[i, m]:,.0f}"
            ])
    
    print(t_balance)
    
    # 3) Resumen Anual (Primeros N_YEARS_TO_PRINT años)
    
    N_años_disponibles = N_meses // 12
    N_años_a_mostrar = min(N_YEARS_TO_PRINT, N_años_disponibles)

    print("\n" + "=" * 80)
    print(f"RESUMEN ANUAL ({politica_seleccionada}) - Primeros {N_años_a_mostrar} Años")
    print("=" * 80)
    
    t_anual = PrettyTable()
    t_anual.set_style(SINGLE_BORDER)
    t_anual.field_names = ["Año", "Región", "Gen Total", "Autoc Total", "Inyec Total", "Ahorro Total [CLP]", "LCOE Prom [CLP]"]
    
    for a in range(N_años_a_mostrar): 
        start_month = a * 12
        end_month = start_month + 12
        
        for i, reg in enumerate(regiones):
            gen_anual = np.sum(energia[i, start_month:end_month])
            autoc_anual = np.sum(autoconsumo[i, start_month:end_month])
            inyec_anual = np.sum(inyeccion[i, start_month:end_month])
            ahorro_anual = np.sum(ahorro[i, start_month:end_month])
            lcoe_prom = np.mean(lcoe[i, start_month:end_month])
            
            t_anual.add_row([
                a + 1,
                reg,
                f"{gen_anual:,.1f} kWh",
                f"{autoc_anual:,.1f} kWh",
                f"{inyec_anual:,.1f} kWh",
                f"${ahorro_anual:,.0f}",
                f"{lcoe_prom:.4f}"
            ])
    
    print(t_anual)
    
    # 4) Estadísticas por región (todos los años)
    print("\n" + "=" * 80)
    print(f"ESTADÍSTICAS COMPLETAS ({politica_seleccionada}) - (Todos los años)")
    print("=" * 80)
    
    t_stats = PrettyTable()
    t_stats.set_style(SINGLE_BORDER)
    t_stats.field_names = ["Región", "Ahorro Total [CLP]", "Ahorro Prom/mes [CLP]", "LCOE Min", "LCOE Max", "LCOE Prom"]
    
    for i, reg in enumerate(regiones):
        ahorro_total = np.sum(ahorro[i, :])
        ahorro_prom = np.mean(ahorro[i, :])
        lcoe_min = np.min(lcoe[i, :])
        lcoe_max = np.max(lcoe[i, :])
        lcoe_prom = np.mean(lcoe[i, :])
        
        t_stats.add_row([
            reg,
            f"${ahorro_total:,.0f}",
            f"${ahorro_prom:,.0f}",
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