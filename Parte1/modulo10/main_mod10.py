# -*- coding: utf-8 -*-
from __future__ import annotations
import numpy as np
from prettytable import PrettyTable
import sys
import os

# Agregar path para importar modulo5
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importar módulos
from modulo5 import modulo5 as m5
from modulo10 import modulo10 as m10


def main():
    print("=" * 80)
    print("MODELO INTEGRADO: LCOE (Módulo 5) + BALANCE ENERGÉTICO (Módulo 10)")
    print("=" * 80)
    
    # ==========================================
    # PASO 1: EJECUTAR MÓDULO 5 (LCOE)
    # ==========================================
    print("\n[1/2] Ejecutando Módulo 5 - LCOE...")
    
    variables_m5 = m5.default_variables()
    variables_m5["use_dynamic_subsidy"] = False
    
    resultados_m5 = m5.correr_modelo("../Datos/costoAño.xlsx", variables_m5)
    
    regiones = resultados_m5["regiones"]
    sstc = resultados_m5["sstc_mensual"]
    energia = resultados_m5["energia_mensual"]
    lcoe = resultados_m5["lcoe_mensual"]
    
    N_meses = sstc.shape[1]
    N_años = N_meses // 12
    
    print(f"✓ Módulo 5 completado: {N_meses} meses ({N_años} años)")
    
    # ==========================================
    # PASO 2: EJECUTAR MÓDULO 10 (BALANCE)
    # ==========================================
    print("\n[2/2] Ejecutando Módulo 10 - Balance Energético...")
    
    variables_balance = m10.default_variables_balance()
    variables_balance["usar_perfil_horario"] = False  # Método simple 60/40
    
    # Rutas a CSVs
    ruta_csvs = {
        "Norte": "../Datos/Antofagasta.csv",
        "Centro": "../Datos/Santiago.csv",
        "Sur": "../Datos/Puerto montt.csv"
    }
    
    resultados_completos = m10.correr_modelo_completo(
        resultados_m5,
        variables_balance,
        ruta_csvs
    )
    
    balance = resultados_completos["balance_energetico"]
    autoconsumo = balance["autoconsumo_mensual"]
    inyeccion = balance["inyeccion_mensual"]
    ahorro = balance["ahorro_mensual"]
    
    print("✓ Módulo 10 completado")
    
    # ==========================================
    # MOSTRAR RESULTADOS
    # ==========================================
    
    # 1) LCOE (primeros 12 meses)
    print("\n" + "=" * 80)
    print("LCOE MENSUAL [USD/kWh] - Primeros 12 meses")
    print("=" * 80)
    
    t_lcoe = PrettyTable()
    t_lcoe.field_names = ["Mes"] + list(regiones)
    
    for m in range(min(12, lcoe.shape[1])):
        fila = [m] + [f"{lcoe[i, m]:.4f}" for i in range(len(regiones))]
        t_lcoe.add_row(fila)
    
    print(t_lcoe)
    
    # 2) Balance Energético (primeros 12 meses)
    print("\n" + "=" * 80)
    print("BALANCE ENERGÉTICO - Primeros 12 meses")
    print("=" * 80)
    
    t_balance = PrettyTable()
    t_balance.field_names = ["Mes", "Región", "Gen [kWh]", "Autoc [kWh]", "Inyec [kWh]", "Ahorro [$]"]
    
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
    print("RESUMEN ANUAL - Año 1")
    print("=" * 80)
    
    t_anual = PrettyTable()
    t_anual.field_names = ["Región", "Gen Total", "Autoc Total", "Inyec Total", "Ahorro Total", "LCOE Prom"]
    
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
    print("ESTADÍSTICAS COMPLETAS")
    print("=" * 80)
    
    t_stats = PrettyTable()
    t_stats.field_names = ["Región", "Ahorro Total", "Ahorro Prom/mes", "LCOE Min", "LCOE Max", "LCOE Prom"]
    
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