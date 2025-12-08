# -*- coding: utf-8 -*-
"""
main_mod9.py — Orquestador con MENÚ DE SELECCIÓN DE ESCENARIO.
"""
from __future__ import annotations
import numpy as np
from prettytable import PrettyTable, SINGLE_BORDER

try:
    from .. import parametros_globales as p_g
    from . import modulo9 as m9
    from ..modulo7 import modulo7 as m7
except ImportError:
    import parametros_globales as p_g
    import modulo9 as m9
    from modulo7 import modulo7 as m7

# =============================================================================
# FUNCIONES DE UTILERÍA (Menú y Visualización)
# =============================================================================

def solicitar_escenario_usuario() -> tuple[str, str]:
    """
    Muestra un menú en consola y captura la elección del usuario.
    Retorna: (código_interno, nombre_display)
    """
    print("\n" + "="*40)
    print("   SELECCIÓN DE ESCENARIO DE EMISIONES")
    print("="*40)
    print("   1. ALTO  (CN Scenario)")
    print("   2. MEDIO (SR Scenario)")
    print("   3. BAJO  (AT Scenario)")
    print("-" * 40)
    
    eleccion = input(">> Ingrese el número de su opción (1, 2 o 3): ").strip()
    
    # Mapa de opciones a códigos que entiende modulo9.py
    opciones = {
        "1": ("ALTO", "Alto (CN)"),
        "2": ("MEDIO", "Medio (SR)"),
        "3": ("BAJO", "Bajo (AT)")
    }
    
    # Si la entrada no es válida, usamos MEDIO por defecto
    seleccion = opciones.get(eleccion, ("MEDIO", "Medio (SR)"))
    
    if eleccion not in opciones:
        print(f"\n[!] Opción '{eleccion}' no reconocida. Usando '{seleccion[1]}' por defecto.")
    else:
        print(f"\n[OK] Escenario seleccionado: {seleccion[1]}")
        
    return seleccion

def mostrar_factores(tiempos: np.ndarray, factors: np.ndarray, titulo: str):
    table = PrettyTable()
    table.set_style(SINGLE_BORDER)
    table.field_names = ["Mes (idx)", "Factor (tCO2/MWh)"]
    # Mostrar solo los primeros 5 y el último para no llenar la pantalla
    indices = [0, 1, 2, 3, 4, len(factors)-1]
    for i in indices:
        if i < len(factors):
            table.add_row([f"{tiempos[i]:.0f}", f"{factors[i]:.4f}"])
    print(f"\n=== Verificación: {titulo} ===")
    print(table)

def mostrar_consumo_mensual(consumo_df):
    table = PrettyTable()
    table.set_style(SINGLE_BORDER)
    table.field_names = ["Mes"] + list(consumo_df.columns)
    # Mostrar solo primeros 3 meses
    for i in range(3):
        row = consumo_df.iloc[i]
        fila = [str(row.name)] + [f"{v:.1f}" for v in row.values]
        table.add_row(fila)
    print("\n=== Verificación: Perfil de Consumo (kWh/mes/hogar) [Primeros 3 meses] ===")
    print(table)

# =============================================================================
# MAIN (ORQUESTADOR)
# =============================================================================

def main():
    # --- MENÚ INTERACTIVO ---
    codigo_escenario, nombre_escenario = solicitar_escenario_usuario()
    
    print("\n-------------------------------------------------------------")
    print(f"   INICIANDO SIMULACIÓN CON ESCENARIO: {nombre_escenario.upper()}")
    print("-------------------------------------------------------------")

    # --- PASO 1: OBTENER DATOS DE MÓDULO 7 (HOGARES) ---
    print("\n>>> PASO 1: Ejecutando simulación de adopción (Módulo 7)...")
    res_m7 = m7.simulate_system(**p_g.MOD7_VARIABLES_INICIALES)
    t_sim = res_m7[0]
    households = res_m7[2]  # Matriz (T, 3)
    
    # --- PASO 2: RECOLECCIÓN DE DATOS DE ENTRADA MÓDULO 9 ---
    print("\n>>> PASO 2: Cargando datos externos (Excel/CSV)...")
    
    # 2.1 Cargar Factores según la elección del usuario
    tiempos_em, factores = m9.cargar_factor_emision(codigo_escenario)
    mostrar_factores(tiempos_em, factores, f"Factores Escenario {nombre_escenario}")

    # 2.2 Cargar Perfil de Consumo
    perfil_consumo = m9.cargar_perfil_consumo_mensual()
    mostrar_consumo_mensual(perfil_consumo)

    # --- PASO 3: CÁLCULO DE EMISIONES (SECCIÓN 9.2) ---
    print("\n>>> PASO 3: Calculando Emisiones de Consumo (Sección 9.2)...")
    
    emisiones_mensuales = m9.calcular_emisiones_consumo(
        factores_emision=factores,
        households=households,
        perfil_consumo_12_meses=perfil_consumo
    )
    
    # --- PASO 4: ACUMULACIÓN ANUAL Y REPORTE ---
    print("\n>>> PASO 4: Generando Reporte Anual...")
    emisiones_anuales = m9.acumular_anualmente(emisiones_mensuales)
    
    # Mostrar Tabla Final
    table = PrettyTable()
    table.set_style(SINGLE_BORDER)
    # Título dinámico según el escenario
    table.title = f"EMISIONES DE CONSUMO ELÉCTRICO - ESCENARIO {nombre_escenario.upper()}"
    table.field_names = ["Año", "Norte (tCO2)", "Centro (tCO2)", "Sur (tCO2)", "TOTAL PAÍS"]
    
    anio_inicio = 2025
    for i in range(min(10, len(emisiones_anuales))):
        anio = anio_inicio + i
        vals = emisiones_anuales[i, :]
        total = np.sum(vals)
        fila = [anio] + [f"{v:,.0f}" for v in vals] + [f"{total:,.0f}"]
        table.add_row(fila)
    
    print(table)
    
    total_periodo = np.sum(emisiones_mensuales)
    print(f"\n[RESULTADO FINAL] Emisiones Totales Acumuladas ({nombre_escenario}): {total_periodo:,.0f} tCO2")

if __name__ == "__main__":
    main()