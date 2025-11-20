# -*- coding: utf-8 -*-
import os
import sys
import pandas as pd

# Truco para poder importar modulo1 estando dentro de la misma carpeta
try:
    from . import modulo1 as m1
except ImportError:
    import modulo1 as m1

# Definimos rutas
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATOS_DIR = os.path.join(BASE_DIR, 'Datos')
ARCHIVO_PRECIOS = "precio_electricidad_vf.xlsx"
RUTA_COMPLETA = os.path.join(DATOS_DIR, ARCHIVO_PRECIOS)

def main():
    print("=" * 60)
    print("   PRUEBA UNITARIA: MÓDULO 1 (Lectura de Precios)")
    print("=" * 60)
    print(f"Ruta de búsqueda: {RUTA_COMPLETA}\n")

    escenarios_a_probar = ["low", "medium", "high", "fake"]

    for esc in escenarios_a_probar:
        print(f"--- Probando Escenario: {esc.upper()} ---")
        try:
            df = m1.obtener_precios(esc, RUTA_COMPLETA)
            
            # Verificar resultados
            print("   ✅ Éxito! Carga correcta.")
            print(f"   -> Dimensiones: {df.shape} (Filas, Columnas)")
            
            # --- LO NUEVO: IMPRIMIR EL DATAFRAME ---
            print("\n   [VISTA PREVIA DE DATOS]")
            print(df.head(5))  # Muestra las primeras 5 filas
            print("   .......................\n")
            
            # Precio promedio
            col_usd = [c for c in df.columns if "USD" in c][0]
            promedio = df[col_usd].mean()
            print(f"   -> Precio Promedio ({col_usd}): {promedio:.4f} USD/kWh")
            print("\n" + "-"*40 + "\n")
            
        except Exception as e:
            print(f"   ❌ Resultado esperado para '{esc}': {e}")
            print("\n" + "-"*40 + "\n")

if __name__ == "__main__":
    main()