# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

# =============================================================================
# CONFIGURACIÓN DE ESCENARIOS
# Mapeamos: ESCENARIO DE PRECIO -> Columna en el CSV de Emisiones
# =============================================================================
CONFIG_ESCENARIOS = {
    "low": {
        "hoja_precio": "low",       
        "col_emision": "AT scenario",   # Asumimos Accelerated Transition (Bajo)
        "descripcion": "Escenario Bajo: Alta penetración renovable"
    },
    "medium": {
        "hoja_precio": "medium",
        "col_emision": "CN scenario",   # Asumimos Carbon Neutrality (Medio)
        "descripcion": "Escenario Medio: Tendencia actual"
    },
    "high": {
        "hoja_precio": "high",
        "col_emision": "SR scenario",   # Asumimos Stated Policies (Alto)
        "descripcion": "Escenario Alto: Retraso en descarbonización"
    }
}

def obtener_datos_escenario(nombre_escenario: str, ruta_archivo_precios: str, ruta_archivo_emisiones: str):
    """
    Carga los datos del escenario seleccionado.
    Retorna: (DataFrame precios, Array factor_emision_mensual)
    """
    key = nombre_escenario.lower()
    
    # 1. Validar escenario
    if key not in CONFIG_ESCENARIOS:
        raise ValueError(f"Escenario no válido: {key}. Opciones: {list(CONFIG_ESCENARIOS.keys())}")
    
    config = CONFIG_ESCENARIOS[key]
    
    # 2. Leer Precios (Excel)
    try:
        # Intentamos leer la hoja específica
        df_precios = pd.read_excel(ruta_archivo_precios, sheet_name=config["hoja_precio"])
    except Exception as e:
        print(f"Error leyendo precios (hoja {config['hoja_precio']}): {e}")
        # Si falla, intentamos leer el archivo genérico (fallback)
        df_precios = pd.read_excel(ruta_archivo_precios)

    # 3. Leer Factores de Emisión (CSV)
    try:
        # El archivo usa punto y coma (;)
        df_emisiones = pd.read_csv(ruta_archivo_emisiones, sep=';')
        
        # Limpiamos espacios en los nombres de las columnas (ej: " CN scenario" -> "CN scenario")
        df_emisiones.columns = df_emisiones.columns.str.strip()
        
        columna_objetivo = config["col_emision"]
        
        # Validación de seguridad
        if columna_objetivo not in df_emisiones.columns:
            # NOTA: Si mañana te dicen que AT=Norte, CN=Centro, SR=Sur,
            # aquí tendríamos que cambiar la lógica para devolver las 3 columnas
            # en lugar de solo una.
            raise ValueError(f"La columna '{columna_objetivo}' no existe en el CSV. Columnas halladas: {df_emisiones.columns.tolist()}")
            
        # Extraemos el vector completo (324 meses)
        factor_emision_vector = df_emisiones[columna_objetivo].astype(float).values
        
    except Exception as e:
        print(f"Error leyendo archivo de emisiones: {e}")
        raise e

    return df_precios, factor_emision_vector