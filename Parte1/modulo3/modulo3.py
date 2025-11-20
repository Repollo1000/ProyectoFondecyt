# -*- coding: utf-8 -*-
import pandas as pd

# =============================================================================
# CONFIGURACIÓN DE ESCENARIOS (CUMPLIENDO LO QUE PIDE LA PROFE)
# Aquí asociamos: ESCENARIO -> (Hoja de Precio, Factor Emisión)
# =============================================================================
CONFIG_ESCENARIOS = {
    "low": {
        "hoja_precio": "low",       
        "factor_emision": 0.3,      # <--- DATO DUMMY (Cámbialo cuando llegue Mónica)
        "descripcion": "Escenario Bajo: Alta penetración renovable"
    },
    "medium": {
        "hoja_precio": "medium",
        "factor_emision": 0.4,      # <--- DATO DUMMY
        "descripcion": "Escenario Medio: Tendencia actual"
    },
    "high": {
        "hoja_precio": "high",
        "factor_emision": 0.6,      # <--- DATO DUMMY
        "descripcion": "Escenario Alto: Retraso en descarbonización"
    }
}

def obtener_datos_escenario(nombre_escenario: str, ruta_archivo_precios: str):
    """
    Entrega el pack completo del escenario: Precios + Emisiones.
    """
    key = nombre_escenario.lower()
    
    # 1. Validar
    if key not in CONFIG_ESCENARIOS:
        raise ValueError(f"Escenario no válido: {key}")
    
    config = CONFIG_ESCENARIOS[key]
    
    # 2. Obtener el Factor de Emisión (Asociado internamente, como pide la profe)
    factor_emision = config["factor_emision"]
    
    # 3. Leer el Precio correspondiente
    try:
        df_precios = pd.read_excel(ruta_archivo_precios, sheet_name=config["hoja_precio"])
    except:
        # Fallback simple por si acaso
        df_precios = pd.read_excel(ruta_archivo_precios)

    return df_precios, factor_emision