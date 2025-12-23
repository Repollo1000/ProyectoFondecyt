# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
from . import demanda_adopcion_vehiculos as logic
from .. import parametros_globales_parte2 as p_g

# 1. Cargar parámetros desde el archivo global
vars_m = p_g.DEMANDA_ADOPCION_INICIALES.copy()

# 2. Crear "Market Share" temporal (Asumimos 10% EV y 90% ICEV para testear)
# Esto simula la flecha que vendrá del módulo de Utilidad en el futuro
T_total = int(vars_m["tf"] + 1)
vars_m["market_share_ev_series"] = np.full((T_total, 3), 0.10)
vars_m["market_share_icev_series"] = np.full((T_total, 3), 0.90)

# 3. Ejecutar la lógica del bucle mes a mes
res = logic.simulate_system(**vars_m)

# 4. Organizar los resultados en una tabla (DataFrame)
df = pd.DataFrame({"Mes": res["t"]})
# Agregamos los Stocks de cada región
# Agregamos los Stocks y redondeamos a 1 decimal
for i, region in enumerate(p_g.REGIONES):
    # .round(1) hace que 961053.48834... se convierta en 961053.5
    df[f"EV_Stock_{region}"] = np.round(res["ev_stock"][:, i], 1)
    df[f"ICEV_Stock_{region}"] = np.round(res["icev_stock"][:, i], 1)
    df[f"Population_{region}"] = np.round(res["population"][:, i], 1)

# --- LÓGICA DE GUARDADO LOCAL ---
ruta_del_main = os.path.dirname(os.path.abspath(__file__))
carpeta_nueva = os.path.join(ruta_del_main, "MIS_RESULTADOS")
os.makedirs(carpeta_nueva, exist_ok=True)

nombre_archivo = "mi_test_adopcion_maite.csv"
ruta_final = os.path.join(carpeta_nueva, nombre_archivo)

# Guardar el CSV (usamos sep=';' para que Excel no confunda decimales con columnas)
df.to_csv(ruta_final, index=False, sep=';', decimal=',')

print(f"\n✅ SIMULACIÓN COMPLETADA Y REDONDEADA")
print(f"👉 Archivo: {ruta_final}")