# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. LEER LOS DATOS OFICIALES DEL CEN (Tu archivo CSV)
# Cambia 'nombre_de_tu_archivo.csv' por el nombre real de tu archivo
archivo_csv = 'f539c535-f00d-46f7-8b3f-2435cdea01b6.csv'
df = pd.read_csv(archivo_csv)

# Limpiamos las comas de los miles y convertimos la columna de demanda a números (Megavatios)
demanda_base_mw = df['Demanda [MWh]'].str.replace(',', '').astype(float).values

# 2. CÁLCULO DE LOS AUTOS ELÉCTRICOS (Región Centro - 2050)
def generar_perfil_diario(estrategia):
    horas = np.arange(24)
    if estrategia == "dumb": perfil = np.exp(-0.5 * ((horas - 19.5) / 2.5)**2)
    elif estrategia == "smart": perfil = np.exp(-0.5 * ((horas - 3.0) / 2.0)**2)
    elif estrategia == "diurna": perfil = np.exp(-0.5 * ((horas - 13.0) / 3.0)**2)
    return perfil / np.sum(perfil)

energia_total = 4722550 * (101.753 / 30.0) # Flota EV consumiendo al día

demanda_dumb = (energia_total * generar_perfil_diario("dumb")) / 1000.0
demanda_diurna = (energia_total * generar_perfil_diario("diurna")) / 1000.0
demanda_smart = (energia_total * generar_perfil_diario("smart")) / 1000.0

# 3. EL ESCENARIO MIXTO (50% Casa, 30% Trabajo, 20% Noche)
demanda_mixta = (0.50 * demanda_dumb) + (0.30 * demanda_diurna) + (0.20 * demanda_smart)

# 4. LA SUPERPOSICIÓN (Ciudad + Autos)
demanda_total_peor_caso = demanda_base_mw + demanda_dumb
demanda_total_mejor_caso = demanda_base_mw + demanda_mixta

# --- DIBUJAMOS EL GRÁFICO FINAL ---
plt.figure(figsize=(12, 7))
horas = np.arange(24)

# Área gris: Curva Real de Chile
plt.fill_between(horas, 0, demanda_base_mw, color='gray', alpha=0.3, label='Demanda Base Real SEN (Julio 2025)')
plt.plot(horas, demanda_base_mw, color='dimgray', linewidth=2)

# Línea Roja: 100% Carga en Casa
plt.plot(horas, demanda_total_peor_caso, color='red', linewidth=3, linestyle='--', 
         label=f'PEOR CASO: Base + Carga Descontrolada (Peak Total: {np.max(demanda_total_peor_caso):.0f} MW)')

# Línea Verde: Política Pública
plt.plot(horas, demanda_total_mejor_caso, color='forestgreen', linewidth=4, 
         label=f'SOLUCIÓN: Base + Escenario Mixto (Peak Total: {np.max(demanda_total_mejor_caso):.0f} MW)')

# Textos y estética
plt.title('Impacto de la Electromovilidad en la Red Nacional (Dato Oficial CEN)', fontsize=15, fontweight='bold')
plt.xlabel('Hora del Día (0 a 23 hrs)', fontsize=12)
plt.ylabel('Demanda Total del Sistema (Megavatios - MW)', fontsize=12)
plt.xticks(np.arange(0, 24, 1))
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend(loc='upper left', fontsize=11)

# Cálculo matemático del ahorro
ahorro_peak = np.max(demanda_total_peor_caso) - np.max(demanda_total_mejor_caso)
plt.annotate(f'Ahorro de Infraestructura:\n¡{ahorro_peak:.0f} MW menos para el SEN!', 
             xy=(19.5, np.max(demanda_total_mejor_caso)), 
             xytext=(9, np.max(demanda_total_peor_caso) - 500),
             arrowprops=dict(facecolor='forestgreen', shrink=0.05, width=2, headwidth=8),
             fontsize=12, fontweight='bold', color='darkgreen')

plt.tight_layout()
plt.savefig('impacto_oficial_sen.png', dpi=300)
print("\n¡Simulación oficial completada! Revisa el gráfico 'impacto_oficial_sen.png'.")