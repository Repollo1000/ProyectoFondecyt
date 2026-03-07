# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

def generar_perfil_diario(estrategia):
    horas = np.arange(24)
    # Matemáticas de la Campana de Gauss para repartir la carga
    if estrategia == "dumb":
        perfil = np.exp(-0.5 * ((horas - 19.5) / 2.5)**2)
    elif estrategia == "smart":
        perfil = np.exp(-0.5 * ((horas - 3.0) / 2.0)**2)
    elif estrategia == "diurna":
        perfil = np.exp(-0.5 * ((horas - 13.0) / 3.0)**2)
    return perfil / np.sum(perfil)

# Datos extraídos del Orquestador (Región Centro - Año 2050)
stock_ev = 4722550 # Los 4.7 millones de vehículos
consumo_diario_ev = 101.753 / 30.0 # kWh al día (Regla de la profe)
energia_total_diaria_kwh = stock_ev * consumo_diario_ev

# Generamos las 24 horas de cada estrategia
horas = np.arange(24)
demanda_dumb = (energia_total_diaria_kwh * generar_perfil_diario("dumb")) / 1000.0
demanda_smart = (energia_total_diaria_kwh * generar_perfil_diario("smart")) / 1000.0
demanda_diurna = (energia_total_diaria_kwh * generar_perfil_diario("diurna")) / 1000.0

# --- DIBUJAMOS EL GRÁFICO ---
plt.figure(figsize=(11, 6))

# Las 3 curvas de colores
plt.plot(horas, demanda_dumb, label=f'Carga Descontrolada (Peak 19:30): {np.max(demanda_dumb):.0f} MW', color='red', linewidth=3)
plt.plot(horas, demanda_smart, label=f'Carga Inteligente (Peak 03:00): {np.max(demanda_smart):.0f} MW', color='blue', linewidth=3, linestyle='--')
plt.plot(horas, demanda_diurna, label=f'Carga Diurna (Peak 13:00): {np.max(demanda_diurna):.0f} MW', color='orange', linewidth=3, linestyle='-.')

# Textos, etiquetas y estilo
plt.title('Impacto Horario en la Red Eléctrica - Flota EV Región Centro (Año 2050)', fontsize=14, fontweight='bold')
plt.xlabel('Hora del Día (0 a 23 hrs)', fontsize=12)
plt.ylabel('Potencia Demandada (Megavatios - MW)', fontsize=12)
plt.xticks(np.arange(0, 24, 1))
plt.grid(True, linestyle='--', alpha=0.5)

# Pintamos suavemente el área bajo la curva para mostrar la equivalencia de energía
plt.fill_between(horas, demanda_dumb, alpha=0.05, color='red')
plt.fill_between(horas, demanda_smart, alpha=0.05, color='blue')
plt.fill_between(horas, demanda_diurna, alpha=0.05, color='orange')

plt.legend(fontsize=11)
plt.tight_layout()

# Guardamos el archivo y lo mostramos en pantalla
plt.savefig('grafico_horario_2050.png', dpi=300)
plt.show()

print("\n¡Gráfico generado exitosamente! Se guardó como 'grafico_horario_2050.png' en tu carpeta.")