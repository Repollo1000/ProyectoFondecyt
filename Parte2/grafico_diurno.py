# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

# 1. Parámetros del modelo (Región Centro, Año 2050)
stock_ev = 4722550
consumo_diario_ev = 101.753 / 30.0
energia_total_diaria_kwh = stock_ev * consumo_diario_ev

# 2. Matemática de la Carga Diurna (Peak 13:00, Dispersión 3.0 hrs - curva más ancha)
horas = np.arange(24)
perfil_diurna = np.exp(-0.5 * ((horas - 13.0) / 3.0)**2)
perfil_diurna_normalizado = perfil_diurna / np.sum(perfil_diurna)

# 3. Cálculo en Megavatios (MW)
demanda_diurna_mw = (energia_total_diaria_kwh * perfil_diurna_normalizado) / 1000.0

# --- DIBUJAMOS EL GRÁFICO ZOOM ---
plt.figure(figsize=(10, 6))

# Dibujar la línea principal (Color Naranja/Sol)
plt.plot(horas, demanda_diurna_mw, color='darkorange', linewidth=4, label='Perfil de Carga Diurna (Trabajo/Solar)')

# Pintar el área bajo la curva (representa la Energía Total, que sigue siendo 101.75 kWh)
plt.fill_between(horas, demanda_diurna_mw, color='orange', alpha=0.3, label='Energía Total Diaria Constante')

# Anotar el Peak (Más bajo y a mediodía)
peak_mw = np.max(demanda_diurna_mw)
hora_peak = horas[np.argmax(demanda_diurna_mw)]
plt.annotate(f'PEAK SUAVIZADO: {peak_mw:.0f} MW\n(Absorbe Generación Solar)', 
             xy=(hora_peak, peak_mw), 
             xytext=(hora_peak - 6, peak_mw - 200),
             arrowprops=dict(facecolor='black', shrink=0.05, width=2, headwidth=8),
             fontsize=11, fontweight='bold', color='saddlebrown')

# Estilos del gráfico
plt.title('ZOOM: Solución de Carga Diurna (Workplace Charging) - Año 2050', fontsize=14, fontweight='bold')
plt.xlabel('Hora del Día (0 a 23 hrs)', fontsize=12)
plt.ylabel('Potencia Demandada (Megavatios - MW)', fontsize=12)
plt.xticks(np.arange(0, 24, 1))
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(loc='upper left', fontsize=11)
plt.tight_layout()

# Guardar
plt.savefig('zoom_diurna_charging.png', dpi=300)
print("\n¡Gráfico Zoom Diurno generado! Revisa el archivo 'zoom_diurna_charging.png'")