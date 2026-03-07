# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

# 1. Parámetros del modelo (Región Centro, Año 2050)
stock_ev = 4722550
consumo_diario_ev = 101.753 / 30.0
energia_total_diaria_kwh = stock_ev * consumo_diario_ev

# 2. Matemática de la Carga Descontrolada (Peak 19:30, Dispersión 2.5 hrs)
horas = np.arange(24)
perfil_dumb = np.exp(-0.5 * ((horas - 19.5) / 2.5)**2)
perfil_dumb_normalizado = perfil_dumb / np.sum(perfil_dumb)

# 3. Cálculo en Megavatios (MW)
demanda_dumb_mw = (energia_total_diaria_kwh * perfil_dumb_normalizado) / 1000.0

# --- DIBUJAMOS EL GRÁFICO ZOOM ---
plt.figure(figsize=(10, 6))

# Dibujar la línea principal
plt.plot(horas, demanda_dumb_mw, color='red', linewidth=4, label='Perfil de Carga Descontrolada')

# Pintar el área bajo la curva (representa la Energía Total)
plt.fill_between(horas, demanda_dumb_mw, color='red', alpha=0.2, label='Energía Total Diaria (101.75 kWh/mes por auto)')

# Anotar el Peak exactamente
peak_mw = np.max(demanda_dumb_mw)
hora_peak = horas[np.argmax(demanda_dumb_mw)]
plt.annotate(f'PEAK CRÍTICO: {peak_mw:.0f} MW', 
             xy=(hora_peak, peak_mw), 
             xytext=(hora_peak - 6, peak_mw - 200),
             arrowprops=dict(facecolor='black', shrink=0.05, width=2, headwidth=8),
             fontsize=12, fontweight='bold', color='darkred')

# Estilos del gráfico
plt.title('ZOOM: Comportamiento de la Carga Descontrolada (Dumb Charging) - Año 2050', fontsize=14, fontweight='bold')
plt.xlabel('Hora del Día (0 a 23 hrs)', fontsize=12)
plt.ylabel('Potencia Demandada (Megavatios - MW)', fontsize=12)
plt.xticks(np.arange(0, 24, 1))
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(loc='upper left', fontsize=11)
plt.tight_layout()

# Guardar
plt.savefig('zoom_dumb_charging.png', dpi=300)
print("\n¡Gráfico Zoom generado! Revisa el archivo 'zoom_dumb_charging.png'")