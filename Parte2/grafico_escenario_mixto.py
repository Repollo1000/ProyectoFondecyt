# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

def generar_perfil_diario(estrategia):
    horas = np.arange(24)
    if estrategia == "dumb":
        perfil = np.exp(-0.5 * ((horas - 19.5) / 2.5)**2)
    elif estrategia == "smart":
        perfil = np.exp(-0.5 * ((horas - 3.0) / 2.0)**2)
    elif estrategia == "diurna":
        perfil = np.exp(-0.5 * ((horas - 13.0) / 3.0)**2)
    return perfil / np.sum(perfil)

# Parámetros (Región Centro - 2050)
stock_ev = 4722550
consumo_diario_ev = 101.753 / 30.0
energia_total = stock_ev * consumo_diario_ev

# Extraemos las 24 horas de cada estrategia 100% pura
horas = np.arange(24)
demanda_dumb = (energia_total * generar_perfil_diario("dumb")) / 1000.0
demanda_smart = (energia_total * generar_perfil_diario("smart")) / 1000.0
demanda_diurna = (energia_total * generar_perfil_diario("diurna")) / 1000.0

# --- APLICAMOS LA MEJORA: MEZCLAMOS LOS PORCENTAJES ---
pct_dumb = 0.50   # 50% Carga Descontrolada (Residencial tarde)
pct_diurna = 0.30 # 30% Carga Diurna (Solar/Trabajo)
pct_smart = 0.20  # 20% Carga Inteligente (Madrugada)

# Sumamos los impactos ponderados por su porcentaje
demanda_mixta = (pct_dumb * demanda_dumb) + (pct_diurna * demanda_diurna) + (pct_smart * demanda_smart)

# --- DIBUJAMOS EL GRÁFICO ---
plt.figure(figsize=(11, 6))

# Dibujamos las curvas "fantasmas" (Puras) como referencia de cuán malas eran
plt.plot(horas, demanda_dumb, color='red', alpha=0.3, linestyle=':', label='Referencia: 100% Descontrolada')
plt.plot(horas, demanda_smart, color='blue', alpha=0.3, linestyle=':', label='Referencia: 100% Inteligente')
plt.plot(horas, demanda_diurna, color='orange', alpha=0.3, linestyle=':', label='Referencia: 100% Diurna')

# Dibujamos la Curva Mixta Real (Morada Gruesa)
plt.plot(horas, demanda_mixta, color='purple', linewidth=4, label=f'ESCENARIO MIXTO (Peak Suavizado: {np.max(demanda_mixta):.0f} MW)')
plt.fill_between(horas, demanda_mixta, color='purple', alpha=0.2, label='Energía Total Repartida (101.75 kWh)')

# Anotación del nuevo Peak Aplanado
peak_mw = np.max(demanda_mixta)
hora_peak = horas[np.argmax(demanda_mixta)]
plt.annotate(f'Punto Máximo Realista:\n{peak_mw:.0f} MW', 
             xy=(hora_peak, peak_mw), 
             xytext=(hora_peak - 4, peak_mw + 300),
             arrowprops=dict(facecolor='black', shrink=0.05, width=2, headwidth=8),
             fontsize=12, fontweight='bold', color='indigo')

# Estilo general
plt.title(f'La Solución: Escenario Mixto de Carga (50% Casa, 30% Trabajo, 20% Noche) - 2050', fontsize=14, fontweight='bold')
plt.xlabel('Hora del Día (0 a 23 hrs)', fontsize=12)
plt.ylabel('Potencia Demandada (Megavatios - MW)', fontsize=12)
plt.xticks(np.arange(0, 24, 1))
plt.grid(True, linestyle='--', alpha=0.6)

# Configuramos la leyenda para que no tape el gráfico
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=10)
plt.tight_layout()

# Guardamos el archivo y avisamos en consola
plt.savefig('escenario_mixto_2050.png', dpi=300)
print("\n¡Mejora 1 Ejecutada! Revisa el gráfico 'escenario_mixto_2050.png' en VS Code.")