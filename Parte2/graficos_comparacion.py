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

# Parámetros base
consumo_diario_ev = 101.753 / 30.0
horas = np.arange(24)

# --- 1. DATOS 2026 (Región Centro) ---
stock_2026 = 7368
energia_2026 = stock_2026 * consumo_diario_ev
dumb_2026 = (energia_2026 * generar_perfil_diario("dumb")) / 1000.0
smart_2026 = (energia_2026 * generar_perfil_diario("smart")) / 1000.0
diurna_2026 = (energia_2026 * generar_perfil_diario("diurna")) / 1000.0

# --- 2. DATOS 2050 (Región Centro) ---
stock_2050 = 4722550
energia_2050 = stock_2050 * consumo_diario_ev
dumb_2050 = (energia_2050 * generar_perfil_diario("dumb")) / 1000.0
smart_2050 = (energia_2050 * generar_perfil_diario("smart")) / 1000.0
diurna_2050 = (energia_2050 * generar_perfil_diario("diurna")) / 1000.0

# --- DIBUJAMOS LA COMPARACIÓN (1 FILA, 2 COLUMNAS) ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Gráfico Izquierdo: 2026
ax1.plot(horas, dumb_2026, label=f'Dumb ({np.max(dumb_2026):.1f} MW)', color='red', lw=3)
ax1.plot(horas, smart_2026, label=f'Smart ({np.max(smart_2026):.1f} MW)', color='blue', lw=3, ls='--')
ax1.plot(horas, diurna_2026, label=f'Diurna ({np.max(diurna_2026):.1f} MW)', color='orange', lw=3, ls='-.')

ax1.set_title('El Problema Hoy: Año 2026\n(7.368 Vehículos)', fontsize=14, fontweight='bold')
ax1.set_xlabel('Hora del Día (0 a 23 hrs)', fontsize=12)
ax1.set_ylabel('Demanda (Megavatios - MW)', fontsize=12)
ax1.set_xticks(np.arange(0, 24, 2))
ax1.grid(True, ls='--', alpha=0.6)
ax1.legend(fontsize=11)

# Gráfico Derecho: 2050
ax2.plot(horas, dumb_2050, label=f'Dumb ({np.max(dumb_2050):.0f} MW)', color='red', lw=3)
ax2.plot(horas, smart_2050, label=f'Smart ({np.max(smart_2050):.0f} MW)', color='blue', lw=3, ls='--')
ax2.plot(horas, diurna_2050, label=f'Diurna ({np.max(diurna_2050):.0f} MW)', color='orange', lw=3, ls='-.')

ax2.set_title('La Explosión Futura: Año 2050\n(4.722.550 Vehículos)', fontsize=14, fontweight='bold')
ax2.set_xlabel('Hora del Día (0 a 23 hrs)', fontsize=12)
ax2.set_ylabel('Demanda (Megavatios - MW)', fontsize=12)
ax2.set_xticks(np.arange(0, 24, 2))
ax2.grid(True, ls='--', alpha=0.6)
ax2.legend(fontsize=11)

# Ajustar espacios y guardar
plt.tight_layout()
plt.savefig('comparacion_2026_vs_2050.png', dpi=300)
print("\n¡Gráfico comparativo generado! Revisa el archivo 'comparacion_2026_vs_2050.png'")