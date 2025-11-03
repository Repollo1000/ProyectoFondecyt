# -*- coding: utf-8 -*-
from __future__ import annotations
import numpy as np
from prettytable import PrettyTable
from . import modulo5 as m5
import os  # <-- 1. IMPORTADO 'os' PARA MANEJAR RUTAS

def tabla_subsidio(info_subsidio, regiones):
    """Muestra tabla con información de subsidios aplicados."""
    t = PrettyTable()
    t.field_names = ["Región", "Difference usado", "Subsidio aplicado"]
    
    diff = info_subsidio["difference"]
    subs = info_subsidio["subsidio"]
    
    for i, reg in enumerate(regiones):
        d = float(diff[i]) if diff is not None and getattr(diff, "ndim", 0) > 0 else (
            float(diff) if diff is not None else float('nan'))
        s = float(subs[i])
        t.add_row([reg, f"{d:.3f}" if diff is not None else "-", f"{s:.3f}"])
    
    return t


def tabla_sstc_mensual(sstc, regiones, meses_mostrar=12, mes_inicio=0):
    """
    Tabla compacta de SSTC mensual.
    
    Args:
        sstc: matriz (3, N_meses)
        meses_mostrar: cuántos meses mostrar
        mes_inicio: desde qué mes empezar (0-indexed)
    """
    N = sstc.shape[1]
    mes_fin = min(mes_inicio + meses_mostrar, N)
    
    t = PrettyTable()
    t.field_names = ["Mes"] + list(regiones)
    
    for m in range(mes_inicio, mes_fin):
        fila = [m] + [f"{float(sstc[j, m]):,.2f}" for j in range(len(regiones))]
        t.add_row(fila)
    
    return t


def tabla_energia_mensual(E, regiones, meses_mostrar=12, mes_inicio=0):
    """Tabla compacta de energía mensual [kWh/mes]."""
    N = E.shape[1]
    mes_fin = min(mes_inicio + meses_mostrar, N)
    
    t = PrettyTable()
    t.field_names = ["Mes"] + list(regiones)
    
    for m in range(mes_inicio, mes_fin):
        fila = [m] + [f"{float(E[j, m]):,.2f}" for j in range(len(regiones))]
        t.add_row(fila)
    
    return t


def tabla_lcoe_mensual(lcoe, regiones, meses_mostrar=12, mes_inicio=0):
    """Tabla compacta de LCOE mensual [USD/kWh]."""
    N = lcoe.shape[1]
    mes_fin = min(mes_inicio + meses_mostrar, N)
    
    t = PrettyTable()
    t.field_names = ["Mes"] + list(regiones)
    
    for m in range(mes_inicio, mes_fin):
        fila = [m] + [f"{float(lcoe[j, m]):.4f}" for j in range(len(regiones))]
        t.add_row(fila)
    
    return t


def detalle_sstc_mensual(resultados, meses_mostrar=3):
    """
    Muestra el detalle de la ecuación SSTC mensual con números.
    """
    regiones = resultados["regiones"]
    tabla_costos = resultados["tabla_costos"]
    anio_inicio = resultados["anio_inicio"]
    variables = resultados["variables"]
    sstc = resultados["sstc_mensual"]
    
    N = int(variables["project_lifetime_months"])
    
    # Obtener costos anuales y convertir a mensuales
    cola = tabla_costos.loc[tabla_costos["Año"] >= anio_inicio]
    
    investment_cols = [
        "Costo _inversion_norte",
        "Costo _inversion_centro", 
        "Costo _inversion_sur"
    ]
    aoc_cols = [
        "Costo _operacion_norte",
        "Costo _operacion_centro",
        "Costo _operacion_sur"
    ]
    
    inv_años = cola[investment_cols].to_numpy(dtype=float).T
    aoc_años = cola[aoc_cols].to_numpy(dtype=float).T
    
    # Convertir a mensual (dividir por 12 y repetir)
    inv_m = np.repeat(inv_años / 12.0, 12, axis=1)[:, :N]
    aoc_m = np.repeat(aoc_años / 12.0, 12, axis=1)[:, :N]
    
    # Tasa mensual
    r_anual = np.asarray(variables["rtasa"], dtype=float)
    r_m = np.power(1.0 + r_anual, 1.0/12.0) - 1.0
    
    subs = np.asarray(variables["Percentage_capital_subsidy"], dtype=float)
    
    print("=== Detalle SSTC Mensual por región (con números) ===")
    
    for m_idx in range(min(meses_mostrar, sstc.shape[1])):
        m = m_idx + 1  # m desde 1
        print(f"\nMes = {m_idx} (t={m})")
        
        for i, reg in enumerate(regiones):
            inv = inv_m[i, m_idx]
            aoc = aoc_m[i, m_idx]
            r_val = r_m[i]
            sub_val = subs[i]
            
            # Factor de anualidad mensual
            factor = (1.0 - np.power(1.0 + r_val, -m)) / r_val
            pv_aoc = aoc * factor
            total = inv + pv_aoc
            sstc_val = total * (1.0 - sub_val)
            
            print(f"  {reg}: SSTC = [{inv:,.2f} + {aoc:,.2f} × {factor:.6f}] × {1-sub_val:.2f}")
            print(f"        = [{inv:,.2f} + {pv_aoc:,.2f}] × {1-sub_val:.2f}")
            print(f"        = {total:,.2f} × {1-sub_val:.2f}")
            print(f"        = {sstc_val:,.2f}")


def detalle_lcoe_mensual(resultados, meses_mostrar=3):
    """
    Muestra el detalle de la ecuación LCOE mensual con números.
    """
    regiones = resultados["regiones"]
    variables = resultados["variables"]
    sstc = resultados["sstc_mensual"]
    lcoe = resultados["lcoe_mensual"]
    E = resultados["energia_mensual"]
    
    # Tasa mensual
    r_anual = np.asarray(variables["rtasa"], dtype=float)
    r_m = np.power(1.0 + r_anual, 1.0/12.0) - 1.0
    
    print("=== Detalle LCOE Mensual por región (con números) ===")
    
    for m_idx in range(min(meses_mostrar, lcoe.shape[1])):
        m = m_idx + 1
        print(f"\nMes = {m_idx} (t={m})")
        
        for i, reg in enumerate(regiones):
            # Calcular energía acumulada descontada
            energy_slice = E[i, :m]
            indices = np.arange(1, m + 1, dtype=float)
            disc = 1.0 / np.power(1.0 + r_m[i], indices)
            energy_pv = np.sum(energy_slice * disc)
            
            sstc_val = sstc[i, m_idx]
            lcoe_val = lcoe[i, m_idx]
            
            print(f"  {reg}: LCOE = {sstc_val:,.2f} / {energy_pv:,.2f} = {lcoe_val:.4f}")


def estadisticas_resumen(sstc, lcoe, regiones):
    """Muestra estadísticas resumen de SSTC y LCOE."""
    print("\n=== Estadísticas Resumen (Mensual) ===")
    stats = PrettyTable()
    stats.field_names = ["Región", "SSTC Prom", "SSTC Min", "SSTC Max", 
                         "LCOE Prom", "LCOE Min", "LCOE Max"]
    
    for i, reg in enumerate(regiones):
        stats.add_row([
            reg,
            f"${np.mean(sstc[i, :]):,.2f}",
            f"${np.min(sstc[i, :]):,.2f}",
            f"${np.max(sstc[i, :]):,.2f}",
            f"{np.mean(lcoe[i, :]):.4f}",
            f"{np.min(lcoe[i, :]):.4f}",
            f"{np.max(lcoe[i, :]):.4f}",
        ])
    print(stats)


def estadisticas_anualizadas(sstc, lcoe, energia, regiones):
    """Muestra promedios anuales para comparación."""
    print("\n=== Promedios Anuales (agrupando cada 12 meses) ===")
    
    N = sstc.shape[1]
    años = N // 12
    
    stats = PrettyTable()
    stats.field_names = ["Año", "Región", "SSTC Prom Anual", "Energía Anual", "LCOE Prom"]
    
    for año in range(min(5, años)):  # Primeros 5 años
        inicio = año * 12
        fin = (año + 1) * 12
        
        for i, reg in enumerate(regiones):
            sstc_anual = np.mean(sstc[i, inicio:fin])
            energia_anual = np.sum(energia[i, inicio:fin])
            lcoe_anual = np.mean(lcoe[i, inicio:fin])
            
            stats.add_row([
                año,
                reg,
                f"${sstc_anual:,.2f}",
                f"{energia_anual:,.2f}",
                f"{lcoe_anual:.4f}"
            ])
    
    print(stats)


def main():
    print("=" * 70)
    print("MODELO LCOE - GRANULARIDAD MENSUAL")
    print("=" * 70)
    
    # 1) Configurar variables
    variables = m5.default_variables()
    variables["use_dynamic_subsidy"] = False
    # variables["difference_init"] = np.array([0.45, 0.72, 0.88], dtype=float)
    
    # 2) Ejecutar modelo
    print("\nCargando datos y ejecutando modelo...")

    # --- INICIO DE CORRECCIÓN DE RUTA ---
    # 2.A) Obtener la ruta del directorio 'Parte1' (que es el padre de 'modulo5')
    #      __file__ es la ruta de este script (main.py)
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    # 2.B) Construir la ruta completa al archivo de datos
    ruta_costos = os.path.join(BASE_DIR, 'Datos', 'costoAño.xlsx')
    
    # 2.C) Usar la ruta robusta
    resultados = m5.correr_modelo(ruta_costos, variables)
    # --- FIN DE CORRECCIÓN ---
    
    regiones = resultados["regiones"]
    info_subsidio = resultados["info_subsidio"]
    sstc = resultados["sstc_mensual"]      # (3, N_meses)
    energia = resultados["energia_mensual"] # (3, N_meses)
    lcoe = resultados["lcoe_mensual"]      # (3, N_meses)
    
    N_meses = sstc.shape[1]
    N_años = N_meses // 12
    
    print(f"Modelo ejecutado: {N_meses} meses ({N_años} años)")
    
    # 3) Mostrar subsidio si aplica
    if variables.get("use_dynamic_subsidy", False):
        print("\n" + "=" * 70)
        print("SUBSIDIO APLICADO")
        print("=" * 70)
        print(f"Fuente: {info_subsidio['fuente']}")
        print(tabla_subsidio(info_subsidio, regiones))
    
    # 4) Tablas de resultados mensuales (primeros 12 meses)
    print("\n" + "=" * 70)
    print("SSTC MENSUAL [USD] - Primeros 12 meses")
    print("=" * 70)
    print(tabla_sstc_mensual(sstc, regiones, meses_mostrar=12))
    
    print("\n" + "=" * 70)
    print("ENERGÍA MENSUAL [kWh/mes por hogar] - Primeros 12 meses")
    print("=" * 70)
    print(tabla_energia_mensual(energia, regiones, meses_mostrar=12))
    
    print("\n" + "=" * 70)
    print("LCOE MENSUAL [USD/kWh] - Primeros 12 meses")
    print("=" * 70)
    print(tabla_lcoe_mensual(lcoe, regiones, meses_mostrar=12))
    
    # 5) Detalles con ecuaciones (primeros 3 meses)
    print("\n" + "=" * 70)
    detalle_sstc_mensual(resultados, meses_mostrar=3)
    
    print("\n" + "=" * 70)
    detalle_lcoe_mensual(resultados, meses_mostrar=3)
    
    # 6) Estadísticas
    print("\n" + "=" * 70)
    estadisticas_resumen(sstc, lcoe, regiones)
    
    print("\n" + "=" * 70)
    estadisticas_anualizadas(sstc, lcoe, energia, regiones)
    
    # 7) Último año (meses finales)
    print("\n" + "=" * 70)
    print(f"LCOE MENSUAL - Último año (meses {N_meses-12} a {N_meses-1})")
    print("=" * 70)
    print(tabla_lcoe_mensual(lcoe, regiones, meses_mostrar=12, mes_inicio=N_meses-12))
    
    print("\n" + "=" * 70)
    print("MODELO COMPLETADO")
    print("=" * 70)


if __name__ == "__main__":
    main()