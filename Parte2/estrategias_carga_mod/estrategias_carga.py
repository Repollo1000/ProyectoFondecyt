# -*- coding: utf-8 -*-
import numpy as np

def generar_perfil_diario(estrategia):
    """
    Crea una Campana de Gauss de 24 horas. 
    Define 'a qué hora' es más probable que la gente enchufe el auto.
    """
    horas = np.arange(24)
    
    # np.exp(...) es la fórmula matemática de la Campana de Gauss
    if estrategia == "dumb":
        # Carga Descontrolada: Peak a las 19:30 hrs (19.5), dispersión de 2.5 horas
        perfil = np.exp(-0.5 * ((horas - 19.5) / 2.5)**2)
        
    elif estrategia == "smart":
        # Carga Inteligente: Peak a las 03:00 am (3.0), dispersión de 2.0 horas
        perfil = np.exp(-0.5 * ((horas - 3.0) / 2.0)**2)
        
    elif estrategia == "diurna":
        # Carga Diurna/Solar: Peak a las 13:00 hrs (13.0), dispersión de 3.0 horas
        perfil = np.exp(-0.5 * ((horas - 13.0) / 3.0)**2)
        
    else:
        perfil = np.ones(24)
        
    # LA REGLA DE ORO DE LA PROFE: Normalizamos la curva.
    # Esto asegura que la suma de las 24 horas dé exactamente 1.0 (el 100% de la energía diaria).
    # Así no inventamos ni perdemos energía.
    perfil_normalizado = perfil / np.sum(perfil)
    
    return perfil_normalizado

def calcular_impacto_red_mw(stock_ev, dias_mes=30):
    """
    Toma el stock de autos, calcula la energía y escupe el Peak de Megavatios (MW).
    """
    # 1. Traemos la constante de Vensim (el modelo manda)
    consumo_mensual_ev = 101.753 # kWh/mes
    
    # 2. Ración diaria por auto
    consumo_diario_ev = consumo_mensual_ev / dias_mes
    
    # 3. Energía de TODA la flota en la región ese día
    energia_total_diaria_kwh = stock_ev * consumo_diario_ev
    
    # 4. Obtenemos cómo se reparte esa energía según el comportamiento humano
    p_dumb = generar_perfil_diario("dumb")
    p_smart = generar_perfil_diario("smart")
    p_diurna = generar_perfil_diario("diurna")
    
    # 5. Multiplicamos la energía total por la tajada de cada hora.
    # Dividimos por 1000 para pasar de Kilovatios (kW) a Megavatios (MW)
    demanda_dumb_mw = (energia_total_diaria_kwh * p_dumb) / 1000.0
    demanda_smart_mw = (energia_total_diaria_kwh * p_smart) / 1000.0
    demanda_diurna_mw = (energia_total_diaria_kwh * p_diurna) / 1000.0
    
    # 6. Al Coordinador Eléctrico le importa el valor máximo (Peak) que estresa la red
    return {
        "peak_dumb": np.max(demanda_dumb_mw),
        "peak_smart": np.max(demanda_smart_mw),
        "peak_diurna": np.max(demanda_diurna_mw)
    }