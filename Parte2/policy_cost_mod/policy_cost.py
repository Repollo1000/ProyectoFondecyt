# -*- coding: utf-8 -*-

def calcular_politicas_y_costos(ev_sales, icev_sales, ev_base_price, icev_base_price,
                                crecimiento_real_cs, crecimiento_base_cs, costo_unitario_cs,
                                pct_subsidio_ev=0.0, impuesto_verde_icev=0.015):
    """
    Módulo 5: Calcula los costos y recaudaciones de las políticas públicas anualmente.
    """
    # 1. Subsidios a los EV (Según Vensim, actualmente es 0%)
    annual_subsidy = ev_base_price * ev_sales * pct_subsidio_ev
    
    # 2. Recaudación del Impuesto Verde a los ICEV
    annual_tax = impuesto_verde_icev * icev_base_price * icev_sales
    
    # 3. Costo de Política de Cargadores
    # Vensim permite que esto sea negativo si el crecimiento orgánico (base) 
    # es mayor que el crecimiento real inducido por la demanda.
    annual_cs_policy = costo_unitario_cs * (crecimiento_real_cs - crecimiento_base_cs)
    
    return {
        "annual_subsidy": annual_subsidy,
        "annual_tax": annual_tax,
        "annual_cs_policy": annual_cs_policy
    }