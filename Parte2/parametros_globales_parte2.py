# -*- coding: utf-8 -*-
import numpy as np

UTILIDAD_EV_PARAMETROS = {
    # --- Economía ---
    "discount_rate": 0.06,
    "price_reduction_rate_ev": -0.136186,
    "initial_base_price_ev": 19475.0,
    "ev_appraisal_SII": 25960.5,
    "ev_subsidies_percent": 0.0,
    "rel_importance_tco": np.array([-0.011259, -0.0236662, -0.0168225]),
    
    # --- Costos Operativos ---
    "anual_insurance_cost_EV": 6.7,
    "anual_technical_revision_cost_ev": 31.31,
    "charging_price": 0.026,
    "yearly_km_traveled": 492.0, # (41 * 12)
    
    # --- Autonomía (Driving Range) ---
    "initial_driving_range_ev": 300.0,
    "dr_growth_rate_ev": 0.0986,
    "rel_importance_driving_range": np.array([0.0942897, 0.0974696, 0.00795017]),
    
    # --- Tiempo de Carga (Charging Time) ---
    "initial_charging_time_ev": 82.88,
    "ev_charge_time_improve_rate": 0.0670057,
    "rel_importance_charging_time": np.array([-2.84744, -5.12323, -2.91483]), # Ajusta según tu Vensim
    
    # --- Infraestructura ---
    "inital_quantity_public_charges": np.array([39, 340, 84]),
    # --- Dentro de UTILIDAD_EV_PARAMETROS en parametros_globales_parte2.py ---
    "rel_importance_infra": np.array([0.000266909, 0.000314744, 0.564]),
}

DEMANDA_ADOPCION_INICIALES = {
    "ev_stock_initial": np.array([1200.0, 10999.0, 1500.0]),
    "icev_stock_initial": np.array([745762.0, 3686350.0, 1420100.0]), # Valores corregidos
    "population_initial": np.array([2.56368e+06,1.30784e+07,4.13657e+06]),
    "population_growth_rate": 0.0143146,
    "motorization_rate": 2.8525  # Valor corregido: ahora el target será ~21M de vehículos
}
UTILIDAD_ICEV_PARAMETROS = {
    "initial_icev_price": 10240.1,
    "price_reduction_rate_icev": -0.07,
    # Reutilizamos las importancias relativas que ya definiste para el TCO
    "rel_importance_tco": np.array([-0.011259, -0.0236662, -0.0168225])
}