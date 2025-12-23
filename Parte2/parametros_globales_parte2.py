# -*- coding: utf-8 -*-
import numpy as np

# Configuración del tiempo (26 años * 12 meses)
PROJECT_LIFETIME_MONTHS = 312
REGIONES = ("Norte", "Centro", "Sur")

# ---Parámetros modulo demanda y adopción ---
DEMANDA_ADOPCION_INICIALES = {
    "population_initial": np.array([2563680, 13078400, 4136570]), 
    "ev_stock_initial": np.array([828, 10999, 1517]),       
    "icev_stock_initial": np.array([745762, 3686350, 1420100]), 
    
    "population_growth_rate": np.array([0.014172, 0.010671, 0.006716]),
    "motorization_rate": np.array([3.168, 2.8525, 2.9216]),
    
    "ev_lifetime": 8.0,
    "icev_lifetime": 8.0,
    "time_delay": 1.0,
    
    "tf": PROJECT_LIFETIME_MONTHS,
    "dt": 1.0
}

# --- Parámetros para el Módulo de Utilidad EV ---

UTILIDAD_EV_PARAMETROS = {
    # Rel. importance of Infrastructure readiness 
    "rel_importance_infra": np.array([0.000266909, 0.000314744, 0.564]),
    
    # Rel. importance of Charging Refuelling Time 
    "rel_importance_charging_time": np.array([-2.84744, -5.12323, -2.91483]),
    
    # Relative importance of Driving Range 
    "rel_importance_driving_range": np.array([0.0942897, 0.0974696, 0.00795017]),
    
    # Relative importance of TCO Factor 
    "rel_importance_tco": np.array([-0.011259, -0.0236662, -0.0168225]),
    
    # Datos para Tiempo de Carga e Infraestructura
    "charging_station_per_ev": 0.04, 
    "inital_quantity_public_charges": np.array([39,340,84]),
    "time_delay_cg": 0.532634,       
    "charging_station_power_rate": 22, 
    "ev_charging_loss": 0.163,        
    "ev_battery_capacity": 1848.0,   
    "ev_charge_time_improve_rate": 0.0670057, 
    
    # Datos para Autonomía (Driving Range)
    "initial_driving_range_ev": 300.0, 
    "dr_growth_rate_ev": 0.0986,      
    
    # Datos para Costos (TCO)
    "discount_rate": 0.06,   
    "price_reduction_rate ICEV": (-0.07),    
    "anual_technical_revision_cost_ev": 31.31,
    "anual_insurance_cost_EV": 6.7,
    "maintance_cost_per_km": 0.037,
    "ev_appraisal_SII": 25960.5,    
    "ev_subsidies_percent": 0.0,       # [cite: 52]
    "yearly_km_traveled": 492.0,    # 41 * 12 [cite: 59]
    "charging_price": 0.026
}