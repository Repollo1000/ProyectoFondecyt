# 🧩 Módulo 10 — Balance Energético

Este módulo integra los resultados del **Módulo 5 (LCOE)** con el cálculo del **balance energético horario o simplificado**, permitiendo estimar el **autoconsumo**, la **inyección a red** y el **ahorro económico** bajo diferentes políticas energéticas:  
**Net Billing**, **Net Metering** y **Feed-in Tariff**.

---

## ⚙️ Carga de Datos

Durante la ejecución del script principal (`main_mod10.py`), se cargan los siguientes archivos desde la carpeta `Datos/`:

| Archivo | Descripción | Formato |
|----------|--------------|----------|
| `precio_electricidad_vf.xlsx` | Contiene los precios de compra e inyección eléctrica (columnas `low1` y `low2`). | Excel |
| `curva_de_carga.xlsx` | Perfil horario de consumo residencial (mes, hora, regiones). | Excel |
| `Factor_capacidad_solar.csv` | Factores de capacidad solar por hora y mes (Antofagasta, Santiago, Puerto Montt). | CSV |

---

### 📂 Sección del Código donde se Cargan los Datos

Los archivos se cargan en la sección:

```python
# ==========================================
# PASO 2: CARGAR DATOS DE BALANCE - Se hace UNA VEZ
# ==========================================

# Archivo de precios
ruta_precios = os.path.join(DATOS_DIR, NOMBRE_ARCHIVO_PRECIOS)
df_precios = pd.read_excel(ruta_precios)

# Perfil de consumo
ruta_consumo = os.path.join(DATOS_DIR, NOMBRE_ARCHIVO_CONSUMO_HORARIO)
df_consumo_horario = pd.read_excel(ruta_consumo)

# Perfil de generación
ruta_generacion = os.path.join(DATOS_DIR, NOMBRE_ARCHIVO_GENERACION_HORARIO)
df_generacion_horario = pd.read_csv(ruta_generacion, sep=';', encoding='latin-1')
