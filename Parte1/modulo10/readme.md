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
```

---

## 🧮 Cálculo

En esta etapa el módulo toma los datos cargados (perfiles horarios, energía mensual del Módulo 5 y precios) y calcula, para **cada región** y **cada mes**, las siguientes magnitudes:

1. **Generación hora a hora del mes**  
   - A partir de la energía mensual generada (`energia_mensual[i, m]`) y del perfil horario de generación (`cf_horario[mes, hora]`).
   - Esto viene del CSV `Factor_capacidad_solar.csv`.

2. **Consumo hora a hora del mes**  
   - A partir del consumo mensual por hogar de la región (`consumo_mensual_hogar[i]`) y del perfil de consumo (`perfil_consumo[mes, hora]`).
   - Esto viene del Excel `curva_de_carga.xlsx`.

3. **Comparación generación vs consumo (hora a hora)**  
   - Si `generación_hora >= consumo_hora` → hay **autoconsumo** y **excedente**.  
   - Si `generación_hora < consumo_hora` → hay **solo autoconsumo** (no hay inyección).

4. **Autoconsumo mensual**  
   - Es la suma de toda la energía que el hogar pudo usar directamente de su generación en ese mes.

5. **Inyección mensual a la red**  
   - Es la suma de todos los excedentes horarios del mes.

6. **Ahorro por inyección**  
   - `inyeccion_mensual × tarifa_inyeccion`
   - La **tarifa de inyección** depende de la política elegida (Net Billing, Net Metering o Feed-in Tariff).

7. **Ahorro por autoconsumo**  
   - `autoconsumo_mensual × (precio_electricidad - lcoe)`
   - Representa lo que dejo de comprarle a la red, descontando mi costo de generación.

8. **Ahorro total mensual**  
   - `ahorro_total = ahorro_inyeccion + ahorro_autoconsumo`

9. **Cálculo separado por región**  
   - Todo lo anterior se hace para: **Norte**, **Centro** y **Sur**.

10. **Cálculo separado por mes**  
    - El modelo recorre todos los meses del horizonte del Módulo 5 (`N_meses`) y guarda matrices de tamaño `(3, N_meses)`.

---

### 📍 Ubicación en el Código

En el código, esta lógica está principalmente en:

- `balance_energetico_horario(...)` → caso con perfiles hora a hora  
- `balance_energetico_simple(...)` → caso 60/40  
- `calcular_ahorro_mensual(...)` → arma los dólares del mes  
- `calcular_balance_energetico(...)` → orquesta todo y aplica la **política**

---

## 🧩 Paso a Paso del Cálculo Horario

Para cada hora del mes seleccionado se hacen estas operaciones:

### Consumo horario
```python
consumo_hora = consumo_mensual * perfil_consumo[mes, hora]
```

### Generación horaria
```python
generacion_hora = generacion_mensual * cf_horario[mes, hora]
```

### Balance horario
```python
diferencia = generacion_hora - consumo_hora

if diferencia > 0:
    # Hay excedente → se inyecta
    autoconsumo_hora = consumo_hora
    inyeccion_hora = diferencia
    demanda_red = 0.0
else:
    # Hay déficit → se toma desde la red
    autoconsumo_hora = generacion_hora
    inyeccion_hora = 0.0
    demanda_red = -diferencia   # equivalente a (consumo_hora - generacion_hora)
```

### Acumulación mensual
```python
autoconsumo_total += autoconsumo_hora
inyeccion_total += inyeccion_hora
```

### Cálculo de ahorros
```python
ahorro_inyeccion = inyeccion * tarifa_inyeccion
ahorro_autoconsumo = autoconsumo * (precio_electricidad - lcoe)
ahorro_total = ahorro_inyeccion + ahorro_autoconsumo
```