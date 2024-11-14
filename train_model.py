import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib
import math

# Paso 1: Cargar el CSV que contiene los datos necesarios para calcular el Effort Score
marathon_data_extra = pd.read_csv('MarathonData.csv')

# Convertir columnas a tipo numérico
marathon_data_extra['km4week'] = pd.to_numeric(marathon_data_extra['km4week'], errors='coerce')
marathon_data_extra['sp4week'] = pd.to_numeric(marathon_data_extra['sp4week'], errors='coerce')
marathon_data_extra['Wall21'] = pd.to_numeric(marathon_data_extra['Wall21'], errors='coerce')
marathon_data_extra['CrossTraining'] = pd.to_numeric(marathon_data_extra['CrossTraining'], errors='coerce')

# Factores para calcular el Effort Score
category_factors = {
    'MAM': 1.0,    # Male Athletes under 40
    'WAM': 1.1,    # Women under 40
    'M40': 1.05    # Male Athletes between 40 and 45 years
}
cross_training_factor = 0.05

# Función para calcular el Effort Score
def calcular_score_entreno(row):
    category_factor = category_factors.get(row['Category'], 1.0)  # 1.0 por defecto
    wall_penalization = row['Wall21'] / 10 if pd.notnull(row['Wall21']) else 0
    effort_score = (row['km4week'] * row['sp4week']) * category_factor * (1 + cross_training_factor * row['CrossTraining']) * (1 - wall_penalization)
    return effort_score

# Calcular el Effort Score y añadirlo al DataFrame
marathon_data_extra['effort_score'] = marathon_data_extra.apply(calcular_score_entreno, axis=1)

# Paso 2: Cargar los datos de clima y maratón
marathon_data = pd.read_csv('Berlin_Marathon_data_1974_2019.csv', low_memory=False)
weather_data = pd.read_csv('Berlin_Marathon_weather_data_since_1974.csv', low_memory=False)

# Convertir la columna 'YEAR' a formato de fecha para unir los datos
marathon_data['YEAR'] = pd.to_datetime(marathon_data['YEAR'], format='%Y')
weather_data['YEAR'] = pd.to_datetime(weather_data['YEAR'], format='%Y')

# Convertir la columna 'TIME' de formato HH:MM:SS a minutos
def convertir_tiempo_minutos(tiempo):
    try:
        h, m, s = map(int, tiempo.split(':'))
        return h * 60 + m + s / 60
    except ValueError:
        return np.nan

marathon_data['TIME'] = marathon_data['TIME'].apply(convertir_tiempo_minutos)

# Unir los datos de maratón y clima por año
combined_data = pd.merge(marathon_data, weather_data, on='YEAR', how='inner')

# Añadir el Effort Score al dataset combinado usando el ID del atleta para la combinación
combined_data = pd.merge(combined_data, marathon_data_extra[['id', 'effort_score']], on='id', how='inner')

# Seleccionar características y el objetivo
features = ['effort_score', 'AVG_TEMP_C', 'PRECIP_mm', 'ATMOS_PRESS_mbar']
target = 'TIME'

# Eliminar filas con valores nulos en las características seleccionadas
combined_data = combined_data.dropna(subset=features + [target])

# Dividir los datos en X (características) y y (objetivo)
X = combined_data[features]
y = combined_data[target]

# Dividir en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar un modelo de regresión para predecir el tiempo de maratón
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Calcular el error cuadrático medio (Root Mean Squared Error)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = math.sqrt(mse)
print(f"Root Mean Squared Error del modelo: {rmse:.2f} minutos")

# Guardar el modelo entrenado
joblib.dump(model, 'marathon_prediction_model.pkl')
print("Modelo entrenado y guardado en 'marathon_prediction_model.pkl'")
