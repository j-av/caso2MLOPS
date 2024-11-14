import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib
import math

# Cargar los datos de maratón y clima
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

# Conectar a la base de datos para obtener el score de esfuerzo de cada atleta
import sqlite3
conn = sqlite3.connect('entrenos.db')
query = 'SELECT atleta_id, effort_score FROM entrenos'
effort_data = pd.read_sql(query, conn)
conn.close()

# Añadir el effort_score al dataset combinado
combined_data['effort_score'] = effort_data['effort_score']

# Seleccionar características y el objetivo (sin presión atmosférica ni horas de sol)
features = ['effort_score', 'AVG_TEMP_C', 'PRECIP_mm']
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
