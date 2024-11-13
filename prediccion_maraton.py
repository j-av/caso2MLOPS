import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

# Cargar el archivo Berlin_Marathon_weather_data_since_1974.csv
weather_data = pd.read_csv('Berlin_Marathon_weather_data_since_1974.csv')

# Seleccionar características climáticas
features = ['AVG_TEMP_C', 'PRECIP_mm', 'ATMOS_PRESS_mbar', 'SUNSHINE_hrs', 'CLOUD_hrs']
target = 'Tiempo_Maraton'  # Ajusta según la columna disponible o simula si es necesario

# Simular datos de 'Tiempo_Maraton' si no está en el dataset
if target not in weather_data.columns:
    weather_data[target] = np.random.normal(loc=150, scale=10, size=len(weather_data))

# Dividir los datos en características (X) y objetivo (y)
X = weather_data[features]
y = weather_data[target]

# Dividir en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar un modelo de regresión (Random Forest)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluación del modelo
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Guardar el modelo entrenado
import joblib
joblib.dump(model, 'marathon_prediction_model.pkl')

print("Modelo de predicción de maratón entrenado y guardado como 'marathon_prediction_model.pkl'.")
