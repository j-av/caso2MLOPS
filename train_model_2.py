import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import joblib

# Cargar los datos
marathon_data = pd.read_csv('Berlin_Marathon_data_1974_2019.csv')
weather_data = pd.read_csv('Berlin_Marathon_weather_data_since_1974.csv')

# Preprocesamiento
marathon_results = marathon_data[['YEAR', 'COUNTRY', 'GENDER', 'AGE', 'TIME']].copy()
marathon_results.rename(columns={
    'YEAR': 'year', 'COUNTRY': 'country', 'GENDER': 'gender', 'AGE': 'age', 'TIME': 'marathon_time'
}, inplace=True)
marathon_results['marathon_time'] = pd.to_timedelta(marathon_results['marathon_time'], errors='coerce').dt.total_seconds() / 60
marathon_results['gender'] = marathon_results['gender'].apply(lambda x: 1 if x == 'M' else 0)

weather_conditions = weather_data[['YEAR', 'PRECIP_mm', 'SUNSHINE_hrs', 'CLOUD_hrs', 'ATMOS_PRESS_mbar', 
                                   'AVG_TEMP_C', 'MAX_TEMP_C', 'MIN_TEMP_C']].copy()
weather_conditions.rename(columns={
    'YEAR': 'year', 'PRECIP_mm': 'precipitation_mm', 'SUNSHINE_hrs': 'sunshine_hours',
    'CLOUD_hrs': 'cloud_hours', 'ATMOS_PRESS_mbar': 'atmospheric_pressure_mbar', 
    'AVG_TEMP_C': 'avg_temp_c', 'MAX_TEMP_C': 'max_temp_c', 'MIN_TEMP_C': 'min_temp_c'
}, inplace=True)

merged_data = pd.merge(marathon_results, weather_conditions, on='year', how='inner')
features = ['age', 'gender', 'precipitation_mm', 'atmospheric_pressure_mbar', 'avg_temp_c']
X = merged_data[features].apply(pd.to_numeric, errors='coerce').dropna()
y = merged_data.loc[X.index, 'marathon_time'].apply(pd.to_numeric, errors='coerce').dropna()
X = X.loc[y.index]

# Entrenamiento del modelo
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Guardar el modelo
joblib.dump(model, 'trained_model_2.pkl')
print("Modelo 2 guardado en 'trained_model_2.pkl'")

