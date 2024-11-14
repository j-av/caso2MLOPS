import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import joblib

# Cargar los datos
data = pd.read_csv('MarathonData.csv')

# Preprocesamiento
data['cross_training'] = data['CrossTraining'].apply(lambda x: 1 if x == 'yes' else 0)
data = pd.get_dummies(data, columns=['Category'], drop_first=True)
data['km4week'] = pd.to_numeric(data['km4week'], errors='coerce')
data['sp4week'] = pd.to_numeric(data['sp4week'], errors='coerce')
data['Wall21'] = pd.to_numeric(data['Wall21'], errors='coerce')
data.dropna(subset=['km4week', 'sp4week', 'Wall21'], inplace=True)
data['training_score'] = (data['km4week'] * data['sp4week']) + (data['cross_training'] * 50) - (data['Wall21'] * 100)

# Entrenamiento del modelo
X = data[['training_score'] + [col for col in data.columns if 'Category_' in col]]
y = data['MarathonTime']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Guardar el modelo
joblib.dump(model, 'trained_model_1.pkl')
print("Modelo 1 guardado en 'trained_model_1.pkl'")
