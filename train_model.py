import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# Cargar el modelo entrenado
model = joblib.load("marathon_prediction_model.pkl")

# Inicializar FastAPI
app = FastAPI()

# Definir el esquema de entrada
class PredictionRequest(BaseModel):
    km4week: float
    sp4week: float
    category: str
    cross_training: int
    wall21: float
    avg_temp: float
    precip_mm: float

# Asignar factores de categoría
category_factors = {
    'MAM': 1.0,
    'WAM': 1.1,
    'M40': 1.05
}

cross_training_factor = 0.05

# Función para calcular el Effort Score
def calcular_score_entreno(km4week, sp4week, category, wall21, cross_training):
    category_factor = category_factors.get(category, 1.0)
    wall_penalization = wall21 / 10 if wall21 else 0
    effort_score = (km4week * sp4week) * category_factor * (1 + cross_training_factor * cross_training) * (1 - wall_penalization)
    return effort_score

# Ruta de predicción
@app.post("/predict")
def predict(request: PredictionRequest):
    # Calcular el Effort Score
    effort_score = calcular_score_entreno(request.km4week, request.sp4week, request.category, request.wall21, request.cross_training)

    # Crear el DataFrame de entrada
    input_data = pd.DataFrame({
        'effort_score': [effort_score],
        'AVG_TEMP_C': [request.avg_temp],
        'PRECIP_mm': [request.precip_mm]
    })

    # Realizar la predicción
    prediction = model.predict(input_data)[0]
    return {"predicted_time_minutes": prediction}
