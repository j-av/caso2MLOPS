from fastapi import FastAPI, HTTPException
import pandas as pd
import joblib
import uvicorn

app = FastAPI()

# Cargar los modelos al inicio
try:
    model_effort_score = joblib.load('trained_model_1.pkl')  # Modelo basado en score de esfuerzo
    model_weather_conditions = joblib.load('trained_model_2.pkl')  # Modelo basado en condiciones climáticas
except:
    model_effort_score = None
    model_weather_conditions = None

@app.get("/")
def read_root():
    return {"message": "Marathon Training API"}

@app.get("/calculate-effort/{km4week}/{sp4week}")
def calculate_effort(km4week: float, sp4week: float):
    effort_score = km4week * sp4week
    return {"effort_score": effort_score}

@app.post("/predict")
def predict(prediction_type: str, features: dict):
    # Validar el tipo de predicción solicitado
    if prediction_type == "effort_score":
        if model_effort_score is None:
            raise HTTPException(status_code=500, detail="Model based on effort score not loaded")
        
        # Verificar que las características necesarias estén presentes
        required_features = ['training_score', 'Category_MAM', 'Category_WAM']
        if not all(feature in features for feature in required_features):
            raise HTTPException(status_code=400, detail="Missing features for effort score model")

        # Convertir características en un DataFrame y hacer la predicción
        df = pd.DataFrame([features])
        prediction = model_effort_score.predict(df)[0]
        return {"predicted_time": float(prediction)}

    elif prediction_type == "weather_conditions":
        if model_weather_conditions is None:
            raise HTTPException(status_code=500, detail="Model based on weather conditions not loaded")

        # Verificar que las características necesarias estén presentes
        required_features = ['age', 'gender', 'precipitation_mm', 'atmospheric_pressure_mbar', 'avg_temp_c']
        if not all(feature in features for feature in required_features):
            raise HTTPException(status_code=400, detail="Missing features for weather conditions model")

        # Convertir características en un DataFrame y hacer la predicción
        df = pd.DataFrame([features])
        prediction = model_weather_conditions.predict(df)[0]
        return {"predicted_time": float(prediction)}

    else:
        raise HTTPException(status_code=400, detail="Invalid prediction type. Use 'effort_score' or 'weather_conditions'.")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
