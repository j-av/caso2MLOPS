from fastapi import FastAPI, HTTPException
import pandas as pd
import joblib
import requests
import os

app = FastAPI()

# URLs de los modelos en GitHub Releases
MODEL_1_URL = "https://github.com/<j-av>/<caso2MLOPS>/releases/download/v1.0.0/trained_model_1.pkl"
MODEL_2_URL = "https://github.com/<j-av>/<caso2MLOPS>/releases/download/v1.0.0/trained_model_2.pkl"

# Función para descargar archivos
def download_model(url, output_path):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Model downloaded: {output_path}")
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        raise

# Descargar y cargar los modelos
try:
    if not os.path.exists("trained_model_1.pkl"):
        download_model(MODEL_1_URL, "trained_model_1.pkl")
    if not os.path.exists("trained_model_2.pkl"):
        download_model(MODEL_2_URL, "trained_model_2.pkl")

    # Cargar los modelos
    model_effort_score = joblib.load("trained_model_1.pkl")
    model_weather_conditions = joblib.load("trained_model_2.pkl")
except Exception as e:
    print(f"Error loading models: {e}")
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
    if prediction_type == "effort_score":
        if model_effort_score is None:
            raise HTTPException(status_code=500, detail="Model based on effort score not loaded")
        
        # Validar las características necesarias
        required_features = ['training_score', 'Category_MAM', 'Category_WAM']
        if not all(feature in features for feature in required_features):
            raise HTTPException(status_code=400, detail="Missing features for effort score model")

        # Hacer predicción
        df = pd.DataFrame([features])
        prediction = model_effort_score.predict(df)[0]
        return {"predicted_time": float(prediction)}

    elif prediction_type == "weather_conditions":
        if model_weather_conditions is None:
            raise HTTPException(status_code=500, detail="Model based on weather conditions not loaded")
        
        # Validar las características necesarias
        required_features = ['age', 'gender', 'precipitation_mm', 'atmospheric_pressure_mbar', 'avg_temp_c']
        if not all(feature in features for feature in required_features):
            raise HTTPException(status_code=400, detail="Missing features for weather conditions model")

        # Hacer predicción
        df = pd.DataFrame([features])
        prediction = model_weather_conditions.predict(df)[0]
        return {"predicted_time": float(prediction)}

    else:
        raise HTTPException(status_code=400, detail="Invalid prediction type. Use 'effort_score' or 'weather_conditions'.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
