from fastapi import FastAPI
import pandas as pd
import joblib
import uvicorn

app = FastAPI()

# Cargar los modelos al inicio
try:
    model = joblib.load('marathon_prediction_model.pkl')
except:
    model = None

@app.get("/")
def read_root():
    return {"message": "Marathon Training API"}

@app.get("/calculate-effort/{km4week}/{sp4week}")
def calculate_effort(km4week: float, sp4week: float):
    effort_score = km4week * sp4week
    return {"effort_score": effort_score}

@app.get("/predict-marathon/{temp}/{precip}/{pressure}/{sunshine}/{cloud}")
def predict_marathon(temp: float, precip: float, pressure: float, sunshine: float, cloud: float):
    if model is None:
        return {"error": "Model not loaded"}
    
    features = [[temp, precip, pressure, sunshine, cloud]]
    prediction = model.predict(features)[0]
    return {"predicted_time": float(prediction)}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)