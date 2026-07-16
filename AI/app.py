from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import numpy as np

app = FastAPI(title="Crop Recommendation API")

model = joblib.load("model/crop_model.pkl")
encoder = joblib.load("model/label_encoder.pkl")


#request
class CropInput(BaseModel):
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float

#API
@app.get("/")
def home():
    return {
        "message": "Crop Recommendation API is Running"
    }

@app.post("/predict")
def predict(data: CropInput):

    sample = pd.DataFrame([{
        "N": data.N,
        "P": data.P,
        "K": data.K,
        "temperature": data.temperature,
        "humidity": data.humidity,
        "ph": data.ph,
        "rainfall": data.rainfall
    }])

    prediction = model.predict(sample)[0]
    probability = model.predict_proba(sample)[0]

    crop = encoder.inverse_transform([prediction])[0]

    confidence = float(np.max(probability) * 100)

    top3_index = np.argsort(probability)[::-1][:3]

    top3 = []

    for idx in top3_index:
        top3.append({
            "crop": encoder.inverse_transform([idx])[0],
            "confidence": round(float(probability[idx] * 100), 2)
        })

    return {
        "recommendation": crop,
        "confidence": round(confidence,2),
        "top3": top3
    }