import os
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from comet_ml import API
import joblib
import numpy as np

from dotenv import load_dotenv
load_dotenv()
COMET_API_KEY = os.getenv("COMET_API_KEY") 
COMET_WORKSPACE = os.getenv("COMET_WORKSPACE", "hugo-alvarez-3377")
COMET_PROJECT = os.getenv("COMET_PROJECT", "mlops-software-inteligente")

app = FastAPI(title="MLOps Credit Risk API - Comet.ml Edition", version="1.0")

model = None
model_version = "unknown"

class CreditRequest(BaseModel):
    features: list[float] = Field(..., min_items=20, max_items=20)

class CreditResponse(BaseModel):
    prediction: int
    risk_probability: float
    status: str
    model_version: str
    timestamp: str

@app.on_event("startup")
def download_champion_from_comet():
    """Conecta a la API de Comet, busca por Tag y descarga el modelo usando download_model"""
    global model, model_version
    print("Conectando con el repositorio central de Comet.ml...")
    
    try:
        api = API(api_key=COMET_API_KEY)
        experiments = api.get_experiments(COMET_WORKSPACE, COMET_PROJECT)
        
        champion_exp = None
        for exp in experiments:
            tags = exp.get_tags()
            if "production-champion" in tags and "archived" not in tags:
                champion_exp = exp
                break
                
        if not champion_exp:
            raise RuntimeError("Fallo crítico: No hay ningún experimento con el tag 'production-champion'")
            
        model_version = str(champion_exp.id)[:8]
        print(f"Champion localizado en Comet (ID: {model_version}). Descargando binario...")
        
        champion_exp.download_model(name="credit-risk-model", output_path="./")
        
        if os.path.exists("model.pkl"):
            model = joblib.load("model.pkl")
            print("¡Modelo cargado en memoria exitosamente desde Comet.ml!")
            os.remove("model.pkl")
        else:
            raise FileNotFoundError("El archivo model.pkl no se encontró tras la descarga de Comet.")
            
    except Exception as e:
        print(f"🚨 ERROR EN STARTUP: {e}")
        raise e

@app.post("/predict", response_model=CreditResponse)
def predict(request: CreditRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo no inicializado.")
    try:
        input_data = np.array(request.features).reshape(1, -1)
        prediction = int(model.predict(input_data)[0])
        risk_probability = float(model.predict_proba(input_data)[0][1])
        
        status = "High Risk Fraud Detected" if prediction == 1 else "Low Risk Approved"
        
        return CreditResponse(
            prediction=prediction,
            risk_probability=round(risk_probability, 4),
            status=status,
            model_version=model_version,
            timestamp=datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))