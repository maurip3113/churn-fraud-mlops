"""API de predicción de churn, servida sobre el pipeline entrenado (sklearn)."""

import logging
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from churn_mlops.config import settings
from churn_mlops.data import FEATURES

logger = logging.getLogger(__name__)

app = FastAPI(title="Churn Prediction API")

_modelo = None


def get_modelo():
    global _modelo
    if _modelo is None:
        if not settings.model_path.exists():
            raise HTTPException(
                status_code=503,
                detail=f"Modelo no encontrado en {settings.model_path}. Corré train.py primero.",
            )
        _modelo = joblib.load(settings.model_path)
        logger.info("Modelo cargado desde %s", settings.model_path)
    return _modelo


SiNo = Literal["Yes", "No"]


class Cliente(BaseModel):
    gender: Literal["Male", "Female"]
    SeniorCitizen: Literal[0, 1]
    Partner: SiNo
    Dependents: SiNo
    tenure: int
    PhoneService: SiNo
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: SiNo
    PaymentMethod: Literal[
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ]
    MonthlyCharges: float
    TotalCharges: float


@app.get("/")
def home():
    return {"status": "ok", "modelo": "churn_predictor v1"}


@app.get("/health")
def health():
    existe = settings.model_path.exists()
    return {"status": "ok" if existe else "modelo_faltante", "model_path": str(settings.model_path)}


@app.post("/predict")
def predecir(cliente: Cliente):
    modelo = get_modelo()
    datos = pd.DataFrame([cliente.model_dump()])[FEATURES]

    probabilidad = modelo.predict_proba(datos)[0][1]
    prediccion = int(probabilidad > 0.5)

    return {
        "va_a_abandonar": bool(prediccion),
        "probabilidad_churn": round(float(probabilidad), 3),
    }
