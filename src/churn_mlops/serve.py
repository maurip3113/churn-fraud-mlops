"""Fábrica de la API de predicción: build_app(usecase) arma una app FastAPI
distinta según el caso de uso (columnas, modelo y umbral propios), sobre el
mismo motor de serving."""

import json
import logging

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from churn_mlops.config import settings
from churn_mlops.usecases.base import UseCase

logger = logging.getLogger(__name__)


def build_app(usecase: UseCase) -> FastAPI:
    app = FastAPI(title=f"{usecase.name.capitalize()} Prediction API")

    ruta_modelo = settings.model_path(usecase.name)
    ruta_umbral = settings.threshold_path(usecase.name)

    estado = {"modelo": None, "umbral": None}

    def get_modelo():
        if estado["modelo"] is None:
            if not ruta_modelo.exists():
                raise HTTPException(
                    status_code=503,
                    detail=(
                        f"Modelo no encontrado en {ruta_modelo}. "
                        f"Corré train.py --usecase {usecase.name} y aprobar_modelo.py primero."
                    ),
                )
            estado["modelo"] = joblib.load(ruta_modelo)
            logger.info("Modelo cargado desde %s", ruta_modelo)
        return estado["modelo"]

    def get_umbral() -> float:
        if estado["umbral"] is None:
            if ruta_umbral.exists():
                estado["umbral"] = json.loads(ruta_umbral.read_text())["umbral"]
                logger.info(
                    "Umbral de decisión cargado desde %s: %.2f", ruta_umbral, estado["umbral"]
                )
            else:
                estado["umbral"] = settings.prediction_threshold_default
                logger.info(
                    "Sin umbral optimizado guardado, usando default: %.2f", estado["umbral"]
                )
        return estado["umbral"]

    @app.get("/")
    def home():
        return {"status": "ok", "usecase": usecase.name, "modelo": usecase.registered_model_name}

    @app.get("/health")
    def health():
        existe = ruta_modelo.exists()
        return {
            "status": "ok" if existe else "modelo_faltante",
            "usecase": usecase.name,
            "model_path": str(ruta_modelo),
            "umbral_decision": get_umbral(),
        }

    @app.post("/predict")
    def predecir(item: usecase.request_model):
        modelo = get_modelo()
        umbral = get_umbral()
        datos = pd.DataFrame([item.model_dump()])[usecase.features]

        probabilidad = modelo.predict_proba(datos)[0][1]
        prediccion = int(probabilidad > umbral)

        return {
            "positivo": bool(prediccion),
            "probabilidad": round(float(probabilidad), 3),
            "umbral_usado": umbral,
        }

    return app
