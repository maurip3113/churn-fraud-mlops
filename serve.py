"""
PASO 5: Serving — exponer el modelo entrenado como API REST.

CÓMO CORRERLO
-------------
uvicorn serve:app --reload

Luego probás en el navegador: http://localhost:8000/docs
(FastAPI genera documentación interactiva automáticamente)
"""

from churn_mlops.logging_config import setup_logging
from churn_mlops.serve import app

setup_logging()

__all__ = ["app"]
