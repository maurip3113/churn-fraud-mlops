"""
PASO 5: Serving — exponer el modelo entrenado como API REST.

uvicorn importa este módulo directamente (no puede recibir argparse), así
que el caso de uso se elige con la variable de entorno USECASE (default:
"churn") — seteala en .env o al vuelo:

CÓMO CORRERLO
-------------
uvicorn serve:app --reload                       # sirve settings.usecase (default "churn")
USECASE=fraude uvicorn serve:app --reload         # sirve el caso de uso "fraude"

Luego probás en el navegador: http://localhost:8000/docs
(FastAPI genera documentación interactiva automáticamente)
"""

from churn_mlops.config import settings
from churn_mlops.logging_config import setup_logging
from churn_mlops.serve import build_app
from churn_mlops.usecases.registry import get_usecase

setup_logging()

app = build_app(get_usecase(settings.usecase))

__all__ = ["app"]
