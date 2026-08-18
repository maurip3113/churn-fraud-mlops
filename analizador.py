"""
Interfaz web: subís un dataset (CSV), elegís la columna target de una
lista, y te devuelve un perfil estadístico (EDA) + un RandomForest de
referencia entrenado al vuelo, con un resumen en lenguaje natural
generado por Ollama (o Claude si LLM_PROVIDER=anthropic en .env).

Es una herramienta de exploración rápida, separada del pipeline de
producción (churn/fraude): no registra nada en MLflow ni pasa por el gate
de aprobación de aprobar_modelo.py.

CÓMO CORRERLO
-------------
uvicorn analizador:app --port 8080
abrir http://localhost:8080
"""

from churn_mlops.logging_config import setup_logging
from churn_mlops.web_analizador import build_app

setup_logging()

app = build_app()

__all__ = ["app"]
