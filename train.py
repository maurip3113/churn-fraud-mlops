"""
PASOS 2 y 3: Entrenamiento del modelo + tracking de experimentos con MLflow.

Corré 'python prepare_data.py' primero para generar los CSV de entrenamiento.

Cada corrida registra en MLflow: hiperparámetros, métricas (accuracy, F1,
precision, recall) y el pipeline entrenado (versionado en el Model Registry).
Al final, se elige automáticamente el mejor modelo entre TODOS los runs
históricos del experimento (según settings.model_selection_metric, default
recall) y se promueve a alias "champion" — ese es el que queda copiado en
models/modelo_actual.pkl para que serve.py lo sirva.

Para ver los resultados en una interfaz visual, corré en otra terminal:
    mlflow ui
y abrí http://localhost:5000
"""

import logging

import pandas as pd

from churn_mlops.config import settings
from churn_mlops.logging_config import setup_logging
from churn_mlops.train import run_experiments, seleccionar_mejor_modelo

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    setup_logging()

    ruta_train = settings.data_dir / "clientes_entrenamiento.csv"
    if not ruta_train.exists():
        raise SystemExit(f"No se encontró {ruta_train}. Corré 'python prepare_data.py' primero.")

    df_train = pd.read_csv(ruta_train)
    run_experiments(df_train)

    ganador = seleccionar_mejor_modelo()
    logger.info(
        "=== Modelo elegido para servir: '%s' (v%s) — %s=%.4f ===",
        ganador["run_name"], ganador["version"], ganador["metric"], ganador["metric_value"],
    )
