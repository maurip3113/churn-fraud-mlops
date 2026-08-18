"""
PASOS 2 y 3: Entrenamiento del modelo + tracking de experimentos con MLflow.

Corré 'python prepare_data.py' primero para generar los CSV de entrenamiento.

Cada corrida registra en MLflow: hiperparámetros, métricas (accuracy, F1,
precision, recall) y el pipeline entrenado (versionado en el Model Registry).
Al final, se identifica automáticamente el mejor candidato entre TODOS los
runs históricos del experimento (según settings.model_selection_metric,
default recall) y queda pendiente de aprobación en
models/candidato_pendiente.json.

Este script NO promueve el modelo a producción por sí solo — correr
'python aprobar_modelo.py' para revisarlo y confirmar el reemplazo de
models/modelo_actual.pkl (lo que sirve serve.py).

Para ver los resultados en una interfaz visual, corré en otra terminal:
    mlflow ui
y abrí http://localhost:5000
"""

import logging

import pandas as pd

from churn_mlops.config import settings
from churn_mlops.logging_config import setup_logging
from churn_mlops.train import identificar_mejor_candidato, run_experiments

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    setup_logging()

    ruta_train = settings.data_dir / "clientes_entrenamiento.csv"
    if not ruta_train.exists():
        raise SystemExit(f"No se encontró {ruta_train}. Corré 'python prepare_data.py' primero.")

    df_train = pd.read_csv(ruta_train)
    run_experiments(df_train)

    candidato = identificar_mejor_candidato()
    logger.info(
        "=== Candidato pendiente de aprobación: '%s' (v%s) — %s=%.4f ===",
        candidato["run_name"], candidato["version"], candidato["metric"],
        candidato["metric_value"],
    )
    logger.info("Corré 'python aprobar_modelo.py' para revisarlo y promoverlo a producción.")
