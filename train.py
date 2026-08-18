"""
PASOS 2 y 3: Entrenamiento del modelo + tracking de experimentos con MLflow.

Corré 'python prepare_data.py --usecase <nombre>' primero para generar los
CSV de entrenamiento.

Cada corrida registra en MLflow: hiperparámetros, métricas (accuracy, F1,
precision, recall) y el pipeline entrenado (versionado en el Model Registry).
Al final, se identifica automáticamente el mejor candidato entre TODOS los
runs históricos del experimento (según settings.model_selection_metric,
default recall) y queda pendiente de aprobación.

Este script NO promueve el modelo a producción por sí solo — correr
'python aprobar_modelo.py --usecase <nombre>' para revisarlo y confirmar
el reemplazo del modelo que sirve serve.py.

Para ver los resultados en una interfaz visual, corré en otra terminal:
    mlflow ui
y abrí http://localhost:5000

CÓMO CORRERLO
-------------
python train.py --usecase churn
python train.py --usecase fraude
"""

import argparse
import logging

import pandas as pd

from churn_mlops.config import settings
from churn_mlops.logging_config import setup_logging
from churn_mlops.train import identificar_mejor_candidato, run_experiments
from churn_mlops.usecases.registry import USECASES, get_usecase

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--usecase", default="churn", choices=sorted(USECASES))
    args = parser.parse_args()

    setup_logging()
    usecase = get_usecase(args.usecase)

    ruta_train = settings.train_csv_path(usecase.name)
    if not ruta_train.exists():
        raise SystemExit(
            f"No se encontró {ruta_train}. Corré 'python prepare_data.py "
            f"--usecase {usecase.name}' primero."
        )

    df_train = pd.read_csv(ruta_train)
    run_experiments(usecase, df_train)

    candidato = identificar_mejor_candidato(usecase)
    logger.info(
        "=== Candidato pendiente de aprobación: '%s' (v%s) — %s=%.4f ===",
        candidato["run_name"], candidato["version"], candidato["metric"],
        candidato["metric_value"],
    )
    logger.info(
        "Corré 'python aprobar_modelo.py --usecase %s' para revisarlo y promoverlo.",
        usecase.name,
    )
