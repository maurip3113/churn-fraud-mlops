"""
PASOS 2 y 3: Entrenamiento del modelo + tracking de experimentos con MLflow.

Corré 'python prepare_data.py' primero para generar los CSV de entrenamiento.

Cada corrida registra en MLflow: hiperparámetros, métricas (accuracy, F1,
precision, recall) y el pipeline entrenado (versionado en el Model Registry).

Para ver los resultados en una interfaz visual, corré en otra terminal:
    mlflow ui
y abrí http://localhost:5000
"""

import pandas as pd

from churn_mlops.config import settings
from churn_mlops.logging_config import setup_logging
from churn_mlops.train import run_experiments

if __name__ == "__main__":
    setup_logging()

    ruta_train = settings.data_dir / "clientes_entrenamiento.csv"
    if not ruta_train.exists():
        raise SystemExit(f"No se encontró {ruta_train}. Corré 'python prepare_data.py' primero.")

    df_train = pd.read_csv(ruta_train)
    run_experiments(df_train)
