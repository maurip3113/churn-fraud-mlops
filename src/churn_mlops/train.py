"""Entrenamiento del modelo de churn + tracking de experimentos con MLflow.

El preprocesamiento (one-hot encoding de las variables categóricas) vive
dentro de un sklearn Pipeline junto con el clasificador, así el modelo
versionado en MLflow y el .pkl para serving reciben directamente las
columnas crudas del dataset — no hay lógica de features duplicada en serve.py.
"""

import logging

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from churn_mlops.config import settings
from churn_mlops.data import CAT_FEATURES, FEATURES, NUM_FEATURES, TARGET

logger = logging.getLogger(__name__)


def build_pipeline(n_estimators: int, max_depth: int) -> Pipeline:
    preprocesador = ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUM_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
        ]
    )
    clasificador = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        class_weight="balanced",  # el dataset real tiene ~20% de churn positivo
    )
    return Pipeline([("preprocesador", preprocesador), ("clasificador", clasificador)])


def entrenar(df_train: pd.DataFrame, n_estimators: int = 100, max_depth: int = 6) -> dict:
    X = df_train[FEATURES]
    y = df_train[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    with mlflow.start_run():
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("n_train_samples", len(X_train))

        pipeline = build_pipeline(n_estimators, max_depth)
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        metricas = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
        }
        for nombre, valor in metricas.items():
            mlflow.log_metric(nombre, valor)

        mlflow.sklearn.log_model(
            pipeline,
            "modelo",
            registered_model_name="churn_predictor",
        )

        settings.models_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, settings.model_path)

        settings.data_dir.mkdir(parents=True, exist_ok=True)
        X_train[NUM_FEATURES].describe().to_csv(settings.train_stats_path)

        run_id = mlflow.active_run().info.run_id
        logger.info("Run ID: %s", run_id)
        logger.info("Métricas: %s", metricas)

    return metricas


def run_experiments(df_train: pd.DataFrame) -> None:
    mlflow.set_experiment(settings.mlflow_experiment_name)

    logger.info("=== Experimento 1: modelo base ===")
    entrenar(df_train, n_estimators=100, max_depth=6)

    logger.info("=== Experimento 2: más árboles, más profundidad ===")
    entrenar(df_train, n_estimators=200, max_depth=10)

    logger.info("=== Experimento 3: modelo más simple ===")
    entrenar(df_train, n_estimators=50, max_depth=3)

    logger.info("Corré 'mlflow ui' para comparar los 3 experimentos visualmente.")
