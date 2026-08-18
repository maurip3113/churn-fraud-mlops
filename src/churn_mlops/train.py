"""Entrenamiento del modelo de churn + tracking de experimentos con MLflow.

El preprocesamiento (one-hot encoding de las variables categóricas) vive
dentro de un sklearn Pipeline junto con el clasificador, así el modelo
versionado en MLflow y el .pkl para serving reciben directamente las
columnas crudas del dataset — no hay lógica de features duplicada en serve.py.
"""

import json
import logging

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
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


def optimizar_umbral(
    y_true, y_proba, costo_falso_negativo: float, costo_falso_positivo: float
) -> dict:
    """Busca, por grilla, el umbral de decisión que minimiza el costo de negocio.

    costo_total(umbral) = falsos_negativos * costo_falso_negativo
                         + falsos_positivos * costo_falso_positivo

    Con costo_falso_negativo > costo_falso_positivo (el caso típico en
    churn: perder un cliente sale más caro que una promo de retención de
    más), el óptimo tiende a un umbral menor a 0.5 — el modelo se vuelve
    más "alerta" a costa de más falsos positivos.
    """
    mejor = None
    for umbral in np.linspace(0.05, 0.95, 91):
        y_pred = (y_proba >= umbral).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        costo_total = fn * costo_falso_negativo + fp * costo_falso_positivo
        if mejor is None or costo_total < mejor["costo_total"]:
            mejor = {
                "umbral": round(float(umbral), 2),
                "costo_total": float(costo_total),
                "falsos_negativos": int(fn),
                "falsos_positivos": int(fp),
            }
    return mejor


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

        y_proba = pipeline.predict_proba(X_test)[:, 1]
        umbral_optimo = optimizar_umbral(
            y_test, y_proba, settings.costo_falso_negativo, settings.costo_falso_positivo
        )
        mlflow.log_param("umbral_optimo", umbral_optimo["umbral"])
        mlflow.log_metric("costo_total_test", umbral_optimo["costo_total"])
        metricas["umbral_optimo"] = umbral_optimo["umbral"]

        mlflow.sklearn.log_model(
            pipeline,
            "modelo",
            registered_model_name=settings.registered_model_name,
        )
        # OJO: acá NO se copia el modelo a settings.model_path — entrenar()
        # solo registra en MLflow. Lo que sirve serve.py se actualiza
        # únicamente vía promover_a_produccion(), que requiere aprobación
        # humana explícita (ver aprobar_modelo.py).

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


def identificar_mejor_candidato(metric: str | None = None) -> dict:
    """Busca, entre TODOS los runs históricos del experimento, el que mejor
    puntúa en `metric` (default: recall) y lo deja marcado como candidato
    pendiente de aprobación en settings.pending_candidate_path.

    A propósito, esta función NO toca el modelo que sirve serve.py ni el
    alias "champion" del Registry — promover un modelo a producción es una
    decisión separada y explícita (ver promover_a_produccion / aprobar_modelo.py).
    No se limita a la última corrida de run_experiments(): recorre todo el
    historial, así que sirve para elegir el ganador entre experimentos
    corridos en momentos distintos.
    """
    metric = metric or settings.model_selection_metric
    client = MlflowClient()

    experimento = client.get_experiment_by_name(settings.mlflow_experiment_name)
    if experimento is None:
        raise RuntimeError(f"No existe el experimento '{settings.mlflow_experiment_name}'.")

    runs = client.search_runs(
        [experimento.experiment_id],
        order_by=[f"metrics.{metric} DESC"],
    )
    runs = [r for r in runs if metric in r.data.metrics]
    if not runs:
        raise RuntimeError(
            f"No hay runs con la métrica '{metric}'. Corré train.py primero."
        )

    mejor_run = runs[0]
    valor_metrica = mejor_run.data.metrics[metric]
    logger.info(
        "Mejor run por '%s' entre %d candidatos: %s (%s=%.4f)",
        metric, len(runs), mejor_run.info.run_name, metric, valor_metrica,
    )

    versiones = client.search_model_versions(f"run_id='{mejor_run.info.run_id}'")
    if not versiones:
        raise RuntimeError(
            f"El run {mejor_run.info.run_id} no tiene un modelo registrado en el Registry."
        )
    version = versiones[0]

    umbral = float(
        mejor_run.data.params.get("umbral_optimo", settings.prediction_threshold_default)
    )

    candidato = {
        "run_id": mejor_run.info.run_id,
        "run_name": mejor_run.info.run_name,
        "version": version.version,
        "metric": metric,
        "metric_value": valor_metrica,
        "umbral": umbral,
    }

    settings.models_dir.mkdir(parents=True, exist_ok=True)
    settings.pending_candidate_path.write_text(json.dumps(candidato, indent=2))

    logger.info(
        "Candidato pendiente de aprobación: %s v%s (run '%s') — %s=%.4f. "
        "Corré 'python aprobar_modelo.py' para revisarlo y promoverlo.",
        settings.registered_model_name, version.version, mejor_run.info.run_name,
        metric, valor_metrica,
    )

    return candidato


def promover_a_produccion(candidato: dict | None = None) -> dict:
    """Promueve un candidato aprobado a producción: alias "champion" en el
    Model Registry + copia el modelo y su umbral de decisión a disco para
    que serve.py lo sirva.

    Es el ÚNICO punto del pipeline que efectivamente cambia lo que la API
    está sirviendo — separado a propósito de identificar_mejor_candidato()
    para que ese cambio requiera una acción humana explícita en vez de
    ocurrir solo porque un entrenamiento terminó.
    """
    if candidato is None:
        if not settings.pending_candidate_path.exists():
            raise RuntimeError(
                "No hay ningún candidato pendiente. Corré identificar_mejor_candidato() "
                "(o train.py / monitor.py) primero."
            )
        candidato = json.loads(settings.pending_candidate_path.read_text())

    client = MlflowClient()
    client.set_registered_model_alias(
        settings.registered_model_name, settings.champion_alias, candidato["version"]
    )

    modelo = mlflow.sklearn.load_model(
        f"models:/{settings.registered_model_name}@{settings.champion_alias}"
    )
    settings.models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(modelo, settings.model_path)

    settings.threshold_path.write_text(
        json.dumps(
            {
                "umbral": candidato["umbral"],
                "run_id": candidato["run_id"],
                "run_name": candidato["run_name"],
            }
        )
    )

    if settings.pending_candidate_path.exists():
        settings.pending_candidate_path.unlink()

    logger.info(
        "Modelo promovido a producción: %s v%s (run '%s') copiado a %s | umbral: %.2f",
        settings.registered_model_name, candidato["version"], candidato["run_name"],
        settings.model_path, candidato["umbral"],
    )

    return candidato


def reentrenar_con_datos_combinados(df_train: pd.DataFrame, df_nuevo: pd.DataFrame) -> dict:
    """Reentrena incorporando la cohorte de monitoreo al set de entrenamiento
    y deja un candidato pendiente de aprobación (NO promueve solo).

    Se dispara desde monitor.py cuando detecta drift significativo
    (ver evaluar_necesidad_reentrenamiento). La cohorte "nueva" no son datos
    sin etiqueta: es historia real con el churn ya conocido, así que
    incorporarla al entrenamiento es información válida — no estamos
    entrenando con el futuro, estamos ampliando la base con la población
    más reciente que el modelo venía sin ver.

    Corre los mismos 3 experimentos de run_experiments() sobre el dataset
    combinado e identifica el mejor candidato entre TODOS los runs
    históricos (viejos + nuevos) — promoverlo a producción sigue
    requiriendo aprobación humana vía aprobar_modelo.py.
    """
    df_combinado = pd.concat([df_train, df_nuevo], ignore_index=True)
    logger.info(
        "Reentrenando con %d filas (%d histórico + %d cohorte de monitoreo)",
        len(df_combinado), len(df_train), len(df_nuevo),
    )
    run_experiments(df_combinado)
    return identificar_mejor_candidato()
