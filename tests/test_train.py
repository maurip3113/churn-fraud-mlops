import json

import mlflow
import mlflow.sklearn
import numpy as np
import pytest

from churn_mlops.data import FEATURES, TARGET
from churn_mlops.train import (
    build_pipeline,
    entrenar,
    identificar_mejor_candidato,
    optimizar_umbral,
    promover_a_produccion,
    reentrenar_con_datos_combinados,
)


def test_optimizar_umbral_prioriza_recall_si_fn_es_mas_caro():
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    # proba alta para los positivos reales, pero por debajo de 0.5 en dos de ellos
    y_proba = np.array([0.9, 0.6, 0.35, 0.3, 0.2, 0.15, 0.1, 0.05])

    resultado = optimizar_umbral(y_true, y_proba, costo_falso_negativo=10, costo_falso_positivo=1)

    # con FN mucho más caro, el umbral óptimo debe bajar para capturar los
    # positivos de proba 0.35 y 0.3, aunque eso implique 0 falsos positivos extra
    assert resultado["umbral"] <= 0.3
    assert resultado["falsos_negativos"] == 0


def test_optimizar_umbral_devuelve_las_claves_esperadas():
    y_true = np.array([1, 0, 1, 0])
    y_proba = np.array([0.8, 0.4, 0.6, 0.2])

    resultado = optimizar_umbral(y_true, y_proba, costo_falso_negativo=5, costo_falso_positivo=1)

    assert set(resultado) == {"umbral", "costo_total", "falsos_negativos", "falsos_positivos"}
    assert 0.05 <= resultado["umbral"] <= 0.95


def test_build_pipeline_predice_probabilidades(df_train_fake):
    pipeline = build_pipeline(n_estimators=10, max_depth=3)
    X = df_train_fake[FEATURES]
    y = df_train_fake[TARGET]
    pipeline.fit(X, y)

    proba = pipeline.predict_proba(X)

    assert proba.shape == (len(X), 2)
    assert (proba >= 0).all() and (proba <= 1).all()


def test_entrenar_devuelve_metricas_validas_y_no_toca_serving(df_train_fake, tmp_path, monkeypatch):
    from churn_mlops.config import settings

    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    mlflow.set_experiment("test_churn_prediction")
    monkeypatch.setattr(settings, "model_path", tmp_path / "modelo.pkl")
    monkeypatch.setattr(settings, "train_stats_path", tmp_path / "stats.csv")

    metricas = entrenar(df_train_fake, n_estimators=10, max_depth=3)

    assert set(metricas) == {"accuracy", "f1_score", "precision", "recall", "umbral_optimo"}
    assert all(0.0 <= v <= 1.0 for v in metricas.values())
    assert (tmp_path / "stats.csv").exists()
    # entrenar() NO debe promover a producción por sí solo
    assert not (tmp_path / "modelo.pkl").exists()


@pytest.fixture
def settings_de_seleccion(tmp_path, monkeypatch):
    from churn_mlops.config import settings

    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    mlflow.set_experiment("test_seleccion")
    monkeypatch.setattr(settings, "mlflow_experiment_name", "test_seleccion")
    monkeypatch.setattr(settings, "model_path", tmp_path / "modelo.pkl")
    monkeypatch.setattr(settings, "threshold_path", tmp_path / "umbral.json")
    monkeypatch.setattr(settings, "pending_candidate_path", tmp_path / "candidato.json")
    monkeypatch.setattr(settings, "registered_model_name", "test_model_registry")
    monkeypatch.setattr(settings, "champion_alias", "champion")
    return settings


def _crear_runs_con_recalls(recalls_por_run: dict, df_train_fake):
    pipeline = build_pipeline(n_estimators=5, max_depth=2)
    pipeline.fit(df_train_fake[FEATURES], df_train_fake[TARGET])

    for nombre, recall in recalls_por_run.items():
        with mlflow.start_run(run_name=nombre):
            mlflow.log_metric("recall", recall)
            mlflow.log_param("umbral_optimo", 0.35)
            mlflow.sklearn.log_model(
                pipeline, "modelo", registered_model_name="test_model_registry"
            )


def test_identificar_mejor_candidato_elige_la_metrica_mas_alta(
    df_train_fake, settings_de_seleccion
):
    _crear_runs_con_recalls({"run_bajo": 0.3, "run_ganador": 0.9, "run_medio": 0.6}, df_train_fake)

    candidato = identificar_mejor_candidato(metric="recall")

    assert candidato["run_name"] == "run_ganador"
    assert candidato["metric_value"] == 0.9


def test_identificar_mejor_candidato_no_toca_serving_ni_registry(
    df_train_fake, settings_de_seleccion
):
    _crear_runs_con_recalls({"run_unico": 0.7}, df_train_fake)

    identificar_mejor_candidato(metric="recall")

    # el candidato queda pendiente, pero NADA se promovió todavía
    assert settings_de_seleccion.pending_candidate_path.exists()
    assert not settings_de_seleccion.model_path.exists()
    assert not settings_de_seleccion.threshold_path.exists()


def test_identificar_mejor_candidato_sin_runs_lanza_error(tmp_path, monkeypatch):
    from churn_mlops.config import settings

    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow_vacio.db'}")
    mlflow.set_experiment("test_seleccion_vacio")
    monkeypatch.setattr(settings, "mlflow_experiment_name", "test_seleccion_vacio")

    with pytest.raises(RuntimeError):
        identificar_mejor_candidato(metric="recall")


def test_promover_a_produccion_copia_el_modelo_y_limpia_pendiente(
    df_train_fake, settings_de_seleccion
):
    _crear_runs_con_recalls({"run_ganador": 0.9}, df_train_fake)
    candidato = identificar_mejor_candidato(metric="recall")
    assert settings_de_seleccion.pending_candidate_path.exists()

    resultado = promover_a_produccion(candidato)

    assert resultado["run_name"] == "run_ganador"
    assert settings_de_seleccion.model_path.exists()
    umbral_guardado = json.loads(settings_de_seleccion.threshold_path.read_text())
    assert umbral_guardado["umbral"] == 0.35
    # aprobado y promovido: ya no debería quedar nada pendiente
    assert not settings_de_seleccion.pending_candidate_path.exists()


def test_promover_a_produccion_lee_el_pendiente_si_no_se_pasa_candidato(
    df_train_fake, settings_de_seleccion
):
    _crear_runs_con_recalls({"run_unico": 0.5}, df_train_fake)
    identificar_mejor_candidato(metric="recall")

    resultado = promover_a_produccion()  # sin argumento: debe leer pending_candidate_path

    assert resultado["run_name"] == "run_unico"
    assert settings_de_seleccion.model_path.exists()


def test_promover_a_produccion_sin_candidato_pendiente_lanza_error(tmp_path, monkeypatch):
    from churn_mlops.config import settings

    monkeypatch.setattr(settings, "pending_candidate_path", tmp_path / "no_existe.json")

    with pytest.raises(RuntimeError):
        promover_a_produccion()


def test_reentrenar_con_datos_combinados_usa_ambos_datasets_y_no_promueve(
    df_train_fake, df_monitor_fake, settings_de_seleccion, monkeypatch
):
    monkeypatch.setattr(settings_de_seleccion, "mlflow_experiment_name", "test_reentrenamiento")
    mlflow.set_experiment("test_reentrenamiento")

    candidato = reentrenar_con_datos_combinados(df_train_fake, df_monitor_fake)

    # deja un candidato pendiente, pero no promueve solo
    assert settings_de_seleccion.pending_candidate_path.exists()
    assert not settings_de_seleccion.model_path.exists()

    client = mlflow.tracking.MlflowClient()
    run = client.get_run(candidato["run_id"])
    n_train_samples = int(run.data.params["n_train_samples"])
    total_combinado = len(df_train_fake) + len(df_monitor_fake)
    # el tamaño de entrenamiento logueado debe reflejar el dataset combinado
    # (80% de train+monitor), no solo el de df_train_fake
    assert int(len(df_train_fake) * 0.8) < n_train_samples <= int(total_combinado * 0.8) + 1
