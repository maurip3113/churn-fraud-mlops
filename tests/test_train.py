import mlflow
import mlflow.sklearn
import numpy as np

from churn_mlops.data import FEATURES, TARGET
from churn_mlops.train import build_pipeline, entrenar, optimizar_umbral, seleccionar_mejor_modelo


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
    from churn_mlops.data import FEATURES, TARGET

    pipeline = build_pipeline(n_estimators=10, max_depth=3)
    X = df_train_fake[FEATURES]
    y = df_train_fake[TARGET]
    pipeline.fit(X, y)

    proba = pipeline.predict_proba(X)

    assert proba.shape == (len(X), 2)
    assert (proba >= 0).all() and (proba <= 1).all()


def test_entrenar_devuelve_metricas_validas(df_train_fake, tmp_path, monkeypatch):
    from churn_mlops.config import settings

    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    mlflow.set_experiment("test_churn_prediction")
    monkeypatch.setattr(settings, "model_path", tmp_path / "modelo.pkl")
    monkeypatch.setattr(settings, "train_stats_path", tmp_path / "stats.csv")

    metricas = entrenar(df_train_fake, n_estimators=10, max_depth=3)

    assert set(metricas) == {"accuracy", "f1_score", "precision", "recall", "umbral_optimo"}
    assert all(0.0 <= v <= 1.0 for v in metricas.values())
    assert (tmp_path / "modelo.pkl").exists()
    assert (tmp_path / "stats.csv").exists()


def test_seleccionar_mejor_modelo_elige_la_metrica_mas_alta(df_train_fake, tmp_path, monkeypatch):
    from churn_mlops.config import settings

    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    mlflow.set_experiment("test_seleccion")
    monkeypatch.setattr(settings, "mlflow_experiment_name", "test_seleccion")
    monkeypatch.setattr(settings, "model_path", tmp_path / "modelo.pkl")
    monkeypatch.setattr(settings, "threshold_path", tmp_path / "umbral.json")
    monkeypatch.setattr(settings, "registered_model_name", "test_model_registry")
    monkeypatch.setattr(settings, "champion_alias", "champion")

    pipeline = build_pipeline(n_estimators=5, max_depth=2)
    pipeline.fit(df_train_fake[FEATURES], df_train_fake[TARGET])

    runs = {
        "run_bajo": {"recall": 0.3, "umbral_optimo": 0.55},
        "run_ganador": {"recall": 0.9, "umbral_optimo": 0.35},
        "run_medio": {"recall": 0.6, "umbral_optimo": 0.45},
    }
    for nombre, params_y_metricas in runs.items():
        with mlflow.start_run(run_name=nombre):
            mlflow.log_metric("recall", params_y_metricas["recall"])
            mlflow.log_param("umbral_optimo", params_y_metricas["umbral_optimo"])
            mlflow.sklearn.log_model(
                pipeline, "modelo", registered_model_name="test_model_registry"
            )

    ganador = seleccionar_mejor_modelo(metric="recall")

    assert ganador["run_name"] == "run_ganador"
    assert ganador["metric_value"] == 0.9
    assert ganador["umbral"] == 0.35
    assert (tmp_path / "modelo.pkl").exists()

    import json

    umbral_guardado = json.loads((tmp_path / "umbral.json").read_text())
    assert umbral_guardado["umbral"] == 0.35
    assert umbral_guardado["run_name"] == "run_ganador"


def test_seleccionar_mejor_modelo_sin_runs_lanza_error(tmp_path, monkeypatch):
    from churn_mlops.config import settings

    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow_vacio.db'}")
    mlflow.set_experiment("test_seleccion_vacio")
    monkeypatch.setattr(settings, "mlflow_experiment_name", "test_seleccion_vacio")

    import pytest

    with pytest.raises(RuntimeError):
        seleccionar_mejor_modelo(metric="recall")
