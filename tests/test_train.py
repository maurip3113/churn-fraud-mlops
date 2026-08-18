import mlflow
import mlflow.sklearn

from churn_mlops.data import FEATURES, TARGET
from churn_mlops.train import build_pipeline, entrenar, seleccionar_mejor_modelo


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

    assert set(metricas) == {"accuracy", "f1_score", "precision", "recall"}
    assert all(0.0 <= v <= 1.0 for v in metricas.values())
    assert (tmp_path / "modelo.pkl").exists()
    assert (tmp_path / "stats.csv").exists()


def test_seleccionar_mejor_modelo_elige_la_metrica_mas_alta(df_train_fake, tmp_path, monkeypatch):
    from churn_mlops.config import settings

    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    mlflow.set_experiment("test_seleccion")
    monkeypatch.setattr(settings, "mlflow_experiment_name", "test_seleccion")
    monkeypatch.setattr(settings, "model_path", tmp_path / "modelo.pkl")
    monkeypatch.setattr(settings, "registered_model_name", "test_model_registry")
    monkeypatch.setattr(settings, "champion_alias", "champion")

    pipeline = build_pipeline(n_estimators=5, max_depth=2)
    pipeline.fit(df_train_fake[FEATURES], df_train_fake[TARGET])

    recalls_por_run = {"run_bajo": 0.3, "run_ganador": 0.9, "run_medio": 0.6}
    for nombre, recall in recalls_por_run.items():
        with mlflow.start_run(run_name=nombre):
            mlflow.log_metric("recall", recall)
            mlflow.sklearn.log_model(
                pipeline, "modelo", registered_model_name="test_model_registry"
            )

    ganador = seleccionar_mejor_modelo(metric="recall")

    assert ganador["run_name"] == "run_ganador"
    assert ganador["metric_value"] == 0.9
    assert (tmp_path / "modelo.pkl").exists()


def test_seleccionar_mejor_modelo_sin_runs_lanza_error(tmp_path, monkeypatch):
    from churn_mlops.config import settings

    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow_vacio.db'}")
    mlflow.set_experiment("test_seleccion_vacio")
    monkeypatch.setattr(settings, "mlflow_experiment_name", "test_seleccion_vacio")

    import pytest

    with pytest.raises(RuntimeError):
        seleccionar_mejor_modelo(metric="recall")
