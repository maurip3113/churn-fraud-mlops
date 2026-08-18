import mlflow

from churn_mlops.informe import (
    _tabla_markdown,
    generar_informe,
    recolectar_runs,
)


def test_tabla_markdown_con_valores_faltantes_no_rompe():
    runs = [
        {
            "run_name": "r1", "n_estimators": "100", "max_depth": "6",
            "n_train_samples": "500", "umbral_optimo": "0.4",
            "accuracy": 0.8, "f1_score": 0.5, "precision": 0.4, "recall": 0.6,
            "costo_total_test": 12.0,
        },
        {
            "run_name": "r2", "n_estimators": None, "max_depth": None,
            "n_train_samples": None, "umbral_optimo": None,
            "accuracy": None, "f1_score": None, "precision": None, "recall": None,
            "costo_total_test": None,
        },
    ]

    tabla = _tabla_markdown(runs)

    assert "r1" in tabla and "r2" in tabla
    assert "0.800" in tabla
    assert "-" in tabla  # valores None se muestran como "-"


def test_recolectar_runs_junta_metricas_de_todos_los_runs(df_train_fake, tmp_path, monkeypatch):
    from churn_mlops.config import settings
    from churn_mlops.train import build_pipeline

    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    mlflow.set_experiment("test_informe")
    monkeypatch.setattr(settings, "mlflow_experiment_name", "test_informe")

    pipeline = build_pipeline(n_estimators=5, max_depth=2)
    from churn_mlops.data import FEATURES, TARGET

    pipeline.fit(df_train_fake[FEATURES], df_train_fake[TARGET])

    for nombre, recall in {"run_a": 0.5, "run_b": 0.7}.items():
        with mlflow.start_run(run_name=nombre):
            mlflow.log_param("n_estimators", 5)
            mlflow.log_metric("recall", recall)

    runs = recolectar_runs()

    nombres = {r["run_name"] for r in runs}
    assert nombres == {"run_a", "run_b"}
    recalls = {r["run_name"]: r["recall"] for r in runs}
    assert recalls["run_a"] == 0.5
    assert recalls["run_b"] == 0.7


def test_generar_informe_usa_ollama_por_default_y_arma_markdown(monkeypatch):
    from churn_mlops.config import settings

    monkeypatch.setattr(settings, "llm_provider", "ollama")
    monkeypatch.setattr(
        "churn_mlops.informe._llamar_ollama", lambda prompt: "Análisis de prueba."
    )

    runs = [
        {
            "run_name": "r1", "n_estimators": "100", "max_depth": "6",
            "n_train_samples": "500", "umbral_optimo": "0.4",
            "accuracy": 0.8, "f1_score": 0.5, "precision": 0.4, "recall": 0.6,
            "costo_total_test": 12.0,
        }
    ]

    texto = generar_informe(runs)

    assert "Análisis de prueba." in texto
    assert "r1" in texto
    assert "# Informe de experimentos" in texto


def test_generar_informe_anthropic_sin_key_no_rompe(monkeypatch):
    from churn_mlops.config import settings

    monkeypatch.setattr(settings, "llm_provider", "anthropic")
    monkeypatch.setattr(settings, "anthropic_api_key", None)

    runs = [
        {
            "run_name": "r1", "n_estimators": "100", "max_depth": "6",
            "n_train_samples": "500", "umbral_optimo": "0.4",
            "accuracy": 0.8, "f1_score": 0.5, "precision": 0.4, "recall": 0.6,
            "costo_total_test": 12.0,
        }
    ]

    texto = generar_informe(runs)

    assert "Sin ANTHROPIC_API_KEY" in texto


def test_generar_informe_provider_invalido_lanza_error(monkeypatch):
    import pytest

    from churn_mlops.config import settings

    monkeypatch.setattr(settings, "llm_provider", "otro_provider_invalido")

    runs = [
        {
            "run_name": "r1", "n_estimators": "100", "max_depth": "6",
            "n_train_samples": "500", "umbral_optimo": "0.4",
            "accuracy": 0.8, "f1_score": 0.5, "precision": 0.4, "recall": 0.6,
            "costo_total_test": 12.0,
        }
    ]

    with pytest.raises(ValueError):
        generar_informe(runs)
