import mlflow

from churn_mlops.informe import _tabla_markdown, generar_informe, recolectar_runs

RUNS_DE_PRUEBA = [
    {
        "run_name": "r1", "n_estimators": "100", "max_depth": "6",
        "n_train_samples": "500", "umbral_optimo": "0.4",
        "accuracy": 0.8, "f1_score": 0.5, "precision": 0.4, "recall": 0.6,
        "costo_total_test": 12.0,
    }
]


def test_tabla_markdown_con_valores_faltantes_no_rompe():
    runs = RUNS_DE_PRUEBA + [
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


def test_recolectar_runs_junta_metricas_de_todos_los_runs(
    churn_usecase, df_train_fake, tmp_path, monkeypatch
):
    from churn_mlops.train import build_pipeline

    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    mlflow.set_experiment(churn_usecase.mlflow_experiment_name)

    pipeline = build_pipeline(churn_usecase, n_estimators=5, max_depth=2)
    pipeline.fit(df_train_fake[churn_usecase.features], df_train_fake[churn_usecase.target])

    for nombre, recall in {"run_a": 0.5, "run_b": 0.7}.items():
        with mlflow.start_run(run_name=nombre):
            mlflow.log_param("n_estimators", 5)
            mlflow.log_metric("recall", recall)

    runs = recolectar_runs(churn_usecase)

    nombres = {r["run_name"] for r in runs}
    assert nombres == {"run_a", "run_b"}
    recalls = {r["run_name"]: r["recall"] for r in runs}
    assert recalls["run_a"] == 0.5
    assert recalls["run_b"] == 0.7


def test_generar_informe_usa_ollama_por_default_y_arma_markdown(churn_usecase, monkeypatch):
    from churn_mlops.config import settings

    monkeypatch.setattr(settings, "llm_provider", "ollama")
    monkeypatch.setattr(
        "churn_mlops.informe._llamar_ollama", lambda prompt: "Análisis de prueba."
    )

    texto = generar_informe(churn_usecase, RUNS_DE_PRUEBA)

    assert "Análisis de prueba." in texto
    assert "r1" in texto
    assert "# Informe de experimentos" in texto


def test_generar_informe_anthropic_sin_key_no_rompe(churn_usecase, monkeypatch):
    from churn_mlops.config import settings

    monkeypatch.setattr(settings, "llm_provider", "anthropic")
    monkeypatch.setattr(settings, "anthropic_api_key", None)

    texto = generar_informe(churn_usecase, RUNS_DE_PRUEBA)

    assert "Sin ANTHROPIC_API_KEY" in texto


def test_generar_informe_provider_invalido_lanza_error(churn_usecase, monkeypatch):
    import pytest

    from churn_mlops.config import settings

    monkeypatch.setattr(settings, "llm_provider", "otro_provider_invalido")

    with pytest.raises(ValueError):
        generar_informe(churn_usecase, RUNS_DE_PRUEBA)
