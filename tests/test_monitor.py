from churn_mlops.monitor import detectar_drift, evaluar_necesidad_reentrenamiento


def test_detectar_drift_sin_cambios_no_marca_drift(churn_usecase, df_train_fake):
    resultados = detectar_drift(churn_usecase, df_train_fake, df_train_fake.copy())

    assert all(not r["drift_detectado"] for r in resultados)


def test_detectar_drift_detecta_cambio_numerico(churn_usecase, df_train_fake, df_monitor_con_drift):
    resultados = detectar_drift(churn_usecase, df_train_fake, df_monitor_con_drift)

    resultado_monthly = next(r for r in resultados if r["feature"] == "MonthlyCharges")
    assert resultado_monthly["drift_detectado"] is True
    assert resultado_monthly["test"] == "KS"


def test_detectar_drift_detecta_cambio_categorico(
    churn_usecase, df_train_fake, df_monitor_con_drift
):
    resultados = detectar_drift(churn_usecase, df_train_fake, df_monitor_con_drift)

    resultado_contract = next(r for r in resultados if r["feature"] == "Contract")
    assert resultado_contract["drift_detectado"] is True
    assert resultado_contract["test"] == "chi2"


def test_detectar_drift_devuelve_todas_las_features(churn_usecase, df_train_fake, df_monitor_fake):
    resultados = detectar_drift(churn_usecase, df_train_fake, df_monitor_fake)

    features_evaluadas = {r["feature"] for r in resultados}
    assert features_evaluadas == set(churn_usecase.features)


def test_detectar_drift_funciona_con_otro_usecase(fraude_usecase, df_fraude_fake):
    """Prueba de genericidad: detectar_drift no debería tener nada hardcodeado de churn."""
    resultados = detectar_drift(fraude_usecase, df_fraude_fake, df_fraude_fake.copy())

    features_evaluadas = {r["feature"] for r in resultados}
    assert features_evaluadas == set(fraude_usecase.features)
    assert all(not r["drift_detectado"] for r in resultados)


def _resultados_con_fraccion_de_drift(fraccion_con_drift: float, total: int = 10) -> list[dict]:
    n_con_drift = round(fraccion_con_drift * total)
    return [
        {
            "feature": f"f{i}", "test": "KS", "p_valor": 0.0,
            "drift_detectado": i < n_con_drift, "resumen_train_vs_nuevo": "x/y",
        }
        for i in range(total)
    ]


def test_evaluar_necesidad_reentrenamiento_dispara_si_supera_umbral(monkeypatch):
    from churn_mlops.config import settings

    monkeypatch.setattr(settings, "drift_retrain_fraction_threshold", 0.5)
    resultados = _resultados_con_fraccion_de_drift(0.7)

    diagnostico = evaluar_necesidad_reentrenamiento(resultados)

    assert diagnostico["debe_reentrenar"] is True
    assert diagnostico["features_con_drift"] == 7
    assert diagnostico["features_totales"] == 10


def test_evaluar_necesidad_reentrenamiento_no_dispara_si_es_bajo(monkeypatch):
    from churn_mlops.config import settings

    monkeypatch.setattr(settings, "drift_retrain_fraction_threshold", 0.5)
    resultados = _resultados_con_fraccion_de_drift(0.2)

    diagnostico = evaluar_necesidad_reentrenamiento(resultados)

    assert diagnostico["debe_reentrenar"] is False


def test_evaluar_necesidad_reentrenamiento_lista_vacia_no_rompe():
    diagnostico = evaluar_necesidad_reentrenamiento([])

    assert diagnostico["debe_reentrenar"] is False
    assert diagnostico["fraccion_drift"] == 0.0


def test_generar_reporte_llm_sin_api_key_no_rompe(churn_usecase, monkeypatch):
    from churn_mlops.config import settings
    from churn_mlops.monitor import generar_reporte_llm

    monkeypatch.setattr(settings, "anthropic_api_key", None)

    reporte = generar_reporte_llm(
        churn_usecase,
        [{"feature": "tenure", "test": "KS", "p_valor": 0.01,
          "drift_detectado": True, "resumen_train_vs_nuevo": "1/2"}],
    )

    assert "Sin API key" in reporte
