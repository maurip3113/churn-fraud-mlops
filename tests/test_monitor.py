from churn_mlops.monitor import detectar_drift


def test_detectar_drift_sin_cambios_no_marca_drift(df_train_fake, df_monitor_fake):
    resultados = detectar_drift(df_train_fake, df_train_fake.copy())

    assert all(not r["drift_detectado"] for r in resultados)


def test_detectar_drift_detecta_cambio_numerico(df_train_fake, df_monitor_con_drift):
    resultados = detectar_drift(df_train_fake, df_monitor_con_drift)

    resultado_monthly = next(r for r in resultados if r["feature"] == "MonthlyCharges")
    assert resultado_monthly["drift_detectado"] is True
    assert resultado_monthly["test"] == "KS"


def test_detectar_drift_detecta_cambio_categorico(df_train_fake, df_monitor_con_drift):
    resultados = detectar_drift(df_train_fake, df_monitor_con_drift)

    resultado_contract = next(r for r in resultados if r["feature"] == "Contract")
    assert resultado_contract["drift_detectado"] is True
    assert resultado_contract["test"] == "chi2"


def test_detectar_drift_devuelve_todas_las_features(df_train_fake, df_monitor_fake):
    from churn_mlops.data import CAT_FEATURES, NUM_FEATURES

    resultados = detectar_drift(df_train_fake, df_monitor_fake)

    features_evaluadas = {r["feature"] for r in resultados}
    assert features_evaluadas == set(NUM_FEATURES + CAT_FEATURES)


def test_generar_reporte_llm_sin_api_key_no_rompe(monkeypatch):
    from churn_mlops.config import settings
    from churn_mlops.monitor import generar_reporte_llm

    monkeypatch.setattr(settings, "anthropic_api_key", None)

    reporte = generar_reporte_llm([{"feature": "tenure", "test": "KS", "p_valor": 0.01,
                                     "drift_detectado": True, "resumen_train_vs_nuevo": "1/2"}])

    assert "Sin API key" in reporte
