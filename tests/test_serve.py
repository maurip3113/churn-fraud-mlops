import joblib
import pytest
from fastapi.testclient import TestClient

from churn_mlops.serve import build_app
from churn_mlops.train import build_pipeline

CLIENTE_VALIDO = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 1,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 90.5,
    "TotalCharges": 90.5,
}


@pytest.fixture
def entorno_modelo(churn_usecase, df_train_fake, tmp_path, monkeypatch):
    """Entrena un modelo chico y lo deja donde settings.model_path() lo espera,
    con threshold_path inexistente a propósito (para probar el default)."""
    from churn_mlops.config import settings

    monkeypatch.setattr(settings, "models_dir", tmp_path / "models")

    pipeline = build_pipeline(churn_usecase, n_estimators=10, max_depth=3)
    pipeline.fit(df_train_fake[churn_usecase.features], df_train_fake[churn_usecase.target])
    ruta_modelo = settings.model_path(churn_usecase.name)
    ruta_modelo.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, ruta_modelo)

    return settings


@pytest.fixture
def client_con_modelo(churn_usecase, entorno_modelo):
    return TestClient(build_app(churn_usecase))


def test_home_ok(client_con_modelo):
    resp = client_con_modelo.get("/")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
    assert resp.json()["usecase"] == "churn"


def test_predict_devuelve_probabilidad(client_con_modelo):
    resp = client_con_modelo.post("/predict", json=CLIENTE_VALIDO)

    assert resp.status_code == 200
    body = resp.json()
    assert "positivo" in body
    assert 0.0 <= body["probabilidad"] <= 1.0


def test_predict_rechaza_valor_invalido(client_con_modelo):
    cliente_invalido = {**CLIENTE_VALIDO, "InternetService": "Cable"}  # no es un valor válido

    resp = client_con_modelo.post("/predict", json=cliente_invalido)

    assert resp.status_code == 422


def test_health_reporta_modelo_presente(client_con_modelo):
    resp = client_con_modelo.get("/health")
    assert resp.json()["status"] == "ok"


def test_predict_usa_umbral_default_sin_archivo_guardado(client_con_modelo, entorno_modelo):
    resp = client_con_modelo.post("/predict", json=CLIENTE_VALIDO)

    assert resp.json()["umbral_usado"] == entorno_modelo.prediction_threshold_default


def test_predict_usa_umbral_guardado_en_threshold_path(churn_usecase, entorno_modelo):
    import json

    entorno_modelo.threshold_path(churn_usecase.name).write_text(
        json.dumps({"umbral": 0.2, "run_id": "x", "run_name": "y"})
    )
    client = TestClient(build_app(churn_usecase))

    resp = client.post("/predict", json=CLIENTE_VALIDO)

    assert resp.json()["umbral_usado"] == 0.2


def test_build_app_funciona_con_otro_usecase(fraude_usecase, df_fraude_fake, tmp_path, monkeypatch):
    """Prueba de genericidad: build_app no debería tener nada hardcodeado de churn."""
    from churn_mlops.config import settings

    monkeypatch.setattr(settings, "models_dir", tmp_path / "models")

    pipeline = build_pipeline(fraude_usecase, n_estimators=10, max_depth=3)
    pipeline.fit(df_fraude_fake[fraude_usecase.features], df_fraude_fake[fraude_usecase.target])
    ruta_modelo = settings.model_path(fraude_usecase.name)
    ruta_modelo.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, ruta_modelo)

    client = TestClient(build_app(fraude_usecase))
    transaccion = {
        "monto": 120.5,
        "hora_del_dia": 3,
        "distancia_a_casa_km": 80.0,
        "veces_tarjeta_hoy": 4,
        "dias_desde_ultima_transaccion": 0.5,
        "es_extranjero": 1,
        "es_online": 1,
    }

    resp = client.post("/predict", json=transaccion)

    assert resp.status_code == 200
    assert 0.0 <= resp.json()["probabilidad"] <= 1.0
