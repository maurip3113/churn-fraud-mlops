import joblib
import pytest
from fastapi.testclient import TestClient

from churn_mlops.data import FEATURES, TARGET
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
def client_con_modelo(df_train_fake, tmp_path, monkeypatch):
    from churn_mlops.config import settings

    pipeline = build_pipeline(n_estimators=10, max_depth=3)
    pipeline.fit(df_train_fake[FEATURES], df_train_fake[TARGET])
    model_path = tmp_path / "modelo.pkl"
    joblib.dump(pipeline, model_path)
    monkeypatch.setattr(settings, "model_path", model_path)

    import churn_mlops.serve as serve_module

    monkeypatch.setattr(serve_module, "_modelo", None)

    return TestClient(serve_module.app)


def test_home_ok(client_con_modelo):
    resp = client_con_modelo.get("/")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_predict_devuelve_probabilidad(client_con_modelo):
    resp = client_con_modelo.post("/predict", json=CLIENTE_VALIDO)

    assert resp.status_code == 200
    body = resp.json()
    assert "va_a_abandonar" in body
    assert 0.0 <= body["probabilidad_churn"] <= 1.0


def test_predict_rechaza_valor_invalido(client_con_modelo):
    cliente_invalido = {**CLIENTE_VALIDO, "InternetService": "Cable"}  # no es un valor válido

    resp = client_con_modelo.post("/predict", json=cliente_invalido)

    assert resp.status_code == 422


def test_health_reporta_modelo_presente(client_con_modelo):
    resp = client_con_modelo.get("/health")
    assert resp.json()["status"] == "ok"
