import numpy as np
import pandas as pd
import pytest

from churn_mlops.usecases import churn as churn_mod
from churn_mlops.usecases import fraude as fraude_mod


@pytest.fixture
def churn_usecase():
    return churn_mod.USECASE


@pytest.fixture
def fraude_usecase():
    return fraude_mod.USECASE


@pytest.fixture
def df_train_fake() -> pd.DataFrame:
    return _fake_churn_df(n=200, tasa_positiva=0.2, seed=1)


@pytest.fixture
def df_monitor_fake() -> pd.DataFrame:
    return _fake_churn_df(n=80, tasa_positiva=0.2, seed=2)


@pytest.fixture
def df_monitor_con_drift() -> pd.DataFrame:
    df = _fake_churn_df(n=80, tasa_positiva=0.2, seed=3)
    df["MonthlyCharges"] = df["MonthlyCharges"] + 500  # desplazamos la distribución a propósito
    df["Contract"] = "Month-to-month"  # colapsamos la categoría a propósito
    return df


@pytest.fixture
def df_fraude_fake() -> pd.DataFrame:
    return _fake_fraude_df(n=200, tasa_positiva=0.05, seed=11)


@pytest.fixture
def df_fraude_monitor_fake() -> pd.DataFrame:
    return _fake_fraude_df(n=80, tasa_positiva=0.05, seed=12)


def _fake_churn_df(n: int, tasa_positiva: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = {
        "tenure": rng.integers(1, 72, n),
        "MonthlyCharges": rng.normal(65, 20, n).clip(18, 120),
        "TotalCharges": rng.normal(2000, 1500, n).clip(0, None),
        "gender": rng.choice(["Male", "Female"], n),
        "SeniorCitizen": rng.integers(0, 2, n),
        "Partner": rng.choice(["Yes", "No"], n),
        "Dependents": rng.choice(["Yes", "No"], n),
        "PhoneService": rng.choice(["Yes", "No"], n),
        "MultipleLines": rng.choice(["Yes", "No", "No phone service"], n),
        "InternetService": rng.choice(["DSL", "Fiber optic", "No"], n),
        "OnlineSecurity": rng.choice(["Yes", "No"], n),
        "OnlineBackup": rng.choice(["Yes", "No"], n),
        "DeviceProtection": rng.choice(["Yes", "No"], n),
        "TechSupport": rng.choice(["Yes", "No"], n),
        "StreamingTV": rng.choice(["Yes", "No"], n),
        "StreamingMovies": rng.choice(["Yes", "No"], n),
        "Contract": rng.choice(["Month-to-month", "One year", "Two year"], n),
        "PaperlessBilling": rng.choice(["Yes", "No"], n),
        "PaymentMethod": rng.choice(
            ["Electronic check", "Mailed check", "Bank transfer (automatic)"], n
        ),
    }
    df = pd.DataFrame(data)
    df[churn_mod.TARGET] = rng.choice([0, 1], n, p=[1 - tasa_positiva, tasa_positiva])
    assert set(churn_mod.NUM_FEATURES + churn_mod.CAT_FEATURES).issubset(df.columns)
    return df


def _fake_fraude_df(n: int, tasa_positiva: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = {
        "monto": rng.lognormal(3.5, 1.0, n).clip(1, 5000),
        "hora_del_dia": rng.integers(0, 24, n),
        "distancia_a_casa_km": rng.exponential(15, n).clip(0, 500),
        "veces_tarjeta_hoy": rng.poisson(1.5, n),
        "dias_desde_ultima_transaccion": rng.exponential(3, n).clip(0, 90),
        "es_extranjero": rng.binomial(1, 0.05, n),
        "es_online": rng.binomial(1, 0.35, n),
    }
    df = pd.DataFrame(data)
    df[fraude_mod.TARGET] = rng.choice([0, 1], n, p=[1 - tasa_positiva, tasa_positiva])
    assert set(fraude_mod.NUM_FEATURES + fraude_mod.CAT_FEATURES).issubset(df.columns)
    return df
