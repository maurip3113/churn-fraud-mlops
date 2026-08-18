import pandas as pd
import pytest

from churn_mlops.data import CAT_FEATURES, NUM_FEATURES, TARGET


@pytest.fixture
def df_train_fake() -> pd.DataFrame:
    return _fake_df(n=200, churn_rate=0.2, seed=1)


@pytest.fixture
def df_monitor_fake() -> pd.DataFrame:
    return _fake_df(n=80, churn_rate=0.2, seed=2)


@pytest.fixture
def df_monitor_con_drift() -> pd.DataFrame:
    df = _fake_df(n=80, churn_rate=0.2, seed=3)
    df["MonthlyCharges"] = df["MonthlyCharges"] + 500  # desplazamos la distribución a propósito
    df["Contract"] = "Month-to-month"  # colapsamos la categoría a propósito
    return df


def _fake_df(n: int, churn_rate: float, seed: int) -> pd.DataFrame:
    import numpy as np

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
    df[TARGET] = rng.choice([0, 1], n, p=[1 - churn_rate, churn_rate])
    assert set(NUM_FEATURES + CAT_FEATURES).issubset(df.columns)
    return df
