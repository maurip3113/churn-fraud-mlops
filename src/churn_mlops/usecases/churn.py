"""Plugin: predicción de churn (IBM Telco Customer Churn — dataset real).

El dataset es una foto estática de ~7000 clientes, sin columna de fecha, así
que no hay forma de simular "el futuro" de manera genuina. En su lugar,
separar_train_monitor() usa una cohorte real y con sentido de negocio: los
clientes más nuevos (tenure baja) contra el resto — el mismo tipo de
comparación que un equipo de MLOps haría en producción para detectar si el
comportamiento de los clientes recién adquiridos se aleja de lo que el
modelo aprendió con la base histórica.
"""

import logging
from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import BaseModel

from churn_mlops.usecases.base import UseCase

logger = logging.getLogger(__name__)

RAW_DATASET_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)

NUM_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
CAT_FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]
TARGET = "churn"

# Cohorte usada para simular un batch "nuevo" a monitorear: clientes con
# tenure <= este valor (meses) se separan del set de entrenamiento y se
# tratan como la población entrante más reciente.
MONITOR_TENURE_CUTOFF_MONTHS = 6


def asegurar_datos_crudos(path: Path) -> None:
    """El dataset real de IBM no se descarga automáticamente (evita bajar
    ~1MB sin que el usuario lo pida explícitamente) — hay que traerlo una
    vez a mano."""
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró {path}. Descargalo con:\n"
            f"curl -o {path} {RAW_DATASET_URL}"
        )


def cargar_y_limpiar(path: Path) -> pd.DataFrame:
    """TotalCharges viene como texto y tiene ~11 filas en blanco (clientes
    con tenure=0, es decir que se dieron de alta y baja el mismo mes): se
    convierten a NaN y se descartan, ya que no aportan señal de churn."""
    df = pd.read_csv(path)

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    filas_antes = len(df)
    df = df.dropna(subset=["TotalCharges"]).reset_index(drop=True)
    if (descartadas := filas_antes - len(df)) > 0:
        logger.info("Se descartaron %d filas con TotalCharges inválido", descartadas)

    df[TARGET] = (df["Churn"] == "Yes").astype(int)
    return df


def separar_train_monitor(
    df: pd.DataFrame, cutoff_months: int = MONITOR_TENURE_CUTOFF_MONTHS
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """df_monitor: clientes con tenure <= cutoff_months (cohorte reciente).
    df_train: el resto, usado para entrenar y como referencia de drift."""
    df_monitor = df[df["tenure"] <= cutoff_months].reset_index(drop=True)
    df_train = df[df["tenure"] > cutoff_months].reset_index(drop=True)
    return df_train, df_monitor


SiNo = Literal["Yes", "No"]


class Cliente(BaseModel):
    gender: Literal["Male", "Female"]
    SeniorCitizen: Literal[0, 1]
    Partner: SiNo
    Dependents: SiNo
    tenure: int
    PhoneService: SiNo
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: SiNo
    PaymentMethod: Literal[
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ]
    MonthlyCharges: float
    TotalCharges: float


USECASE = UseCase(
    name="churn",
    descripcion="Predicción de abandono de clientes (IBM Telco Customer Churn, dataset real)",
    num_features=NUM_FEATURES,
    cat_features=CAT_FEATURES,
    target=TARGET,
    request_model=Cliente,
    # Perder un cliente sale ~5 veces más caro que una promo de retención
    # de más (ofrecida a alguien que igual se iba a quedar).
    costo_falso_negativo=5.0,
    costo_falso_positivo=1.0,
    asegurar_datos_crudos=asegurar_datos_crudos,
    cargar_y_limpiar=cargar_y_limpiar,
    separar_train_monitor=separar_train_monitor,
)
