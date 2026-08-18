"""Carga y preparación del dataset real de churn (IBM Telco Customer Churn).

El dataset es una foto estática de ~7000 clientes, sin columna de fecha, así
que no hay forma de simular "el futuro" de manera genuina. En su lugar,
separamos una cohorte real y con sentido de negocio para el monitoreo de
drift: los clientes más nuevos (tenure baja) contra el resto, que es
exactamente el tipo de comparación que un equipo de MLOps haría en
producción para detectar si el comportamiento de los clientes recién
adquiridos se está alejando de lo que el modelo aprendió a partir de la
base histórica.
"""

import logging

import pandas as pd

from churn_mlops.config import settings

logger = logging.getLogger(__name__)

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
FEATURES = NUM_FEATURES + CAT_FEATURES
TARGET = "churn"


def load_and_clean(path=None) -> pd.DataFrame:
    """Lee el CSV crudo y aplica la limpieza mínima necesaria.

    TotalCharges viene como texto y tiene ~11 filas en blanco (clientes con
    tenure=0, es decir que se dieron de alta y baja el mismo mes): se
    convierten a NaN y se descartan, ya que no aportan señal de churn.
    """
    path = path or settings.raw_dataset_path
    df = pd.read_csv(path)

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    filas_antes = len(df)
    df = df.dropna(subset=["TotalCharges"]).reset_index(drop=True)
    if (descartadas := filas_antes - len(df)) > 0:
        logger.info("Se descartaron %d filas con TotalCharges inválido", descartadas)

    df[TARGET] = (df["Churn"] == "Yes").astype(int)
    return df


def split_train_monitor(
    df: pd.DataFrame, cutoff_months: int | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separa la base histórica (entrenamiento) de la cohorte de clientes nuevos.

    df_monitor: clientes con tenure <= cutoff_months (cohorte reciente).
    df_train: el resto, usado para entrenar y como referencia de drift.
    """
    cutoff = cutoff_months if cutoff_months is not None else settings.monitor_tenure_cutoff_months
    df_monitor = df[df["tenure"] <= cutoff].reset_index(drop=True)
    df_train = df[df["tenure"] > cutoff].reset_index(drop=True)
    return df_train, df_monitor


def prepare_datasets() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pipeline completo: cargar, limpiar y separar train/monitor."""
    df = load_and_clean()
    df_train, df_monitor = split_train_monitor(df)
    logger.info(
        "Dataset preparado: %d filas de entrenamiento, %d de monitoreo (tenure <= %d meses)",
        len(df_train),
        len(df_monitor),
        settings.monitor_tenure_cutoff_months,
    )
    return df_train, df_monitor
