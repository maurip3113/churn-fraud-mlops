"""Plugin: detección de fraude en transacciones (dataset SINTÉTICO).

A diferencia de churn, acá no usamos un dataset real: los datasets públicos
de fraude de tarjetas requieren cuenta/login en Kaggle, así que generamos
uno sintético — igual que hacía el proyecto original antes de migrar churn
a datos reales. El objetivo de este plugin no es la fidelidad del dataset,
sino probar que el motor genérico (train/monitor/serve/informe) funciona
igual de bien con un caso de uso genuinamente distinto a churn:

- Desbalance extremo (~2% fraude vs. ~20% churn)
- Asimetría de costos mucho más marcada (no detectar un fraude es MUCHO
  más caro que bloquear una transacción legítima por error)
- Drift real por CANAL en vez de por antigüedad: las transacciones online
  tienen un perfil de riesgo distinto a las presenciales, y ese cambio de
  mix de canal es el análogo de "clientes nuevos" en churn.
"""

import logging
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel

from churn_mlops.usecases.base import UseCase

logger = logging.getLogger(__name__)

NUM_FEATURES = [
    "monto",
    "hora_del_dia",
    "distancia_a_casa_km",
    "veces_tarjeta_hoy",
    "dias_desde_ultima_transaccion",
]
CAT_FEATURES = ["es_extranjero", "es_online"]
TARGET = "es_fraude"

N_TRANSACCIONES = 20000
SEMILLA = 7


def asegurar_datos_crudos(path: Path, n: int = N_TRANSACCIONES, seed: int = SEMILLA) -> None:
    """Genera el dataset sintético de transacciones si todavía no existe."""
    if path.exists():
        return

    rng = np.random.default_rng(seed)

    canal = rng.choice(["pos", "online", "atm"], n, p=[0.55, 0.35, 0.10])
    es_online = (canal == "online").astype(int)
    es_extranjero = rng.binomial(1, 0.05, n)
    hora = rng.integers(0, 24, n)
    distancia = rng.exponential(scale=15, size=n).clip(0, 500)
    veces_hoy = rng.poisson(1.5, n)
    dias_desde_ultima = rng.exponential(scale=3, size=n).clip(0, 90)

    # las transacciones online tienden a ser de mayor monto en este dataset
    monto_base = rng.lognormal(mean=3.5, sigma=1.0, size=n)
    monto = (monto_base * np.where(es_online == 1, 1.6, 1.0)).clip(1, 5000)

    logit_fraude = (
        -6.2
        + 0.0009 * monto
        + 0.03 * distancia
        + 1.8 * es_extranjero
        + 0.9 * es_online
        + 0.4 * (hora < 5).astype(float)
        - 0.15 * dias_desde_ultima
        + rng.normal(0, 0.3, n)
    )
    prob_fraude = 1 / (1 + np.exp(-logit_fraude))
    es_fraude = (rng.random(n) < prob_fraude).astype(int)

    df = pd.DataFrame(
        {
            "monto": monto.round(2),
            "hora_del_dia": hora,
            "distancia_a_casa_km": distancia.round(1),
            "veces_tarjeta_hoy": veces_hoy,
            "dias_desde_ultima_transaccion": dias_desde_ultima.round(1),
            "es_extranjero": es_extranjero,
            "es_online": es_online,
            "canal": canal,
            TARGET: es_fraude,
        }
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info(
        "Dataset sintético de fraude generado: %d filas, %.1f%% fraude", n, df[TARGET].mean() * 100
    )


def cargar_y_limpiar(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def separar_train_monitor(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """df_monitor: transacciones online (perfil de riesgo distinto, el
    análogo de "clientes nuevos" en churn). df_train: pos + atm."""
    df_monitor = df[df["canal"] == "online"].reset_index(drop=True)
    df_train = df[df["canal"] != "online"].reset_index(drop=True)
    return df_train, df_monitor


class Transaccion(BaseModel):
    monto: float
    hora_del_dia: int
    distancia_a_casa_km: float
    veces_tarjeta_hoy: int
    dias_desde_ultima_transaccion: float
    es_extranjero: Literal[0, 1]
    es_online: Literal[0, 1]


USECASE = UseCase(
    name="fraude",
    descripcion="Detección de fraude en transacciones (dataset sintético)",
    num_features=NUM_FEATURES,
    cat_features=CAT_FEATURES,
    target=TARGET,
    request_model=Transaccion,
    # No detectar un fraude sale mucho más caro que bloquear por error una
    # transacción legítima (fricción para el cliente, pero reversible).
    costo_falso_negativo=20.0,
    costo_falso_positivo=1.0,
    asegurar_datos_crudos=asegurar_datos_crudos,
    cargar_y_limpiar=cargar_y_limpiar,
    separar_train_monitor=separar_train_monitor,
)
