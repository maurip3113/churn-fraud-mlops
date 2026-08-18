"""Contrato que debe cumplir cada caso de uso (plugin) del motor de MLOps.

El motor genérico (train.py, monitor.py, serve.py, informe.py) no conoce
columnas, targets ni costos de negocio concretos — todo eso vive acá,
encapsulado en una instancia de UseCase. Agregar un caso de uso nuevo
significa escribir un módulo en usecases/ que construya uno de estos y
registrarlo en usecases/registry.py; el resto del pipeline no cambia.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd
from pydantic import BaseModel


@dataclass(frozen=True)
class UseCase:
    # Identidad: se usa para nombrar el experimento de MLflow, el modelo
    # registrado, y las subcarpetas de data/ y models/ — dos casos de uso
    # nunca comparten artefactos entre sí.
    name: str
    descripcion: str

    # Features y target del dataset crudo (ya limpio por cargar_y_limpiar).
    num_features: list[str]
    cat_features: list[str]
    target: str

    # Schema de entrada de POST /predict para este caso de uso.
    request_model: type[BaseModel]

    # Costos relativos de negocio para optimizar el umbral de decisión:
    # costo_falso_negativo > costo_falso_positivo típicamente, pero la
    # magnitud de la asimetría es específica de cada dominio (no es lo
    # mismo perder un cliente que no detectar un fraude).
    costo_falso_negativo: float
    costo_falso_positivo: float

    # Genera o valida que exista el CSV crudo en `path` (descarga para
    # datasets reales, generación sintética para demos sin dataset público).
    asegurar_datos_crudos: Callable[[Path], None]

    # Lee el CSV crudo y aplica la limpieza mínima (tipos, nulos, target).
    cargar_y_limpiar: Callable[[Path], pd.DataFrame]

    # Separa la base histórica (entrenamiento) de una cohorte con sentido de
    # negocio para monitoreo de drift (ej. clientes nuevos, canal distinto).
    separar_train_monitor: Callable[[pd.DataFrame], tuple[pd.DataFrame, pd.DataFrame]]

    @property
    def features(self) -> list[str]:
        return self.num_features + self.cat_features

    @property
    def mlflow_experiment_name(self) -> str:
        return f"{self.name}_prediction"

    @property
    def registered_model_name(self) -> str:
        return f"{self.name}_predictor"
