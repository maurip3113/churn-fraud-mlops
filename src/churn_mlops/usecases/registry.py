"""Registro de casos de uso disponibles. Agregar uno nuevo: escribir el
módulo en usecases/, exportar una instancia UseCase llamada USECASE, y
sumarla acá — el resto del motor (train/monitor/serve/informe) no cambia."""

from churn_mlops.usecases import churn, fraude
from churn_mlops.usecases.base import UseCase

USECASES: dict[str, UseCase] = {
    churn.USECASE.name: churn.USECASE,
    fraude.USECASE.name: fraude.USECASE,
}


def get_usecase(name: str) -> UseCase:
    try:
        return USECASES[name]
    except KeyError:
        opciones = ", ".join(sorted(USECASES))
        raise ValueError(f"Caso de uso desconocido: '{name}'. Opciones: {opciones}") from None
