"""
PASO 1: Preparación de datos para un caso de uso.

Para "churn" (dataset real), asegurar_datos_crudos() valida que ya hayas
descargado el CSV a mano (ver README). Para "fraude" (dataset sintético),
lo genera automáticamente si no existe.

Separa el CSV crudo en:
- data/<usecase>/entrenamiento.csv: base histórica usada para entrenar
- data/<usecase>/monitoreo.csv: cohorte reciente, usada por monitor.py

CÓMO CORRERLO
-------------
python prepare_data.py --usecase churn
python prepare_data.py --usecase fraude
"""

import argparse
import logging

from churn_mlops.config import settings
from churn_mlops.logging_config import setup_logging
from churn_mlops.usecases.registry import USECASES, get_usecase

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--usecase", default="churn", choices=sorted(USECASES))
    args = parser.parse_args()

    setup_logging()
    usecase = get_usecase(args.usecase)

    ruta_cruda = settings.raw_dataset_path(usecase.name)
    ruta_cruda.parent.mkdir(parents=True, exist_ok=True)
    usecase.asegurar_datos_crudos(ruta_cruda)

    df = usecase.cargar_y_limpiar(ruta_cruda)
    df_train, df_nuevo = usecase.separar_train_monitor(df)

    df_train.to_csv(settings.train_csv_path(usecase.name), index=False)
    df_nuevo.to_csv(settings.monitor_csv_path(usecase.name), index=False)

    tasa_train = df_train[usecase.target].mean() * 100
    tasa_nuevo = df_nuevo[usecase.target].mean() * 100
    logger.info(
        "[%s] Entrenamiento: %d filas | tasa positiva: %.1f%%",
        usecase.name, len(df_train), tasa_train,
    )
    logger.info(
        "[%s] Nuevos (monitoreo): %d filas | tasa positiva: %.1f%%",
        usecase.name, len(df_nuevo), tasa_nuevo,
    )
