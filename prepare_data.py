"""
PASO 1: Preparación del dataset real (IBM Telco Customer Churn).

Lee data/telco_churn_raw.csv, lo limpia y lo separa en:
- data/clientes_entrenamiento.csv: base histórica usada para entrenar
- data/clientes_nuevos.csv: cohorte de clientes recientes (tenure baja),
  usada más adelante por monitor.py para detectar drift real
"""

import logging

from churn_mlops.config import settings
from churn_mlops.data import prepare_datasets
from churn_mlops.logging_config import setup_logging

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    setup_logging()

    if not settings.raw_dataset_path.exists():
        raise SystemExit(
            f"No se encontró {settings.raw_dataset_path}. "
            "Descargá el dataset Telco Customer Churn (IBM) antes de continuar."
        )

    df_train, df_nuevo = prepare_datasets()

    settings.data_dir.mkdir(parents=True, exist_ok=True)
    df_train.to_csv(settings.data_dir / "clientes_entrenamiento.csv", index=False)
    df_nuevo.to_csv(settings.data_dir / "clientes_nuevos.csv", index=False)

    tasa_train = df_train["churn"].mean() * 100
    tasa_nuevo = df_nuevo["churn"].mean() * 100
    logger.info("Entrenamiento: %d filas | tasa de churn: %.1f%%", len(df_train), tasa_train)
    logger.info("Nuevos (monitoreo): %d filas | tasa de churn: %.1f%%", len(df_nuevo), tasa_nuevo)
