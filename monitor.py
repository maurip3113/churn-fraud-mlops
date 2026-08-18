"""
PASO 6: Monitoreo de Data Drift + reporte automático generado por un LLM.

Compara estadísticamente la cohorte de clientes nuevos contra la base
histórica de entrenamiento (KS para features numéricas, chi-cuadrado para
categóricas) y le pide a Claude que redacte un resumen del hallazgo en
lenguaje natural.

CÓMO CORRERLO
-------------
python monitor.py
(si tenés ANTHROPIC_API_KEY seteada en .env, genera el reporte real; si no,
igual muestra los resultados numéricos del test de drift)
"""

import logging

import pandas as pd

from churn_mlops.config import settings
from churn_mlops.logging_config import setup_logging
from churn_mlops.monitor import detectar_drift, generar_reporte_llm

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    setup_logging()

    ruta_train = settings.data_dir / "clientes_entrenamiento.csv"
    ruta_nuevo = settings.data_dir / "clientes_nuevos.csv"
    if not ruta_train.exists() or not ruta_nuevo.exists():
        raise SystemExit("Faltan los CSV de datos. Corré 'python prepare_data.py' primero.")

    df_train = pd.read_csv(ruta_train)
    df_nuevo = pd.read_csv(ruta_nuevo)

    resultados = detectar_drift(df_train, df_nuevo)

    print("=" * 70)
    print("RESULTADOS DEL TEST DE DRIFT")
    print("=" * 70)
    for r in resultados:
        estado = "DRIFT" if r["drift_detectado"] else "estable"
        print(
            f"{r['feature']:22s} [{r['test']:>3s}] {estado:8s} "
            f"p={r['p_valor']:<8} train/nuevo={r['resumen_train_vs_nuevo']}"
        )

    print("\n" + "=" * 70)
    print("REPORTE GENERADO POR LLM")
    print("=" * 70)
    print(generar_reporte_llm(resultados))
