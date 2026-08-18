"""
PASO 6: Monitoreo de Data Drift + reporte automático generado por un LLM +
reentrenamiento automático si el drift es significativo.

Compara estadísticamente la cohorte de clientes nuevos contra la base
histórica de entrenamiento (KS para features numéricas, chi-cuadrado para
categóricas), le pide a Claude que redacte un resumen del hallazgo en
lenguaje natural, y si la fracción de features con drift supera
settings.drift_retrain_fraction_threshold, reentrena automáticamente
incorporando la cohorte de monitoreo (ya con el churn real conocido) al
set de entrenamiento.

Ese reentrenamiento deja un candidato pendiente de aprobación — este
script NUNCA promueve un modelo nuevo a producción por sí solo. Correr
'python aprobar_modelo.py' para revisar el candidato y confirmar el
reemplazo del modelo que sirve la API.

CÓMO CORRERLO
-------------
python monitor.py                 # reporta y reentrena si hace falta
python monitor.py --no-retrain    # solo reporta, nunca reentrena

(si tenés ANTHROPIC_API_KEY seteada en .env, genera el reporte real; si no,
igual muestra los resultados numéricos del test de drift)
"""

import argparse
import logging

import pandas as pd

from churn_mlops.config import settings
from churn_mlops.logging_config import setup_logging
from churn_mlops.monitor import (
    detectar_drift,
    evaluar_necesidad_reentrenamiento,
    generar_reporte_llm,
)
from churn_mlops.train import reentrenar_con_datos_combinados

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-retrain",
        action="store_true",
        help="Solo reporta el drift, nunca dispara el reentrenamiento automático.",
    )
    args = parser.parse_args()

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

    diagnostico = evaluar_necesidad_reentrenamiento(resultados)
    print("\n" + "=" * 70)
    print("DIAGNÓSTICO DE REENTRENAMIENTO")
    print("=" * 70)
    print(
        f"{diagnostico['features_con_drift']}/{diagnostico['features_totales']} features con "
        f"drift ({diagnostico['fraccion_drift']:.0%}) — umbral de reentrenamiento: "
        f"{diagnostico['umbral']:.0%}"
    )

    if not diagnostico["debe_reentrenar"]:
        logger.info("Drift por debajo del umbral de reentrenamiento — no se dispara nada.")
    elif args.no_retrain:
        logger.warning(
            "Drift suficiente para reentrenar, pero se corrió con --no-retrain: no se dispara."
        )
    else:
        logger.warning("Drift significativo detectado — disparando reentrenamiento automático.")
        candidato = reentrenar_con_datos_combinados(df_train, df_nuevo)
        logger.info(
            "Reentrenamiento completo. Candidato pendiente de aprobación: '%s' (v%s) — "
            "%s=%.4f, umbral=%.2f",
            candidato["run_name"], candidato["version"], candidato["metric"],
            candidato["metric_value"], candidato["umbral"],
        )
        logger.info("Corré 'python aprobar_modelo.py' para revisarlo y promoverlo a producción.")
