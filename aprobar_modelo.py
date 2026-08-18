"""
Gate humano: aprobación explícita antes de promover un modelo a producción.

train.py y monitor.py (al reentrenar por drift) identifican un candidato y
lo dejan pendiente en models/candidato_pendiente.json, pero NUNCA tocan el
modelo que sirve serve.py ni el alias "champion" del Model Registry — ese
cambio requiere correr este script y confirmar explícitamente.

CÓMO CORRERLO
-------------
python aprobar_modelo.py            # muestra el candidato y pide confirmación
python aprobar_modelo.py --yes      # aprueba sin preguntar (uso en scripts/CI)
"""

import argparse
import json
import logging

from churn_mlops.config import settings
from churn_mlops.logging_config import setup_logging
from churn_mlops.train import promover_a_produccion

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--yes", action="store_true", help="Aprobar sin pedir confirmación interactiva."
    )
    args = parser.parse_args()

    setup_logging()

    if not settings.pending_candidate_path.exists():
        raise SystemExit(
            "No hay ningún candidato pendiente de aprobación. Corré train.py o monitor.py primero."
        )

    candidato = json.loads(settings.pending_candidate_path.read_text())

    print("=" * 70)
    print("CANDIDATO PENDIENTE DE APROBACIÓN")
    print("=" * 70)
    print(f"Run:                          {candidato['run_name']} (v{candidato['version']})")
    print(f"Métrica de selección:         {candidato['metric']} = {candidato['metric_value']:.4f}")
    print(f"Umbral de decisión propuesto: {candidato['umbral']:.2f}")
    print()
    print(f"Al aprobar, este modelo reemplaza a {settings.model_path}")
    print(f"y queda con el alias '{settings.champion_alias}' en el Model Registry")
    print("— es el modelo que POST /predict sirve a partir de ahí.")
    print()

    if not args.yes:
        respuesta = input("¿Aprobás promover este modelo a producción? [y/N]: ").strip().lower()
        if respuesta not in ("y", "yes", "s", "si", "sí"):
            print(f"Cancelado. El candidato sigue pendiente en {settings.pending_candidate_path}")
            raise SystemExit(0)

    promover_a_produccion(candidato)
    print("Modelo promovido a producción.")
