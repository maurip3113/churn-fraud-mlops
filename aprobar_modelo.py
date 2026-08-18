"""
Gate humano: aprobación explícita antes de promover un modelo a producción.

train.py y monitor.py (al reentrenar por drift) identifican un candidato y
lo dejan pendiente, pero NUNCA tocan el modelo que sirve serve.py ni el
alias "champion" del Model Registry — ese cambio requiere correr este
script y confirmar explícitamente.

CÓMO CORRERLO
-------------
python aprobar_modelo.py --usecase churn            # muestra el candidato y pide confirmación
python aprobar_modelo.py --usecase churn --yes      # aprueba sin preguntar (uso en scripts/CI)
"""

import argparse
import json
import logging

from churn_mlops.config import settings
from churn_mlops.logging_config import setup_logging
from churn_mlops.train import promover_a_produccion
from churn_mlops.usecases.registry import USECASES, get_usecase

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--usecase", default="churn", choices=sorted(USECASES))
    parser.add_argument(
        "--yes", action="store_true", help="Aprobar sin pedir confirmación interactiva."
    )
    args = parser.parse_args()

    setup_logging()
    usecase = get_usecase(args.usecase)

    ruta_pendiente = settings.pending_candidate_path(usecase.name)
    if not ruta_pendiente.exists():
        raise SystemExit(
            f"No hay ningún candidato pendiente para '{usecase.name}'. Corré "
            f"train.py --usecase {usecase.name} o monitor.py --usecase {usecase.name} primero."
        )

    candidato = json.loads(ruta_pendiente.read_text())

    print("=" * 70)
    print(f"CANDIDATO PENDIENTE DE APROBACIÓN — {usecase.name}")
    print("=" * 70)
    print(f"Run:                          {candidato['run_name']} (v{candidato['version']})")
    print(f"Métrica de selección:         {candidato['metric']} = {candidato['metric_value']:.4f}")
    print(f"Umbral de decisión propuesto: {candidato['umbral']:.2f}")
    print()
    print(f"Al aprobar, este modelo reemplaza a {settings.model_path(usecase.name)}")
    print(f"y queda con el alias '{settings.champion_alias}' en el Model Registry")
    print("— es el modelo que POST /predict sirve a partir de ahí.")
    print()

    if not args.yes:
        respuesta = input("¿Aprobás promover este modelo a producción? [y/N]: ").strip().lower()
        if respuesta not in ("y", "yes", "s", "si", "sí"):
            print(f"Cancelado. El candidato sigue pendiente en {ruta_pendiente}")
            raise SystemExit(0)

    promover_a_produccion(usecase, candidato)
    print("Modelo promovido a producción.")
