"""
Informe descriptivo de todos los modelos entrenados + conclusiones.

Junta todos los runs históricos del experimento en MLflow y le pide a un
LLM que redacte un análisis: qué se probó, qué configuración rindió mejor,
tendencias observadas, y una conclusión.

Por default usa Ollama corriendo localmente (gratis, sin API key). Hace
falta tenerlo instalado (https://ollama.com/download), corriendo, y con el
modelo bajado:
    ollama pull llama3.2

Para usar Claude en cambio, seteá en .env:
    LLM_PROVIDER=anthropic
(necesita ANTHROPIC_API_KEY con crédito)

CÓMO CORRERLO
-------------
python generar_informe.py
"""

import logging
from datetime import datetime

from churn_mlops.config import settings
from churn_mlops.informe import generar_informe
from churn_mlops.logging_config import setup_logging

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    setup_logging()
    logger.info("Generando informe con proveedor: %s", settings.llm_provider)

    texto = generar_informe()
    print(texto)

    carpeta_informes = settings.data_dir.parent / "informes"
    carpeta_informes.mkdir(exist_ok=True)
    archivo = carpeta_informes / f"informe_{datetime.now():%Y%m%d_%H%M%S}.md"
    archivo.write_text(texto, encoding="utf-8")
    logger.info("Informe guardado en %s", archivo)
