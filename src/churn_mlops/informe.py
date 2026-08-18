"""Informe descriptivo de todos los modelos entrenados + conclusiones.

Junta hiperparámetros y métricas de TODOS los runs históricos del
experimento en MLflow (no solo la última corrida) y le pide a un LLM que
redacte un análisis en lenguaje natural: qué se probó, qué configuración
rindió mejor y por qué, tendencias observadas, y una conclusión.

Soporta dos proveedores (settings.llm_provider):
- "ollama" (default): local, gratis, sin API key — requiere tener Ollama
  corriendo (`ollama serve`) y el modelo bajado (`ollama pull llama3.2`).
- "anthropic": requiere ANTHROPIC_API_KEY con crédito.
"""

import logging

import requests
from mlflow.tracking import MlflowClient

from churn_mlops.config import settings

logger = logging.getLogger(__name__)


def recolectar_runs() -> list[dict]:
    """Junta hiperparámetros y métricas de TODOS los runs del experimento,
    ordenados cronológicamente."""
    client = MlflowClient()
    experimento = client.get_experiment_by_name(settings.mlflow_experiment_name)
    if experimento is None:
        raise RuntimeError(f"No existe el experimento '{settings.mlflow_experiment_name}'.")

    runs = client.search_runs([experimento.experiment_id], order_by=["start_time ASC"])
    if not runs:
        raise RuntimeError("No hay runs en el experimento. Corré train.py primero.")

    return [
        {
            "run_name": r.info.run_name,
            "n_estimators": r.data.params.get("n_estimators"),
            "max_depth": r.data.params.get("max_depth"),
            "n_train_samples": r.data.params.get("n_train_samples"),
            "umbral_optimo": r.data.params.get("umbral_optimo"),
            "accuracy": r.data.metrics.get("accuracy"),
            "f1_score": r.data.metrics.get("f1_score"),
            "precision": r.data.metrics.get("precision"),
            "recall": r.data.metrics.get("recall"),
            "costo_total_test": r.data.metrics.get("costo_total_test"),
        }
        for r in runs
    ]


def _fmt(valor, decimales=3) -> str:
    return f"{valor:.{decimales}f}" if valor is not None else "-"


def _tabla_markdown(runs: list[dict]) -> str:
    encabezado = (
        "| Run | n_estimators | max_depth | n_train | accuracy | f1 | "
        "precision | recall | umbral | costo_test |\n"
    )
    separador = "|---|---|---|---|---|---|---|---|---|---|\n"
    filas = "".join(
        f"| {r['run_name']} | {r['n_estimators']} | {r['max_depth']} | "
        f"{r['n_train_samples']} | {_fmt(r['accuracy'])} | {_fmt(r['f1_score'])} | "
        f"{_fmt(r['precision'])} | {_fmt(r['recall'])} | {r['umbral_optimo']} | "
        f"{_fmt(r['costo_total_test'], 1)} |\n"
        for r in runs
    )
    return encabezado + separador + filas


def _construir_prompt(tabla: str) -> str:
    return (
        "Sos un ingeniero de ML senior escribiendo un informe interno sobre "
        "los experimentos de un modelo de predicción de churn (RandomForest "
        "sobre datos reales de clientes de telecom, IBM Telco Customer Churn). "
        f"Te paso una tabla con todos los runs entrenados hasta ahora:\n\n{tabla}\n\n"
        "Escribí un informe descriptivo en español, con estas secciones:\n"
        "1. Resumen de qué se probó (rango de hiperparámetros, cuántos runs)\n"
        "2. Qué configuración rindió mejor y por qué (priorizando recall, ya "
        "que en churn un falso negativo —cliente que se va sin ser detectado— "
        "cuesta más que un falso positivo)\n"
        "3. Tendencias observadas (¿más árboles/profundidad ayudó o no? "
        "¿hay señales de overfitting u underfitting?)\n"
        "4. Conclusión y recomendación final\n"
        "Sé concreto, citá números de la tabla. Máximo 300 palabras."
    )


def _llamar_ollama(prompt: str) -> str:
    respuesta = requests.post(
        f"{settings.ollama_base_url}/api/generate",
        json={"model": settings.ollama_model, "prompt": prompt, "stream": False},
        timeout=180,
    )
    respuesta.raise_for_status()
    return respuesta.json()["response"].strip()


def _llamar_anthropic(prompt: str) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    respuesta = client.messages.create(
        model=settings.anthropic_model,
        max_tokens=800,
        messages=[{"role": "user", "content": prompt}],
    )
    return respuesta.content[0].text


def generar_informe(runs: list[dict] | None = None) -> str:
    """Genera el informe completo (tabla + análisis del LLM) en Markdown."""
    runs = runs if runs is not None else recolectar_runs()
    tabla = _tabla_markdown(runs)
    prompt = _construir_prompt(tabla)

    if settings.llm_provider == "ollama":
        analisis = _llamar_ollama(prompt)
    elif settings.llm_provider == "anthropic":
        if not settings.anthropic_api_key:
            analisis = (
                "[Sin ANTHROPIC_API_KEY configurada — seteá LLM_PROVIDER=ollama "
                "o completá la key]"
            )
        else:
            analisis = _llamar_anthropic(prompt)
    else:
        raise ValueError(
            f"llm_provider desconocido: '{settings.llm_provider}' (usar 'ollama' o 'anthropic')"
        )

    titulo = f"# Informe de experimentos — {settings.registered_model_name}"
    return f"{titulo}\n\n{tabla}\n## Análisis\n\n{analisis}\n"
