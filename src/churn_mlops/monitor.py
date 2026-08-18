"""Monitoreo de data drift + reporte automático generado por un LLM.

Numéricas (tenure, MonthlyCharges, TotalCharges): test de Kolmogorov-Smirnov.
Categóricas (Contract, InternetService, etc.): test de chi-cuadrado sobre las
frecuencias de cada categoría. Mezclar ambos tipos de features bajo un único
test (como haría un enfoque naive) no es estadísticamente correcto — KS
asume variables continuas.
"""

import logging

import pandas as pd
from scipy import stats

from churn_mlops.config import settings
from churn_mlops.data import CAT_FEATURES, NUM_FEATURES

logger = logging.getLogger(__name__)


def _test_numerico(train: pd.Series, nuevo: pd.Series) -> tuple[float, str, str]:
    _, p_valor = stats.ks_2samp(train, nuevo)
    return p_valor, "KS", f"{train.mean():.2f} / {nuevo.mean():.2f}"


def _test_categorico(train: pd.Series, nuevo: pd.Series) -> tuple[float, str, str]:
    categorias = sorted(set(train.unique()) | set(nuevo.unique()))
    tabla = pd.DataFrame(
        {
            "train": train.value_counts().reindex(categorias, fill_value=0),
            "nuevo": nuevo.value_counts().reindex(categorias, fill_value=0),
        }
    )
    _, p_valor, _, _ = stats.chi2_contingency(tabla.T)
    top_train = train.value_counts(normalize=True).idxmax()
    top_nuevo = nuevo.value_counts(normalize=True).idxmax()
    return p_valor, "chi2", f"'{top_train}' / '{top_nuevo}'"


def detectar_drift(df_train: pd.DataFrame, df_nuevo: pd.DataFrame) -> list[dict]:
    resultados = []
    umbral = settings.drift_p_value_threshold

    for feature in NUM_FEATURES:
        p_valor, test, resumen = _test_numerico(df_train[feature], df_nuevo[feature])
        resultados.append(
            {"feature": feature, "test": test, "p_valor": round(p_valor, 4),
             "drift_detectado": bool(p_valor < umbral), "resumen_train_vs_nuevo": resumen}
        )

    for feature in CAT_FEATURES:
        p_valor, test, resumen = _test_categorico(df_train[feature], df_nuevo[feature])
        resultados.append(
            {"feature": feature, "test": test, "p_valor": round(p_valor, 4),
             "drift_detectado": bool(p_valor < umbral), "resumen_train_vs_nuevo": resumen}
        )

    return resultados


def evaluar_necesidad_reentrenamiento(resultados: list[dict]) -> dict:
    """Decide si el drift detectado amerita reentrenar automáticamente.

    Se dispara por FRACCIÓN de features con drift, no por "alguna feature
    tiene drift" — con datos reales casi siempre hay alguna señal de cambio
    en algo, y reentrenar en cada corrida de monitor.py sería ruidoso e
    inútil. El umbral (settings.drift_retrain_fraction_threshold) es la
    perilla que separa "ruido normal" de "la población realmente cambió".
    """
    total = len(resultados)
    con_drift = sum(1 for r in resultados if r["drift_detectado"])
    fraccion = con_drift / total if total else 0.0
    umbral = settings.drift_retrain_fraction_threshold

    return {
        "features_con_drift": con_drift,
        "features_totales": total,
        "fraccion_drift": round(fraccion, 3),
        "umbral": umbral,
        "debe_reentrenar": fraccion >= umbral,
    }


def generar_reporte_llm(resultados: list[dict]) -> str:
    """Usa Claude para redactar un resumen ejecutivo del drift detectado."""
    resumen_datos = "\n".join(
        f"- {r['feature']} (test {r['test']}): p-valor={r['p_valor']}, "
        f"drift={'SÍ' if r['drift_detectado'] else 'NO'}, "
        f"train vs nuevo={r['resumen_train_vs_nuevo']}"
        for r in resultados
    )

    prompt = (
        "Sos un ingeniero de ML analizando resultados de un test de data drift "
        "sobre un modelo de predicción de churn de telecom. Te paso los resultados "
        "por feature (comparando la base histórica de entrenamiento contra la "
        "cohorte de clientes más nuevos). Escribí un resumen ejecutivo breve "
        "(máximo 4 líneas) para un equipo no técnico, explicando qué cambió y "
        "si recomendás reentrenar el modelo.\n\n"
        f"RESULTADOS:\n{resumen_datos}"
    )

    if not settings.anthropic_api_key:
        return "[Sin API key configurada — reporte no generado. Seteá ANTHROPIC_API_KEY en .env]"

    try:
        import anthropic

        client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        respuesta = client.messages.create(
            model=settings.anthropic_model,
            max_tokens=250,
            messages=[{"role": "user", "content": prompt}],
        )
        return respuesta.content[0].text
    except Exception as e:
        logger.exception("Error generando el reporte con el LLM")
        return f"[Error generando el reporte: {e}]"
