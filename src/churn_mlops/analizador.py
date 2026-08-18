"""Analizador ad-hoc de datasets: subís un CSV, elegís la columna target, y
te devuelve un perfil estadístico (EDA) + un RandomForest de referencia
entrenado al vuelo, con un resumen en lenguaje natural generado por un LLM.

A diferencia del resto del proyecto (churn/fraude), esto NO registra nada
en MLflow ni pasa por el gate de aprobación de aprobar_modelo.py — es una
herramienta de exploración rápida ("¿qué tiene este dataset, se puede
predecir algo interesante?"), no un pipeline de producción.
"""

import logging
import uuid

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from churn_mlops.config import settings
from churn_mlops.informe import llamar_llm

logger = logging.getLogger(__name__)

UPLOADS_DIR = settings.data_dir / "_uploads"
MAX_FILAS_MUESTRA = 200_000  # evita colgar el server con un CSV gigante


def guardar_csv_temporal(contenido: bytes) -> str:
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    dataset_id = uuid.uuid4().hex
    (UPLOADS_DIR / f"{dataset_id}.csv").write_bytes(contenido)
    return dataset_id


def cargar_csv_temporal(dataset_id: str) -> pd.DataFrame:
    ruta = UPLOADS_DIR / f"{dataset_id}.csv"
    if not ruta.exists():
        raise FileNotFoundError("Dataset no encontrado o expirado — subilo de nuevo.")
    return pd.read_csv(ruta, nrows=MAX_FILAS_MUESTRA)


def perfilar_dataset(df: pd.DataFrame, target: str | None = None) -> dict:
    """Perfil estadístico básico: shape, tipos, nulos, describe() de
    numéricas, top categorías de categóricas, y balance de clases si se
    indica un target."""
    numericas = df.select_dtypes(include="number").columns.tolist()
    categoricas = [c for c in df.columns if c not in numericas]

    perfil = {
        "filas": len(df),
        "columnas": len(df.columns),
        "nulos_por_columna": {c: int(v) for c, v in df.isna().sum().items() if v > 0},
        "numericas": numericas,
        "categoricas": categoricas,
        "describe_numericas": df[numericas].describe().round(2).to_dict() if numericas else {},
        "top_categorias": {
            c: df[c].value_counts(normalize=True).head(3).round(3).to_dict()
            for c in categoricas
            if c != target
        },
    }
    if target and target in df.columns:
        perfil["balance_target"] = df[target].value_counts(normalize=True).round(4).to_dict()
    return perfil


def inferir_features(df: pd.DataFrame, target: str) -> tuple[list[str], list[str]]:
    """Separa el resto de las columnas en numéricas/categóricas por dtype."""
    candidatas = [c for c in df.columns if c != target]
    numericas = df[candidatas].select_dtypes(include="number").columns.tolist()
    categoricas = [c for c in candidatas if c not in numericas]
    return numericas, categoricas


def entrenar_rapido(
    df: pd.DataFrame, target: str, num_features: list[str], cat_features: list[str]
) -> dict:
    """Entrena un único RandomForest de referencia (no 3 experimentos, no
    MLflow) — el objetivo es una señal rápida de "esto se puede predecir o
    no", no un modelo para producción."""
    y = df[target]
    if y.nunique() < 2:
        raise ValueError(f"La columna target '{target}' tiene un solo valor distinto.")
    if y.nunique() > 20:
        raise ValueError(
            f"La columna target '{target}' tiene {y.nunique()} valores distintos — "
            "elegí una columna categórica con pocas clases (esto entrena un clasificador)."
        )

    X = df[num_features + cat_features]
    conteos = y.value_counts()
    puede_estratificar = (conteos >= 2).all()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if puede_estratificar else None
    )

    preprocesador = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
        ]
    )
    pipeline = Pipeline(
        [
            ("preprocesador", preprocesador),
            (
                "clasificador",
                RandomForestClassifier(
                    n_estimators=150, max_depth=8, random_state=42, class_weight="balanced"
                ),
            ),
        ]
    )
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # average="weighted" funciona igual de bien para target binario o
    # multiclase, sin tener que adivinar cuál es la "clase positiva" en un
    # dataset arbitrario subido por el usuario.
    return {
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_clases": int(y.nunique()),
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
    }


def _construir_prompt_ad_hoc(nombre: str, target: str, perfil: dict, metricas: dict | None) -> str:
    resumen_perfil = (
        f"Filas: {perfil['filas']}, columnas: {perfil['columnas']}\n"
        f"Numéricas: {', '.join(perfil['numericas']) or '(ninguna)'}\n"
        f"Categóricas: {', '.join(perfil['categoricas']) or '(ninguna)'}\n"
        f"Nulos por columna: {perfil['nulos_por_columna'] or '(sin nulos)'}\n"
        f"Balance del target '{target}': {perfil.get('balance_target', {})}\n"
    )
    resumen_metricas = (
        f"\nModelo RandomForest de referencia entrenado sobre '{target}':\n"
        f"accuracy={metricas['accuracy']:.3f}, f1={metricas['f1_score']:.3f}, "
        f"precision={metricas['precision']:.3f}, recall={metricas['recall']:.3f} "
        f"({metricas['n_train']} filas de train, {metricas['n_test']} de test)\n"
        if metricas
        else "\nNo se entrenó ningún modelo.\n"
    )

    return (
        f"Sos un data scientist analizando un dataset subido por un usuario, "
        f"llamado '{nombre}'. Te paso un perfil estadístico y, si corresponde, "
        f"el resultado de un modelo de clasificación de referencia.\n\n"
        f"{resumen_perfil}{resumen_metricas}\n"
        "Escribí un informe breve en español con estas secciones:\n"
        "1. Qué tipo de dataset parece ser y qué se podría predecir con la "
        f"columna '{target}' como target\n"
        "2. Calidad de los datos (nulos, balance de clases, tamaño)\n"
        "3. Si se entrenó un modelo: qué tan bien predijo y si vale la pena "
        "invertir en mejorarlo o si el dataset no tiene señal suficiente\n"
        "4. Próximos pasos recomendados\n"
        "Sé concreto, citá los números del resumen. Máximo 250 palabras."
    )


def generar_informe_ad_hoc(nombre: str, target: str, perfil: dict, metricas: dict | None) -> str:
    prompt = _construir_prompt_ad_hoc(nombre, target, perfil, metricas)
    analisis = llamar_llm(prompt)
    return analisis
