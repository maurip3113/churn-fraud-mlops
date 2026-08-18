"""Interfaz web del analizador ad-hoc: subís un CSV, elegís el target de
una lista, y te devuelve EDA + un modelo de referencia + resumen del LLM.

Todo HTML server-rendered con formularios planos (sin JavaScript ni
frameworks de frontend) — consistente con el resto del proyecto, que es
Python de punta a punta.
"""

import html
import logging

from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse

from churn_mlops.analizador import (
    cargar_csv_temporal,
    entrenar_rapido,
    generar_informe_ad_hoc,
    guardar_csv_temporal,
    inferir_features,
    perfilar_dataset,
)

logger = logging.getLogger(__name__)


def _esc(valor) -> str:
    """Escapa cualquier valor antes de interpolarlo en HTML — nombres de
    columna, nombre de archivo y hasta el texto del LLM vienen de un CSV
    subido por el usuario (o de contenido generado a partir de él), así que
    se tratan como no confiables."""
    return html.escape(str(valor))


def _pagina(titulo: str, cuerpo: str) -> str:
    return f"""<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<title>{titulo}</title>
<style>
  body {{ font-family: system-ui, sans-serif; max-width: 720px; margin: 40px auto;
          padding: 0 16px; color: #222; }}
  h1 {{ font-size: 1.4rem; }}
  form {{ margin: 24px 0; padding: 16px; border: 1px solid #ddd; border-radius: 8px; }}
  label {{ display: block; margin: 8px 0 4px; font-weight: 600; }}
  select, input[type=file] {{ width: 100%; padding: 8px; box-sizing: border-box; }}
  button {{ margin-top: 16px; padding: 10px 18px; background: #222; color: #fff; border: none;
            border-radius: 6px; cursor: pointer; }}
  button:hover {{ background: #444; }}
  pre {{ white-space: pre-wrap; background: #f7f7f7; padding: 16px; border-radius: 8px; }}
  table {{ border-collapse: collapse; width: 100%; margin: 12px 0; }}
  th, td {{ border: 1px solid #ddd; padding: 6px 10px; text-align: left; font-size: 0.9rem; }}
  a {{ color: #222; }}
</style>
</head>
<body>
<h1>{titulo}</h1>
{cuerpo}
</body>
</html>"""


def _pagina_subida(error: str | None = None) -> str:
    aviso = f'<p style="color:#b00">{_esc(error)}</p>' if error else ""
    return _pagina(
        "Analizador de datasets",
        f"""
<p>Subí un CSV, elegí la columna target, y te devuelvo un perfil
estadístico + un RandomForest de referencia entrenado al vuelo, con un
resumen en lenguaje natural.</p>
{aviso}
<form action="/subir" method="post" enctype="multipart/form-data">
  <label for="archivo">Archivo CSV</label>
  <input type="file" id="archivo" name="archivo" accept=".csv" required>
  <button type="submit">Subir</button>
</form>
""",
    )


def _pagina_columnas(dataset_id: str, nombre: str, columnas: list[str]) -> str:
    opciones = "".join(f'<option value="{_esc(c)}">{_esc(c)}</option>' for c in columnas)
    return _pagina(
        "Elegí la columna target",
        f"""
<p>Dataset: <strong>{_esc(nombre)}</strong> ({len(columnas)} columnas)</p>
<form action="/analizar" method="post">
  <input type="hidden" name="dataset_id" value="{_esc(dataset_id)}">
  <input type="hidden" name="nombre" value="{_esc(nombre)}">
  <label for="target">Columna a predecir (target)</label>
  <select id="target" name="target" required>{opciones}</select>
  <button type="submit">Analizar</button>
</form>
<p><a href="/">&larr; Subir otro dataset</a></p>
""",
    )


def _pagina_informe(nombre: str, perfil: dict, metricas: dict | None, analisis: str) -> str:
    filas_perfil = f"""
<table>
<tr><th>Filas</th><td>{perfil['filas']}</td></tr>
<tr><th>Columnas</th><td>{perfil['columnas']}</td></tr>
<tr><th>Numéricas</th><td>{_esc(', '.join(perfil['numericas']) or '—')}</td></tr>
<tr><th>Categóricas</th><td>{_esc(', '.join(perfil['categoricas']) or '—')}</td></tr>
<tr><th>Nulos</th><td>{_esc(perfil['nulos_por_columna'] or 'sin nulos')}</td></tr>
</table>"""

    filas_metricas = ""
    if metricas:
        filas_metricas = f"""
<h3>Modelo de referencia</h3>
<table>
<tr><th>Accuracy</th><td>{metricas['accuracy']:.3f}</td></tr>
<tr><th>F1 (weighted)</th><td>{metricas['f1_score']:.3f}</td></tr>
<tr><th>Precision (weighted)</th><td>{metricas['precision']:.3f}</td></tr>
<tr><th>Recall (weighted)</th><td>{metricas['recall']:.3f}</td></tr>
<tr><th>Filas train / test</th><td>{metricas['n_train']} / {metricas['n_test']}</td></tr>
</table>"""

    return _pagina(
        f"Informe — {_esc(nombre)}",
        f"""
{filas_perfil}
{filas_metricas}
<h3>Análisis</h3>
<pre>{_esc(analisis)}</pre>
<p><a href="/">&larr; Analizar otro dataset</a></p>
""",
    )


def build_app() -> FastAPI:
    app = FastAPI(title="Analizador de datasets")

    @app.get("/", response_class=HTMLResponse)
    def home():
        return _pagina_subida()

    @app.post("/subir", response_class=HTMLResponse)
    async def subir(archivo: UploadFile):
        if not archivo.filename or not archivo.filename.lower().endswith(".csv"):
            return _pagina_subida(error="Subí un archivo .csv válido.")

        contenido = await archivo.read()
        dataset_id = guardar_csv_temporal(contenido)
        try:
            df = cargar_csv_temporal(dataset_id)
        except Exception as e:
            logger.exception("Error leyendo el CSV subido")
            return _pagina_subida(error=f"No se pudo leer el CSV: {e}")

        if df.shape[1] < 2:
            return _pagina_subida(error="El CSV necesita al menos 2 columnas.")

        return _pagina_columnas(dataset_id, archivo.filename, df.columns.tolist())

    @app.post("/analizar", response_class=HTMLResponse)
    async def analizar(
        dataset_id: str = Form(...), nombre: str = Form(...), target: str = Form(...)
    ):
        try:
            df = cargar_csv_temporal(dataset_id)
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e

        perfil = perfilar_dataset(df, target)

        metricas = None
        try:
            num_features, cat_features = inferir_features(df, target)
            metricas = entrenar_rapido(df, target, num_features, cat_features)
        except ValueError as e:
            logger.info("No se entrenó modelo de referencia: %s", e)

        analisis = generar_informe_ad_hoc(nombre, target, perfil, metricas)

        return _pagina_informe(nombre, perfil, metricas, analisis)

    return app
