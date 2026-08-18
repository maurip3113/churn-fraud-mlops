# MLOps end-to-end: predicción de churn (Telco Customer Churn, IBM)

Pipeline completo de ML en producción: datos reales, tracking de
experimentos, model registry, serving vía API, monitoreo de drift y un
reporte generado por un LLM.

## Dataset

[IBM Telco Customer Churn](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv):
~7000 clientes reales de una empresa de telecom, con 19 features (contrato,
servicios contratados, forma de pago, gasto) y la variable objetivo `Churn`.

El dataset es una foto estática sin columna de fecha, así que para el
monitoreo de drift no se simula "el futuro" de forma artificial: se separa
una cohorte real y con sentido de negocio — los clientes con `tenure <= 6`
meses quedan afuera del entrenamiento y se usan como el "batch nuevo" a
monitorear. Es el mismo tipo de comparación (clientes recién adquiridos vs.
base histórica) que un equipo de MLOps haría en producción.

## Estructura del proyecto

```
src/churn_mlops/       paquete instalable (pip install -e .)
  config.py              settings tipadas, leídas de .env
  data.py                carga, limpieza y split train/monitor
  train.py                pipeline sklearn (ColumnTransformer + RandomForest) + MLflow
  serve.py                lógica de la API FastAPI
  monitor.py              detección de drift (KS + chi-cuadrado) + reporte LLM
  logging_config.py       logging compartido

prepare_data.py         entrypoint: genera los CSV de train/monitor
train.py                entrypoint: corre los 3 experimentos, trackea en MLflow
                          y elige automáticamente el mejor modelo
serve.py                entrypoint: expone `app` para uvicorn
monitor.py              entrypoint: corre el test de drift + reporte

tests/                  pytest (datos, drift, entrenamiento, API)
Dockerfile              imagen para servir la API
.github/workflows/ci.yml  lint (ruff) + tests en cada push/PR
```

## Cómo correrlo, en orden

```bash
# Setup
pip install -r requirements-dev.txt   # runtime + pytest/ruff
pip install -e .                      # instala el paquete churn_mlops
cp .env.example .env                  # completá ANTHROPIC_API_KEY si la tenés

# 1. Descargar el dataset real (una sola vez)
curl -o data/telco_churn_raw.csv \
  https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv

# 2. Preparar los datos (limpieza + split train/monitor)
python prepare_data.py

# 3. Entrenar (corre 3 experimentos con distintos hiperparámetros y elige
#    automáticamente el mejor modelo — ver sección de abajo)
python train.py

# 4. Ver los experimentos comparados visualmente
mlflow ui
# abrir http://localhost:5000

# 5. Levantar la API de predicción
uvicorn serve:app --reload
# abrir http://localhost:8000/docs

# 6. Correr el monitoreo de drift (necesita ANTHROPIC_API_KEY en .env para
#    el reporte en lenguaje natural; sin ella igual muestra los resultados
#    numéricos de los tests estadísticos)
python monitor.py
```

**Nunca hardcodees la API key en el README ni en ningún archivo versionado**
— va en `.env` (que está en `.gitignore`) o como variable de entorno.

### Selección automática del mejor modelo

Al final de `train.py`, `seleccionar_mejor_modelo()` recorre **todos los runs
históricos** del experimento en MLflow (no solo los 3 que se acaban de
correr), elige el que mejor puntúa en `settings.model_selection_metric`
(default: **recall** — en churn, un falso negativo cuesta más que un falso
positivo, ver discusión en el código de `config.py`), lo promueve a alias
`"champion"` en el Model Registry, y copia ese modelo a
`models/modelo_actual.pkl` para que `serve.py` lo sirva.

Se puede correr por separado en cualquier momento, sin reentrenar:

```bash
python -c "from churn_mlops.train import seleccionar_mejor_modelo; print(seleccionar_mejor_modelo())"
```

O cambiar el criterio (por ejemplo, priorizar F1 en vez de recall):

```bash
python -c "from churn_mlops.train import seleccionar_mejor_modelo; print(seleccionar_mejor_modelo(metric='f1_score'))"
```

### Umbral de decisión ajustable por costo de negocio

`serve.py` no usa `probabilidad > 0.5` a secas: usa el umbral que minimiza
el costo esperado, calculado en `entrenar()` con `optimizar_umbral()`
(barrido de umbrales evaluando `falsos_negativos * costo_falso_negativo +
falsos_positivos * costo_falso_positivo`). Los costos son configurables en
`settings` (`costo_falso_negativo=5.0`, `costo_falso_positivo=1.0` por
default: perder un cliente sale ~5 veces más caro que una promo de
retención de más).

Ese umbral óptimo se loguea como parámetro de cada run en MLflow, y
`seleccionar_mejor_modelo()` lo persiste en `models/umbral.json` junto con
el modelo campeón — `serve.py` lo carga en el arranque (`GET /health` lo
expone como `umbral_decision`, y cada respuesta de `POST /predict` incluye
`umbral_usado`). Si no existe ese archivo (por ejemplo, antes del primer
`train.py`), cae al default de `settings.prediction_threshold_default`
(0.5).

### Tests y lint

```bash
pytest -v
ruff check .
```

### Docker

```bash
docker build -t churn-api .
docker run -p 8000:8000 churn-api
```

Requiere que `models/modelo_actual.pkl` ya exista (correr `train.py` antes
del build).

## Qué conecta con cada materia
- **MLOps**: todo el pipeline (tracking, registry, serving, monitoreo, CI, containerización)
- **Big Data**: si el dataset fuera masivo, `prepare_data.py` se reemplazaría
  por un preprocesamiento en Spark antes de entrenar
- **LLMs**: `monitor.py` usa Claude para redactar el reporte de drift en
  lenguaje natural, a partir del contexto numérico de los tests estadísticos
