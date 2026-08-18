# MLOps end-to-end: predicción de churn (Telco Customer Churn, IBM)

Pipeline completo de ML en producción: datos reales, tracking de
experimentos, model registry, serving vía API, monitoreo de drift,
reentrenamiento automático y un gate humano de aprobación antes de tocar
producción — con un reporte generado por un LLM en el medio.

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

## Qué haría este modelo en producción

El modelo no "decide" nada por sí mismo — solo puntúa clientes. Lo que
dispara una acción real es el sistema que lo consume:

- **Scoring por lotes** (ej. cada noche): un job llama a `/predict` para
  toda la base de clientes activos y guarda `probabilidad_churn` +
  `va_a_abandonar` en un data warehouse.
- **Scoring en tiempo real**: cuando cambia algo relevante del cliente
  (deja de pagar, cierra un producto, llama mucho a soporte), el sistema
  que dispara ese evento llama a `/predict` al vuelo.
- **Downstream**: la lista de clientes con `va_a_abandonar=true` (según el
  umbral optimizado, ver más abajo) alimenta al equipo de retención —
  dispara una campaña automática (mail, descuento) o una alerta para que
  un agente llame proactivamente.

El modelo nunca decide "retener" por sí mismo — decide a quién mirar
primero. La política de negocio (a quién ofrecerle qué, con qué
presupuesto) queda afuera del modelo, en reglas separadas. Justamente por
eso importa el gate humano de aprobación: si el modelo alimenta decisiones
reales de negocio (a quién ofrecerle descuentos; en un caso fintech, a
quién ajustarle un límite de crédito), un cambio de modelo sin revisión
puede alterar esas decisiones a gran escala de un día para el otro.

## Estructura del proyecto

```
src/churn_mlops/       paquete instalable (pip install -e .)
  config.py              settings tipadas, leídas de .env
  data.py                carga, limpieza y split train/monitor
  train.py                pipeline sklearn (ColumnTransformer + RandomForest) + MLflow
                            + selección de candidato + promoción a producción
  serve.py                lógica de la API FastAPI
  monitor.py              detección de drift (KS + chi-cuadrado) + reporte LLM
                            + diagnóstico de reentrenamiento
  logging_config.py       logging compartido

prepare_data.py         entrypoint: genera los CSV de train/monitor
train.py                entrypoint: corre los 3 experimentos, trackea en MLflow
                          y deja el mejor candidato pendiente de aprobación
monitor.py              entrypoint: corre el test de drift + reporte, y si el
                          drift es significativo reentrena (también queda
                          pendiente de aprobación, nunca se auto-promueve)
aprobar_modelo.py       entrypoint: gate humano — revisa el candidato pendiente
                          y, si se confirma, lo promueve a producción
serve.py                entrypoint: expone `app` para uvicorn

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

# 3. Entrenar (corre 3 experimentos con distintos hiperparámetros y deja el
#    mejor candidato pendiente de aprobación — ver sección de abajo)
python train.py

# 4. Ver los experimentos comparados visualmente
mlflow ui
# abrir http://localhost:5000

# 5. Revisar y aprobar el candidato (gate humano, ver sección de abajo)
python aprobar_modelo.py

# 6. Levantar la API de predicción
uvicorn serve:app --reload
# abrir http://localhost:8000/docs

# 7. Correr el monitoreo de drift (necesita ANTHROPIC_API_KEY en .env para
#    el reporte en lenguaje natural; sin ella igual muestra los resultados
#    numéricos de los tests estadísticos). Si el drift es significativo,
#    reentrena y deja otro candidato pendiente — repetir el paso 5.
python monitor.py
```

**Nunca hardcodees la API key en el README ni en ningún archivo versionado**
— va en `.env` (que está en `.gitignore`) o como variable de entorno.

### Selección de candidato + gate humano de aprobación

`train.py` corre los 3 experimentos y, al final, `identificar_mejor_candidato()`
recorre **todos los runs históricos** del experimento en MLflow (no solo los
3 que se acaban de correr), elige el que mejor puntúa en
`settings.model_selection_metric` (default: **recall** — en churn, un falso
negativo cuesta más que un falso positivo, ver discusión en el código de
`config.py`), y lo deja anotado en `models/candidato_pendiente.json`.

**Ahí se frena.** Ni `train.py` ni `monitor.py` tocan el modelo que sirve
`serve.py`, ni el alias `"champion"` del Model Registry — eso requiere
correr explícitamente:

```bash
python aprobar_modelo.py            # muestra el candidato y pide confirmación
python aprobar_modelo.py --yes      # aprueba sin preguntar (uso en scripts/CI)
```

Recién ahí `promover_a_produccion()` setea el alias `"champion"`, copia el
modelo a `models/modelo_actual.pkl` y su umbral a `models/umbral.json` — lo
que efectivamente sirve la API.

Se puede correr la identificación de candidato por separado, sin reentrenar:

```bash
python -c "from churn_mlops.train import identificar_mejor_candidato; print(identificar_mejor_candidato())"
```

O cambiar el criterio (por ejemplo, priorizar F1 en vez de recall):

```bash
python -c "from churn_mlops.train import identificar_mejor_candidato; print(identificar_mejor_candidato(metric='f1_score'))"
```

### Reentrenamiento automático por drift

`monitor.py` compara la cohorte de clientes nuevos contra la base histórica
y calcula la **fracción de features con drift**. Si supera
`settings.drift_retrain_fraction_threshold` (default: 50%), dispara
`reentrenar_con_datos_combinados()`: incorpora la cohorte de monitoreo (que
ya tiene el churn real conocido, no es "el futuro" sin etiqueta) al set de
entrenamiento, corre los 3 experimentos de nuevo, e identifica un candidato
— que, igual que en `train.py`, **queda pendiente de aprobación**, nunca se
promueve solo.

```bash
python monitor.py                 # reporta y reentrena si hace falta
python monitor.py --no-retrain    # solo reporta, nunca reentrena
```

El umbral de fracción es una perilla a propósito: con datos reales casi
siempre hay *algo* de drift en alguna feature, así que reentrenar por
"cualquier feature cambió" sería ruidoso. Se dispara solo cuando cambió una
porción sustancial de la población.

### Umbral de decisión ajustable por costo de negocio

`serve.py` no usa `probabilidad > 0.5` a secas: usa el umbral que minimiza
el costo esperado, calculado en `entrenar()` con `optimizar_umbral()`
(barrido de umbrales evaluando `falsos_negativos * costo_falso_negativo +
falsos_positivos * costo_falso_positivo`). Los costos son configurables en
`settings` (`costo_falso_negativo=5.0`, `costo_falso_positivo=1.0` por
default: perder un cliente sale ~5 veces más caro que una promo de
retención de más).

Ese umbral óptimo se loguea como parámetro de cada run en MLflow, viaja con
el candidato, y `promover_a_produccion()` lo persiste en `models/umbral.json`
junto con el modelo aprobado — `serve.py` lo carga en el arranque
(`GET /health` lo expone como `umbral_decision`, y cada respuesta de
`POST /predict` incluye `umbral_usado`). Si no existe ese archivo (por
ejemplo, antes de la primera aprobación), cae al default de
`settings.prediction_threshold_default` (0.5).

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

Requiere que `models/modelo_actual.pkl` ya exista — correr `train.py` +
`aprobar_modelo.py` antes del build.

## Qué conecta con cada materia
- **MLOps**: todo el pipeline (tracking, registry, serving, monitoreo,
  reentrenamiento automático, gate de aprobación, CI, containerización)
- **Big Data**: si el dataset fuera masivo, `prepare_data.py` se reemplazaría
  por un preprocesamiento en Spark antes de entrenar
- **LLMs**: `monitor.py` usa Claude para redactar el reporte de drift en
  lenguaje natural, a partir del contexto numérico de los tests estadísticos
