# MLOps end-to-end multi-caso-de-uso (churn + fraude)

Motor genérico de ML en producción — tracking de experimentos, model
registry, serving vía API, monitoreo de drift, reentrenamiento automático
y un gate humano de aprobación antes de tocar producción, con un informe
generado por un LLM — que sirve para **cualquier** problema de
clasificación binaria enchufado como plugin. Vienen dos casos de uso ya
armados: **churn** (abandono de clientes, dataset real) y **fraude**
(detección de fraude en transacciones, dataset sintético).

## Arquitectura de plugins

El motor (`train.py`, `monitor.py`, `serve.py`, `informe.py` dentro de
`src/churn_mlops/`) no sabe nada de columnas, targets ni costos de negocio
concretos — todo eso vive en una instancia de `UseCase`
([usecases/base.py](src/churn_mlops/usecases/base.py)) que cada plugin
construye:

```python
@dataclass(frozen=True)
class UseCase:
    name: str                    # nombra el experimento de MLflow y las subcarpetas
    num_features: list[str]
    cat_features: list[str]
    target: str
    request_model: type[BaseModel]      # schema de POST /predict
    costo_falso_negativo: float
    costo_falso_positivo: float
    asegurar_datos_crudos: Callable     # descarga o genera el CSV crudo
    cargar_y_limpiar: Callable
    separar_train_monitor: Callable     # cohorte de monitoreo con sentido de negocio
```

Agregar un caso de uso nuevo: escribir un módulo en `usecases/` que arme
uno de estos, y sumarlo a `usecases/registry.py`. El resto del pipeline
—MLflow, selección de candidato, umbral por costo, gate de aprobación,
detección de drift, informe con LLM— no cambia una línea.

```
src/churn_mlops/
  usecases/
    base.py       contrato UseCase
    churn.py       plugin: abandono de clientes (IBM Telco, dataset real)
    fraude.py      plugin: fraude en transacciones (dataset sintético)
    registry.py    get_usecase(nombre)
  config.py        settings + paths parametrizados por usecase
  train.py         motor: pipeline sklearn + MLflow + candidato + promoción
  serve.py         motor: build_app(usecase) → FastAPI
  monitor.py       motor: drift (KS + chi-cuadrado) + reentrenamiento
  informe.py       motor: informe descriptivo con LLM (Ollama/Claude)

prepare_data.py / train.py / monitor.py / aprobar_modelo.py / generar_informe.py
  entrypoints — todos aceptan --usecase churn|fraude
serve.py (raíz)
  entrypoint — uvicorn no acepta argparse, así que el usecase se elige con
  la variable de entorno USECASE (default "churn")
```

## Los dos casos de uso

| | **churn** | **fraude** |
|---|---|---|
| Dataset | [IBM Telco Customer Churn](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv), real, ~7000 filas | Sintético, generado por `asegurar_datos_crudos()`, 100000 filas |
| Desbalance | ~20% positivo | ~1.8% positivo (fraude real es así de raro) |
| Cohorte de monitoreo | Clientes con `tenure <= 6` meses | Transacciones por canal `online` |
| Asimetría de costo | Falso negativo 5x más caro | Falso negativo 20x más caro |

No hay un dataset público de fraude gratis sin login (los de Kaggle piden
cuenta), así que `fraude` genera uno sintético — documentado como tal en
[usecases/fraude.py](src/churn_mlops/usecases/fraude.py). El objetivo de
ese plugin no es la fidelidad del dataset: es probar que el motor
funciona igual de bien con un caso de uso genuinamente distinto a churn
(desbalance extremo, costos invertidos, drift por canal en vez de por
antigüedad).

**Bug real que encontramos probando la API**: al principio `es_online`
estaba en `CAT_FEATURES` del modelo, pero como `separar_train_monitor()`
la usa como criterio de corte (monitor = canal online, train = el resto),
esa columna queda **constante** dentro del set de entrenamiento — el
modelo nunca pudo aprender nada de ella (`feature_importance = 0`), y las
predicciones eran casi idénticas sin importar el perfil de riesgo. Se
sacó de las features del modelo (queda solo como criterio interno del
split); con eso y más volumen de datos (100k filas, ~1200 positivos en
training vs. los 42 originales), el recall subió de 0.37 a ~0.52 y las
probabilidades ahora discriminan de verdad entre transacciones normales
(~0.35) y de alto riesgo (~0.63).

## Qué haría un modelo de estos en producción

El modelo no "decide" nada por sí mismo — solo puntúa. Lo que dispara una
acción real es el sistema que lo consume:

- **Scoring por lotes** (ej. cada noche): un job llama a `/predict` para
  toda la base activa y guarda `probabilidad` + `positivo` en un data
  warehouse.
- **Scoring en tiempo real**: cuando pasa algo relevante (un cliente deja
  de pagar; una transacción se procesa), el sistema que dispara ese evento
  llama a `/predict` al vuelo.
- **Downstream**: la lista de casos con `positivo=true` (según el umbral
  optimizado por costo) alimenta al equipo correspondiente — retención en
  churn, revisión manual en fraude.

El modelo nunca actúa por sí solo — decide a quién mirar primero. La
política de negocio (a quién ofrecerle qué, a quién bloquearle una
transacción) queda afuera del modelo, en reglas separadas. Justamente por
eso importa el gate humano de aprobación: si el modelo alimenta decisiones
reales (descuentos, límites de crédito, bloqueos de tarjeta), un cambio de
modelo sin revisión puede alterar esas decisiones a gran escala de un día
para el otro.

## Cómo correrlo, en orden

```bash
# Setup
pip install -r requirements-dev.txt   # runtime + pytest/ruff
pip install -e .                      # instala el paquete churn_mlops
cp .env.example .env                  # completá ANTHROPIC_API_KEY si la tenés

# 1. Datos: para churn hay que traer el dataset real a mano (una sola vez);
#    para fraude se genera solo, no hace falta este paso.
curl -o data/churn/datos_crudos.csv \
  https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv

# 2. Preparar los datos (limpieza + split train/monitor)
python prepare_data.py --usecase churn      # o --usecase fraude

# 3. Entrenar (corre 3 experimentos y deja el mejor candidato pendiente)
python train.py --usecase churn

# 4. Ver los experimentos comparados visualmente
mlflow ui
# abrir http://localhost:5000

# 5. Revisar y aprobar el candidato (gate humano, ver sección de abajo)
python aprobar_modelo.py --usecase churn

# 6. Levantar la API de predicción
uvicorn serve:app --reload                    # sirve settings.usecase (default "churn")
USECASE=fraude uvicorn serve:app --port 8001  # o levantar fraude en otro puerto
# abrir http://localhost:8000/docs

# 7. Monitoreo de drift + reentrenamiento automático si hace falta
python monitor.py --usecase churn
```

**Nunca hardcodees la API key en el README ni en ningún archivo versionado**
— va en `.env` (que está en `.gitignore`) o como variable de entorno.

### Selección de candidato + gate humano de aprobación

`train.py` corre los 3 experimentos y, al final, `identificar_mejor_candidato()`
recorre **todos los runs históricos** del experimento de ese caso de uso en
MLflow, elige el que mejor puntúa en `settings.model_selection_metric`
(default: **recall**), y lo deja anotado en
`models/<usecase>/candidato_pendiente.json`.

**Ahí se frena.** Ni `train.py` ni `monitor.py` tocan el modelo que sirve
la API, ni el alias `"champion"` del Model Registry — eso requiere correr
explícitamente:

```bash
python aprobar_modelo.py --usecase churn            # muestra el candidato y pide confirmación
python aprobar_modelo.py --usecase churn --yes      # aprueba sin preguntar (uso en scripts/CI)
```

Recién ahí `promover_a_produccion()` setea el alias `"champion"`, copia el
modelo a `models/<usecase>/modelo_actual.pkl` y su umbral a
`models/<usecase>/umbral.json` — lo que efectivamente sirve la API.

### Reentrenamiento automático por drift

`monitor.py` compara la cohorte reciente contra la base histórica y
calcula la **fracción de features con drift**. Si supera
`settings.drift_retrain_fraction_threshold` (default: 50%), dispara
`reentrenar_con_datos_combinados()`: incorpora la cohorte de monitoreo (que
ya tiene el resultado real conocido, no es "el futuro" sin etiqueta) al set
de entrenamiento, corre los 3 experimentos de nuevo, e identifica un
candidato — que, igual que en `train.py`, **queda pendiente de
aprobación**, nunca se promueve solo.

```bash
python monitor.py --usecase churn                  # reporta y reentrena si hace falta
python monitor.py --usecase churn --no-retrain     # solo reporta, nunca reentrena
```

El umbral de fracción es una perilla a propósito: con datos reales casi
siempre hay *algo* de drift en alguna feature, así que reentrenar por
"cualquier feature cambió" sería ruidoso. Se dispara solo cuando cambió una
porción sustancial de la población.

### Umbral de decisión ajustable por costo de negocio

`serve.py` no usa `probabilidad > 0.5` a secas: usa el umbral que minimiza
el costo esperado, calculado en `entrenar()` con `optimizar_umbral()`
(barrido de umbrales evaluando `falsos_negativos * costo_falso_negativo +
falsos_positivos * costo_falso_positivo`). Los costos son propios de cada
`UseCase` — en churn, un falso negativo sale ~5x más caro que uno
positivo; en fraude, ~20x, porque no detectar un fraude es mucho más grave
que bloquear por error una transacción legítima.

Ese umbral óptimo se loguea como parámetro de cada run en MLflow, viaja con
el candidato, y `promover_a_produccion()` lo persiste junto con el modelo
aprobado — la API lo carga en el arranque (`GET /health` lo expone como
`umbral_decision`, y cada respuesta de `POST /predict` incluye
`umbral_usado`). Si no existe ese archivo (antes de la primera aprobación),
cae al default de `settings.prediction_threshold_default` (0.5).

### Informe descriptivo de todos los experimentos

`generar_informe.py` junta hiperparámetros y métricas de **todos los runs
históricos** del experimento de un caso de uso y le pide a un LLM que
redacte un análisis: qué se probó, qué configuración rindió mejor y por
qué, tendencias (over/underfitting), y una conclusión.

Por default usa **Ollama local** (`settings.llm_provider = "ollama"`) —
gratis, sin API key, sin depender de crédito de ningún proveedor:

```bash
# 1. Instalar Ollama: https://ollama.com/download
# 2. Bajar un modelo chico (una sola vez)
ollama pull llama3.2
# 3. Generar el informe
python generar_informe.py --usecase churn
```

Para usar Claude en cambio (mejor calidad de análisis, pero requiere
`ANTHROPIC_API_KEY` con crédito), seteá en `.env`:

```
LLM_PROVIDER=anthropic
```

**Limitación conocida**: `llama3.2` (3B parámetros) es liviano y corre en
cualquier PC, pero razona peor sobre tablas numéricas que un modelo
grande — en pruebas reales (tanto en churn como en fraude) llegó a señalar
como "mejor en recall" un run que no tenía el recall más alto de la tabla.
Sirve para un borrador rápido y gratis; para un informe que se vaya a usar
en una decisión real, conviene revisar los números contra la tabla (que sí
es exacta, generada directamente desde MLflow) o usar Claude.

### Tests y lint

```bash
pytest -v
ruff check .
```

Los tests cubren el motor genérico con `churn_usecase` y, en varios casos,
también con `fraude_usecase` — para verificar que el motor no tiene nada
hardcodeado de un caso de uso en particular.

### Docker

```bash
docker build -t churn-api .
docker run -p 8000:8000 churn-api
```

Requiere que `models/churn/modelo_actual.pkl` ya exista — correr `train.py`
+ `aprobar_modelo.py` antes del build. (El Dockerfile sirve `churn` por
default; para `fraude` habría que parametrizar la imagen con `USECASE`.)

## Qué conecta con cada materia
- **MLOps**: todo el pipeline (tracking, registry, serving, monitoreo,
  reentrenamiento automático, gate de aprobación, CI, containerización) —
  y la arquitectura de plugins que lo hace reusable entre casos de uso
- **Big Data**: si un dataset fuera masivo, `asegurar_datos_crudos()` de
  ese plugin se reemplazaría por un preprocesamiento en Spark antes de
  entrenar
- **LLMs**: `monitor.py` usa Claude para redactar el reporte de drift, y
  `generar_informe.py` usa Ollama (o Claude) para el informe descriptivo
  de experimentos
