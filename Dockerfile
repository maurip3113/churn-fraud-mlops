# Imagen para servir la API de predicción. Sirve el caso de uso "churn" por
# default (ver USECASE en serve.py) — requiere que exista
# models/churn/modelo_actual.pkl (train.py + aprobar_modelo.py) antes del
# build; no se entrena dentro de la imagen, se copia el artefacto ya
# aprobado. Para otro caso de uso: -e USECASE=fraude al correr el container.
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt pyproject.toml ./
COPY src/ ./src/
COPY serve.py ./

RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir --no-deps -e .

# El modelo se copia en build time; para desarrollo local podés montarlo
# como volumen en su lugar (ver docker-compose.yml).
COPY models/ ./models/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]
