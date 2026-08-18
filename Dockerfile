# Imagen para servir la API de predicción de churn.
# Requiere que exista models/modelo_actual.pkl (entrenado con train.py) — no
# se entrena dentro del build, se monta o se copia el artefacto ya generado.
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
