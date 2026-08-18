"""Configuración centralizada del proyecto, leída de variables de entorno / .env.

Los paths son métodos parametrizados por nombre de caso de uso (no campos
fijos): cada plugin (churn, fraude, ...) tiene su propia subcarpeta bajo
data/ y models/, y su propio experimento de MLflow — nunca comparten
artefactos entre sí.
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ROOT_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Caso de uso activo por default para los entrypoints que no reciben
    # --usecase explícito (por ejemplo serve.py, que uvicorn importa
    # directamente sin poder pasarle argumentos de línea de comandos).
    usecase: str = "churn"

    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-sonnet-4-6"

    # Proveedor de LLM para generar_informe.py: "ollama" (local, gratis, sin
    # API key — requiere tener Ollama corriendo) o "anthropic" (requiere
    # ANTHROPIC_API_KEY con crédito).
    llm_provider: str = "ollama"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"

    data_dir: Path = ROOT_DIR / "data"
    models_dir: Path = ROOT_DIR / "models"

    drift_p_value_threshold: float = 0.05

    # Si esta fracción (o más) de las features monitoreadas muestra drift,
    # monitor.py dispara un reentrenamiento automático incorporando la
    # cohorte de monitoreo (que ya tiene el resultado real conocido) al set
    # de entrenamiento. Con datasets reales casi siempre hay ALGO de drift
    # en alguna feature — por eso el umbral es una fracción del total, no
    # "cualquier feature con drift ya alcanza".
    drift_retrain_fraction_threshold: float = 0.5

    # Métrica usada para elegir automáticamente el mejor modelo entre runs.
    # Se prioriza recall: en los dos casos de uso actuales (churn, fraude),
    # un falso negativo cuesta más que un falso positivo.
    model_selection_metric: str = "recall"
    champion_alias: str = "champion"

    # Umbral de decisión usado por serve.py si no hay un umbral optimizado
    # guardado (por ejemplo, antes de la primera aprobación).
    prediction_threshold_default: float = 0.5

    def raw_dataset_path(self, usecase: str) -> Path:
        return self.data_dir / usecase / "datos_crudos.csv"

    def train_csv_path(self, usecase: str) -> Path:
        return self.data_dir / usecase / "entrenamiento.csv"

    def monitor_csv_path(self, usecase: str) -> Path:
        return self.data_dir / usecase / "monitoreo.csv"

    def train_stats_path(self, usecase: str) -> Path:
        return self.data_dir / usecase / "estadisticas_entrenamiento.csv"

    def model_path(self, usecase: str) -> Path:
        return self.models_dir / usecase / "modelo_actual.pkl"

    def threshold_path(self, usecase: str) -> Path:
        return self.models_dir / usecase / "umbral.json"

    def pending_candidate_path(self, usecase: str) -> Path:
        return self.models_dir / usecase / "candidato_pendiente.json"


settings = Settings()
