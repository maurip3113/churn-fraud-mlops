"""Configuración centralizada del proyecto, leída de variables de entorno / .env."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ROOT_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-sonnet-4-6"

    data_dir: Path = ROOT_DIR / "data"
    models_dir: Path = ROOT_DIR / "models"

    raw_dataset_path: Path = ROOT_DIR / "data" / "telco_churn_raw.csv"
    model_path: Path = ROOT_DIR / "models" / "modelo_actual.pkl"
    train_stats_path: Path = ROOT_DIR / "data" / "estadisticas_entrenamiento.csv"

    mlflow_experiment_name: str = "churn_prediction"
    drift_p_value_threshold: float = 0.05

    # Métrica usada para elegir automáticamente el mejor modelo entre runs.
    # Se prioriza recall: en churn, un falso negativo (cliente que se va y no
    # lo detectamos) cuesta más que un falso positivo (oferta de retención
    # de más a alguien que igual se quedaba).
    model_selection_metric: str = "recall"
    registered_model_name: str = "churn_predictor"
    champion_alias: str = "champion"

    # Cohorte usada para simular un batch "nuevo" a monitorear: clientes con
    # tenure <= este valor (meses) se separan del set de entrenamiento y se
    # tratan como la población entrante más reciente.
    monitor_tenure_cutoff_months: int = 6


settings = Settings()
