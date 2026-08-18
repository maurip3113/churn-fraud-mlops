import mlflow

from churn_mlops.train import build_pipeline, entrenar


def test_build_pipeline_predice_probabilidades(df_train_fake):
    from churn_mlops.data import FEATURES, TARGET

    pipeline = build_pipeline(n_estimators=10, max_depth=3)
    X = df_train_fake[FEATURES]
    y = df_train_fake[TARGET]
    pipeline.fit(X, y)

    proba = pipeline.predict_proba(X)

    assert proba.shape == (len(X), 2)
    assert (proba >= 0).all() and (proba <= 1).all()


def test_entrenar_devuelve_metricas_validas(df_train_fake, tmp_path, monkeypatch):
    from churn_mlops.config import settings

    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    mlflow.set_experiment("test_churn_prediction")
    monkeypatch.setattr(settings, "model_path", tmp_path / "modelo.pkl")
    monkeypatch.setattr(settings, "train_stats_path", tmp_path / "stats.csv")

    metricas = entrenar(df_train_fake, n_estimators=10, max_depth=3)

    assert set(metricas) == {"accuracy", "f1_score", "precision", "recall"}
    assert all(0.0 <= v <= 1.0 for v in metricas.values())
    assert (tmp_path / "modelo.pkl").exists()
    assert (tmp_path / "stats.csv").exists()
