import pandas as pd
import pytest
from fastapi.testclient import TestClient

from churn_mlops.analizador import (
    cargar_csv_temporal,
    entrenar_rapido,
    generar_informe_ad_hoc,
    guardar_csv_temporal,
    inferir_features,
    perfilar_dataset,
)
from churn_mlops.web_analizador import build_app


@pytest.fixture
def uploads_dir_aislado(tmp_path, monkeypatch):
    import churn_mlops.analizador as analizador_mod

    monkeypatch.setattr(analizador_mod, "UPLOADS_DIR", tmp_path / "_uploads")
    return analizador_mod


def test_guardar_y_cargar_csv_temporal(uploads_dir_aislado):
    contenido = b"a,b\n1,2\n3,4\n"

    dataset_id = guardar_csv_temporal(contenido)
    df = cargar_csv_temporal(dataset_id)

    assert list(df.columns) == ["a", "b"]
    assert len(df) == 2


def test_cargar_csv_temporal_inexistente_lanza_error(uploads_dir_aislado):
    with pytest.raises(FileNotFoundError):
        cargar_csv_temporal("no-existe")


def test_perfilar_dataset_detecta_nulos_y_tipos():
    df = pd.DataFrame({
        "edad": [20, 30, None, 40],
        "categoria": ["a", "b", "a", "c"],
        "target": [0, 1, 0, 1],
    })

    perfil = perfilar_dataset(df, target="target")

    assert perfil["filas"] == 4
    assert perfil["numericas"] == ["edad", "target"]
    assert perfil["categoricas"] == ["categoria"]
    assert perfil["nulos_por_columna"] == {"edad": 1}
    assert perfil["balance_target"] == {0: 0.5, 1: 0.5}


def test_inferir_features_separa_por_dtype():
    df = pd.DataFrame({
        "num1": [1, 2, 3],
        "num2": [1.5, 2.5, 3.5],
        "cat1": ["x", "y", "z"],
        "target": [0, 1, 0],
    })

    num, cat = inferir_features(df, target="target")

    assert set(num) == {"num1", "num2"}
    assert cat == ["cat1"]


def test_entrenar_rapido_devuelve_metricas_validas():
    import numpy as np

    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({
        "x1": rng.normal(size=n),
        "x2": rng.choice(["a", "b"], n),
        "y": rng.choice([0, 1], n),
    })

    metricas = entrenar_rapido(df, target="y", num_features=["x1"], cat_features=["x2"])

    assert set(metricas) == {
        "n_train", "n_test", "n_clases", "accuracy", "f1_score", "precision", "recall",
    }
    assert metricas["n_clases"] == 2
    assert 0.0 <= metricas["accuracy"] <= 1.0


def test_entrenar_rapido_target_constante_lanza_error():
    df = pd.DataFrame({"x1": [1, 2, 3], "y": [0, 0, 0]})

    with pytest.raises(ValueError):
        entrenar_rapido(df, target="y", num_features=["x1"], cat_features=[])


def test_entrenar_rapido_target_con_demasiadas_clases_lanza_error():
    df = pd.DataFrame({"x1": range(30), "y": range(30)})  # 30 clases distintas

    with pytest.raises(ValueError):
        entrenar_rapido(df, target="y", num_features=["x1"], cat_features=[])


def test_generar_informe_ad_hoc_usa_el_llm_configurado(monkeypatch):
    from churn_mlops.config import settings

    monkeypatch.setattr(settings, "llm_provider", "ollama")
    monkeypatch.setattr(
        "churn_mlops.analizador.llamar_llm", lambda prompt: "Análisis de prueba del dataset."
    )

    df = pd.DataFrame({"x1": [1, 2, 3], "target": [0, 1, 0]})
    perfil = perfilar_dataset(df, target="target")

    resultado = generar_informe_ad_hoc("mi_dataset.csv", "target", perfil, None)

    assert resultado == "Análisis de prueba del dataset."


# --- endpoints web ---


@pytest.fixture
def client_analizador(uploads_dir_aislado, monkeypatch):
    monkeypatch.setattr(
        "churn_mlops.web_analizador.generar_informe_ad_hoc",
        lambda nombre, target, perfil, metricas: "Informe simulado.",
    )
    return TestClient(build_app())


def test_home_muestra_formulario_de_subida(client_analizador):
    resp = client_analizador.get("/")

    assert resp.status_code == 200
    assert "Subí un CSV" in resp.text or "csv" in resp.text.lower()


def test_subir_rechaza_archivo_no_csv(client_analizador):
    resp = client_analizador.post(
        "/subir", files={"archivo": ("notas.txt", b"hola", "text/plain")}
    )

    assert resp.status_code == 200
    assert "válido" in resp.text.lower() or "csv" in resp.text.lower()


def test_subir_csv_valido_muestra_columnas(client_analizador):
    csv_bytes = b"col_a,col_b,target\n1,x,0\n2,y,1\n"

    resp = client_analizador.post(
        "/subir", files={"archivo": ("datos.csv", csv_bytes, "text/csv")}
    )

    assert resp.status_code == 200
    assert "col_a" in resp.text
    assert "col_b" in resp.text
    assert "target" in resp.text


def test_flujo_completo_subir_y_analizar(client_analizador):
    csv_bytes = b"x1,x2,y\n1,a,0\n2,b,1\n3,a,0\n4,b,1\n5,a,0\n6,b,1\n"

    resp_subir = client_analizador.post(
        "/subir", files={"archivo": ("datos.csv", csv_bytes, "text/csv")}
    )
    assert resp_subir.status_code == 200

    import re

    dataset_id = re.search(r'name="dataset_id" value="([a-f0-9]+)"', resp_subir.text).group(1)

    resp_analizar = client_analizador.post(
        "/analizar",
        data={"dataset_id": dataset_id, "nombre": "datos.csv", "target": "y"},
    )

    assert resp_analizar.status_code == 200
    assert "Informe simulado." in resp_analizar.text


def test_pagina_columnas_escapa_html_en_nombre_de_columna(client_analizador):
    csv_bytes = b'<script>alert(1)</script>,y\n1,0\n2,1\n'

    resp = client_analizador.post(
        "/subir", files={"archivo": ("datos.csv", csv_bytes, "text/csv")}
    )

    assert "<script>alert(1)</script>" not in resp.text
    assert "&lt;script&gt;" in resp.text
