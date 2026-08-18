"""Tests de cada plugin de caso de uso: su lógica de datos específica."""

import pandas as pd

from churn_mlops.usecases import churn as churn_mod
from churn_mlops.usecases import fraude as fraude_mod
from churn_mlops.usecases.registry import USECASES, get_usecase


def test_registry_expone_churn_y_fraude():
    assert set(USECASES) == {"churn", "fraude"}


def test_get_usecase_desconocido_lanza_error():
    import pytest

    with pytest.raises(ValueError):
        get_usecase("no_existe")


def test_usecase_features_es_num_mas_cat(churn_usecase, fraude_usecase):
    for usecase in (churn_usecase, fraude_usecase):
        assert usecase.features == usecase.num_features + usecase.cat_features


def test_usecase_nombres_derivados_son_unicos(churn_usecase, fraude_usecase):
    assert churn_usecase.mlflow_experiment_name != fraude_usecase.mlflow_experiment_name
    assert churn_usecase.registered_model_name != fraude_usecase.registered_model_name


# --- churn ---


def test_churn_cargar_y_limpiar_convierte_totalcharges_a_numerico(tmp_path):
    csv = tmp_path / "raw.csv"
    csv.write_text(
        "customerID,tenure,MonthlyCharges,TotalCharges,Churn\n"
        "1,0,50.0, ,No\n"  # TotalCharges en blanco: fila a descartar
        "2,10,60.0,600.0,Yes\n"
    )

    df = churn_mod.cargar_y_limpiar(csv)

    assert len(df) == 1
    assert df.loc[0, "TotalCharges"] == 600.0
    assert df.loc[0, churn_mod.TARGET] == 1


def test_churn_separar_train_monitor_por_tenure():
    df = pd.DataFrame({"tenure": [1, 3, 6, 7, 20, 50]})

    df_train, df_monitor = churn_mod.separar_train_monitor(df, cutoff_months=6)

    assert sorted(df_monitor["tenure"]) == [1, 3, 6]
    assert sorted(df_train["tenure"]) == [7, 20, 50]
    assert len(df_train) + len(df_monitor) == len(df)


def test_churn_asegurar_datos_crudos_sin_archivo_lanza_error(tmp_path):
    import pytest

    with pytest.raises(FileNotFoundError):
        churn_mod.asegurar_datos_crudos(tmp_path / "no_existe.csv")


# --- fraude ---


def test_fraude_asegurar_datos_crudos_genera_csv_sintetico(tmp_path):
    ruta = tmp_path / "datos_crudos.csv"

    fraude_mod.asegurar_datos_crudos(ruta, n=500, seed=1)

    assert ruta.exists()
    df = pd.read_csv(ruta)
    assert len(df) == 500
    assert set(fraude_mod.NUM_FEATURES + fraude_mod.CAT_FEATURES).issubset(df.columns)
    # desbalance esperado: fraude es la clase minoritaria
    assert 0 < df[fraude_mod.TARGET].mean() < 0.15


def test_fraude_asegurar_datos_crudos_no_pisa_si_ya_existe(tmp_path):
    ruta = tmp_path / "datos_crudos.csv"
    ruta.write_text("contenido_preexistente")

    fraude_mod.asegurar_datos_crudos(ruta, n=500, seed=1)

    assert ruta.read_text() == "contenido_preexistente"


def test_fraude_separar_train_monitor_por_canal():
    df = pd.DataFrame(
        {
            "canal": ["pos", "online", "atm", "online"],
            "monto": [10, 20, 30, 40],
            fraude_mod.TARGET: [0, 1, 0, 0],
        }
    )

    df_train, df_monitor = fraude_mod.separar_train_monitor(df)

    assert list(df_monitor["canal"].unique()) == ["online"]
    assert "online" not in df_train["canal"].unique()
    assert len(df_train) + len(df_monitor) == len(df)
