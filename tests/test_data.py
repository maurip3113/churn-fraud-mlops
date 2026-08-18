import pandas as pd

from churn_mlops.data import TARGET, load_and_clean, split_train_monitor


def test_load_and_clean_convierte_totalcharges_a_numerico(tmp_path):
    csv = tmp_path / "raw.csv"
    csv.write_text(
        "customerID,tenure,MonthlyCharges,TotalCharges,Churn\n"
        "1,0,50.0, ,No\n"  # TotalCharges en blanco: fila a descartar
        "2,10,60.0,600.0,Yes\n"
    )

    df = load_and_clean(csv)

    assert len(df) == 1
    assert df.loc[0, "TotalCharges"] == 600.0
    assert df.loc[0, TARGET] == 1


def test_split_train_monitor_separa_por_tenure():
    df = pd.DataFrame({"tenure": [1, 3, 6, 7, 20, 50]})

    df_train, df_monitor = split_train_monitor(df, cutoff_months=6)

    assert sorted(df_monitor["tenure"]) == [1, 3, 6]
    assert sorted(df_train["tenure"]) == [7, 20, 50]


def test_split_train_monitor_no_pierde_filas():
    df = pd.DataFrame({"tenure": range(1, 101)})

    df_train, df_monitor = split_train_monitor(df, cutoff_months=6)

    assert len(df_train) + len(df_monitor) == len(df)
