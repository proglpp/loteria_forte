import pandas as pd

from data_loader import LotomaniaDataLoader
from config import DATA_FILE, DATABASE_FILE


def test_load_data():
    loader = LotomaniaDataLoader(DATA_FILE, DATABASE_FILE)
    df = loader.load_data()

    assert not df.empty
    assert "Concurso" in df.columns
    assert "Data Sorteio" in df.columns
    draw_cols = [f"Bola{i}" for i in range(1, 21)]
    assert all(col in df.columns for col in draw_cols)
    assert df["Concurso"].dtype == "Int64"
    assert pd.api.types.is_datetime64_any_dtype(df["Data Sorteio"])
    assert df[draw_cols].apply(lambda col: col.astype(str).str.len() == 2).all().all()
