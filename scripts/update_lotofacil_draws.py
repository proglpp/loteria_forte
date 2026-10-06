import tempfile
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT_DIR / "data" / "Lotofacil.csv"
API_ROOT = "https://servicebus3.caixa.gov.br/portaldeloterias"
DRAW_COLUMNS = [f"Bola{index}" for index in range(1, 16)]


def main() -> None:
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    download_url = f"{API_ROOT}/api/resultados/download?{urlencode({'modalidade': 'Lotofácil'})}"
    request = Request(
        download_url,
        headers={
            "Accept": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "Referer": "https://loterias.caixa.gov.br/Paginas/Lotofacil.aspx",
            "User-Agent": "Mozilla/5.0",
        },
    )
    with tempfile.TemporaryDirectory() as temporary_directory:
        workbook_path = Path(temporary_directory) / "Lotofacil.xlsx"
        with urlopen(request, timeout=60) as response:
            workbook_path.write_bytes(response.read())
        history = pd.read_excel(
            workbook_path,
            usecols=["Concurso", "Data Sorteio", *DRAW_COLUMNS],
            dtype=str,
        )

    history["Concurso"] = pd.to_numeric(history["Concurso"], errors="coerce")
    numbers = history[DRAW_COLUMNS].apply(lambda column: pd.to_numeric(column, errors="coerce"))
    if history["Concurso"].isna().any() or history["Data Sorteio"].isna().any():
        raise ValueError("A planilha oficial contém concurso ou data inválidos")
    if numbers.isna().values.any() or ((numbers < 1) | (numbers > 25)).values.any():
        raise ValueError("A planilha oficial contém dezenas fora do intervalo de 01 a 25")
    if numbers.apply(lambda row: row.nunique() != 15, axis=1).any():
        raise ValueError("A planilha oficial contém concurso com dezenas repetidas")
    if history["Concurso"].duplicated().any():
        raise ValueError("A planilha oficial contém concursos duplicados")

    history[DRAW_COLUMNS] = numbers.apply(lambda column: column.map(lambda number: f"{int(number):02d}"))
    history["Concurso"] = history["Concurso"].astype(int).astype(str)
    history = history.sort_values("Concurso", key=lambda values: values.astype(int)).reset_index(drop=True)
    temporary_file = DATA_FILE.with_suffix(".csv.tmp")
    history.to_csv(temporary_file, sep=";", index=False, encoding="utf-8-sig")
    temporary_file.replace(DATA_FILE)
    last_draw = history.iloc[-1]
    print(f"Concursos carregados: {len(history)}")
    print(f"Último resultado oficial: {last_draw['Concurso']} em {last_draw['Data Sorteio']}")
    print(f"Arquivo atualizado: {DATA_FILE}")


if __name__ == "__main__":
    main()