import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

DATA_FILE = Path(__file__).resolve().parents[1] / 'data' / 'Lotomania.csv'
API_URL = 'https://servicebus2.caixa.gov.br/portaldeloterias/api/lotomania'
DRAW_COLUMNS = [f'Bola{index}' for index in range(1, 21)]


def fetch_draw(contest: int | None) -> dict:
    url = API_URL if contest is None else f'{API_URL}/{contest}'
    request = Request(
        url,
        headers={'Accept': 'application/json', 'User-Agent': 'Mozilla/5.0'},
    )
    with urlopen(request, timeout=30) as response:
        draw = json.load(response)

    numbers = draw.get('listaDezenas')
    if (contest is not None and draw.get('numero') != contest) or not isinstance(numbers, list) or len(numbers) != 20:
        raise ValueError(f'Resultado inválido recebido para o concurso {contest or "mais recente"}')
    if len(set(numbers)) != 20 or any(not str(number).isdigit() or not 0 <= int(number) <= 99 for number in numbers):
        raise ValueError(f'Dezenas inválidas recebidas para o concurso {contest}')
    if not draw.get('dataApuracao'):
        raise ValueError(f'Data de apuração ausente para o concurso {contest}')
    return draw


def build_row(draw: dict, columns: list[str]) -> dict[str, str]:
    row = {column: '' for column in columns}
    row['Concurso'] = str(draw['numero'])
    row['Data Sorteio'] = draw['dataApuracao']
    for index, number in enumerate(draw['listaDezenas'], start=1):
        row[f'Bola{index}'] = str(number).zfill(2)
    return row


def main() -> None:
    df = pd.read_csv(DATA_FILE, sep=';', encoding='utf-8-sig', dtype=str, keep_default_na=False)
    latest = fetch_draw(None)
    latest_contest = int(latest['numero'])
    current_contest = int(df['Concurso'].astype(int).max()) if not df.empty else 0

    if latest_contest <= current_contest:
        print(f'Base já está atualizada: concurso {current_contest}.')
        return

    contests = list(range(current_contest + 1, latest_contest + 1))
    draws_by_contest = {latest_contest: latest}
    with ThreadPoolExecutor(max_workers=5) as executor:
        for draw in executor.map(fetch_draw, [contest for contest in contests if contest != latest_contest]):
            draws_by_contest[int(draw['numero'])] = draw

    new_rows = [build_row(draws_by_contest[contest], list(df.columns)) for contest in contests]
    updated = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    updated = updated.sort_values('Concurso', key=lambda values: values.astype(int)).reset_index(drop=True)
    updated.to_csv(DATA_FILE, sep=';', index=False, encoding='utf-8-sig')

    last_draw = draws_by_contest[latest_contest]
    print(f'Concursos adicionados: {contests[0]} a {latest_contest}')
    print(f'Último resultado: {last_draw["numero"]} em {last_draw["dataApuracao"]}')
    print(f'Dezenas: {";".join(last_draw["listaDezenas"])}')
    print(f'Arquivo atualizado: {DATA_FILE}')


if __name__ == '__main__':
    main()
