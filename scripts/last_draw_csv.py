import csv
from pathlib import Path
import pandas as pd

DATA_FILE = Path('data/Lotomania.csv')
if not DATA_FILE.exists():
    print(f"ERROR: Data file not found: {DATA_FILE}")
    raise SystemExit(1)

with open(DATA_FILE, 'r', encoding='utf-8-sig', errors='replace') as fh:
    sample = fh.read(8192)
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=[',',';','\t','|'])
        sep = dialect.delimiter
    except csv.Error:
        sep = ';'

try:
    df = pd.read_csv(DATA_FILE, sep=sep, engine='python', parse_dates=['Data Sorteio'], dayfirst=True, encoding='utf-8-sig')
    if df.empty:
        print('No rows in CSV')
    else:
        last = df.iloc[-1]
        concurso = last.get('Concurso')
        draw_date = last.get('Data Sorteio')
        print(f"LAST_CSV_CONCURSO:{concurso}")
        print(f"LAST_CSV_DATE:{draw_date}")
except Exception as e:
    print('Failed to read CSV:', e)
    raise
