from pathlib import Path
import pandas as pd
file = Path('data/Lotomania.csv')
print('exists', file.exists())
df = pd.read_csv(file, sep=';', encoding='utf-8-sig', engine='python', parse_dates=['Data Sorteio'], dayfirst=True, keep_default_na=False)
print('rows', len(df))
print('last_concurso', df.iloc[-1]['Concurso'])
print('last_date', df.iloc[-1]['Data Sorteio'])
print(df.tail(3).to_string(index=False))
