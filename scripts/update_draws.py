import pandas as pd
from pathlib import Path

DATA_FILE = Path('data/Lotomania.csv')

# Read current CSV with all columns as strings
df = pd.read_csv(DATA_FILE, sep=';', encoding='utf-8-sig', engine='python', dtype=str)

# Data mapping do usuário
updates = {
    '2943': ('29/06/2026', '03;09;10;11;13;14;18;23;29;32;51;64;65;70;78;84;85;87;95;96'),
    '2944': ('01/07/2026', '12;20;24;29;35;37;38;46;53;60;63;75;79;80;82;86;90;91;96;98'),
    '2945': ('03/07/2026', '00;04;05;07;10;14;18;22;24;40;46;47;57;59;64;74;76;81;84;97'),
}

# Update existing rows
for concurso, (data_sorteio, numeros) in updates.items():
    row_idx = df[df['Concurso'] == concurso].index
    if len(row_idx) > 0:
        i = row_idx[0]
        df.at[i, 'Data Sorteio'] = data_sorteio
        # Update balls (Bola1 to Bola20)
        numeros_list = numeros.split(';')
        for j, num in enumerate(numeros_list, 1):
            df.at[i, f'Bola{j}'] = str(int(num)).zfill(2)
        print(f"Updated concurso {concurso} to {data_sorteio}")

# Add new row for 06/07 (concurso 2946)
novo_concurso = {
    'Concurso': '2946',
    'Data Sorteio': '06/07/2026',
}
numeros_novo = ['01', '03', '15', '17', '20', '22', '30', '31', '35', '40', '45', '51', '53', '59', '64', '65', '72', '82', '86', '89']
for j, num in enumerate(numeros_novo, 1):
    novo_concurso[f'Bola{j}'] = str(int(num)).zfill(2)

# Fill missing columns with defaults
for col in df.columns:
    if col not in novo_concurso:
        if 'Ganhadores' in col or 'Rateio' in col or 'Rateio' in col or 'Arrecadação' in col or 'Estimativa' in col or 'Acumulado' in col:
            if 'Rateio' in col or 'Arrecadação' in col or 'Estimativa' in col or 'Acumulado' in col:
                novo_concurso[col] = 'R$0,00'
            else:
                novo_concurso[col] = 0
        elif col == 'Observação':
            novo_concurso[col] = ''
        elif col == 'Cidade / UF':
            novo_concurso[col] = ''
        else:
            novo_concurso[col] = ''

# Add new row
df = pd.concat([df, pd.DataFrame([novo_concurso])], ignore_index=True)
print(f"Added concurso 2946 for 06/07/2026")

# Save
df.to_csv(DATA_FILE, sep=';', index=False, encoding='utf-8-sig')
print(f"\nFile saved: {DATA_FILE}")
print(f"Total rows: {len(df)}")
print(f"\nLast 5 rows:")
for idx in df.tail(5).index:
    row = df.loc[idx]
    bolas = ';'.join([str(row[f'Bola{i}']) for i in range(1, 21)])
    print(f"  {row['Concurso']}: {row['Data Sorteio']} - {bolas}")
