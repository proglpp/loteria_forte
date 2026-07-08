import pandas as pd
from pathlib import Path
from collections import Counter
import numpy as np

DATA_FILE = Path('data/Lotomania.csv')

# Read CSV
df = pd.read_csv(DATA_FILE, sep=';', encoding='utf-8-sig', engine='python', dtype=str)

print("=" * 80)
print("ANÁLISE DE PROBABILIDADES - LOTOMANIA")
print("=" * 80)
print(f"Total de concursos: {len(df)}")
print(f"Período: {df.iloc[0]['Data Sorteio']} até {df.iloc[-1]['Data Sorteio']}")
print()

# Extract all drawn numbers
DRAW_COLUMNS = [f'Bola{i}' for i in range(1, 21)]
all_numbers = []
for idx, row in df.iterrows():
    for col in DRAW_COLUMNS:
        num = str(row[col]).zfill(2)
        all_numbers.append(num)

# Calculate frequency
counter = Counter(all_numbers)
total_appearances = len(all_numbers)

# Create frequency dataframe
freq_data = []
for num in [f"{n:02d}" for n in range(100)]:
    count = counter.get(num, 0)
    probability = count / total_appearances * 100
    freq_data.append({
        'number': num,
        'count': count,
        'probability': probability,
    })

freq_df = pd.DataFrame(freq_data).sort_values('probability', ascending=False)

print("TOP 50 NÚMEROS MAIS FREQUENTES:")
print("-" * 80)
print(f"{'Rank':<5} {'Número':<8} {'Aparições':<12} {'Probabilidade':<15}")
print("-" * 80)

top_50 = freq_df.head(50)
top_50_numbers = set(top_50['number'].values)
top_50_prob = top_50['probability'].mean()

for idx, (i, row) in enumerate(top_50.iterrows(), 1):
    print(f"{idx:<5} {row['number']:<8} {int(row['count']):<12} {row['probability']:.4f}%")

print()
print("=" * 80)
print("ANÁLISE DE ACERTOS - 20 NÚMEROS SORTEADOS vs 50 MARCADOS")
print("=" * 80)

# For each draw, count how many hits if we had marked the top 50
hits_list = []
for idx, row in df.iterrows():
    draw_numbers = set()
    for col in DRAW_COLUMNS:
        num = str(row[col]).zfill(2)
        draw_numbers.add(num)
    
    hits = len(draw_numbers & top_50_numbers)
    hits_list.append(hits)

mean_hits = np.mean(hits_list)
median_hits = np.median(hits_list)
min_hits = min(hits_list)
max_hits = max(hits_list)
std_hits = np.std(hits_list)

print(f"Se marcássemos os 50 números mais frequentes:")
print(f"  Média de acertos:      {mean_hits:.2f} números em 20")
print(f"  Mediana de acertos:    {median_hits:.2f} números em 20")
print(f"  Mínimo de acertos:     {min_hits} números")
print(f"  Máximo de acertos:     {max_hits} números")
print(f"  Desvio padrão:         {std_hits:.2f}")
print(f"  Taxa esperada:         {(mean_hits / 20) * 100:.2f}%")
print()

# Distribution of hits
hit_counts = Counter(hits_list)
print("DISTRIBUIÇÃO DE ACERTOS NOS 2946 CONCURSOS:")
print("-" * 80)
for hits in sorted(hit_counts.keys()):
    count = hit_counts[hits]
    percentage = (count / len(df)) * 100
    print(f"  {hits} acertos: {count:4d} vezes ({percentage:5.2f}%)")

print()
print("=" * 80)
print("NÚMEROS MAIS FRACOS (MENOS FREQUENTES):")
print("-" * 80)
bottom_10 = freq_df.tail(10).sort_values('probability')
for idx, (i, row) in enumerate(bottom_10.iterrows(), 1):
    print(f"{row['number']}: {int(row['count']):3d} aparições ({row['probability']:.4f}%)")

print()
print("=" * 80)
print("CONCLUSÃO")
print("=" * 80)
print(f"Marcando os 50 números com MAIOR probabilidade:")
print(f"  → Esperaríamos acertar ~{mean_hits:.1f} em cada sorteio de 20")
print(f"  → Taxa de sucesso: ~{(mean_hits / 20) * 100:.1f}% em média")
print(f"  → Probabilidade matemática (sem viés): {(50/100) * 100:.1f}%")
print(f"  → Ganho real vs acaso: {((mean_hits / 20) - 0.5) * 100:+.1f}% pontos percentuais")
print()
