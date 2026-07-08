import json
from pathlib import Path

EXPORT_DIR = Path('exports')
REPORT_FILE = EXPORT_DIR / 'elite_portfolio_report.json'
GAMES_CSV = EXPORT_DIR / 'elite_portfolio_games.csv'

def display_report():
    if not REPORT_FILE.exists():
        print(f"Relatório não encontrado: {REPORT_FILE}")
        return
    
    with open(REPORT_FILE, encoding='utf-8') as f:
        report = json.load(f)
    
    print("=" * 100)
    print("PORTFÓLIO ELITE - 20 JOGOS DE 50 NÚMEROS")
    print("=" * 100)
    print(f"Concurso analisado: {report['latest_draw_concurso']}")
    print(f"Data: {report['latest_draw_date']}")
    print(f"Total de concursos históricos: {report['total_draws']}")
    print()
    
    print("PERFIL DE OTIMIZAÇÃO:")
    profile = report['elite_profile']
    print(f"  Target de acertos: {profile['target_hits']} números em 20")
    print(f"  Média histórica: {profile['avg_hits']:.2f} acertos")
    print(f"  Taxa de acerto do target: {profile['target_rate']*100:.2f}%")
    print(f"  Taxa de 19+ acertos: {profile['hit_19_rate']*100:.2f}%")
    print(f"  Taxa de 18+ acertos: {profile['hit_18_rate']*100:.2f}%")
    print(f"  Taxa de 15+ acertos: {profile['hit_15_rate']*100:.2f}%")
    print(f"  Máximo histórico: {profile['max_hits']} acertos")
    print()
    
    print("ANÁLISE DO PORTFÓLIO:")
    print(f"  Números únicos cobertos: {report['portfolio_unique_numbers']}")
    print(f"  Frequência de cobertura dos 20 jogos:")
    freq = report['portfolio_number_frequency']
    for num in sorted(freq.keys(), key=lambda x: freq[x], reverse=True)[:20]:
        print(f"    {num:02d}: {freq[num]:2d} jogos")
    print()
    
    print("RESULTADO NA AVALIAÇÃO (Último sorteio):")
    print(f"  Melhor jogo: {report['latest_draw_best_game_hits']} acertos em 20")
    print(f"  Média de acertos: {report['latest_draw_avg_hits']:.2f}")
    print(f"  Jogos com 15+ acertos: {report['latest_draw_games_with_15_plus']}")
    print()
    
    print("TOP 20 NÚMEROS RANQUEADOS PELO ENSEMBLE ML:")
    print("-" * 100)
    for idx, item in enumerate(report['top_20_ensemble'], 1):
        print(f"{idx:2d}. {item['number']}: {item['prob_final']:.6f}")
    print()
    
    print("TOP 50 NÚMEROS RANQUEADOS PELO ELITE SCORE:")
    print("-" * 100)
    for idx, item in enumerate(report['top_rated_numbers'], 1):
        print(f"{idx:2d}. {item['number']}: {item['elite_score']:.6f}")
    print()
    
    if GAMES_CSV.exists():
        import pandas as pd
        games_df = pd.read_csv(GAMES_CSV)
        unique_games = games_df['game'].max()
        print(f"JOGOS GERADOS: {unique_games}")
        print("-" * 100)
        for game_idx in range(1, min(6, unique_games+1)):
            game_data = games_df[games_df['game'] == game_idx].sort_values('position')['number'].tolist()
            print(f"Jogo {game_idx:02d}: {', '.join(game_data)}")
        print(f"... ({unique_games} total de jogos)")

if __name__ == '__main__':
    display_report()
