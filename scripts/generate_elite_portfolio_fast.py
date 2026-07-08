import json
import sys
from pathlib import Path
from typing import List

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data_loader import LotomaniaDataLoader
from elite_optimizer import generate_elite_portfolio, evaluate_portfolio_on_draw
from preprocessing import preprocess_draws, cache_draws, load_cached_draws
from ai_learning_engine import load_learning_records


DATA_FILE = ROOT / 'data' / 'Lotomania.csv'
EXPORT_DIR = ROOT / 'exports'
EXPORT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_FILE = EXPORT_DIR / 'elite_portfolio_report.json'
GAMES_FILE = EXPORT_DIR / 'elite_portfolio_games.txt'


def main():
    print("Loading historical draws...")
    loader = LotomaniaDataLoader(DATA_FILE)
    raw_draws = loader.load_data()
    clean_draws = preprocess_draws(raw_draws)
    
    cached = load_cached_draws(clean_draws)
    if cached is not None:
        clean_draws = cached
    else:
        cache_draws(clean_draws)
    
    print(f"Loaded {len(clean_draws)} draws")
    
    print("Loading AI learning feedback (if available)...")
    learning_records = load_learning_records()
    if learning_records:
        print(f"  Found {len(learning_records)} feedback records")
    
    print("Generating 20 elite games of 50 numbers each...")
    games, profile, scores = generate_elite_portfolio(
        clean_draws,
        learning_records=learning_records or None,
        n_games=20,
        ticket_size=50,
        seed=8609,
        target_hits=19,
    )
    
    print(f"Generated {len(games)} games")
    
    print("Evaluating games on latest draw...")
    latest_draw = clean_draws.iloc[-1]
    latest_numbers = {str(latest_draw[col]).zfill(2) for col in [f'Bola{i}' for i in range(1, 21)]}
    evaluation = evaluate_portfolio_on_draw(games, latest_numbers)
    
    unique_numbers = sorted({number for game in games for number in game}, key=lambda x: int(x))
    coverage_freq = {}
    for game in games:
        for number in game:
            coverage_freq[number] = coverage_freq.get(number, 0) + 1
    
    report = {
        'title': 'Elite Portfolio - 20 Games de 50 Números',
        'total_draws_analyzed': len(clean_draws),
        'latest_draw_concurso': int(latest_draw['Concurso']),
        'latest_draw_date': str(latest_draw['Data Sorteio']),
        'elite_profile': {
            'name': profile.name,
            'target_hits': profile.target_hits,
            'avg_hits': profile.avg_hits,
            'target_rate': f"{profile.target_rate*100:.2f}%",
            'hit_19_rate': f"{profile.hit_19_rate*100:.2f}%",
            'hit_18_rate': f"{profile.hit_18_rate*100:.2f}%",
            'hit_15_rate': f"{profile.hit_15_rate*100:.2f}%",
            'max_hits': profile.max_hits,
        },
        'portfolio_analysis': {
            'num_games': len(games),
            'ticket_size': 50,
            'unique_numbers_covered': len(unique_numbers),
        },
        'evaluation_on_latest_draw': {
            'best_game_hits': evaluation['best'],
            'average_hits': f"{evaluation['avg']:.2f}",
            'games_with_15_plus_hits': evaluation['count_15_plus'],
        },
        'top_50_elite_scored_numbers': scores[['number', 'elite_score']].head(50).to_dict(orient='records'),
        'portfolio_number_coverage': sorted(unique_numbers, key=lambda x: int(x)),
        'portfolio_number_frequency': {k: v for k, v in sorted(coverage_freq.items(), key=lambda x: x[1], reverse=True)},
    }
    
    with open(REPORT_FILE, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    with open(GAMES_FILE, 'w', encoding='utf-8') as f:
        f.write(f"{'=' * 100}\n")
        f.write("ELITE PORTFOLIO - 20 JOGOS DE 50 NÚMEROS\n")
        f.write(f"{'=' * 100}\n\n")
        f.write(f"Concurso: {report['latest_draw_concurso']}\n")
        f.write(f"Data: {report['latest_draw_date']}\n")
        f.write(f"Período analisado: {len(clean_draws)} concursos\n\n")
        
        f.write(f"{'=' * 100}\n")
        f.write("PERFIL DE OTIMIZAÇÃO\n")
        f.write(f"{'=' * 100}\n\n")
        for k, v in report['elite_profile'].items():
            f.write(f"{k}: {v}\n")
        f.write("\n")
        
        f.write(f"{'=' * 100}\n")
        f.write("ANÁLISE DO PORTFÓLIO\n")
        f.write(f"{'=' * 100}\n\n")
        f.write(f"Números únicos cobertos: {report['portfolio_analysis']['unique_numbers_covered']}\n")
        f.write(f"Cobertura em {len(games)} jogos: {report['portfolio_number_coverage']}\n\n")
        
        f.write(f"{'=' * 100}\n")
        f.write("20 JOGOS GERADOS\n")
        f.write(f"{'=' * 100}\n\n")
        for idx, game in enumerate(games, 1):
            f.write(f"Jogo {idx:02d}: {', '.join(game)}\n")
        
        f.write("\n")
        f.write(f"{'=' * 100}\n")
        f.write("AVALIAÇÃO NA ÚLTIMA RODADA\n")
        f.write(f"{'=' * 100}\n\n")
        f.write(f"Melhor jogo: {report['evaluation_on_latest_draw']['best_game_hits']} acertos em 20\n")
        f.write(f"Média de acertos: {report['evaluation_on_latest_draw']['average_hits']}\n")
        f.write(f"Jogos com 15+ acertos: {report['evaluation_on_latest_draw']['games_with_15_plus_hits']}\n\n")
    
    print(f"\n✅ Report saved to: {REPORT_FILE}")
    print(f"✅ Games saved to: {GAMES_FILE}")
    
    print(f"\nSUMMARY:")
    print(f"  Games generated: {len(games)}")
    print(f"  Unique numbers covered: {len(unique_numbers)}")
    print(f"  Best hit count on latest draw: {evaluation['best']}")
    print(f"  Average hit count: {evaluation['avg']:.2f}")
    print(f"  Target hits: {profile.target_hits}")
    print(f"  Historical target hit rate: {profile.target_rate*100:.2f}%")


if __name__ == '__main__':
    main()
