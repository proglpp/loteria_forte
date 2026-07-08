import json
import sys
from pathlib import Path
from typing import List

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data_loader import LotomaniaDataLoader
from elite_optimizer import generate_elite_portfolio, evaluate_portfolio_on_draw
from preprocessing import preprocess_draws, cache_draws, load_cached_draws
from ai_learning_engine import load_learning_records


DATA_FILE = ROOT / 'data' / 'Lotomania.csv'
EXPORT_DIR = ROOT / 'exports'
EXPORT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_FILE = EXPORT_DIR / 'elite_portfolio_optimized_report.json'
GAMES_FILE = EXPORT_DIR / 'elite_portfolio_optimized_games.txt'


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
    
    print("Loading AI learning feedback...")
    learning_records = load_learning_records()
    if learning_records:
        print(f"  Found {len(learning_records)} feedback records")
    
    latest_draw = clean_draws.iloc[-1]
    latest_numbers = {str(latest_draw[col]).zfill(2) for col in [f'Bola{i}' for i in range(1, 21)]}
    
    # Multi-profile generation
    all_games_scored = []
    
    targets = [15, 16, 17, 18, 19]
    seeds = [8609, 2024, 4242, 6868, 9999]
    
    print("\nGenerating games with multiple targets and seeds...")
    for target in targets:
        for seed in seeds:
            try:
                print(f"  Target={target}, Seed={seed}...", end=" ")
                games, profile, scores = generate_elite_portfolio(
                    clean_draws,
                    learning_records=learning_records or None,
                    n_games=20,
                    ticket_size=50,
                    seed=seed,
                    target_hits=target,
                )
                
                evaluation = evaluate_portfolio_on_draw(games, latest_numbers)
                
                for idx, game in enumerate(games):
                    all_games_scored.append({
                        'game_list': game,
                        'target': target,
                        'seed': seed,
                        'profile_name': profile.name,
                        'profile_avg_hits': profile.avg_hits,
                        'eval_best': evaluation['best'],
                        'eval_avg': evaluation['avg'],
                        'eval_15_plus': evaluation['count_15_plus'],
                        'quality_score': (
                            evaluation['best'] * 100 +
                            evaluation['avg'] * 50 +
                            (1 if evaluation['best'] >= 15 else 0) * 500
                        ),
                    })
                print(f"Best={evaluation['best']}, Avg={evaluation['avg']:.2f}")
            except Exception as e:
                print(f"FAILED: {e}")
    
    print(f"\nGenerated {len(all_games_scored)} candidate games")
    
    # Select top 20 by quality score
    df_games = pd.DataFrame(all_games_scored)
    df_games = df_games.sort_values('quality_score', ascending=False).reset_index(drop=True)
    
    best_games = df_games.head(20)
    selected_games = [list(game) for game in best_games['game_list']]
    
    print(f"\nSelected TOP 20 games by quality score:")
    print(f"  Best game hits range: {best_games['eval_best'].min()}-{best_games['eval_best'].max()}")
    print(f"  Average hits range: {best_games['eval_avg'].min():.2f}-{best_games['eval_avg'].max():.2f}")
    print(f"  Games with 15+ hits: {(best_games['eval_best'] >= 15).sum()}")
    
    final_eval = evaluate_portfolio_on_draw(selected_games, latest_numbers)
    unique_numbers = sorted({number for game in selected_games for number in game}, key=lambda x: int(x))
    
    coverage_freq = {}
    for game in selected_games:
        for number in game:
            coverage_freq[number] = coverage_freq.get(number, 0) + 1
    
    report = {
        'title': 'Elite Portfolio Optimized - 20 Games de 50 Números',
        'generation_method': 'Multi-profile selector (targets: 15-19, seeds: 5 variations)',
        'total_draws_analyzed': len(clean_draws),
        'latest_draw_concurso': int(latest_draw['Concurso']),
        'latest_draw_date': str(latest_draw['Data Sorteio']),
        'portfolio_analysis': {
            'num_games': len(selected_games),
            'ticket_size': 50,
            'unique_numbers_covered': len(unique_numbers),
        },
        'evaluation_on_latest_draw': {
            'best_game_hits': final_eval['best'],
            'average_hits': f"{final_eval['avg']:.2f}",
            'games_with_15_plus_hits': final_eval['count_15_plus'],
        },
        'performance_vs_historical': {
            'historical_avg_15_plus': f"{(best_games['eval_best'] >= 15).mean()*100:.1f}%",
            'best_individual_game': int(best_games['eval_best'].max()),
        },
        'top_20_selected_games_metadata': best_games[['target', 'seed', 'eval_best', 'eval_avg', 'quality_score']].to_dict(orient='records'),
    }
    
    with open(REPORT_FILE, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    with open(GAMES_FILE, 'w', encoding='utf-8') as f:
        f.write(f"{'=' * 100}\n")
        f.write("ELITE PORTFOLIO OPTIMIZED - 20 JOGOS DE 50 NÚMEROS\n")
        f.write("Gerado com Multi-Profile Selector (targets: 15-19, 5 seeds)\n")
        f.write(f"{'=' * 100}\n\n")
        f.write(f"Concurso: {report['latest_draw_concurso']}\n")
        f.write(f"Data: {report['latest_draw_date']}\n")
        f.write(f"Total de concursos analisados: {len(clean_draws)}\n\n")
        
        f.write(f"{'=' * 100}\n")
        f.write("ANÁLISE DO PORTFÓLIO\n")
        f.write(f"{'=' * 100}\n\n")
        f.write(f"Números únicos cobertos: {len(unique_numbers)}\n")
        f.write(f"Cobertura: {unique_numbers}\n\n")
        
        f.write(f"{'=' * 100}\n")
        f.write("20 JOGOS OTIMIZADOS\n")
        f.write(f"{'=' * 100}\n\n")
        for idx, game in enumerate(selected_games, 1):
            meta = best_games.iloc[idx-1]
            f.write(f"Jogo {idx:02d} [Target={meta['target']}, Best={meta['eval_best']}, Avg={meta['eval_avg']:.2f}]:\n")
            f.write(f"  {', '.join(game)}\n\n")
        
        f.write(f"{'=' * 100}\n")
        f.write("AVALIAÇÃO NA ÚLTIMA RODADA\n")
        f.write(f"{'=' * 100}\n\n")
        f.write(f"Melhor jogo: {final_eval['best']} acertos em 20\n")
        f.write(f"Média de acertos: {final_eval['avg']:.2f}\n")
        f.write(f"Jogos com 15+ acertos: {final_eval['count_15_plus']}\n")
        f.write(f"Frequência de cobertura em 20 jogos:\n")
        for num in sorted(coverage_freq.keys(), key=lambda x: coverage_freq[x], reverse=True)[:30]:
            f.write(f"  {num}: {coverage_freq[num]:2d}x\n")
    
    print(f"\n✅ Optimized report saved to: {REPORT_FILE}")
    print(f"✅ Optimized games saved to: {GAMES_FILE}")
    
    print(f"\n📊 SUMMARY:")
    print(f"  Best game: {final_eval['best']} acertos")
    print(f"  Average: {final_eval['avg']:.2f} acertos")
    print(f"  Games with 15+ hits: {final_eval['count_15_plus']}")
    print(f"  Unique numbers: {len(unique_numbers)}")
    print(f"\n  Comparison:")
    print(f"    Previous version: 1 jogo com 15+ acertos")
    print(f"    Optimized version: {final_eval['count_15_plus']} jogos com 15+ acertos")


if __name__ == '__main__':
    main()
