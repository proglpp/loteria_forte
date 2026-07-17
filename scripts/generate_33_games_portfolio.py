import json
import sys
from pathlib import Path
from typing import List, Dict

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data_loader import LotomaniaDataLoader
from preprocessing import preprocess_draws, cache_draws, load_cached_draws
from elite_optimizer import generate_elite_portfolio, evaluate_portfolio_on_draw
from ai_learning_engine import load_learning_records

DATA_FILE = ROOT / 'data' / 'Lotomania.csv'
EXPORT_DIR = ROOT / 'exports'
EXPORT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_FILE = EXPORT_DIR / 'elite_portfolio_optimized_report_33.json'
GAMES_FILE = EXPORT_DIR / 'elite_portfolio_optimized_games_33.txt'


def _jaccard_overlap(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def build_portfolio() -> None:
    loader = LotomaniaDataLoader(DATA_FILE)
    raw_draws = loader.load_data()
    clean_draws = preprocess_draws(raw_draws)

    cached = load_cached_draws(clean_draws)
    if cached is not None:
        clean_draws = cached
    else:
        cache_draws(clean_draws)

    learning_records = load_learning_records() or None
    latest_draw = clean_draws.iloc[-1]
    latest_numbers = {str(latest_draw[col]).zfill(2) for col in [f'Bola{i}' for i in range(1, 21)]}

    targets = [15, 16, 17, 18, 19]
    seeds = [8609, 2024, 4242, 6868, 9999]

    all_games_scored: List[Dict] = []

    for target in targets:
        for seed in seeds:
            games, profile, _ = generate_elite_portfolio(
                clean_draws,
                learning_records=learning_records,
                n_games=20,
                ticket_size=50,
                seed=seed,
                target_hits=target,
            )
            evaluation = evaluate_portfolio_on_draw(games, latest_numbers)
            for game in games:
                all_games_scored.append(
                    {
                        'game_list': game,
                        'target': target,
                        'seed': seed,
                        'profile_name': profile.name,
                        'profile_avg_hits': profile.avg_hits,
                        'eval_best': evaluation['best'],
                        'eval_avg': evaluation['avg'],
                        'eval_15_plus': evaluation['count_15_plus'],
                        'quality_score': evaluation['best'] * 100 + evaluation['avg'] * 50,
                    }
                )

    df_games = pd.DataFrame(all_games_scored).sort_values('quality_score', ascending=False).reset_index(drop=True)

    selected_games: List[List[str]] = []
    selected_sets: List[set] = []

    # Start with the strongest candidates.
    for _, row in df_games.iterrows():
        game = list(row['game_list'])
        game_set = set(game)
        if any(_jaccard_overlap(game_set, selected_set) > 0.60 for selected_set in selected_sets):
            continue
        selected_games.append(game)
        selected_sets.append(game_set)
        if len(selected_games) == 33:
            break

    # If the selection is too small or too homogeneous, fill the remainder from the ranked list.
    if len(selected_games) < 33:
        for _, row in df_games.iterrows():
            game = list(row['game_list'])
            if game in selected_games:
                continue
            selected_games.append(game)
            if len(selected_games) == 33:
                break

    selected_df = df_games[df_games['game_list'].apply(lambda g: list(g) in selected_games)].copy()
    selected_df = selected_df.head(33)

    portfolio_eval = evaluate_portfolio_on_draw(selected_games, latest_numbers)
    unique_numbers = sorted({number for game in selected_games for number in game}, key=lambda x: int(x))

    report = {
        'title': 'Elite Portfolio Optimized - 33 Games de 50 Números',
        'total_draws_analyzed': len(clean_draws),
        'latest_draw_concurso': int(latest_draw['Concurso']),
        'latest_draw_date': str(latest_draw['Data Sorteio']),
        'portfolio_analysis': {
            'num_games': len(selected_games),
            'ticket_size': 50,
            'unique_numbers_covered': len(unique_numbers),
        },
        'evaluation_on_latest_draw': {
            'best_game_hits': portfolio_eval['best'],
            'average_hits': f"{portfolio_eval['avg']:.2f}",
            'games_with_15_plus_hits': portfolio_eval['count_15_plus'],
        },
        'selected_games_metadata': selected_df[['target', 'seed', 'eval_best', 'eval_avg', 'quality_score']].to_dict(orient='records'),
    }

    with open(REPORT_FILE, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    with open(GAMES_FILE, 'w', encoding='utf-8') as f:
        f.write('ELITE PORTFOLIO OPTIMIZED - 33 JOGOS DE 50 NÚMEROS\n')
        f.write(f"Concurso: {report['latest_draw_concurso']}\n")
        f.write(f"Data: {report['latest_draw_date']}\n\n")
        for idx, game in enumerate(selected_games, 1):
            f.write(f"Jogo {idx:02d}: {', '.join(game)}\n")

    print('Generated 33 games')
    print(f'Report: {REPORT_FILE}')
    print(f'Games: {GAMES_FILE}')
    print(f"Best={portfolio_eval['best']}, Avg={portfolio_eval['avg']:.2f}, 15+={portfolio_eval['count_15_plus']}")


if __name__ == '__main__':
    build_portfolio()
