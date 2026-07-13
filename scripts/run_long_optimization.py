import json
import sys
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data_loader import LotomaniaDataLoader
from preprocessing import preprocess_draws, cache_draws, load_cached_draws
from elite_optimizer import generate_elite_portfolio, evaluate_portfolio_on_draw
from ai_learning_engine import load_learning_records


DATA_FILE = ROOT / 'data' / 'Lotomania.csv'
EXPORT_DIR = ROOT / 'exports'
EXPORT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_FILE = EXPORT_DIR / 'long_optimization_results.json'
GAMES_FILE = EXPORT_DIR / 'long_optimization_top_games.txt'


def main():
    print('Loading draws...')
    loader = LotomaniaDataLoader(DATA_FILE)
    raw_draws = loader.load_data()
    draws = preprocess_draws(raw_draws)

    cached = load_cached_draws(draws)
    if cached is not None:
        draws = cached
    else:
        cache_draws(draws)

    print(f'Loaded {len(draws)} draws')
    learning = load_learning_records()
    if learning:
        print(f'Loaded {len(learning)} learning records')

    latest_draw = draws.iloc[-1]
    latest_numbers = {str(latest_draw[col]).zfill(2) for col in [f'Bola{i}' for i in range(1,21)]}

    # Long-run parameters (tunable)
    targets = [16, 17, 18]
    seed_start = 100
    seed_end = 10000
    seed_step = 50
    n_games_per_run = 20

    candidates = []
    total_runs = sum(1 for _ in range(seed_start, seed_end, seed_step)) * len(targets)
    run_idx = 0
    t0 = time.time()

    for target in targets:
        for seed in range(seed_start, seed_end, seed_step):
            run_idx += 1
            try:
                if run_idx % 50 == 0:
                    elapsed = time.time() - t0
                    print(f'Progress: {run_idx}/{total_runs} runs, elapsed {elapsed:.0f}s')

                games, profile, scores = generate_elite_portfolio(
                    draws,
                    learning_records=learning or None,
                    n_games=n_games_per_run,
                    ticket_size=50,
                    seed=seed,
                    target_hits=target,
                )

                evaluation = evaluate_portfolio_on_draw(games, latest_numbers)
                for game in games:
                    candidates.append({
                        'game': game,
                        'seed': seed,
                        'target': target,
                        'profile': profile.name,
                        'profile_avg': profile.avg_hits,
                        'eval_best': evaluation['best'],
                        'eval_avg': evaluation['avg'],
                        'eval_15_plus': evaluation['count_15_plus'],
                        'quality': evaluation['best'] * 100 + evaluation['avg'] * 50,
                    })
            except Exception as e:
                print(f'Run failed target={target} seed={seed}: {e}')

    # Select top candidates by quality
    candidates_sorted = sorted(candidates, key=lambda x: x['quality'], reverse=True)
    top_games = candidates_sorted[:200]

    report = {
        'total_candidate_games': len(candidates),
        'top_sample': len(top_games),
        'selected_at': time.ctime(),
    }

    with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
        json.dump({'report': report, 'top_games': top_games}, f, ensure_ascii=False, indent=2)

    with open(GAMES_FILE, 'w', encoding='utf-8') as f:
        f.write('LONG OPTIMIZATION - TOP GAMES\n')
        f.write(f"Total candidates: {len(candidates)}\n\n")
        for idx, item in enumerate(top_games, 1):
            f.write(f'Game {idx:03d} [Seed={item["seed"]} Target={item["target"]} Quality={item["quality"]:.2f}]:\n')
            f.write(', '.join(item['game']) + '\n\n')

    print('Long optimization finished')
    print(f'Candidates: {len(candidates)}, top saved to: {RESULTS_FILE}, {GAMES_FILE}')


if __name__ == '__main__':
    main()
