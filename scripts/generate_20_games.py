import json
import sys
from pathlib import Path
from typing import List

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ai_learning_engine import load_learning_records
from data_loader import LotomaniaDataLoader
from elite_optimizer import generate_elite_portfolio, evaluate_portfolio_on_draw, build_elite_scores, EliteProfile
from ensemble_engine import EnsembleEngine
from feature_engineering import build_feature_matrix
from graph_engine import build_graph_metrics
from hmm_engine import HMMEngine
from markov_engine import MarkovEngine
from ml_engine import MachineLearningEngine
from preprocessing import cache_draws, load_cached_draws, preprocess_draws
from exporter import _save_dataframe
DATA_FILE = ROOT / 'data' / 'Lotomania.csv'
EXPORT_DIR = ROOT / 'exports'
EXPORT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_FILE = EXPORT_DIR / 'elite_portfolio_report.json'
GAMES_CSV = EXPORT_DIR / 'elite_portfolio_games.csv'


def _read_draw_data() -> pd.DataFrame:
    loader = LotomaniaDataLoader(DATA_FILE)
    raw_draws = loader.load_data()
    clean_draws = preprocess_draws(raw_draws)
    cached = load_cached_draws(clean_draws)
    if cached is not None:
        clean_draws = cached
    else:
        cache_draws(clean_draws)
    return clean_draws


def _build_prediction_artifacts(draws: pd.DataFrame):
    graph_metrics = build_graph_metrics(draws)
    markov_report = MarkovEngine(draws).build_all_orders()
    hmm_report = HMMEngine(draws).train_and_score()
    features = build_feature_matrix(
        draws,
        graph_metrics=graph_metrics,
        markov_report=markov_report,
        hmm_report=hmm_report,
        include_last_draw=True,
    )
    ml_engine = MachineLearningEngine(draws, features)
    ml_next = ml_engine.predict_next_draw()
    ensemble = EnsembleEngine(features, ml_next, ml_engine.model_weights_).build_stack(fit_targets=False)
    return features, ensemble


def _save_games(games: List[List[str]], profile: EliteProfile, scores: pd.DataFrame):
    rows = []
    for idx, game in enumerate(games, start=1):
        for position, number in enumerate(game, start=1):
            rows.append({'game': idx, 'position': position, 'number': number})
    games_df = pd.DataFrame(rows)
    _save_dataframe(games_df, 'elite_portfolio_games')
    return games_df


def _portfolio_analysis(draws: pd.DataFrame, games: List[List[str]], profile: EliteProfile, ensemble: pd.DataFrame, scores: pd.DataFrame):
    unique_numbers = sorted({number for game in games for number in game}, key=lambda x: int(x))
    frequency = pd.Series([number for game in games for number in game]).value_counts().sort_index()
    top_20_pred = ensemble.head(20)[['number', 'prob_final']].to_dict(orient='records')
    latest_draw = draws.iloc[-1]
    latest_draw_numbers = {str(latest_draw[col]).zfill(2) for col in latest_draw.index if col.startswith('Bola')}
    evaluation = evaluate_portfolio_on_draw(games, latest_draw_numbers)
    return {
        'total_draws': len(draws),
        'latest_draw_concurso': int(latest_draw['Concurso']),
        'latest_draw_date': str(latest_draw['Data Sorteio']),
        'portfolio_unique_numbers': len(unique_numbers),
        'portfolio_number_coverage': unique_numbers,
        'portfolio_number_frequency': frequency.to_dict(),
        'top_20_ensemble': top_20_pred,
        'latest_draw_best_game_hits': evaluation['best'],
        'latest_draw_avg_hits': evaluation['avg'],
        'latest_draw_games_with_15_plus': evaluation['count_15_plus'],
        'elite_profile': {
            'name': profile.name,
            'target_hits': profile.target_hits,
            'avg_hits': profile.avg_hits,
            'target_rate': profile.target_rate,
            'hit_19_rate': profile.hit_19_rate,
            'hit_18_rate': profile.hit_18_rate,
            'hit_15_rate': profile.hit_15_rate,
            'max_hits': profile.max_hits,
        },
        'top_rated_numbers': scores[['number', 'elite_score']].head(50).to_dict(orient='records'),
    }


def main() -> None:
    draws = _read_draw_data()
    learning_records = load_learning_records()
    features, ensemble = _build_prediction_artifacts(draws)
    games, profile, scores = generate_elite_portfolio(
        draws,
        learning_records=learning_records or None,
        n_games=20,
        ticket_size=50,
        seed=8609,
        target_hits=19,
    )
    games_df = _save_games(games, profile, scores)
    report = _portfolio_analysis(draws, games, profile, ensemble, scores)
    with open(REPORT_FILE, 'w', encoding='utf-8') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    print('Generated 20 elite games of 50 numbers each. Report saved to:', REPORT_FILE)
    print('Games saved to:', GAMES_CSV)
    print('Portfolio unique numbers:', report['portfolio_unique_numbers'])
    print('Best hit count on latest draw (historical evaluation):', report['latest_draw_best_game_hits'])
    print('Average hit count on latest draw:', report['latest_draw_avg_hits'])
    print('Number of games with >=15 hits on latest draw:', report['latest_draw_games_with_15_plus'])
    print('\nTop 20 ensemble candidates:')
    for item in report['top_20_ensemble']:
        print(f"{item['number']}: {item['prob_final']:.6f}")
    print('\nFirst 5 generated games:')
    for idx, game in enumerate(games[:5], start=1):
        print(f"Game {idx:02d}: {', '.join(game)}")


if __name__ == '__main__':
    main()
