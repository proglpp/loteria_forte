import argparse
import logging
import sys
from pathlib import Path

from config import DATA_FILE, DATABASE_FILE, LOG_DIR, ROOT_DIR, SEED
from data_loader import LotomaniaDataLoader
from preprocessing import cache_draws, load_cached_draws, preprocess_draws
from feature_engineering import build_feature_matrix
from statistics_engine import build_statistics_report
from graph_engine import build_graph_metrics
from correlation_engine import build_correlation_report
from markov_engine import MarkovEngine
from hmm_engine import HMMEngine
from montecarlo_engine import MonteCarloSimulator
from ml_engine import MachineLearningEngine
from ensemble_engine import EnsembleEngine
from backtest_engine import BacktestEngine
from genetic_engine import GeneticGameOptimizer
from exporter import Exporter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "pipeline.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("lotomania_quant")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lotomania Quant Research Engine v2.0")
    parser.add_argument("--dashboard", action="store_true", help="Start the Streamlit dashboard")
    parser.add_argument("--export", action="store_true", help="Export generated data to CSV, JSON, and SQLite")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger.info("Starting Lotomania Quant Research Engine v2.0")

    loader = LotomaniaDataLoader(DATA_FILE, DATABASE_FILE)
    raw_draws = loader.load_data()
    clean_draws = preprocess_draws(raw_draws)
    cached = load_cached_draws(clean_draws)

    if cached is not None:
        logger.info("Loaded preprocessed draws from cache")
        clean_draws = cached
    else:
        cache_draws(clean_draws)

    loader.persist_draws(clean_draws)

    stats = build_statistics_report(clean_draws)
    graph_metrics = build_graph_metrics(clean_draws)
    correlation_report = build_correlation_report(clean_draws)

    markov = MarkovEngine(clean_draws)
    markov_report = markov.build_all_orders()

    hmm = HMMEngine(clean_draws)
    hmm_report = hmm.train_and_score()

    features = build_feature_matrix(
        clean_draws,
        graph_metrics=graph_metrics,
        markov_report=markov_report,
        hmm_report=hmm_report,
    )

    ml_engine = MachineLearningEngine(clean_draws, features)
    ml_report = ml_engine.train_models()

    ensemble = EnsembleEngine(features, ml_report)
    ensemble_report = ensemble.build_stack()

    simulator = MonteCarloSimulator(clean_draws)
    montecarlo_report = simulator.run_simulation(n_simulations=10000)

    backtester = BacktestEngine(clean_draws, features)
    backtest_report = backtester.walk_forward_validation()

    optimizer = GeneticGameOptimizer(clean_draws, features)
    generated_games = optimizer.run_evolutionary_search()

    exporter = Exporter(DATABASE_FILE)
    exporter.save_statistics(stats)
    exporter.save_predictions(ml_report)
    exporter.save_games(generated_games)
    exporter.save_backtests(backtest_report)

    if args.export:
        exporter.export_all(
            clean_draws,
            features,
            stats,
            predictions=ml_report,
            backtests=backtest_report,
            generated_games=generated_games,
        )

    if args.dashboard:
        import dashboard as dashboard_module

        dashboard_module.run_dashboard()

    logger.info("Lotomania Quant Research Engine execution completed")


if __name__ == "__main__":
    main()
