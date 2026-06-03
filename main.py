import argparse
import logging
import sys
from pathlib import Path

from config import DATA_FILE, DATABASE_FILE, LOG_DIR
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
from evaluation_engine import evaluate_last_draw
from exporter import Exporter, _save_dataframe

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
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Skip Monte Carlo and genetic search (faster refresh after new draws)",
    )
    return parser.parse_args()


def _build_shared_reports(draws):
    stats = build_statistics_report(draws)
    graph_metrics = build_graph_metrics(draws)
    correlation_report = build_correlation_report(draws)
    markov = MarkovEngine(draws)
    markov_report = markov.build_all_orders()
    hmm = HMMEngine(draws)
    hmm_report = hmm.train_and_score()
    return stats, graph_metrics, correlation_report, markov_report, hmm_report


def _build_prediction_pipeline(draws, graph_metrics, markov_report, hmm_report):
    eval_draws = draws.iloc[:-1].copy() if len(draws) > 1 else draws.copy()
    eval_graph_metrics = build_graph_metrics(eval_draws)
    eval_markov_report = MarkovEngine(eval_draws).build_all_orders()
    eval_hmm_report = HMMEngine(eval_draws).train_and_score()

    features_eval = build_feature_matrix(
        draws,
        graph_metrics=eval_graph_metrics,
        markov_report=eval_markov_report,
        hmm_report=eval_hmm_report,
        include_last_draw=False,
    )
    ml_eval = MachineLearningEngine(draws, features_eval)
    ml_eval_report = ml_eval.train_models()
    ensemble_eval = EnsembleEngine(features_eval, ml_eval_report, ml_eval.model_weights_).build_stack(
        fit_targets=True
    )
    last_draw_evaluation = evaluate_last_draw(draws, ensemble_eval, prob_column="prob_final")

    features_next = build_feature_matrix(
        draws,
        graph_metrics=graph_metrics,
        markov_report=markov_report,
        hmm_report=hmm_report,
        include_last_draw=True,
    )
    ml_next = MachineLearningEngine(draws, features_next)
    ml_next_report = ml_next.predict_next_draw()
    ensemble_next = EnsembleEngine(features_next, ml_next_report, ml_next.model_weights_).build_stack(
        fit_targets=False
    )
    return features_next, ensemble_eval, ensemble_next, last_draw_evaluation


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

    stats, graph_metrics, correlation_report, markov_report, hmm_report = _build_shared_reports(clean_draws)
    features, ensemble_eval, ensemble_next, last_draw_evaluation = _build_prediction_pipeline(
        clean_draws, graph_metrics, markov_report, hmm_report
    )

    backtester = BacktestEngine(clean_draws, features)
    backtest_report = backtester.walk_forward_validation()

    generated_games = None
    montecarlo_report = None
    if not args.quick:
        simulator = MonteCarloSimulator(clean_draws)
        montecarlo_report = simulator.run_simulation(n_simulations=10000)
        optimizer = GeneticGameOptimizer(clean_draws, features)
        generated_games = optimizer.run_evolutionary_search()

    exporter = Exporter(DATABASE_FILE)
    exporter.save_statistics(stats)
    exporter.save_predictions(ensemble_next)
    exporter.save_last_draw_evaluation(last_draw_evaluation)
    if generated_games is not None:
        exporter.save_games(generated_games)
    exporter.save_backtests(backtest_report)

    hits = last_draw_evaluation["top_slices"]["20"]["hits"]
    logger.info(
        "Last draw %s: %s/20 hits in top-20 (prob_final). Next-draw top picks: %s",
        last_draw_evaluation.get("concurso"),
        hits,
        ", ".join(ensemble_next.head(20)["number"].tolist()),
    )

    if args.export:
        exporter.export_all(
            clean_draws,
            features,
            stats,
            predictions=ensemble_next,
            backtests=backtest_report,
            generated_games=generated_games,
        )
        _save_dataframe(ensemble_eval, "predictions_last_draw_eval")

    if args.dashboard:
        import dashboard as dashboard_module

        dashboard_module.run_dashboard()

    logger.info("Lotomania Quant Research Engine execution completed")


if __name__ == "__main__":
    main()
