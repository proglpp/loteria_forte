import logging
import math
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
import threading
import time

from config import CACHE_DIR, DATA_FILE, DATABASE_FILE, DRAW_COLUMNS
from correlation_engine import build_correlation_report, strongest_pairs
from data_loader import LotomaniaDataLoader
from exporter import Exporter
from feature_engineering import build_feature_matrix
from graph_engine import build_graph_metrics
from hmm_engine import HMMEngine
from markov_engine import MarkovEngine
from montecarlo_engine import MonteCarloSimulator
from backtest_engine import BacktestEngine
from genetic_engine import GeneticGameOptimizer
from ml_engine import MachineLearningEngine
from ensemble_engine import EnsembleEngine
from evaluation_engine import evaluate_last_draw
from statistics_engine import build_statistics_report
import ml_engine as ml_engine_module
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from lightgbm import LGBMClassifier
from game_ui import render_game_selector_panel, render_generated_games_gallery

logger = logging.getLogger(__name__)

CACHE_DIR_PATH = Path(CACHE_DIR)
CACHE_DIR_PATH.mkdir(parents=True, exist_ok=True)
FEATURES_CACHE_FILE = CACHE_DIR_PATH / "dashboard_features.pkl"
GRAPH_CACHE_FILE = CACHE_DIR_PATH / "dashboard_graph_metrics.pkl"
MARKOV_CACHE_FILE = CACHE_DIR_PATH / "dashboard_markov.pkl"
HMM_CACHE_FILE = CACHE_DIR_PATH / "dashboard_hmm.pkl"
CORRELATION_CACHE_FILE = CACHE_DIR_PATH / "dashboard_correlation.pkl"

_background_features = None
_background_graph_metrics = None
_background_markov_report = None
_background_hmm_report = None
_background_running = False


def _get_cache_signature(draws: pd.DataFrame) -> tuple[int, int, str]:
    last_concurso = int(draws["Concurso"].max()) if "Concurso" in draws.columns else 0
    last_date = str(draws["Data Sorteio"].max()) if "Data Sorteio" in draws.columns else ""
    return len(draws), last_concurso, last_date


def _load_from_disk_cache(cache_file: Path, signature: tuple[int, int, str]) -> Any | None:
    if not cache_file.exists():
        return None
    try:
        with open(cache_file, "rb") as handle:
            payload = pickle.load(handle)
        if payload.get("signature") == signature:
            return payload.get("value")
    except Exception as exc:
        logger.warning("Failed to load dashboard cache %s: %s", cache_file, exc)
    return None


def _save_to_disk_cache(cache_file: Path, signature: tuple[int, int, str], value: Any) -> None:
    try:
        with open(cache_file, "wb") as handle:
            pickle.dump({"signature": signature, "value": value}, handle)
    except Exception as exc:
        logger.warning("Failed to save dashboard cache %s: %s", cache_file, exc)


def render_table(title: str, df: pd.DataFrame, max_rows: int = 10) -> None:
    st.markdown(f"### {title}")
    st.dataframe(df.head(max_rows))


def render_chart(df: pd.DataFrame, x: str, y: str, title: str) -> None:
    if df.empty:
        return
    st.markdown(f"### {title}")
    chart_data = df.set_index(x)[y]
    st.bar_chart(chart_data)


def render_multi_series_chart(df: pd.DataFrame, x: str, y_cols: list[str], title: str) -> None:
    if df.empty or not y_cols:
        return
    st.markdown(f"### {title}")
    chart_data = df.set_index(x)[y_cols]
    st.line_chart(chart_data)


@st.cache_data(show_spinner=False)
def load_basic_draws():
    loader = LotomaniaDataLoader(DATA_FILE)
    draws = loader.load_data()
    statistics = build_statistics_report(draws)
    return draws, statistics


@st.cache_data(show_spinner=False)
def compute_graph_metrics(draws):
    signature = _get_cache_signature(draws)
    cached = _load_from_disk_cache(GRAPH_CACHE_FILE, signature)
    if cached is not None:
        return cached
    graph_metrics = build_graph_metrics(draws)
    _save_to_disk_cache(GRAPH_CACHE_FILE, signature, graph_metrics)
    return graph_metrics


@st.cache_data(show_spinner=False)
def compute_correlation(draws):
    signature = _get_cache_signature(draws)
    cached = _load_from_disk_cache(CORRELATION_CACHE_FILE, signature)
    if cached is not None:
        return cached
    correlation_report = build_correlation_report(draws)
    _save_to_disk_cache(CORRELATION_CACHE_FILE, signature, correlation_report)
    return correlation_report


@st.cache_data(show_spinner=False)
def compute_markov(draws):
    signature = _get_cache_signature(draws)
    cached = _load_from_disk_cache(MARKOV_CACHE_FILE, signature)
    if cached is not None:
        return cached
    markov_report = MarkovEngine(draws).build_all_orders()
    _save_to_disk_cache(MARKOV_CACHE_FILE, signature, markov_report)
    return markov_report


@st.cache_data(show_spinner=False)
def compute_hmm(draws):
    signature = _get_cache_signature(draws)
    cached = _load_from_disk_cache(HMM_CACHE_FILE, signature)
    if cached is not None:
        return cached
    hmm_report = HMMEngine(draws).train_and_score()
    _save_to_disk_cache(HMM_CACHE_FILE, signature, hmm_report)
    return hmm_report


@st.cache_data(show_spinner=False)
def prepare_features(draws, graph_metrics, markov_report, hmm_report):
    signature = _get_cache_signature(draws)
    cached = _load_from_disk_cache(FEATURES_CACHE_FILE, signature)
    if cached is not None:
        return cached
    features = build_feature_matrix(
        draws,
        graph_metrics=graph_metrics,
        markov_report=markov_report,
        hmm_report=hmm_report,
    )
    _save_to_disk_cache(FEATURES_CACHE_FILE, signature, features)
    return features


def ensure_features(draws):
    features = st.session_state.get("features")
    global _background_features
    if features is not None:
        return features

    if _background_features is not None:
        st.session_state.features = _background_features
        _background_features = None
        return st.session_state.features

    # synchronous prepare (keeps compatibility when explicitly requested)
    with st.spinner("Preparando métricas avançadas e matriz de features..."):
        graph_metrics = st.session_state.get("graph_metrics")
        if graph_metrics is None:
            graph_metrics = compute_graph_metrics(draws)
            st.session_state.graph_metrics = graph_metrics

        markov_report = st.session_state.get("markov_report")
        if markov_report is None:
            markov_report = compute_markov(draws)
            st.session_state.markov_report = markov_report

        hmm_report = st.session_state.get("hmm_report")
        if hmm_report is None:
            hmm_report = compute_hmm(draws)
            st.session_state.hmm_report = hmm_report

        features = prepare_features(draws, graph_metrics, markov_report, hmm_report)
        st.session_state.features = features

    return features


def _sync_background_state():
    global _background_features, _background_graph_metrics, _background_markov_report, _background_hmm_report, _background_running

    if st.session_state.get("features") is None and _background_features is not None:
        st.session_state.features = _background_features
        if _background_graph_metrics is not None:
            st.session_state.graph_metrics = _background_graph_metrics
        if _background_markov_report is not None:
            st.session_state.markov_report = _background_markov_report
        if _background_hmm_report is not None:
            st.session_state.hmm_report = _background_hmm_report

    if st.session_state.get("features_background_running") and not _background_running:
        st.session_state["features_background_running"] = False
    if not st.session_state.get("features_background_running") and _background_running:
        st.session_state["features_background_running"] = True


def _background_prepare_features(draws):
    """Run heavy feature preparation in a background thread and cache to session_state."""
    global _background_features, _background_graph_metrics, _background_markov_report, _background_hmm_report, _background_running
    _background_running = True
    signature = _get_cache_signature(draws)
    try:
        graph_metrics = build_graph_metrics(draws)
        _background_graph_metrics = graph_metrics
        _save_to_disk_cache(GRAPH_CACHE_FILE, signature, graph_metrics)
    except Exception as exc:
        logger.exception("Background graph metrics failed: %s", exc)
        _background_running = False
        return

    try:
        markov_report = MarkovEngine(draws).build_all_orders()
        _background_markov_report = markov_report
        _save_to_disk_cache(MARKOV_CACHE_FILE, signature, markov_report)
    except Exception as exc:
        logger.exception("Background markov report failed: %s", exc)
        _background_running = False
        return

    try:
        hmm_report = HMMEngine(draws).train_and_score()
        _background_hmm_report = hmm_report
        _save_to_disk_cache(HMM_CACHE_FILE, signature, hmm_report)
    except Exception as exc:
        logger.exception("Background HMM report failed: %s", exc)
        _background_running = False
        return

    try:
        features = build_feature_matrix(draws, graph_metrics=graph_metrics, markov_report=markov_report, hmm_report=hmm_report)
        _background_features = features
        _save_to_disk_cache(FEATURES_CACHE_FILE, signature, features)
    except Exception as exc:
        logger.exception("Background feature matrix build failed: %s", exc)
        _background_running = False
        return

    try:
        st.session_state.features = features
        st.session_state.graph_metrics = graph_metrics
        st.session_state.markov_report = markov_report
        st.session_state.hmm_report = hmm_report
    except Exception:
        pass

    _background_running = False
    logger.info("Background features preparation completed. Attempting to trigger page refresh...")
    try:
        # Try newer st.rerun() first (Streamlit >= 1.27)
        if hasattr(st, "rerun"):
            st.rerun()
        # Fallback to experimental version
        elif hasattr(st, "experimental_rerun"):
            st.experimental_rerun()
    except Exception as exc:
        logger.warning("Could not trigger page refresh: %s", exc)


def ensure_correlation(draws):
    correlation = st.session_state.get("correlation")
    if correlation is None:
        correlation = compute_correlation(draws)
        st.session_state.correlation = correlation
    return correlation


@st.cache_data(show_spinner=False)
def compute_ml_report(draws, features):
    ml_engine = MachineLearningEngine(draws, features)
    ml_report = ml_engine.predict_next_draw()
    ensemble_report = EnsembleEngine(features, ml_report, ml_engine.model_weights_).build_stack(fit_targets=False)
    return ml_engine, ml_report, ensemble_report


@st.cache_data(show_spinner=False)
def compute_last_draw_evaluation(draws, features_eval):
    ml_engine = MachineLearningEngine(draws, features_eval)
    ml_report = ml_engine.train_models()
    ensemble_eval = EnsembleEngine(features_eval, ml_report, ml_engine.model_weights_).build_stack(fit_targets=True)
    return evaluate_last_draw(draws, ensemble_eval, prob_column="prob_final")


def compute_ml_report_fast(draws, features):
    """Train a lightweight set of models to provide faster results."""
    # Prepare a lightweight set of models
    FAST_MODELS = {
        "random_forest": RandomForestClassifier(n_estimators=20, random_state=42, n_jobs=-1),
        "extra_trees": ExtraTreesClassifier(n_estimators=20, random_state=42, n_jobs=-1),
        "lightgbm": LGBMClassifier(random_state=42, n_jobs=-1, n_estimators=50),
    }

    # Temporarily replace BASE_MODELS in ml_engine module
    try:
        original_models = getattr(ml_engine_module, "BASE_MODELS")
    except Exception:
        original_models = None

    try:
        setattr(ml_engine_module, "BASE_MODELS", FAST_MODELS)
        ml_engine = MachineLearningEngine(draws, features)
        ml_report = ml_engine.predict_next_draw()
        ensemble_report = EnsembleEngine(features, ml_report, ml_engine.model_weights_).build_stack(fit_targets=False)
    finally:
        # restore original models
        if original_models is not None:
            setattr(ml_engine_module, "BASE_MODELS", original_models)

    return ml_engine, ml_report, ensemble_report


@st.cache_data(show_spinner=False)
def compute_backtest(draws, features):
    return BacktestEngine(draws, features).walk_forward_validation()


@st.cache_data(show_spinner=False)
def compute_montecarlo(draws, n_simulations):
    return MonteCarloSimulator(draws).run_simulation(n_simulations=n_simulations)


def _normalize_game_numbers(selection, limit: int | None = None):
    normalized = []
    for value in selection or []:
        try:
            number = int(value)
        except (TypeError, ValueError):
            continue
        if 0 <= number <= 99:
            normalized.append(f"{number:02d}")
    normalized = list(dict.fromkeys(normalized))
    return normalized[:limit] if limit is not None else normalized


def _rank_numbers_for_generation(features, method: str, variation: int = 0, ensemble_report=None):
    df = features.copy()
    df["number"] = df["number"].astype(str).str.zfill(2)
    df["number_int"] = df["number"].astype(int)
    rng = np.random.default_rng(42 + int(variation))

    score = pd.to_numeric(df.get("score_total", 0), errors="coerce").fillna(0.0)
    score_norm = (score - score.min()) / (score.max() - score.min() + 1e-12)
    df["_score_norm"] = score_norm

    if ensemble_report is not None and "prob_final" in ensemble_report.columns:
        probs = ensemble_report.copy()
        probs["number"] = probs["number"].astype(str).str.zfill(2)
        df = df.merge(probs[["number", "prob_final"]], on="number", how="left")
        df["_generation_score"] = pd.to_numeric(df["prob_final"], errors="coerce").fillna(df["_score_norm"])
        return df.sort_values("_generation_score", ascending=False)["number"].tolist()

    if method == "random":
        return df.sample(frac=1, random_state=42 + int(variation))["number"].tolist()

    if method == "post_result":
        last_draw = pd.to_numeric(df.get("last_draw_hit", 0), errors="coerce").fillna(0.0)
        recent_5 = pd.to_numeric(df.get("last_5_draws_hits", 0), errors="coerce").fillna(0.0)
        recent_5_norm = (recent_5 - recent_5.min()) / (recent_5.max() - recent_5.min() + 1e-12)
        delay = pd.to_numeric(df.get("delay_last", 0), errors="coerce").fillna(0.0)
        delay_norm = (delay - delay.min()) / (delay.max() - delay.min() + 1e-12)
        df["_generation_score"] = (df["_score_norm"] * 0.60) + (last_draw * 0.18) + (recent_5_norm * 0.14) + ((1 - delay_norm) * 0.08)
    elif method == "balanced":
        freq = pd.to_numeric(df.get("freq_global", 0), errors="coerce").fillna(0.0)
        freq_norm = (freq - freq.min()) / (freq.max() - freq.min() + 1e-12)
        parity_balance = np.where(df["number_int"] % 2 == 0, 0.5, 0.55)
        df["_generation_score"] = (df["_score_norm"] * 0.72) + ((1 - abs(freq_norm - 0.5)) * 0.18) + (parity_balance * 0.10)
    elif method == "coverage":
        delay = pd.to_numeric(df.get("delay_last", 0), errors="coerce").fillna(0.0)
        delay_norm = (delay - delay.min()) / (delay.max() - delay.min() + 1e-12)
        df["_generation_score"] = (df["_score_norm"] * 0.55) + (delay_norm * 0.35) + rng.normal(0, 0.03, len(df))
    else:
        df["_generation_score"] = df["_score_norm"] + rng.normal(0, 0.04, len(df))

    return df.sort_values("_generation_score", ascending=False)["number"].tolist()


def _generation_score_map(features, method: str, variation: int = 0, ensemble_report=None):
    ranked = _rank_numbers_for_generation(features, method, variation, ensemble_report=ensemble_report)
    total = max(len(ranked) - 1, 1)
    return {number: 1.0 - (index / total) for index, number in enumerate(ranked)}


def build_fast_generation_features(draws: pd.DataFrame, statistics: pd.DataFrame) -> pd.DataFrame:
    numbers = [f"{n:02d}" for n in range(100)]
    df = statistics.copy()
    df["number"] = df["number"].astype(str).str.zfill(2)
    if "freq_abs" not in df.columns:
        values = draws[DRAW_COLUMNS].values.flatten()
        df = pd.DataFrame({
            "number": numbers,
            "freq_abs": pd.Series(values).astype(str).str.zfill(2).value_counts().reindex(numbers, fill_value=0).values,
        })

    last_seen = {number: -1 for number in numbers}
    for idx, row in draws.reset_index(drop=True).iterrows():
        present = {str(row[col]).zfill(2) for col in DRAW_COLUMNS}
        for number in present:
            last_seen[number] = int(idx)

    recent_25 = pd.Series(draws.tail(25)[DRAW_COLUMNS].values.flatten()).astype(str).str.zfill(2).value_counts()
    recent_5 = pd.Series(draws.tail(5)[DRAW_COLUMNS].values.flatten()).astype(str).str.zfill(2).value_counts()
    last_draw = {str(draws.iloc[-1][col]).zfill(2) for col in DRAW_COLUMNS} if not draws.empty else set()

    df = df.set_index("number").reindex(numbers).reset_index()
    numeric_cols = df.select_dtypes(include=["number"]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0.0)
    df["freq_global"] = pd.to_numeric(df.get("freq_abs", 0), errors="coerce").fillna(0.0)
    df["freq_25"] = df["number"].map(recent_25).fillna(0).astype(float)
    df["last_5_draws_hits"] = df["number"].map(recent_5).fillna(0).astype(float)
    df["last_draw_hit"] = df["number"].isin(last_draw).astype(float)
    df["delay_last"] = df["number"].map(
        lambda number: len(draws) - last_seen[number] if last_seen[number] >= 0 else len(draws)
    ).astype(float)

    freq_norm = (df["freq_global"] - df["freq_global"].min()) / (df["freq_global"].max() - df["freq_global"].min() + 1e-12)
    recent_norm = (df["freq_25"] - df["freq_25"].min()) / (df["freq_25"].max() - df["freq_25"].min() + 1e-12)
    delay_norm = (df["delay_last"] - df["delay_last"].min()) / (df["delay_last"].max() - df["delay_last"].min() + 1e-12)
    df["score_total"] = (0.45 * freq_norm) + (0.30 * recent_norm) + (0.20 * delay_norm) + (0.05 * df["last_5_draws_hits"])
    return df


def generate_game(features, selection, method: str, game_size: int = 50, variation: int = 0, ensemble_report=None):
    """Generate a game of up to `game_size` numbers (strings like '00'..'99').

    - `selection` may contain preselected numbers (strings or ints converted elsewhere).
    - `method` supports scoring, balanced, coverage, and random profiles.
    - Ensures we never request more random choices than available.
    """
    selected = _normalize_game_numbers(selection, limit=game_size)

    needed = max(0, game_size - len(selected))
    remaining = [
        n
        for n in _rank_numbers_for_generation(features, method, variation, ensemble_report=ensemble_report)
        if n not in selected
    ]
    if method == "random":
        take = min(len(remaining), needed)
        rng = np.random.default_rng(42 + int(variation))
        chosen = list(rng.choice(remaining, size=take, replace=False)) if take > 0 else []
    else:
        offset = int(variation) % max(len(remaining), 1)
        rotated = remaining[offset:] + remaining[:offset]
        chosen = rotated[:needed]

    result = selected + chosen
    return sorted(result[:game_size])


def generate_games(features, selection, method: str, n_games: int = 1, game_size: int = 50, variation: int = 0, ensemble_report=None):
    """Generate multiple distinct games of size `game_size`."""
    games = []
    seen = set()
    selected = _normalize_game_numbers(selection, limit=game_size)

    if n_games <= 1:
        return [generate_game(features, selected, method, game_size, variation=variation, ensemble_report=ensemble_report)]

    top_numbers = _rank_numbers_for_generation(features, method, variation, ensemble_report=ensemble_report)
    remaining = [n for n in top_numbers if n not in selected]
    needed = max(0, game_size - len(selected))

    if method == "random":
        attempts = 0
        while len(games) < n_games and attempts < n_games * 10:
            attempts += 1
            chosen = list(np.random.choice(remaining, size=needed, replace=False)) if needed > 0 else []
            game = tuple(sorted(selected + chosen))[:game_size]
            if game not in seen:
                seen.add(game)
                games.append(list(game))
    else:
        for i in range(n_games):
            if needed <= 0:
                game = tuple(sorted(selected)[:game_size])
            else:
                offset = (i * max(1, needed // max(1, n_games))) % max(1, len(remaining))
                numbers = remaining[offset:] + remaining[:offset]
                chosen = numbers[:needed]
                game = tuple(sorted(selected + chosen))[:game_size]
            if game not in seen:
                seen.add(game)
                games.append(list(game))

    # if we could not build enough distinct games, fill with deterministic variations
    i = 0
    while len(games) < n_games and i < len(remaining):
        taken = remaining[i : i + needed]
        game = tuple(sorted(selected + taken))[:game_size]
        if game not in seen:
            seen.add(game)
            games.append(list(game))
        i += max(1, needed // 2 if needed > 1 else 1)

    return games


def generate_power_portfolio(
    features,
    selection,
    method: str,
    n_games: int = 10,
    universe_size: int = 90,
    game_size: int = 50,
    variation: int = 0,
    ensemble_report=None,
    min_power: float = 0.78,
):
    """Generate strong 50-number games while spreading risk across the portfolio."""
    fixed = _normalize_game_numbers(selection, limit=game_size)
    n_games = max(1, int(n_games))
    universe_size = max(game_size, min(100, int(universe_size)))
    ranked = _rank_numbers_for_generation(features, method, variation, ensemble_report=ensemble_report)
    score_map = _generation_score_map(features, method, variation, ensemble_report=ensemble_report)
    pool = list(dict.fromkeys(fixed + [n for n in ranked if n not in fixed]))[:universe_size]
    flexible_pool = [n for n in pool if n not in fixed]
    needed = max(0, game_size - len(fixed))
    if needed == 0:
        return [sorted(fixed[:game_size]) for _ in range(n_games)]

    top_reference = sum(score_map.get(n, 0.0) for n in ranked[:game_size])
    min_score = top_reference * float(min_power)
    usage = {number: 0 for number in pool}
    games = []
    seen = set()

    for game_index in range(n_games):
        chosen = list(fixed)
        remaining = [n for n in flexible_pool if n not in chosen]
        coverage_seed_count = min(max(4, len(flexible_pool) // max(n_games, 1)), max(needed // 3, 1))
        coverage_seed = sorted(
            remaining,
            key=lambda n: (usage.get(n, 0), -score_map.get(n, 0.0), (int(n) + game_index + variation) % 17),
        )[:coverage_seed_count]
        chosen.extend(coverage_seed)
        remaining = [n for n in remaining if n not in coverage_seed]

        while len(chosen) < game_size and remaining:
            best_number = None
            best_value = -1e9
            current = set(chosen)
            min_usage = min(usage.get(n, 0) for n in remaining)
            for number in remaining:
                candidate = current | {number}
                overlap_penalty = sum(len(candidate & set(game)) / game_size for game in games)
                usage_penalty = usage.get(number, 0) / max(game_index + 1, 1)
                underuse_bonus = 0.28 if usage.get(number, 0) == min_usage else 0.0
                rotation_bonus = ((int(number) + game_index + variation) % 17) / 500.0
                value = score_map.get(number, 0.0) + underuse_bonus - (0.58 * usage_penalty) - (0.20 * overlap_penalty) + rotation_bonus
                if value > best_value:
                    best_value = value
                    best_number = number
            if best_number is None:
                break
            chosen.append(best_number)
            remaining.remove(best_number)

        game_score = sum(score_map.get(n, 0.0) for n in chosen)
        if game_score < min_score:
            stronger = [n for n in ranked if n not in chosen]
            replaceable = [n for n in chosen if n not in fixed]
            replaceable.sort(key=lambda n: score_map.get(n, 0.0))
            for weak, strong in zip(replaceable, stronger):
                if game_score >= min_score:
                    break
                chosen.remove(weak)
                chosen.append(strong)
                game_score += score_map.get(strong, 0.0) - score_map.get(weak, 0.0)

        game = sorted(chosen[:game_size])
        attempts = 0
        while tuple(game) in seen and attempts < len(flexible_pool):
            attempts += 1
            swap_out = [n for n in game if n not in fixed]
            in_candidates = [n for n in flexible_pool if n not in game]
            if not swap_out or not in_candidates:
                break
            out_number = swap_out[attempts % len(swap_out)]
            in_number = in_candidates[attempts % len(in_candidates)]
            game = sorted([n for n in game if n != out_number] + [in_number])

        seen.add(tuple(game))
        games.append(game)
        for number in game:
            usage[number] = usage.get(number, 0) + 1

    return games


def generate_covering_games(
    features,
    selection,
    method: str,
    n_games: int = 4,
    universe_size: int = 80,
    game_size: int = 50,
    variation: int = 0,
):
    """Build a fechamento: several 50-number games from a larger ranked universe."""
    fixed = _normalize_game_numbers(selection, limit=game_size)
    universe_size = max(game_size, min(100, int(universe_size)))
    ranked = _rank_numbers_for_generation(features, method, variation)
    pool = list(dict.fromkeys(fixed + [n for n in ranked if n not in fixed]))[:universe_size]
    flexible_pool = [n for n in pool if n not in fixed]
    needed = max(0, game_size - len(fixed))

    games = []
    seen = set()
    for index in range(max(1, int(n_games))):
        if needed <= 0:
            game = tuple(sorted(fixed[:game_size]))
        elif len(flexible_pool) <= needed:
            chosen = flexible_pool[:needed]
            game = tuple(sorted((fixed + chosen)[:game_size]))
        else:
            step = max(1, int(np.ceil(len(flexible_pool) / max(1, n_games))))
            start = (index * step + variation) % len(flexible_pool)
            rotated = flexible_pool[start:] + flexible_pool[:start]
            chosen = rotated[:needed]
            if len(chosen) < needed:
                chosen += [n for n in flexible_pool if n not in chosen][: needed - len(chosen)]
            game = tuple(sorted((fixed + chosen)[:game_size]))
        if game not in seen:
            seen.add(game)
            games.append(list(game))

    return games


def estimate_portfolio_prize_chances(games, n_simulations: int = 4000, seed: int = 42):
    games = [set(_normalize_game_numbers(game, limit=50)) for game in games if game]
    if not games:
        return {
            "tiers": {"17": 0.0, "18": 0.0, "19": 0.0, "20": 0.0},
            "single_ticket_tiers": {"17": 0.0, "18": 0.0, "19": 0.0, "20": 0.0},
            "avg_best_hits": 0.0,
            "covered_numbers": 0,
            "avg_overlap": 0.0,
        }

    rng = np.random.default_rng(seed)
    numbers = np.array([f"{n:02d}" for n in range(100)])
    tiers = {17: 0, 18: 0, 19: 0, 20: 0}
    best_hits_total = 0
    for _ in range(int(n_simulations)):
        draw = set(rng.choice(numbers, size=20, replace=False))
        best_hits = max(len(game & draw) for game in games)
        best_hits_total += best_hits
        for tier in tiers:
            if best_hits >= tier:
                tiers[tier] += 1

    denominator = math.comb(100, 20)
    single = {}
    for tier in tiers:
        single[tier] = sum(
            math.comb(50, hits) * math.comb(50, 20 - hits)
            for hits in range(tier, 21)
        ) / denominator

    overlaps = [
        len(games[i] & games[j])
        for i in range(len(games))
        for j in range(i + 1, len(games))
    ]
    return {
        "tiers": {str(tier): tiers[tier] / n_simulations for tier in tiers},
        "single_ticket_tiers": {str(tier): single[tier] for tier in tiers},
        "avg_best_hits": best_hits_total / n_simulations,
        "covered_numbers": len(set().union(*games)),
        "avg_overlap": float(np.mean(overlaps)) if overlaps else 0.0,
    }


def append_generated_games(new_games):
    current_games = st.session_state.get("generated_games") or []
    st.session_state.generated_games = current_games + list(new_games)
    return st.session_state.generated_games


def run_dashboard() -> None:
    st.set_page_config(page_title="Lotomania Quant Research Engine", layout="wide")
    st.title("Lotomania Quant Research Engine v2.0")

    st.sidebar.header("Controles")
    montecarlo_runs = st.sidebar.slider("Simulações Monte Carlo", min_value=1000, max_value=20000, value=5000, step=1000)
    fitness_criteria = st.sidebar.selectbox(
        "Critério de fitness genético",
        ["score_total", "diversity", "coverage", "balanced"],
        format_func=lambda v: {
            "score_total": "Score Total",
            "diversity": "Diversidade",
            "coverage": "Cobertura",
            "balanced": "Balanceado",
        }[v],
    )
    population_size = st.sidebar.slider("Tamanho da população genética", min_value=10, max_value=100, value=50, step=10)
    generations = st.sidebar.slider("Gerações genéticas", min_value=5, max_value=100, value=20, step=5)
    run_features = st.sidebar.button("Preparar features avançadas")
    run_montecarlo = st.sidebar.button("Executar Monte Carlo")
    run_model = st.sidebar.button("Treinar Modelos")
    run_model_fast = st.sidebar.button("Treinar modelos rápido")
    num_games = st.sidebar.slider("Número de jogos a gerar", min_value=1, max_value=100, value=8, step=1)
    run_backtest = st.sidebar.button("Executar Backtest")
    run_genetic = st.sidebar.button("Executar Genetic Search")
    run_correlation = st.sidebar.button("Calcular Correlações")
    export_csv = st.sidebar.button("Exportar CSV/JSON")
    export_sqlite = st.sidebar.button("Exportar SQLite")
    show_model_metrics = st.sidebar.checkbox("Mostrar métricas de ML", value=True)
    st.sidebar.markdown("---")
    st.sidebar.write("Use os botões e seleções acima para gerar simulações, otimização genética e exportar resultados.")

    exporter = Exporter(DATABASE_FILE)

    with st.spinner("Carregando dados e métricas básicas..."):
        draws, statistics = load_basic_draws()

    # Try to pre-load cached heavy results into session_state so UI shows immediately
    try:
        signature = _get_cache_signature(draws)
        cached_features = _load_from_disk_cache(FEATURES_CACHE_FILE, signature)
        if cached_features is not None:
            st.session_state.features = cached_features
        cached_graph = _load_from_disk_cache(GRAPH_CACHE_FILE, signature)
        if cached_graph is not None:
            st.session_state.graph_metrics = cached_graph
        cached_markov = _load_from_disk_cache(MARKOV_CACHE_FILE, signature)
        if cached_markov is not None:
            st.session_state.markov_report = cached_markov
        cached_hmm = _load_from_disk_cache(HMM_CACHE_FILE, signature)
        if cached_hmm is not None:
            st.session_state.hmm_report = cached_hmm
        cached_corr = _load_from_disk_cache(CORRELATION_CACHE_FILE, signature)
        if cached_corr is not None:
            st.session_state.correlation = cached_corr
        if any(v is not None for v in (cached_features, cached_graph, cached_markov, cached_hmm, cached_corr)):
            logger.info("Pre-loaded dashboard caches into session_state")
    except Exception as exc:
        logger.warning("Failed pre-loading dashboard cache: %s", exc)

    if "montecarlo_results" not in st.session_state:
        st.session_state.montecarlo_results = None

    if "ml_engine" not in st.session_state:
        st.session_state.ml_engine = None
    if "ml_report" not in st.session_state:
        st.session_state.ml_report = None
    if "ensemble_report" not in st.session_state:
        st.session_state.ensemble_report = None
    if "backtest_report" not in st.session_state:
        st.session_state.backtest_report = None
    if "genetic_report" not in st.session_state:
        st.session_state.genetic_report = None
    if "features_background_running" not in st.session_state:
        st.session_state["features_background_running"] = False

    _sync_background_state()

    features = st.session_state.get("features")
    graph_metrics = st.session_state.get("graph_metrics")
    correlation = st.session_state.get("correlation")
    markov_report = st.session_state.get("markov_report")
    hmm_report = st.session_state.get("hmm_report")
    ml_engine = st.session_state.get("ml_engine")
    ml_report = st.session_state.get("ml_report")
    ensemble_report = st.session_state.get("ensemble_report")
    backtest_report = st.session_state.get("backtest_report")

    if run_features and features is None:
        if not st.session_state["features_background_running"]:
            st.session_state["features_background_running"] = True
            threading.Thread(target=_background_prepare_features, args=(draws,), daemon=True).start()
            st.success("Preparação de features em background iniciada. Use Atualizar agora para ver o resultado quando estiver pronto.")
        else:
            st.info("A preparação de features já está rodando em background.")

    if run_correlation and correlation is None:
        correlation = ensure_correlation(draws)
        st.success("Correlação calculada e armazenada em cache.")

    if run_model:
        if features is None:
            features = ensure_features(draws)
        try:
            compute_ml_report.clear()
        except Exception:
            pass
        with st.spinner("Treinando modelos e construindo ensemble..."):
            ml_engine, ml_report, ensemble_report = compute_ml_report(draws, features)
        st.session_state.ml_engine = ml_engine
        st.session_state.ml_report = ml_report
        st.session_state.ensemble_report = ensemble_report
    if run_model_fast:
        if features is None:
            features = ensure_features(draws)
        try:
            compute_ml_report.clear()
        except Exception:
            pass
        with st.spinner("Treinamento rápido em andamento..."):
            ml_engine, ml_report, ensemble_report = compute_ml_report_fast(draws, features)
        st.session_state.ml_engine = ml_engine
        st.session_state.ml_report = ml_report
        st.session_state.ensemble_report = ensemble_report
    elif "ensemble_report" in st.session_state:
        ensemble_report = st.session_state.ensemble_report
        ml_report = st.session_state.ml_report
        ml_engine = st.session_state.ml_engine

    if run_backtest:
        if features is None:
            features = ensure_features(draws)
        with st.spinner("Executando backtest..."):
            backtest_report = compute_backtest(draws, features)
        st.session_state.backtest_report = backtest_report
    elif "backtest_report" in st.session_state:
        backtest_report = st.session_state.backtest_report

    if export_csv:
        if features is None:
            features = ensure_features(draws)
        if ensemble_report is None:
            try:
                compute_ml_report.clear()
            except Exception:
                pass
            with st.spinner("Treinando modelos para exportação..."):
                ml_engine, ml_report, ensemble_report = compute_ml_report(draws, features)
            st.session_state.ml_engine = ml_engine
            st.session_state.ml_report = ml_report
            st.session_state.ensemble_report = ensemble_report
        exporter.export_all(
            draws,
            features,
            statistics,
            predictions=ensemble_report,
            backtests=backtest_report,
            generated_games=None,
        )
        st.success("Dados exportados para CSV/JSON em exports/")

    if export_sqlite:
        if features is None:
            features = ensure_features(draws)
        if ensemble_report is None:
            try:
                compute_ml_report.clear()
            except Exception:
                pass
            with st.spinner("Treinando modelos para exportação..."):
                ml_engine, ml_report, ensemble_report = compute_ml_report(draws, features)
            st.session_state.ml_engine = ml_engine
            st.session_state.ml_report = ml_report
            st.session_state.ensemble_report = ensemble_report
        exporter.save_database_exports(
            {
                "draws": draws,
                "features": features,
                "statistics": statistics,
                "predictions": ensemble_report,
                "backtests": backtest_report,
            }
        )
        st.success("Dados exportados para SQLite em database/")

    tabs = st.tabs(["Resumo", "Ranking", "Previsões", "Modelos", "Gerador", "Monte Carlo", "Backtest", "Genético", "Correlações"])

    with tabs[0]:
        st.subheader("Resumo histórico")
        st.dataframe(draws.tail(20))
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Concursos", len(draws))
        col2.metric("Último concurso", int(draws.iloc[-1]["Concurso"]))
        col3.metric("Números únicos sorteados", int(draws[draws.columns[2:22]].nunique().sum()))
        col4.metric("Média de frequência", round(float(statistics["freq_abs"].mean()), 2))
        eval_path = Path("exports") / "last_draw_evaluation.json"
        if eval_path.exists():
            import json

            with open(eval_path, encoding="utf-8") as handle:
                last_eval = json.load(handle)
            if int(last_eval.get("concurso", 0)) == int(draws.iloc[-1]["Concurso"]):
                top20 = last_eval.get("top_slices", {}).get("20", {})
                st.metric(
                    "Acertos top-20 no último sorteio",
                    f"{top20.get('hits', 0)}/20",
                    help="Avaliação honesta (features sem vazar o resultado do último concurso).",
                )
        render_chart(statistics.sort_values("freq_abs", ascending=False).head(50), "number", "freq_abs", "Top 50 números por frequência")

    with tabs[1]:
        st.subheader("Ranking de Números e Score Total")
        if features is None:
            # start background prepare on-demand when the user opens this tab
            if not st.session_state.get("features_background_running"):
                st.session_state["features_background_running"] = True
                threading.Thread(target=_background_prepare_features, args=(draws,), daemon=True).start()
                st.info("Preparação de features iniciada em background. Use 'Atualizar agora' para ver quando terminar.")
            else:
                st.info("Preparação de features rodando em background...")

            refresh_ranking = st.button("Atualizar agora", key="refresh_ranking")
            if refresh_ranking:
                logger.info("Botão 'Atualizar agora' (ranking) pressionado pelo usuário")
                signature = _get_cache_signature(draws)
                # try to load features and dependencies from disk cache first
                f = _load_from_disk_cache(FEATURES_CACHE_FILE, signature)
                gm = _load_from_disk_cache(GRAPH_CACHE_FILE, signature)
                mr = _load_from_disk_cache(MARKOV_CACHE_FILE, signature)
                hm = _load_from_disk_cache(HMM_CACHE_FILE, signature)
                if f is not None:
                    st.session_state.features = f
                    if gm is not None:
                        st.session_state.graph_metrics = gm
                    if mr is not None:
                        st.session_state.markov_report = mr
                    if hm is not None:
                        st.session_state.hmm_report = hm
                    logger.info("Loaded features from disk cache via 'Atualizar agora'")
                    st.rerun() if hasattr(st, "rerun") else st.experimental_rerun()
                else:
                    features = ensure_features(draws)
                    if features is not None:
                        st.rerun() if hasattr(st, "rerun") else st.experimental_rerun()
        else:
            render_chart(features.sort_values("score_total", ascending=False).head(50), "number", "score_total", "Top 50 score_total")
            st.markdown("**Top 50 por score_total**")
            render_table("Ranking de maior potencial", features.sort_values("score_total", ascending=False), max_rows=50)

    with tabs[2]:
        st.subheader("Previsões de números")
        st.write("Os números abaixo são ordenados pela probabilidade final da pilha de modelos.")
        if ensemble_report is None:
            st.info("Clique em 'Treinar modelos' no painel lateral para calcular as previsões.")
        else:
            display_cols = [c for c in ("number", "prob_final", "prob_ensemble", "prob_weighted", "prob_mean") if c in ensemble_report.columns]
            render_table("Top 50 para o próximo concurso", ensemble_report[display_cols], max_rows=50)

    with tabs[3]:
        st.subheader("Avaliação de Modelos")
        if ensemble_report is None:
            st.info("Treine os modelos para ver a comparação entre eles.")
        else:
            model_columns = [
                col
                for col in ensemble_report.columns
                if col.startswith("prob_") and col not in ("prob_mean", "prob_final")
            ]
            if model_columns:
                comparison = ensemble_report.set_index("number")[model_columns].head(10)
                st.markdown("**Comparação entre modelos (top 10)**")
                render_multi_series_chart(comparison.reset_index(), "number", model_columns, "Probabilidades por modelo")
                st.write("Probabilidades dos modelos para os 10 números com maior probabilidade média.")
                st.dataframe(comparison)

            if show_model_metrics and ml_engine is not None and hasattr(ml_engine, "model_metrics_") and ml_engine.model_metrics_ is not None:
                st.markdown("**Métricas de validação por modelo**")
                st.dataframe(ml_engine.model_metrics_)
            if show_model_metrics and ml_engine is not None and hasattr(ml_engine, "feature_importances_") and ml_engine.feature_importances_ is not None:
                st.markdown("**Principais features do modelo**")
                st.dataframe(ml_engine.feature_importances_.head(50).rename("importance").to_frame())

    with tabs[4]:
        st.subheader("Gerador de Jogos")

        # Use new visual game selector panel
        game_config = render_game_selector_panel()
        selected_numbers = game_config.get("numbers", [])
        generation_features = features if features is not None else build_fast_generation_features(draws, statistics)
        generation_ensemble = ensemble_report if features is not None else None
        using_fast_generator = features is None

        st.divider()

        # Generation options
        col1, col2 = st.columns(2)
        with col1:
            generate_method = st.radio(
                "Método de cada jogo",
                ["post_result", "top_score", "balanced", "coverage", "random"],
                format_func=lambda v: {
                    "post_result": "Pós-resultado",
                    "top_score": "Pontuação Máxima",
                    "balanced": "Balanceado",
                    "coverage": "Cobertura/Atraso",
                    "random": "Aleatório",
                }[v],
                horizontal=False,
            )
            if "use_closure_games" not in st.session_state:
                st.session_state.use_closure_games = True
            use_closure = st.checkbox("Usar fechamento", key="use_closure_games")
            closure_universe = st.slider("Universo do fechamento", min_value=50, max_value=100, value=90, step=1)
            closure_games = st.slider("Jogos no fechamento", min_value=2, max_value=30, value=max(4, min(10, num_games)), step=1)
            use_power_portfolio = st.checkbox("Otimizar carteira de jogos", value=True)
            power_floor = st.slider("Potência mínima", min_value=0.60, max_value=0.95, value=0.78, step=0.01)
            st.caption("No fechamento, o sistema escolhe um universo maior e distribui esses números em jogos de 50 para aumentar a cobertura.")

        with col2:
            st.markdown("### Geração")
            if using_fast_generator:
                if not st.session_state.get("features_background_running"):
                    st.session_state["features_background_running"] = True
                    threading.Thread(target=_background_prepare_features, args=(draws,), daemon=True).start()
                st.info("Modo rápido ativo: você já pode gerar jogos. As métricas avançadas seguem carregando em segundo plano.")
            else:
                st.success("Modo avançado ativo.")

            if st.button(f"Gerador automático: {num_games} jogos", key="button_generate_auto", use_container_width=True):
                fixed_numbers = game_config.get("fixed", [])
                base_selection = fixed_numbers if len(fixed_numbers) > 0 else selected_numbers
                st.session_state.generation_variation = st.session_state.get("generation_variation", 0) + 1
                variation = st.session_state.generation_variation
                generated_games = generate_power_portfolio(
                    generation_features,
                    base_selection,
                    generate_method,
                    n_games=num_games,
                    universe_size=closure_universe,
                    game_size=50,
                    variation=variation,
                    ensemble_report=generation_ensemble,
                    min_power=power_floor,
                )
                append_generated_games(generated_games)
                covered = len(set().union(*[set(game) for game in generated_games])) if generated_games else 0
                st.success(f"Gerados {len(generated_games)} jogos diferentes cobrindo {covered} números")
                if len(fixed_numbers) == 0:
                    st.session_state.ui_selected_numbers = set()

            if st.button("Autocompletar 1 jogo", key="button_autocomplete_game", use_container_width=True):
                fixed_numbers = game_config.get("fixed", [])
                base_selection = fixed_numbers if len(fixed_numbers) > 0 else selected_numbers
                st.session_state.generation_variation = st.session_state.get("generation_variation", 0) + 1
                variation = st.session_state.generation_variation
                generated_game = generate_game(
                    generation_features,
                    base_selection,
                    generate_method,
                    game_size=50,
                    variation=variation,
                    ensemble_report=generation_ensemble,
                )
                append_generated_games([generated_game])
                st.success(f"Jogo completo com {len(generated_game)} números")
                if len(fixed_numbers) == 0:
                    st.session_state.ui_selected_numbers = set()
            
            if features is None:
                if not st.session_state.get("features_background_running"):
                    st.session_state["features_background_running"] = True
                    threading.Thread(target=_background_prepare_features, args=(draws,), daemon=True).start()
                    st.info("Preparação de features iniciada em background.")
                else:
                    st.info("Preparação de features rodando...")

                refresh_generator_tab = st.button("Atualizar agora", key="refresh_generator_tab")
                if refresh_generator_tab:
                    logger.info("Botão 'Atualizar agora' (gerador) pressionado pelo usuário")
                    signature = _get_cache_signature(draws)
                    f = _load_from_disk_cache(FEATURES_CACHE_FILE, signature)
                    gm = _load_from_disk_cache(GRAPH_CACHE_FILE, signature)
                    mr = _load_from_disk_cache(MARKOV_CACHE_FILE, signature)
                    hm = _load_from_disk_cache(HMM_CACHE_FILE, signature)
                    if f is not None:
                        st.session_state.features = f
                        if gm is not None:
                            st.session_state.graph_metrics = gm
                        if mr is not None:
                            st.session_state.markov_report = mr
                        if hm is not None:
                            st.session_state.hmm_report = hm
                        logger.info("Loaded features from disk cache via 'Atualizar agora' (gerador)")
                        st.rerun() if hasattr(st, "rerun") else st.experimental_rerun()
                    else:
                        features = ensure_features(draws)
                        if features is not None:
                            st.rerun() if hasattr(st, "rerun") else st.experimental_rerun()
            else:
                if st.button("Gerar 1 Jogo", key="button_generate_game", use_container_width=True):
                    # Get fixed numbers from game_config
                    fixed_numbers = game_config.get("fixed", [])
                    base_selection = fixed_numbers if len(fixed_numbers) > 0 else selected_numbers
                    st.session_state.generation_variation = st.session_state.get("generation_variation", 0) + 1
                    variation = st.session_state.generation_variation
                    
                    if use_power_portfolio:
                        generated_games = generate_power_portfolio(
                            features,
                            base_selection,
                            generate_method,
                            n_games=num_games,
                            universe_size=closure_universe,
                            game_size=50,
                            variation=variation,
                            ensemble_report=ensemble_report,
                            min_power=power_floor,
                        )
                        append_generated_games(generated_games)
                        covered = len(set().union(*[set(game) for game in generated_games])) if generated_games else 0
                        st.success(f"✓ Carteira otimizada: {len(generated_games)} jogos de 50 cobrindo {covered} números")
                    elif use_closure:
                        generated_games = generate_covering_games(
                            features,
                            base_selection,
                            generate_method,
                            n_games=closure_games,
                            universe_size=closure_universe,
                            game_size=50,
                            variation=variation,
                        )
                        append_generated_games(generated_games)
                        covered = len(set().union(*[set(game) for game in generated_games])) if generated_games else 0
                        st.success(f"✓ Fechamento gerado: {len(generated_games)} jogos de 50 cobrindo {covered} números")
                    else:
                        generated_game = generate_game(
                            features,
                            base_selection,
                            generate_method,
                            game_size=50,
                            variation=variation,
                            ensemble_report=ensemble_report,
                        )
                        append_generated_games([generated_game])
                        st.success(f"✓ Jogo gerado com {len(generated_game)} números")
                    
                    # Clear selection unless numbers are fixed
                    if len(fixed_numbers) == 0:
                        st.session_state.ui_selected_numbers = set()

                if st.button(f"Gerar {num_games} Jogos", key="button_generate_games", use_container_width=True):
                    # Get fixed numbers from game_config
                    fixed_numbers = game_config.get("fixed", [])
                    base_selection = fixed_numbers if len(fixed_numbers) > 0 else selected_numbers
                    st.session_state.generation_variation = st.session_state.get("generation_variation", 0) + 1
                    variation = st.session_state.generation_variation
                    
                    if use_closure:
                        generated_games = generate_covering_games(
                            features,
                            base_selection,
                            generate_method,
                            n_games=closure_games,
                            universe_size=closure_universe,
                            game_size=50,
                            variation=variation,
                        )
                        covered = len(set().union(*[set(game) for game in generated_games])) if generated_games else 0
                        st.success(f"✓ Fechamento gerado: {len(generated_games)} jogos de 50 cobrindo {covered} números")
                    else:
                        generated_games = generate_games(
                            features,
                            base_selection,
                            generate_method,
                            n_games=num_games,
                            game_size=50,
                            variation=variation,
                            ensemble_report=ensemble_report,
                        )
                    
                    append_generated_games(generated_games)
                    if not use_closure and not use_power_portfolio:
                        st.success(f"✓ {len(generated_games)} jogos gerados com sucesso")
                    
                    # Clear selection unless numbers are fixed
                    if len(fixed_numbers) == 0:
                        st.session_state.ui_selected_numbers = set()

                if st.button("Completar com Pontuação", key="button_complete_top_score", use_container_width=True):
                    # Get fixed numbers from game_config
                    fixed_numbers = game_config.get("fixed", [])
                    base_selection = fixed_numbers if len(fixed_numbers) > 0 else selected_numbers
                    st.session_state.generation_variation = st.session_state.get("generation_variation", 0) + 1
                    variation = st.session_state.generation_variation
                    
                    if use_closure:
                        generated_games = generate_covering_games(
                            features,
                            base_selection,
                            "top_score",
                            n_games=closure_games,
                            universe_size=closure_universe,
                            game_size=50,
                            variation=variation,
                        )
                        append_generated_games(generated_games)
                        covered = len(set().union(*[set(game) for game in generated_games])) if generated_games else 0
                        st.success(f"✓ Fechamento por pontuação gerado cobrindo {covered} números")
                    else:
                        generated_game = generate_game(
                            features,
                            base_selection,
                            "top_score",
                            game_size=50,
                            variation=variation,
                            ensemble_report=ensemble_report,
                        )
                        append_generated_games([generated_game])
                        st.success("✓ Jogo completo gerado")
                    
                    # Clear selection unless numbers are fixed
                    if len(fixed_numbers) == 0:
                        st.session_state.ui_selected_numbers = set()

        st.divider()

        # Display generated games
        generated_games = st.session_state.get("generated_games")
        if generated_games is not None:
            portfolio_stats = estimate_portfolio_prize_chances(generated_games)
            st.markdown("### Matemática da carteira")
            stat_cols = st.columns(6)
            stat_cols[0].metric("Jogos", len(generated_games))
            stat_cols[1].metric("Cobertura", portfolio_stats["covered_numbers"])
            stat_cols[2].metric("Sobreposição média", round(portfolio_stats["avg_overlap"], 1))
            stat_cols[3].metric("Média melhor acerto", round(portfolio_stats["avg_best_hits"], 2))
            stat_cols[4].metric("Chance 17+", f"{portfolio_stats['tiers']['17'] * 100:.3f}%")
            stat_cols[5].metric("Chance 18+", f"{portfolio_stats['tiers']['18'] * 100:.4f}%")
            odds_rows = []
            for tier in ("17", "18", "19", "20"):
                portfolio_prob = portfolio_stats["tiers"][tier]
                single_prob = portfolio_stats["single_ticket_tiers"][tier]
                odds_rows.append(
                    {
                        "faixa": f"{tier}+",
                        "carteira_pct": portfolio_prob * 100,
                        "jogo_unico_pct": single_prob * 100,
                        "multiplicador": portfolio_prob / single_prob if single_prob > 0 else 0,
                    }
                )
            st.dataframe(pd.DataFrame(odds_rows), use_container_width=True, hide_index=True)
            render_generated_games_gallery(generated_games)

    with tabs[5]:
        st.subheader("Simulação Monte Carlo")
        if run_montecarlo:
            with st.spinner("Executando Monte Carlo..."):
                st.session_state.montecarlo_results = compute_montecarlo(draws, n_simulations=montecarlo_runs)
        if st.session_state.montecarlo_results is None:
            st.info("Clique em 'Executar Monte Carlo' no painel lateral para iniciar a simulação.")
        else:
            mc_results = st.session_state.montecarlo_results
            probabilities = pd.DataFrame(mc_results["probabilities"].items(), columns=["number", "probability"]).sort_values("probability", ascending=False)
            st.metric("Simulações", mc_results["simulations"])
            render_chart(probabilities.head(50), "number", "probability", "Top 50 probabilidades em Monte Carlo")
            render_table("Top 50 números por probabilidade de Monte Carlo", probabilities.head(50), max_rows=50)

    with tabs[6]:
        st.subheader("Resultados do Backtest")
        st.write("Validação walk-forward com previsão por frequência histórica.")
        if run_backtest:
            if features is None:
                features = ensure_features(draws)
            with st.spinner("Executando backtest..."):
                backtest_report = compute_backtest(draws, features)
            st.session_state.backtest_report = backtest_report
        if backtest_report is None:
            st.info("Clique em 'Executar Backtest' no painel lateral para iniciar o backtest.")
        elif backtest_report.empty:
            st.write("Backtest não retornou resultados suficientes.")
        else:
            summary = backtest_report[["hits", "precision", "recall", "f1", "roi_theoretical"]].describe().transpose()
            st.dataframe(summary)
            render_chart(backtest_report, "split", "precision", "Precisão por split")
            render_chart(backtest_report, "split", "roi_theoretical", "ROI teórico por split")

    with tabs[7]:
        st.subheader("Otimização Genética")
        st.markdown(f"**Critério selecionado:** {fitness_criteria}")
        st.markdown(f"**População:** {population_size} | **Gerações:** {generations}")
        if run_genetic:
            if features is None:
                # trigger background prepare if not running
                if not st.session_state.get("features_background_running"):
                    st.session_state["features_background_running"] = True
                    threading.Thread(target=_background_prepare_features, args=(draws,), daemon=True).start()
                    st.info("Preparação de features iniciada em background. Clique em 'Executar Genetic Search' novamente quando pronto ou use 'Atualizar agora'.")
                    refresh_genetic_start = st.button("Atualizar agora", key="refresh_genetic_start")
                    if refresh_genetic_start:
                        logger.info("Botão 'Atualizar agora' (genetic start) pressionado pelo usuário")
                        signature = _get_cache_signature(draws)
                        f = _load_from_disk_cache(FEATURES_CACHE_FILE, signature)
                        gm = _load_from_disk_cache(GRAPH_CACHE_FILE, signature)
                        mr = _load_from_disk_cache(MARKOV_CACHE_FILE, signature)
                        hm = _load_from_disk_cache(HMM_CACHE_FILE, signature)
                        if f is not None:
                            st.session_state.features = f
                            if gm is not None:
                                st.session_state.graph_metrics = gm
                            if mr is not None:
                                st.session_state.markov_report = mr
                            if hm is not None:
                                st.session_state.hmm_report = hm
                            logger.info("Loaded features from disk cache via 'Atualizar agora' (genetic start)")
                            st.rerun() if hasattr(st, "rerun") else st.experimental_rerun()
                        else:
                            features = ensure_features(draws)
                            if features is not None:
                                st.rerun() if hasattr(st, "rerun") else st.experimental_rerun()
                else:
                    st.info("Preparação de features rodando em background...")
                    refresh_genetic_running = st.button("Atualizar agora", key="refresh_genetic_running")
                    if refresh_genetic_running:
                        logger.info("Botão 'Atualizar agora' (genetic running) pressionado pelo usuário")
                        signature = _get_cache_signature(draws)
                        f = _load_from_disk_cache(FEATURES_CACHE_FILE, signature)
                        gm = _load_from_disk_cache(GRAPH_CACHE_FILE, signature)
                        mr = _load_from_disk_cache(MARKOV_CACHE_FILE, signature)
                        hm = _load_from_disk_cache(HMM_CACHE_FILE, signature)
                        if f is not None:
                            st.session_state.features = f
                            if gm is not None:
                                st.session_state.graph_metrics = gm
                            if mr is not None:
                                st.session_state.markov_report = mr
                            if hm is not None:
                                st.session_state.hmm_report = hm
                            logger.info("Loaded features from disk cache via 'Atualizar agora' (genetic running)")
                            st.rerun() if hasattr(st, "rerun") else st.experimental_rerun()
                        else:
                            features = ensure_features(draws)
                            if features is not None:
                                st.rerun() if hasattr(st, "rerun") else st.experimental_rerun()

            if features is not None:
                with st.spinner("Executando otimização genética..."):
                    st.session_state.genetic_report = GeneticGameOptimizer(draws, features).run_evolutionary_search(
                        population_size=population_size,
                        generations=generations,
                        fitness_criteria=fitness_criteria,
                    )
                    st.session_state.genetic_config = {
                        "criteria": fitness_criteria,
                        "population_size": population_size,
                        "generations": generations,
                    }
            else:
                st.warning("Aguarde a preparação de features terminar antes de executar a busca genética.")

        if "genetic_report" in st.session_state and st.session_state.genetic_report is not None:
            config = st.session_state.get("genetic_config", {})
            st.write(f"Último resultado gerado com critério: {config.get('criteria', 'score_total')}")
            render_table("Jogos gerados", st.session_state.genetic_report.sort_values("score_total", ascending=False), max_rows=10)
        else:
            st.info("Clique no botão na barra lateral para gerar combinações genéticas usando o critério selecionado.")

    with tabs[8]:
        st.subheader("Correlações e padrões")
        st.write("Matriz de lift e pares de números mais correlacionados.")
        if correlation is None:
            correlation = ensure_correlation(draws)
        render_table("Top 50 pares de coocorrência", pd.DataFrame(strongest_pairs(correlation["cooccurrence"], top_n=50), columns=["number_a", "number_b", "count"]), max_rows=50)
        st.markdown("**Amostra da matriz de lift**")
        st.dataframe(correlation["lift"].iloc[:20, :20])

    st.success("Dashboard carregado com sucesso.")


if __name__ == "__main__":
    run_dashboard()
