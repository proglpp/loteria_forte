import logging
import math
import pickle
from datetime import date
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
from elite_optimizer import generate_elite_portfolio
from ai_learning_engine import (
    DEFAULT_EVAL_DRAW,
    apply_learning_to_features,
    build_learning_profile,
    evaluate_games_against_draw,
    load_learning_records,
    normalize_numbers,
    record_learning_feedback,
)
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


def _draw_number_sets(draws: pd.DataFrame) -> list[set[str]]:
    return [
        {str(row[col]).zfill(2) for col in DRAW_COLUMNS}
        for _, row in draws.reset_index(drop=True).iterrows()
    ]


def build_movement_report(draws: pd.DataFrame, rows: int = 12, strength_window: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    numbers = [f"{number:02d}" for number in range(100)]
    rows = max(1, min(int(rows), len(draws)))
    strength_window = max(3, min(int(strength_window), len(draws)))
    draw_sets = _draw_number_sets(draws)
    recent_sets = draw_sets[-strength_window:]
    previous_sets = draw_sets[-(strength_window * 2) : -strength_window] if len(draw_sets) > strength_window else []

    recent_counts = pd.Series(0, index=numbers, dtype=float)
    previous_counts = pd.Series(0, index=numbers, dtype=float)
    global_counts = pd.Series(0, index=numbers, dtype=float)
    for draw_set in recent_sets:
        for number in draw_set:
            recent_counts[number] += 1
    for draw_set in previous_sets:
        for number in draw_set:
            previous_counts[number] += 1
    for draw_set in draw_sets:
        for number in draw_set:
            global_counts[number] += 1

    delay = {}
    for number in numbers:
        last_seen = next((idx for idx, draw_set in enumerate(reversed(draw_sets), start=0) if number in draw_set), len(draw_sets))
        delay[number] = float(last_seen)
    delay_series = pd.Series(delay)

    expected_recent = strength_window * 20 / 100
    freq_norm = _minmax(recent_counts)
    trend_norm = _minmax(recent_counts - previous_counts.reindex(numbers, fill_value=0))
    delay_norm = _minmax(delay_series)
    global_norm = _minmax(global_counts)
    balance_pull = 1.0 - ((recent_counts - expected_recent).abs() / max(expected_recent, 1.0)).clip(upper=1.0)
    score = (
        0.34 * freq_norm
        + 0.25 * delay_norm
        + 0.18 * trend_norm
        + 0.13 * balance_pull
        + 0.10 * global_norm
    )

    q25, q55, q78 = score.quantile([0.25, 0.55, 0.78]).tolist()
    summary_rows = []
    for number in numbers:
        value = float(score[number])
        if value >= q78:
            status = "FORTE"
        elif value >= q55:
            status = "BOM"
        elif value <= q25:
            status = "FRACO"
        else:
            status = "OBS"
        summary_rows.append(
            {
                "dezena": number,
                "forca": round(value * 100, 1),
                "status": status,
                "freq_janela": int(recent_counts[number]),
                "tendencia": int(recent_counts[number] - previous_counts.get(number, 0)),
                "atraso": int(delay_series[number]),
                "freq_total": int(global_counts[number]),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values(["forca", "freq_janela"], ascending=False).reset_index(drop=True)

    movement_rows = []
    for _, row in draws.tail(rows).iterrows():
        present = {str(row[col]).zfill(2) for col in DRAW_COLUMNS}
        movement_row = {"Concurso": str(row["Concurso"])}
        for number in numbers:
            movement_row[number] = number if number in present else ""
        movement_rows.append(movement_row)

    cycle_window = min(5, len(draw_sets))
    cycle_sets = draw_sets[-cycle_window:]
    cycle_seen = set().union(*cycle_sets) if cycle_sets else set()
    missing_cycle = sorted(set(numbers) - cycle_seen)
    movement_rows.append({"Concurso": "FREQ"} | {number: int(recent_counts[number]) for number in numbers})
    movement_rows.append({"Concurso": "ATRASO"} | {number: int(delay_series[number]) for number in numbers})
    movement_rows.append({"Concurso": "FORCA"} | {number: int(round(score[number] * 100)) for number in numbers})
    movement_rows.append({"Concurso": "FALTA CICLO"} | {number: ("*" if number in missing_cycle else "") for number in numbers})
    movement = pd.DataFrame(movement_rows).astype(str)
    movement.attrs["score_map"] = score.to_dict()
    movement.attrs["status_map"] = summary.set_index("dezena")["status"].to_dict()
    return movement, summary


def style_movement_report(df: pd.DataFrame):
    score_map = df.attrs.get("score_map", {})
    status_map = df.attrs.get("status_map", {})

    def style_cell(value, column, row_label: str):
        if column == "Concurso":
            return "background-color: #5b159f; color: white; font-weight: 700; text-align: center;"
        if value == "":
            status = status_map.get(column, "OBS")
            background = {
                "FORTE": "#e8f7ee",
                "BOM": "#edf4ff",
                "OBS": "#fff8dc",
                "FRACO": "#fdecec",
            }.get(status, "#ffffff")
            return f"background-color: {background}; color: transparent;"
        if value == "*":
            return "background-color: #fff0b3; color: #805800; font-weight: 800; text-align: center;"
        try:
            numeric = float(value)
            if row_label in {"FREQ", "ATRASO", "FORCA"}:
                intensity = min(max(numeric / 100, 0.15), 1.0) if numeric > 20 else min(max(numeric / 10, 0.1), 1.0)
                return f"background-color: rgba(91, 21, 159, {0.18 + 0.42 * intensity}); color: #111827; font-weight: 700; text-align: center;"
        except Exception:
            pass
        status = status_map.get(column, "OBS")
        if status == "FORTE":
            return "background-color: #009966; color: white; font-weight: 800; text-align: center;"
        if status == "BOM":
            return "background-color: #55b7ff; color: #07121f; font-weight: 800; text-align: center;"
        if status == "FRACO":
            return "background-color: #e85d75; color: white; font-weight: 800; text-align: center;"
        return "background-color: #d6b4ec; color: #2b0f3f; font-weight: 800; text-align: center;"

    def apply_styles(data: pd.DataFrame):
        styles = pd.DataFrame("", index=data.index, columns=data.columns)
        for row_index in data.index:
            row_label = str(data.loc[row_index, "Concurso"])
            for column in data.columns:
                value = data.loc[row_index, column]
                if row_label in {"FREQ", "ATRASO", "FORCA"} and column != "Concurso":
                    styles.loc[row_index, column] = "background-color: #efe3ff; color: #1f1235; font-weight: 800; text-align: center;"
                else:
                    styles.loc[row_index, column] = style_cell(value, column, row_label)
        return styles

    return (
        df.style.apply(apply_styles, axis=None)
        .set_properties(**{"font-size": "11px", "min-width": "34px", "height": "24px"})
        .hide(axis="index")
    )


def render_movement_tab(draws: pd.DataFrame) -> None:
    st.subheader("Movimentacao das dezenas")
    col1, col2 = st.columns(2)
    with col1:
        movement_rows = st.slider("Concursos na tabela", min_value=5, max_value=30, value=12, step=1)
    with col2:
        strength_window = st.slider("Janela de forca", min_value=5, max_value=50, value=10, step=1)

    movement, summary = build_movement_report(draws, rows=movement_rows, strength_window=strength_window)
    status_counts = summary["status"].value_counts().reindex(["FORTE", "BOM", "OBS", "FRACO"], fill_value=0)
    metric_cols = st.columns(4)
    metric_cols[0].metric("Fortes", int(status_counts["FORTE"]))
    metric_cols[1].metric("Bons", int(status_counts["BOM"]))
    metric_cols[2].metric("Observacao", int(status_counts["OBS"]))
    metric_cols[3].metric("Fracos", int(status_counts["FRACO"]))

    st.dataframe(style_movement_report(movement), width="stretch", height=560)

    legend = pd.DataFrame(
        [
            {"classe": "FORTE", "leitura": "alta forca combinando frequencia recente, atraso, tendencia e historico"},
            {"classe": "BOM", "leitura": "boa sustentacao, mas abaixo do grupo mais forte"},
            {"classe": "OBS", "leitura": "zona intermediaria para compor cobertura"},
            {"classe": "FRACO", "leitura": "baixa forca no momento pela combinacao matematica atual"},
            {"classe": "FALTA CICLO", "leitura": "dezena ausente nos ultimos concursos usados como ciclo curto"},
        ]
    )
    st.dataframe(legend, width="stretch", hide_index=True)
    st.markdown("### Ranking da movimentacao")
    st.dataframe(summary.head(100), width="stretch", hide_index=True)


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
    with st.spinner("Preparando mÃ©tricas avanÃ§adas e matriz de features..."):
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


def _minmax(series, neutral: float = 0.0):
    values = pd.to_numeric(series, errors="coerce").fillna(neutral)
    spread = values.max() - values.min()
    if abs(float(spread)) < 1e-12:
        return pd.Series(np.full(len(values), neutral), index=values.index)
    return (values - values.min()) / (spread + 1e-12)


def _apply_learning_blend(df: pd.DataFrame, base_column: str, base_weight: float = 0.64) -> pd.Series:
    base = pd.to_numeric(df[base_column], errors="coerce").fillna(0.0)
    if "ai_learning_score" not in df.columns:
        return base

    learned = pd.to_numeric(df["ai_learning_score"], errors="coerce").fillna(0.0)
    if learned.abs().sum() <= 1e-12:
        return base

    confidence = pd.to_numeric(df.get("ai_learning_confidence", 0.0), errors="coerce").fillna(0.0).clip(0.0, 1.0)
    learned_norm = _minmax(learned, neutral=0.5)
    signed_boost = np.tanh(learned * 3.0)
    confidence_weight = float(confidence.mean()) if len(confidence) else 0.0
    learning_weight = min(0.42, max(0.18, 1.0 - base_weight) + (0.18 * confidence_weight))
    blended = ((1.0 - learning_weight) * base) + (learning_weight * learned_norm)
    return blended + (0.06 * confidence * signed_boost)


def _learning_anchor_numbers(features, max_count: int = 20) -> list[str]:
    if features is None or "ai_learning_score" not in features.columns:
        return []
    df = features.copy()
    df["number"] = df["number"].astype(str).str.zfill(2)
    df["ai_learning_score"] = pd.to_numeric(df["ai_learning_score"], errors="coerce").fillna(0.0)
    df["ai_learning_confidence"] = pd.to_numeric(df.get("ai_learning_confidence", 0.0), errors="coerce").fillna(0.0)
    df["sorteado_no_resultado_mais_recente"] = pd.to_numeric(
        df.get("sorteado_no_resultado_mais_recente", 0.0),
        errors="coerce",
    ).fillna(0.0)
    anchors = df[(df["ai_learning_score"] > 0.015) & (df["ai_learning_confidence"] >= 0.55)]
    if anchors.empty:
        return []
    anchors = anchors.sort_values(["ai_learning_score", "ai_learning_confidence"], ascending=False)
    recent_limit = max(3, int(max_count * 0.45))
    recent = anchors[anchors["sorteado_no_resultado_mais_recente"] > 0].head(recent_limit)
    non_recent = anchors[anchors["sorteado_no_resultado_mais_recente"] <= 0]
    balanced = pd.concat([recent, non_recent], ignore_index=True)
    balanced = balanced.drop_duplicates("number").sort_values(
        ["ai_learning_score", "ai_learning_confidence"],
        ascending=False,
    )
    return balanced["number"].head(max_count).tolist()


def _rank_numbers_for_generation(features, method: str, variation: int = 0, ensemble_report=None):
    df = features.copy()
    df["number"] = df["number"].astype(str).str.zfill(2)
    df["number_int"] = df["number"].astype(int)
    rng = np.random.default_rng(42 + int(variation))

    score = pd.to_numeric(df.get("score_total", 0), errors="coerce").fillna(0.0)
    df["_score_norm"] = _minmax(score)
    df["_score_norm"] = _apply_learning_blend(df, "_score_norm", base_weight=0.66)

    if ensemble_report is not None and "prob_final" in ensemble_report.columns:
        probs = ensemble_report.copy()
        probs["number"] = probs["number"].astype(str).str.zfill(2)
        df = df.merge(probs[["number", "prob_final"]], on="number", how="left")
        df["_generation_score"] = pd.to_numeric(df["prob_final"], errors="coerce").fillna(df["_score_norm"])
        df["_generation_score"] = _minmax(df["_generation_score"])
        df["_generation_score"] = _apply_learning_blend(df, "_generation_score", base_weight=0.68)
        return df.sort_values("_generation_score", ascending=False)["number"].tolist()

    if method == "random":
        return df.sample(frac=1, random_state=42 + int(variation))["number"].tolist()

    if method == "post_result":
        last_draw = pd.to_numeric(df.get("last_draw_hit", 0), errors="coerce").fillna(0.0)
        recent_5 = pd.to_numeric(df.get("last_5_draws_hits", 0), errors="coerce").fillna(0.0)
        recent_5_norm = _minmax(recent_5)
        delay = pd.to_numeric(df.get("delay_last", 0), errors="coerce").fillna(0.0)
        delay_norm = _minmax(delay)
        df["_generation_score"] = (df["_score_norm"] * 0.62) + (last_draw * 0.08) + (recent_5_norm * 0.15) + ((1 - delay_norm) * 0.15)
    elif method == "balanced":
        freq = pd.to_numeric(df.get("freq_global", 0), errors="coerce").fillna(0.0)
        freq_norm = _minmax(freq)
        parity_balance = np.where(df["number_int"] % 2 == 0, 0.5, 0.55)
        df["_generation_score"] = (df["_score_norm"] * 0.72) + ((1 - abs(freq_norm - 0.5)) * 0.18) + (parity_balance * 0.10)
    elif method == "coverage":
        delay = pd.to_numeric(df.get("delay_last", 0), errors="coerce").fillna(0.0)
        delay_norm = _minmax(delay)
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
    for anchor in _learning_anchor_numbers(features, max_count=min(10, game_size)):
        if len(selected) >= game_size:
            break
        if anchor not in selected:
            selected.append(anchor)

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
    anchors = _learning_anchor_numbers(features, max_count=min(12, game_size))
    n_games = max(1, int(n_games))
    universe_size = max(game_size, min(100, int(universe_size)))
    ranked = _rank_numbers_for_generation(features, method, variation, ensemble_report=ensemble_report)
    score_map = _generation_score_map(features, method, variation, ensemble_report=ensemble_report)
    pool = list(dict.fromkeys(fixed + anchors + [n for n in ranked if n not in fixed]))[:universe_size]
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
        coverage_seed_count = min(max(8, len(flexible_pool) // max(n_games, 1)), max(needed // 2, 1))
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
                underuse_bonus = 0.36 if usage.get(number, 0) == min_usage else 0.0
                rotation_bonus = ((int(number) + game_index + variation) % 17) / 500.0
                value = score_map.get(number, 0.0) + underuse_bonus - (0.72 * usage_penalty) - (0.32 * overlap_penalty) + rotation_bonus
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
    anchors = _learning_anchor_numbers(features, max_count=min(10, game_size))
    universe_size = max(game_size, min(100, int(universe_size)))
    ranked = _rank_numbers_for_generation(features, method, variation)
    pool = list(dict.fromkeys(fixed + anchors + [n for n in ranked if n not in fixed]))[:universe_size]
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
    montecarlo_runs = st.sidebar.slider("SimulaÃ§Ãµes Monte Carlo", min_value=1000, max_value=20000, value=5000, step=1000)
    fitness_criteria = st.sidebar.selectbox(
        "CritÃ©rio de fitness genÃ©tico",
        ["score_total", "diversity", "coverage", "balanced"],
        format_func=lambda v: {
            "score_total": "Score Total",
            "diversity": "Diversidade",
            "coverage": "Cobertura",
            "balanced": "Balanceado",
        }[v],
    )
    population_size = st.sidebar.slider("Tamanho da populaÃ§Ã£o genÃ©tica", min_value=10, max_value=100, value=50, step=10)
    generations = st.sidebar.slider("GeraÃ§Ãµes genÃ©ticas", min_value=5, max_value=100, value=20, step=5)
    run_features = st.sidebar.button("Preparar features avanÃ§adas")
    run_montecarlo = st.sidebar.button("Executar Monte Carlo")
    run_model = st.sidebar.button("Treinar Modelos")
    run_model_fast = st.sidebar.button("Treinar modelos rÃ¡pido")
    num_games = st.sidebar.slider("NÃºmero de jogos a gerar", min_value=1, max_value=100, value=1, step=1)
    run_backtest = st.sidebar.button("Executar Backtest")
    run_genetic = st.sidebar.button("Executar Genetic Search")
    run_correlation = st.sidebar.button("Calcular CorrelaÃ§Ãµes")
    export_csv = st.sidebar.button("Exportar CSV/JSON")
    export_sqlite = st.sidebar.button("Exportar SQLite")
    show_model_metrics = st.sidebar.checkbox("Mostrar mÃ©tricas de ML", value=True)
    st.sidebar.markdown("---")
    st.sidebar.write("Use os botÃµes e seleÃ§Ãµes acima para gerar simulaÃ§Ãµes, otimizaÃ§Ã£o genÃ©tica e exportar resultados.")

    exporter = Exporter(DATABASE_FILE)

    with st.spinner("Carregando dados e mÃ©tricas bÃ¡sicas..."):
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
    learning_records = load_learning_records()
    if features is not None:
        features = apply_learning_to_features(features, learning_records)

    if run_features and features is None:
        with st.spinner("Preparando features avancadas..."):
            features = ensure_features(draws)
        st.success("Features avancadas preparadas.")

    if run_correlation and correlation is None:
        correlation = ensure_correlation(draws)
        st.success("Correlacao calculada e armazenada em cache.")

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
        with st.spinner("Treinamento rÃ¡pido em andamento..."):
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
            with st.spinner("Treinando modelos para exportaÃ§Ã£o..."):
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
            with st.spinner("Treinando modelos para exportaÃ§Ã£o..."):
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

    tabs = st.tabs([
        "Resumo",
        "Ranking",
        "Movimentacao",
        "PrevisÃµes",
        "Modelos",
        "Gerador",
        "IA Aprendizado",
        "Monte Carlo",
        "Backtest",
        "GenÃ©tico",
        "CorrelaÃ§Ãµes",
    ])

    with tabs[0]:
        st.subheader("Resumo histÃ³rico")
        st.dataframe(draws.tail(20))
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Concursos", len(draws))
        col2.metric("Ãšltimo concurso", int(draws.iloc[-1]["Concurso"]))
        col3.metric("NÃºmeros Ãºnicos sorteados", int(draws[draws.columns[2:22]].nunique().sum()))
        col4.metric("MÃ©dia de frequÃªncia", round(float(statistics["freq_abs"].mean()), 2))
        eval_path = Path("exports") / "last_draw_evaluation.json"
        if eval_path.exists():
            import json

            with open(eval_path, encoding="utf-8") as handle:
                last_eval = json.load(handle)
            if int(last_eval.get("concurso", 0)) == int(draws.iloc[-1]["Concurso"]):
                top20 = last_eval.get("top_slices", {}).get("20", {})
                st.metric(
                    "Acertos top-20 no Ãºltimo sorteio",
                    f"{top20.get('hits', 0)}/20",
                    help="AvaliaÃ§Ã£o honesta (features sem vazar o resultado do Ãºltimo concurso).",
                )
        render_chart(statistics.sort_values("freq_abs", ascending=False).head(50), "number", "freq_abs", "Top 50 nÃºmeros por frequÃªncia")

    with tabs[1]:
        st.subheader("Ranking de NÃºmeros e Score Total")
        if features is None:
            quick_features = build_fast_generation_features(draws, statistics)
            quick_features = apply_learning_to_features(quick_features, learning_records)
            st.info("Modo rapido ativo. Use 'Preparar features avancadas' na lateral apenas quando quiser calcular metricas pesadas.")
            render_chart(quick_features.sort_values("score_total", ascending=False).head(50), "number", "score_total", "Top 50 score_total rapido")
            st.markdown("**Top 50 por score_total rapido**")
            render_table("Ranking rapido de maior potencial", quick_features.sort_values("score_total", ascending=False), max_rows=50)
        else:
            render_chart(features.sort_values("score_total", ascending=False).head(50), "number", "score_total", "Top 50 score_total")
            st.markdown("**Top 50 por score_total**")
            render_table("Ranking de maior potencial", features.sort_values("score_total", ascending=False), max_rows=50)

    with tabs[2]:
        render_movement_tab(draws)

    with tabs[3]:
        st.subheader("PrevisÃµes de nÃºmeros")
        st.write("Os nÃºmeros abaixo sÃ£o ordenados pela probabilidade final da pilha de modelos.")
        if ensemble_report is None:
            st.info("Clique em 'Treinar modelos' no painel lateral para calcular as previsoes.")
        else:
            display_cols = [c for c in ("number", "prob_final", "prob_ensemble", "prob_weighted", "prob_mean") if c in ensemble_report.columns]
            render_table("Top 50 para o prÃ³ximo concurso", ensemble_report[display_cols], max_rows=50)

    with tabs[4]:
        st.subheader("AvaliaÃ§Ã£o de Modelos")
        if ensemble_report is None:
            st.info("Treine os modelos para ver a comparaÃ§Ã£o entre eles.")
        else:
            model_columns = [
                col
                for col in ensemble_report.columns
                if col.startswith("prob_") and col not in ("prob_mean", "prob_final")
            ]
            if model_columns:
                comparison = ensemble_report.set_index("number")[model_columns].head(10)
                st.markdown("**ComparaÃ§Ã£o entre modelos (top 10)**")
                render_multi_series_chart(comparison.reset_index(), "number", model_columns, "Probabilidades por modelo")
                st.write("Probabilidades dos modelos para os 10 nÃºmeros com maior probabilidade mÃ©dia.")
                st.dataframe(comparison)

            if show_model_metrics and ml_engine is not None and hasattr(ml_engine, "model_metrics_") and ml_engine.model_metrics_ is not None:
                st.markdown("**MÃ©tricas de validaÃ§Ã£o por modelo**")
                st.dataframe(ml_engine.model_metrics_)
            if show_model_metrics and ml_engine is not None and hasattr(ml_engine, "feature_importances_") and ml_engine.feature_importances_ is not None:
                st.markdown("**Principais features do modelo**")
                st.dataframe(ml_engine.feature_importances_.head(50).rename("importance").to_frame())

    with tabs[5]:
        st.subheader("Gerador de Jogos")

        generation_features = features if features is not None else build_fast_generation_features(draws, statistics)
        generation_features = apply_learning_to_features(generation_features, learning_records)
        generation_ensemble = ensemble_report if features is not None else None
        using_fast_generator = features is None
        _, generator_movement_summary = build_movement_report(draws, rows=10, strength_window=10)
        generator_status_map = generator_movement_summary.set_index("dezena")["status"].to_dict()

        game_config = render_game_selector_panel(status_map=generator_status_map)
        selected_numbers = game_config.get("numbers", [])

        st.divider()

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Configuracao da geracao")
            generate_method = st.radio(
                "MÃ©todo de cada jogo",
                ["elite_19", "elite_15", "post_result", "top_score", "balanced", "coverage", "random"],
                format_func=lambda v: {
                    "elite_19": "Ultra 19+",
                    "elite_15": "Elite 15+",
                    "post_result": "PÃ³s-resultado",
                    "top_score": "PontuaÃ§Ã£o MÃ¡xima",
                    "balanced": "Balanceado",
                    "coverage": "Cobertura/Atraso",
                    "random": "AleatÃ³rio",
                }[v],
                horizontal=False,
            )
            if "use_closure_games" not in st.session_state:
                st.session_state.use_closure_games = True
            use_closure = st.checkbox("Usar fechamento", key="use_closure_games")
            closure_universe = st.slider("Universo do fechamento", min_value=50, max_value=100, value=90, step=1)
            closure_games = st.slider("Jogos no fechamento", min_value=2, max_value=30, value=max(4, min(10, num_games)), step=1)
            use_power_portfolio = st.checkbox("Otimizar carteira de jogos", value=True)
            power_floor = st.slider("PotÃªncia mÃ­nima", min_value=0.60, max_value=0.95, value=0.78, step=0.01)
            st.caption("O fechamento escolhe um universo maior e distribui os numeros em jogos de 50.")

        with col2:
            st.markdown("#### Acoes")
            generation_count = st.number_input(
                "Quantidade de jogos",
                min_value=1,
                max_value=100,
                value=int(num_games),
                step=1,
                key="generation_count",
            )
            if using_fast_generator:
                st.info("Modo rapido ativo: voce ja pode gerar jogos sem aguardar as metricas avancadas.")
            else:
                st.success("Modo avanÃ§ado ativo.")

            if st.button(f"Gerador automÃ¡tico: {generation_count} jogo(s)", key="button_generate_auto", width="stretch"):
                fixed_numbers = game_config.get("fixed", [])
                base_selection = fixed_numbers if len(fixed_numbers) > 0 else selected_numbers
                st.session_state.generation_variation = st.session_state.get("generation_variation", 0) + 1
                st.session_state.last_generation_method = generate_method
                variation = st.session_state.generation_variation
                if generate_method in ("elite_15", "elite_19"):
                    target_hits = 19 if generate_method == "elite_19" else 15
                    generated_games, elite_profile, _ = generate_elite_portfolio(
                        draws,
                        learning_records=learning_records,
                        n_games=generation_count,
                        ticket_size=50,
                        seed=8609 + variation,
                        target_hits=target_hits,
                    )
                    if base_selection:
                        fixed_set = set(_normalize_game_numbers(base_selection, limit=50))
                        adjusted_games = []
                        for game in generated_games:
                            remaining = [number for number in game if number not in fixed_set]
                            adjusted_games.append(sorted(list(fixed_set) + remaining[: max(0, 50 - len(fixed_set))]))
                        generated_games = adjusted_games
                    st.caption(
                        "Perfil elite: "
                        f"{elite_profile.name} | media backtest {elite_profile.avg_hits:.2f} | "
                        f"{target_hits}+ historico {elite_profile.target_rate * 100:.4f}% | "
                        f"18+ historico {elite_profile.hit_18_rate * 100:.4f}%"
                    )
                else:
                    generated_games = generate_power_portfolio(
                        generation_features,
                        base_selection,
                        generate_method,
                        n_games=generation_count,
                        universe_size=closure_universe,
                        game_size=50,
                        variation=variation,
                        ensemble_report=generation_ensemble,
                        min_power=power_floor,
                    )
                append_generated_games(generated_games)
                covered = len(set().union(*[set(game) for game in generated_games])) if generated_games else 0
                st.success(f"Gerados {len(generated_games)} jogos diferentes cobrindo {covered} nÃºmeros")
                if len(fixed_numbers) == 0:
                    st.session_state.ui_selected_numbers = set()

            if st.button("Autocompletar 1 jogo", key="button_autocomplete_game", width="stretch"):
                fixed_numbers = game_config.get("fixed", [])
                base_selection = fixed_numbers if len(fixed_numbers) > 0 else selected_numbers
                st.session_state.generation_variation = st.session_state.get("generation_variation", 0) + 1
                st.session_state.last_generation_method = generate_method
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
                st.success(f"Jogo completo com {len(generated_game)} nÃºmeros")
                if len(fixed_numbers) == 0:
                    st.session_state.ui_selected_numbers = set()
            
            if features is None:
                st.info("Modo rapido ativo na nuvem. Os botoes acima ja podem gerar jogos; metricas avancadas ficam sob demanda pela barra lateral.")
            else:
                if st.button("Gerar 1 Jogo", key="button_generate_game", width="stretch"):
                    # Get fixed numbers from game_config
                    fixed_numbers = game_config.get("fixed", [])
                    base_selection = fixed_numbers if len(fixed_numbers) > 0 else selected_numbers
                    st.session_state.generation_variation = st.session_state.get("generation_variation", 0) + 1
                    st.session_state.last_generation_method = generate_method
                    variation = st.session_state.generation_variation
                    
                    generated_game = generate_game(
                        features,
                        base_selection,
                        generate_method,
                        game_size=50,
                        variation=variation,
                        ensemble_report=ensemble_report,
                    )
                    append_generated_games([generated_game])
                    st.success(f"âœ“ Jogo gerado com {len(generated_game)} nÃºmeros")
                    
                    # Clear selection unless numbers are fixed
                    if len(fixed_numbers) == 0:
                        st.session_state.ui_selected_numbers = set()

                if st.button(f"Gerar {generation_count} Jogo(s)", key="button_generate_games", width="stretch"):
                    # Get fixed numbers from game_config
                    fixed_numbers = game_config.get("fixed", [])
                    base_selection = fixed_numbers if len(fixed_numbers) > 0 else selected_numbers
                    st.session_state.generation_variation = st.session_state.get("generation_variation", 0) + 1
                    st.session_state.last_generation_method = generate_method
                    variation = st.session_state.generation_variation
                    
                    if use_closure:
                        generated_games = generate_covering_games(
                            features,
                            base_selection,
                            generate_method,
                            n_games=generation_count,
                            universe_size=closure_universe,
                            game_size=50,
                            variation=variation,
                        )
                        covered = len(set().union(*[set(game) for game in generated_games])) if generated_games else 0
                        st.success(f"âœ“ Fechamento gerado: {len(generated_games)} jogos de 50 cobrindo {covered} nÃºmeros")
                    else:
                        generated_games = generate_games(
                            features,
                            base_selection,
                            generate_method,
                            n_games=generation_count,
                            game_size=50,
                            variation=variation,
                            ensemble_report=ensemble_report,
                        )
                    
                    append_generated_games(generated_games)
                    if not use_closure and not use_power_portfolio:
                        st.success(f"âœ“ {len(generated_games)} jogos gerados com sucesso")
                    
                    # Clear selection unless numbers are fixed
                    if len(fixed_numbers) == 0:
                        st.session_state.ui_selected_numbers = set()

                if st.button("Completar com PontuaÃ§Ã£o", key="button_complete_top_score", width="stretch"):
                    # Get fixed numbers from game_config
                    fixed_numbers = game_config.get("fixed", [])
                    base_selection = fixed_numbers if len(fixed_numbers) > 0 else selected_numbers
                    st.session_state.generation_variation = st.session_state.get("generation_variation", 0) + 1
                    st.session_state.last_generation_method = "top_score"
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
                        st.success(f"âœ“ Fechamento por pontuaÃ§Ã£o gerado cobrindo {covered} nÃºmeros")
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
                        st.success("âœ“ Jogo completo gerado")
                    
                    # Clear selection unless numbers are fixed
                    if len(fixed_numbers) == 0:
                        st.session_state.ui_selected_numbers = set()

        st.divider()

        # Display generated games
        generated_games = st.session_state.get("generated_games")
        if generated_games is not None:
            if st.button("Limpar jogos gerados", key="button_clear_generated_games"):
                st.session_state.generated_games = []
                st.rerun() if hasattr(st, "rerun") else st.experimental_rerun()

        if generated_games:
            portfolio_stats = estimate_portfolio_prize_chances(generated_games)
            st.markdown("### MatemÃ¡tica da carteira")
            stat_cols = st.columns(6)
            stat_cols[0].metric("Jogos", len(generated_games))
            stat_cols[1].metric("Cobertura", portfolio_stats["covered_numbers"])
            stat_cols[2].metric("SobreposiÃ§Ã£o mÃ©dia", round(portfolio_stats["avg_overlap"], 1))
            stat_cols[3].metric("MÃ©dia melhor acerto", round(portfolio_stats["avg_best_hits"], 2))
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
            st.dataframe(pd.DataFrame(odds_rows), width="stretch", hide_index=True)
            render_generated_games_gallery(generated_games)

    with tabs[6]:
        st.subheader("IA Aprendizado")

        default_draw_text = "-".join(DEFAULT_EVAL_DRAW)
        actual_draw_text = st.text_input(
            "Resultado real para avaliar",
            value=default_draw_text,
            key="ai_actual_draw_text",
        )
        feedback_date = st.date_input(
            "Data do sorteio avaliado",
            value=date.today(),
            key="ai_feedback_draw_date",
        )
        actual_draw = normalize_numbers(actual_draw_text, limit=20)
        st.caption(f"Resultado usado: {'-'.join(actual_draw)}")

        generated_games = st.session_state.get("generated_games") or []
        if not generated_games:
            st.info("Gere jogos na aba Gerador para avaliar acertos, erros e alimentar o aprendizado.")
        elif len(actual_draw) != 20:
            st.warning("Informe exatamente 20 nÃºmeros do resultado real.")
        else:
            evaluation_df = evaluate_games_against_draw(generated_games, actual_draw)
            best_hits = int(evaluation_df["acertos"].max()) if not evaluation_df.empty else 0
            avg_hits = float(evaluation_df["acertos"].mean()) if not evaluation_df.empty else 0.0
            expected_hits = 50 * 20 / 100
            denominator = math.comb(100, 20)
            chance_best_or_more = sum(
                math.comb(50, hits) * math.comb(50, 20 - hits)
                for hits in range(best_hits, 21)
            ) / denominator

            result_cols = st.columns(5)
            result_cols[0].metric("Jogos avaliados", len(generated_games))
            result_cols[1].metric("Melhor acerto", best_hits)
            result_cols[2].metric("MÃ©dia de acertos", round(avg_hits, 2))
            result_cols[3].metric("Base matemÃ¡tica esperada", round(expected_hits, 2))
            result_cols[4].metric("Chance â‰¥ melhor", f"{chance_best_or_more * 100:.2f}%")

            if best_hits >= expected_hits:
                st.success(f"Melhor jogo ficou {round(best_hits - expected_hits, 2)} ponto(s) acima da base esperada.")
            else:
                st.warning(f"Melhor jogo ficou {round(expected_hits - best_hits, 2)} ponto(s) abaixo da base esperada.")

            st.dataframe(evaluation_df, width="stretch", hide_index=True)

            if st.button("Salvar avaliaÃ§Ã£o e treinar IA", key="button_save_ai_feedback", width="stretch"):
                learning_records = record_learning_feedback(
                    generated_games,
                    actual_draw,
                    draw_date=feedback_date.isoformat(),
                    method=st.session_state.get("last_generation_method", "manual"),
                )
                st.success(f"Aprendizado salvo com {len(learning_records)} registro(s). Os prÃ³ximos jogos jÃ¡ usam esse ajuste.")
                st.rerun() if hasattr(st, "rerun") else st.experimental_rerun()

        learning_records = load_learning_records()
        if learning_records:
            profile = build_learning_profile(learning_records)
            profile_cols = [
                "number",
                "ai_learning_score",
                "vezes_escolhido",
                "acertos_quando_escolhido",
                "erros_quando_escolhido",
                "sorteado_fora_do_jogo",
            ]
            st.markdown("### MemÃ³ria aprendida")
            st.write(f"Registros salvos: {len(learning_records)}")
            st.dataframe(profile[profile_cols].head(25), width="stretch", hide_index=True)
        else:
            st.info("A memÃ³ria da IA ainda estÃ¡ vazia. Salve uma avaliaÃ§Ã£o para comeÃ§ar.")

    with tabs[7]:
        st.subheader("SimulaÃ§Ã£o Monte Carlo")
        if run_montecarlo:
            with st.spinner("Executando Monte Carlo..."):
                st.session_state.montecarlo_results = compute_montecarlo(draws, n_simulations=montecarlo_runs)
        if st.session_state.montecarlo_results is None:
            st.info("Clique em 'Executar Monte Carlo' no painel lateral para iniciar a simulaÃ§Ã£o.")
        else:
            mc_results = st.session_state.montecarlo_results
            probabilities = pd.DataFrame(mc_results["probabilities"].items(), columns=["number", "probability"]).sort_values("probability", ascending=False)
            st.metric("SimulaÃ§Ãµes", mc_results["simulations"])
            render_chart(probabilities.head(50), "number", "probability", "Top 50 probabilidades em Monte Carlo")
            render_table("Top 50 nÃºmeros por probabilidade de Monte Carlo", probabilities.head(50), max_rows=50)

    with tabs[8]:
        st.subheader("Resultados do Backtest")
        st.write("Validacao walk-forward com previsao por frequencia historica.")
        if run_backtest:
            if features is None:
                features = ensure_features(draws)
            with st.spinner("Executando backtest..."):
                backtest_report = compute_backtest(draws, features)
            st.session_state.backtest_report = backtest_report
        if backtest_report is None:
            st.info("Clique em 'Executar Backtest' no painel lateral para iniciar o backtest.")
        elif backtest_report.empty:
            st.write("Backtest nÃ£o retornou resultados suficientes.")
        else:
            summary = backtest_report[["hits", "precision", "recall", "f1", "roi_theoretical"]].describe().transpose()
            st.dataframe(summary)
            render_chart(backtest_report, "split", "precision", "PrecisÃ£o por split")
            render_chart(backtest_report, "split", "roi_theoretical", "ROI teÃ³rico por split")

    with tabs[9]:
        st.subheader("Otimizacao Genetica")
        st.markdown(f"**Criterio selecionado:** {fitness_criteria}")
        st.markdown(f"**Populacao:** {population_size} | **Geracoes:** {generations}")
        if run_genetic:
            genetic_features = features
            if genetic_features is None:
                st.info("Usando ranking rapido para a busca genetica. Prepare features avancadas somente se quiser o modo completo.")
                genetic_features = build_fast_generation_features(draws, statistics)
                genetic_features = apply_learning_to_features(genetic_features, learning_records)

            with st.spinner("Executando otimizacao genetica..."):
                st.session_state.genetic_report = GeneticGameOptimizer(draws, genetic_features).run_evolutionary_search(
                    population_size=population_size,
                    generations=generations,
                    fitness_criteria=fitness_criteria,
                )
                st.session_state.genetic_config = {
                    "criteria": fitness_criteria,
                    "population_size": population_size,
                    "generations": generations,
                }

        if "genetic_report" in st.session_state and st.session_state.genetic_report is not None:
            config = st.session_state.get("genetic_config", {})
            st.write(f"Ultimo resultado gerado com criterio: {config.get('criteria', 'score_total')}")
            render_table("Jogos gerados", st.session_state.genetic_report.sort_values("score_total", ascending=False), max_rows=10)
        else:
            st.info("Clique no botao na barra lateral para gerar combinacoes geneticas usando o criterio selecionado.")
    with tabs[10]:
        st.subheader("CorrelaÃ§Ãµes e padrÃµes")
        st.write("Matriz de lift e pares de nÃºmeros mais correlacionados.")
        if correlation is None:
            correlation = ensure_correlation(draws)
        render_table("Top 50 pares de coocorrÃªncia", pd.DataFrame(strongest_pairs(correlation["cooccurrence"], top_n=50), columns=["number_a", "number_b", "count"]), max_rows=50)
        st.markdown("**Amostra da matriz de lift**")
        st.dataframe(correlation["lift"].iloc[:20, :20])

    st.success("Dashboard carregado com sucesso.")


if __name__ == "__main__":
    run_dashboard()

