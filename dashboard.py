import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
import threading
import time

from config import CACHE_DIR, DATA_FILE, DATABASE_FILE
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
from statistics_engine import build_statistics_report
import ml_engine as ml_engine_module
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from lightgbm import LGBMClassifier

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
    ml_report = ml_engine.train_models()
    ensemble_report = EnsembleEngine(features, ml_report).build_stack()
    return ml_engine, ml_report, ensemble_report


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
        ml_report = ml_engine.train_models()
        ensemble_report = EnsembleEngine(features, ml_report).build_stack()
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


@st.cache_data(show_spinner=False)
def generate_game(features, selection, method: str, game_size: int = 50):
    """Generate a game of up to `game_size` numbers (strings like '00'..'99').

    - `selection` may contain preselected numbers (strings or ints converted elsewhere).
    - `method` supports 'top_score' and 'random'.
    - Ensures we never request more random choices than available.
    """
    top_numbers = features.sort_values("score_total", ascending=False)["number"].tolist()
    selected = list(selection)
    # normalize selection as strings and unique
    selected = [str(s).zfill(2) for s in selected]
    selected = list(dict.fromkeys(selected))

    needed = max(0, game_size - len(selected))
    if method == "top_score":
        numbers = [n for n in top_numbers if n not in selected]
        chosen = numbers[:needed]
    elif method == "random":
        remaining = [n for n in top_numbers if n not in selected]
        take = min(len(remaining), needed)
        chosen = list(np.random.choice(remaining, size=take, replace=False)) if take > 0 else []
    else:
        chosen = []

    result = selected + chosen
    result = sorted(result)[:game_size]
    return result


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
            render_table("Top 50 preditos", ensemble_report[["number", "prob_ensemble", "prob_mean", "target"]], max_rows=50)

    with tabs[3]:
        st.subheader("Avaliação de Modelos")
        if ensemble_report is None:
            st.info("Treine os modelos para ver a comparação entre eles.")
        else:
            model_columns = [col for col in ensemble_report.columns if col.startswith("prob_") and col != "prob_mean"]
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
        selected_numbers = st.multiselect(
            "Selecione até 50 números",
            [f"{n:02d}" for n in range(100)],
            max_selections=50,
        )
        generate_method = st.radio(
            "Método de geração",
            ["top_score", "random"],
            format_func=lambda v: {
                "top_score": "Top Score",
                "random": "Aleatório",
            }[v],
        )

        if len(selected_numbers) < 50:
            st.info(
                f"Selecione até 50 números. O jogo será completado automaticamente para 50 números. Atualmente selecionados: {len(selected_numbers)}."
            )
        elif len(selected_numbers) == 50:
            st.success("Selecionou 50 números. Pronto para gerar o jogo.")

        if features is None:
            if not st.session_state.get("features_background_running"):
                st.session_state["features_background_running"] = True
                threading.Thread(target=_background_prepare_features, args=(draws,), daemon=True).start()
                st.info("Preparação de features iniciada em background. Use 'Atualizar agora' quando terminar.")
            else:
                st.info("Preparação de features rodando em background...")

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

        if st.button("Gerar jogo", key="button_generate_game"):
            if features is None:
                if not st.session_state.get("features_background_running"):
                    st.session_state["features_background_running"] = True
                    threading.Thread(target=_background_prepare_features, args=(draws,), daemon=True).start()
                    st.info("Preparação de features iniciada em background. Use 'Atualizar agora' quando terminar.")
                else:
                    st.info("Preparação de features rodando em background...")
                with st.spinner("Preparando features para geração de jogo..."):
                    features = ensure_features(draws)
            generated_game = generate_game(features, selected_numbers, generate_method)
            st.success(f"Jogo gerado com {len(generated_game)} números")
            st.write(sorted(generated_game))

        if st.button("Completar com top score", key="button_complete_top_score"):
            if features is None:
                if not st.session_state.get("features_background_running"):
                    st.session_state["features_background_running"] = True
                    threading.Thread(target=_background_prepare_features, args=(draws,), daemon=True).start()
                    st.info("Preparação de features iniciada em background. Use 'Atualizar agora' quando terminar.")
                else:
                    st.info("Preparação de features rodando em background...")
                with st.spinner("Preparando features para completar o jogo..."):
                    features = ensure_features(draws)
            generated_game = generate_game(features, selected_numbers, "top_score")
            st.success("Jogo completo gerado usando top score")
            st.write(sorted(generated_game))

        if features is None:
            st.info("Clique em 'Preparar features avançadas' para habilitar o gerador de jogos ou aguarde a preparação em background.")

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
