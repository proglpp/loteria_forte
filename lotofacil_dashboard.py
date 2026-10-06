import json
from datetime import date
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor

from genetic_engine import GeneticGameOptimizer

ROOT_DIR = Path(__file__).resolve().parent
DATA_FILE = ROOT_DIR / "data" / "Lotofacil.csv"
DRAW_COLUMNS = [f"Bola{index}" for index in range(1, 16)]
NUMBERS = [f"{number:02d}" for number in range(1, 26)]
PRIZE_TIERS = {15: "1ª faixa", 14: "2ª faixa", 13: "3ª faixa", 12: "4ª faixa", 11: "5ª faixa"}
LEARNING_FILE = ROOT_DIR / "database" / "lotofacil_ai_learning.json"
PATTERN_OPTIONS = [
    "Pares e ímpares",
    "Faixa de soma",
    "Linhas e colunas",
    "Repetidas do último",
    "Sequências",
    "Primos",
    "Fibonacci",
    "Espelho",
]
FIBONACCI_NUMBERS = {"01", "02", "03", "05", "08", "13", "21"}
PRIME_NUMBERS = {"02", "03", "05", "07", "11", "13", "17", "19", "23"}


@st.cache_data(show_spinner=False)
def load_lotofacil_history(data_file: str = str(DATA_FILE)) -> pd.DataFrame:
    draws = pd.read_csv(data_file, sep=";", encoding="utf-8-sig", dtype=str)
    required = {"Concurso", "Data Sorteio", *DRAW_COLUMNS}
    missing = required.difference(draws.columns)
    if missing:
        raise ValueError(f"Colunas ausentes no histórico da Lotofácil: {sorted(missing)}")

    draws["Concurso"] = pd.to_numeric(draws["Concurso"], errors="coerce")
    draws["Data Sorteio"] = pd.to_datetime(draws["Data Sorteio"], dayfirst=True, errors="coerce")
    values = draws[DRAW_COLUMNS].apply(lambda column: pd.to_numeric(column, errors="coerce"))
    if draws[["Concurso", "Data Sorteio"]].isna().values.any() or values.isna().values.any():
        raise ValueError("O histórico da Lotofácil possui concurso, data ou dezena inválidos")
    if ((values < 1) | (values > 25)).values.any():
        raise ValueError("A Lotofácil aceita dezenas de 01 a 25")
    if values.apply(lambda row: row.nunique() != 15, axis=1).any():
        raise ValueError("Cada resultado da Lotofácil deve conter 15 dezenas distintas")
    if draws["Concurso"].duplicated().any():
        raise ValueError("O histórico da Lotofácil contém concursos duplicados")

    draws[DRAW_COLUMNS] = values.apply(lambda column: column.map(lambda number: f"{int(number):02d}"))
    return draws.sort_values("Concurso").reset_index(drop=True)


def build_lotofacil_ranking(draws: pd.DataFrame) -> pd.DataFrame:
    all_values = draws[DRAW_COLUMNS].values.flatten()
    recent_values = draws.tail(100)[DRAW_COLUMNS].values.flatten()
    recent_25_values = draws.tail(25)[DRAW_COLUMNS].values.flatten()
    counts = pd.Series(all_values).value_counts().reindex(NUMBERS, fill_value=0)
    recent_counts = pd.Series(recent_values).value_counts().reindex(NUMBERS, fill_value=0)
    recent_25_counts = pd.Series(recent_25_values).value_counts().reindex(NUMBERS, fill_value=0)
    last_draw = set(draws.iloc[-1][DRAW_COLUMNS])
    last_seen: dict[str, int] = {number: len(draws) for number in NUMBERS}
    for offset, row in enumerate(draws[DRAW_COLUMNS].itertuples(index=False, name=None)):
        for number in row:
            last_seen[number] = len(draws) - offset - 1

    ranking = pd.DataFrame(
        {
            "dezena": NUMBERS,
            "freq_total": counts.values,
            "freq_100": recent_counts.values,
            "freq_25": recent_25_counts.values,
            "atraso": [last_seen[number] for number in NUMBERS],
            "saiu_ultimo": [number in last_draw for number in NUMBERS],
        }
    )
    frequency_score = ranking["freq_total"].rank(pct=True)
    recent_score = ranking["freq_100"].rank(pct=True)
    delay_score = ranking["atraso"].rank(pct=True)
    ranking["score_historico"] = (0.45 * frequency_score) + (0.35 * recent_score) + (0.20 * delay_score)
    ranking["score_total"] = ranking["score_historico"]
    return ranking.sort_values(["score_historico", "dezena"], ascending=[False, True]).reset_index(drop=True)


def lotofacil_ticket_odds(ticket_size: int) -> dict[int, float]:
    if not 15 <= ticket_size <= 20:
        raise ValueError("A aposta da Lotofácil deve ter de 15 a 20 dezenas")
    total_draws = comb(25, 15)
    minimum_hits = max(0, ticket_size - 10)
    maximum_hits = min(ticket_size, 15)
    return {
        tier: sum(
            comb(ticket_size, hits) * comb(25 - ticket_size, 15 - hits)
            for hits in range(max(tier, minimum_hits), maximum_hits + 1)
        )
        / total_draws
        for tier in PRIZE_TIERS
    }


def _lotofacil_pattern_bounds(draws: pd.DataFrame, ticket_size: int) -> dict[str, float]:
    draw_values = draws[DRAW_COLUMNS].astype(int)
    scale = ticket_size / 15
    sums = draw_values.sum(axis=1).to_numpy(dtype=float) * scale
    repeats = [
        len(set(current) & set(previous)) * scale
        for previous, current in zip(draw_values.iloc[:-1].values, draw_values.iloc[1:].values)
    ]
    max_runs = []
    for row in draw_values.values:
        values = set(row)
        longest = current = 0
        for number in range(1, 26):
            current = current + 1 if number in values else 0
            longest = max(longest, current)
        max_runs.append(longest)
    return {
        "sum_min": float(np.quantile(sums, 0.03)),
        "sum_max": float(np.quantile(sums, 0.97)),
        "repeat_min": float(np.quantile(repeats, 0.03)) if repeats else 0.0,
        "repeat_max": float(np.quantile(repeats, 0.97)) if repeats else float(ticket_size),
        "max_run": float(np.quantile(max_runs, 0.99)) + np.ceil((ticket_size - 15) / 2),
    }


def _passes_lotofacil_patterns(
    game: list[str],
    draws: pd.DataFrame | None,
    patterns: list[str],
    bounds: dict[str, float] | None,
) -> bool:
    if not patterns:
        return True
    values = {int(number) for number in game}
    if "Pares e ímpares" in patterns:
        even_count = sum(number % 2 == 0 for number in values)
        if not 0.35 * len(game) <= even_count <= 0.65 * len(game):
            return False
    if draws is None:
        return True
    bounds = bounds or _lotofacil_pattern_bounds(draws, len(game))
    if "Faixa de soma" in patterns and not bounds["sum_min"] <= sum(values) <= bounds["sum_max"]:
        return False
    if "Linhas e colunas" in patterns:
        rows = [sum((number - 1) // 5 == row for number in values) for row in range(5)]
        columns = [sum((number - 1) % 5 == column for number in values) for column in range(5)]
        if min(rows) < 1 or min(columns) < 1 or max(rows) > 5 or max(columns) > 5:
            return False
    if "Repetidas do último" in patterns:
        previous = set(draws.iloc[-1][DRAW_COLUMNS].tolist())
        repeats = len(values & {int(number) for number in previous})
        if not bounds["repeat_min"] <= repeats <= bounds["repeat_max"]:
            return False
    if "Sequências" in patterns:
        longest = current = 0
        for number in range(1, 26):
            current = current + 1 if number in values else 0
            longest = max(longest, current)
        if longest > bounds["max_run"]:
            return False
    if "Primos" in patterns and len(values & {int(number) for number in PRIME_NUMBERS}) < 2:
        return False
    if "Fibonacci" in patterns and len(values & {int(number) for number in FIBONACCI_NUMBERS}) < 2:
        return False
    if "Espelho" in patterns and draws is not None:
        last_draw = {int(number) for number in draws.iloc[-1][DRAW_COLUMNS]}
        mirrored = {26 - number for number in last_draw}
        if len(values & mirrored) < 2:
            return False
    return True


def generate_lotofacil_games(
    ranking: pd.DataFrame,
    ticket_size: int,
    game_count: int,
    method: str,
    selected: list[str] | None = None,
    seed: int = 42,
    draws: pd.DataFrame | None = None,
    patterns: list[str] | None = None,
    previous_games: list[list[str]] | None = None,
) -> list[list[str]]:
    if not 15 <= ticket_size <= 20:
        raise ValueError("A aposta da Lotofácil deve ter de 15 a 20 dezenas")
    fixed = list(dict.fromkeys(number for number in (selected or []) if number in NUMBERS))
    if len(fixed) > ticket_size:
        raise ValueError("A seleção manual excede o tamanho da aposta")
    patterns = patterns or []
    bounds = _lotofacil_pattern_bounds(draws, ticket_size) if draws is not None else None
    if len(fixed) == ticket_size:
        game = sorted(fixed)
        return [game] if _passes_lotofacil_patterns(game, draws, patterns, bounds) else []

    scores = ranking.set_index("dezena").reindex(NUMBERS)
    if method == "Frequência recente":
        weights = scores["freq_100"].to_numpy(dtype=float) + 1.0
    elif method == "Mais atrasadas":
        weights = scores["atraso"].to_numpy(dtype=float) + 1.0
    elif method == "Pós-resultado":
        weights = scores["score_historico"].to_numpy(dtype=float) + (~scores["saiu_ultimo"].to_numpy(dtype=bool)) * 0.5
    elif method == "Pontuação máxima":
        weights = np.exp(scores["score_historico"].to_numpy(dtype=float) * 4.0)
    elif method == "Cobertura/diversidade":
        usage = {number: 0 for number in NUMBERS}
        for previous in previous_games or []:
            for number in previous:
                usage[number] += 1
        weights = (scores["score_historico"].to_numpy(dtype=float) + 0.05) / np.array(
            [1.0 + usage[number] for number in NUMBERS]
        )
    elif method == "Aprendizado" and "ai_learning_score" in scores.columns:
        weights = scores["ai_learning_score"].to_numpy(dtype=float) - scores["ai_learning_score"].min() + 0.05
    elif method == "Aleatório":
        weights = np.ones(len(NUMBERS), dtype=float)
    else:
        weights = scores["score_historico"].to_numpy(dtype=float) + 0.05
    weights /= weights.sum()

    rng = np.random.default_rng(seed)
    games: list[list[str]] = []
    seen = set()
    for variation in range(max(game_count * 250, 500)):
        needed = ticket_size - len(fixed)
        adjusted_weights = weights.copy()
        if method != "Aleatório" and method != "Pontuação máxima" and variation:
            noise = rng.uniform(0.85, 1.15, len(NUMBERS))
            adjusted_weights *= noise
        available = [number for number in NUMBERS if number not in fixed]
        available_weights = np.array([adjusted_weights[NUMBERS.index(number)] for number in available])
        available_weights /= available_weights.sum()
        added = rng.choice(available, size=needed, replace=False, p=available_weights).tolist()
        game = tuple(sorted([*fixed, *added]))
        if game not in seen and _passes_lotofacil_patterns(list(game), draws, patterns, bounds):
            seen.add(game)
            games.append(list(game))
        if len(games) >= game_count:
            break
    return games


def build_lotofacil_movement(draws: pd.DataFrame, ranking: pd.DataFrame | None = None) -> pd.DataFrame:
    movement = (ranking if ranking is not None else build_lotofacil_ranking(draws)).copy()
    movement["tendencia_25_vs_100"] = (movement["freq_25"] * 4.0) - movement["freq_100"]
    movement["score_movimento"] = (
        0.35 * movement["freq_25"].rank(pct=True)
        + 0.25 * movement["tendencia_25_vs_100"].rank(pct=True)
        + 0.20 * movement["atraso"].rank(pct=True)
        + 0.20 * movement["score_historico"]
    )
    lower, middle, upper = movement["score_movimento"].quantile([0.25, 0.55, 0.80])
    movement["status"] = np.select(
        [movement["score_movimento"] >= upper, movement["score_movimento"] >= middle, movement["score_movimento"] <= lower],
        ["FORTE", "BOM", "FRACO"],
        default="OBS",
    )
    return movement.sort_values("score_movimento", ascending=False).reset_index(drop=True)


def build_lotofacil_movement_table(
    draws: pd.DataFrame,
    rows: int = 12,
    strength_window: int = 10,
) -> pd.DataFrame:
    movement = build_lotofacil_movement(draws)
    status_map = movement.set_index("dezena")["status"].to_dict()
    recent_draws = draws.tail(max(1, min(int(rows), len(draws))))
    table_rows = []
    for _, draw in recent_draws.iterrows():
        present = set(draw[DRAW_COLUMNS].tolist())
        table_rows.append(
            {"Concurso": str(int(draw["Concurso"])), **{number: number if number in present else "" for number in NUMBERS}}
        )

    window = max(1, min(int(strength_window), len(draws)))
    recent = draws.tail(window)[DRAW_COLUMNS].values.flatten().tolist()
    counts = pd.Series(recent).value_counts().reindex(NUMBERS, fill_value=0)
    delay = movement.set_index("dezena")["atraso"]
    trend = movement.set_index("dezena")["tendencia_25_vs_100"]
    score = movement.set_index("dezena")["score_movimento"]
    table_rows.extend(
        [
            {"Concurso": "FREQ", **{number: int(counts[number]) for number in NUMBERS}},
            {"Concurso": "ATRASO", **{number: int(delay[number]) for number in NUMBERS}},
            {"Concurso": "TENDÊNCIA", **{number: int(round(trend[number])) for number in NUMBERS}},
            {"Concurso": "FORÇA", **{number: int(round(score[number] * 100)) for number in NUMBERS}},
        ]
    )
    table = pd.DataFrame(table_rows).astype(str)
    table.attrs["status_map"] = status_map
    return table


def style_lotofacil_movement_table(table: pd.DataFrame):
    status_map = table.attrs.get("status_map", {})
    status_styles = {
        "FORTE": ("#009966", "#ffffff"),
        "BOM": ("#55b7ff", "#07121f"),
        "OBS": ("#d6b4ec", "#2b0f3f"),
        "FRACO": ("#e85d75", "#ffffff"),
    }

    def apply_styles(data: pd.DataFrame) -> pd.DataFrame:
        styles = pd.DataFrame("", index=data.index, columns=data.columns)
        for row_index in data.index:
            label = str(data.loc[row_index, "Concurso"])
            for column in data.columns:
                value = str(data.loc[row_index, column])
                if column == "Concurso":
                    styles.loc[row_index, column] = "background-color: #243b53; color: white; font-weight: 700; text-align: center;"
                elif label in {"FREQ", "ATRASO", "TENDÊNCIA", "FORÇA"}:
                    styles.loc[row_index, column] = "background-color: #e8edf2; color: #172b4d; font-weight: 700; text-align: center;"
                elif value:
                    background, foreground = status_styles.get(status_map.get(column, "OBS"), status_styles["OBS"])
                    styles.loc[row_index, column] = f"background-color: {background}; color: {foreground}; font-weight: 800; text-align: center;"
                else:
                    background = {
                        "FORTE": "#e8f7ee",
                        "BOM": "#edf4ff",
                        "OBS": "#f6effa",
                        "FRACO": "#fdecec",
                    }.get(status_map.get(column, "OBS"), "#ffffff")
                    styles.loc[row_index, column] = f"background-color: {background}; color: transparent;"
        return styles

    return (
        table.style.apply(apply_styles, axis=None)
        .set_properties(**{"font-size": "11px", "min-width": "38px", "height": "25px", "text-align": "center"})
        .hide(axis="index")
    )


def build_lotofacil_pattern_report(draws: pd.DataFrame) -> pd.DataFrame:
    results = []
    previous: set[int] = set()
    for _, row in draws.iterrows():
        values = {int(row[column]) for column in DRAW_COLUMNS}
        ordered = sorted(values)
        longest = current = 0
        for number in range(1, 26):
            current = current + 1 if number in values else 0
            longest = max(longest, current)
        results.append(
            {
                "Concurso": int(row["Concurso"]),
                "Soma": sum(values),
                "Pares": sum(number % 2 == 0 for number in values),
                "Ímpares": sum(number % 2 != 0 for number in values),
                "Repetidas do anterior": len(values & previous) if previous else np.nan,
                "Maior sequência": longest,
                "Primos": sum(f"{number:02d}" in PRIME_NUMBERS for number in values),
                "Fibonacci": sum(f"{number:02d}" in FIBONACCI_NUMBERS for number in values),
                "Linhas ocupadas": len({(number - 1) // 5 for number in values}),
                "Colunas ocupadas": len({(number - 1) % 5 for number in values}),
            }
        )
        previous = values
    return pd.DataFrame(results)


def _lotofacil_indicator_matrix(draws: pd.DataFrame) -> np.ndarray:
    indicators = np.zeros((len(draws), 25), dtype=np.uint8)
    row_indices = np.repeat(np.arange(len(draws)), 15)
    number_indices = draws[DRAW_COLUMNS].astype(int).to_numpy().reshape(-1) - 1
    indicators[row_indices, number_indices] = 1
    return indicators


def _lotofacil_observation_features(indicators: np.ndarray, end_index: int) -> np.ndarray:
    history = indicators[:end_index]
    features = []
    for window in (5, 20, 100):
        features.extend(history[-window:].mean(axis=0).tolist())
    features.extend(history[-1].astype(float).tolist())
    delays = []
    for number_index in range(25):
        seen = np.flatnonzero(history[:, number_index])
        delays.append(min(end_index - 1 - int(seen[-1]), 100) / 100 if len(seen) else 1.0)
    features.extend(delays)
    return np.asarray(features, dtype=float)


def train_lotofacil_models(draws: pd.DataFrame, quick: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    indicators = _lotofacil_indicator_matrix(draws)
    first_target = 100
    if len(indicators) <= first_target + 20:
        raise ValueError("São necessários pelo menos 121 concursos para treinar os modelos")
    target_indices = range(first_target, len(indicators))
    features = np.vstack([_lotofacil_observation_features(indicators, index) for index in target_indices])
    targets = indicators[first_target:].astype(float)
    latest_features = _lotofacil_observation_features(indicators, len(indicators)).reshape(1, -1)
    split = max(1, int(len(features) * 0.8))
    if split >= len(features):
        split = len(features) - 1

    estimator_specs = {
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=35 if quick else 100,
            max_depth=8 if quick else 12,
            min_samples_leaf=5,
            n_jobs=4,
            random_state=42,
        )
    }
    if not quick:
        estimator_specs["Random Forest"] = RandomForestRegressor(
            n_estimators=100,
            max_depth=12,
            min_samples_leaf=5,
            n_jobs=4,
            random_state=43,
        )

    model_outputs = []
    metric_rows = []
    for name, estimator in estimator_specs.items():
        estimator.fit(features[:split], targets[:split])
        validation = np.clip(estimator.predict(features[split:]), 0.0, 1.0)
        brier = float(np.mean((validation - targets[split:]) ** 2))
        hits = [
            len(set(np.argsort(prediction)[-15:]) & set(np.flatnonzero(actual)))
            for prediction, actual in zip(validation, targets[split:])
        ]
        estimator.fit(features, targets)
        latest_scores = np.clip(estimator.predict(latest_features).reshape(-1), 0.0, 1.0)
        model_outputs.append(latest_scores)
        metric_rows.append(
            {
                "Modelo": name,
                "Brier (validação temporal)": round(brier, 5),
                "Média de acertos no top-15": round(float(np.mean(hits)), 3),
                "Concursos de validação": len(validation),
            }
        )

    prediction = np.mean(model_outputs, axis=0)
    report = pd.DataFrame({"dezena": NUMBERS, "score_modelos": prediction})
    report = report.sort_values("score_modelos", ascending=False).reset_index(drop=True)
    return report, pd.DataFrame(metric_rows)


def run_lotofacil_backtest(draws: pd.DataFrame, window: int = 100, test_count: int = 150) -> pd.DataFrame:
    first_test = max(window, len(draws) - test_count)
    step = max(1, (len(draws) - first_test) // 75)
    rows = []
    for index in range(first_test, len(draws), step):
        history = draws.iloc[max(0, index - window) : index]
        ranking = build_lotofacil_ranking(history)
        predicted = set(ranking.head(15)["dezena"])
        actual = set(draws.iloc[index][DRAW_COLUMNS].tolist())
        hits = len(predicted & actual)
        rows.append(
            {
                "Concurso": int(draws.iloc[index]["Concurso"]),
                "Acertos no top-15": hits,
                "Precisão": hits / 15,
                "Acertou 11+": hits >= 11,
            }
        )
    return pd.DataFrame(rows)


def run_lotofacil_montecarlo(
    ranking: pd.DataFrame,
    games: list[list[str]] | None = None,
    simulations: int = 5000,
    seed: int = 42,
) -> dict[str, object]:
    scores = ranking.set_index("dezena").reindex(NUMBERS)
    weights = scores["freq_total"].to_numpy(dtype=float) + 1.0
    weights /= weights.sum()
    rng = np.random.default_rng(seed)
    number_counts = np.zeros(25, dtype=int)
    tier_counts = {tier: 0 for tier in PRIZE_TIERS}
    best_hits_total = 0
    game_sets = [set(game) for game in games or []]
    for _ in range(int(simulations)):
        draw_indices = rng.choice(25, size=15, replace=False, p=weights)
        number_counts[draw_indices] += 1
        if game_sets:
            draw = {NUMBERS[index] for index in draw_indices}
            best_hits = max(len(game & draw) for game in game_sets)
            best_hits_total += best_hits
            for tier in tier_counts:
                if best_hits >= tier:
                    tier_counts[tier] += 1
    return {
        "simulations": int(simulations),
        "number_rates": {number: count / (simulations * 15) for number, count in zip(NUMBERS, number_counts)},
        "portfolio_tiers": {tier: count / simulations for tier, count in tier_counts.items()},
        "average_best_hits": best_hits_total / simulations if game_sets else 0.0,
        "sampling": "ponderada pelas frequências históricas; não substitui as chances combinatórias oficiais",
    }


def build_lotofacil_correlations(draws: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    indicators = _lotofacil_indicator_matrix(draws).astype(float)
    cooccurrence = indicators.T @ indicators
    marginal = indicators.mean(axis=0)
    joint = cooccurrence / max(len(draws), 1)
    expected = np.outer(marginal, marginal)
    lift = np.divide(joint, expected, out=np.zeros_like(joint), where=expected > 0)
    np.fill_diagonal(lift, 1.0)
    pairs = []
    for left in range(25):
        for right in range(left + 1, 25):
            pairs.append(
                {
                    "Dezena A": NUMBERS[left],
                    "Dezena B": NUMBERS[right],
                    "Coocorrências": int(cooccurrence[left, right]),
                    "Lift": float(lift[left, right]),
                }
            )
    pairs_frame = pd.DataFrame(pairs).sort_values(["Lift", "Coocorrências"], ascending=False).reset_index(drop=True)
    lift_frame = pd.DataFrame(lift, index=NUMBERS, columns=NUMBERS)
    return pairs_frame, lift_frame


def load_lotofacil_learning_records() -> list[dict]:
    if not LEARNING_FILE.exists():
        return []
    try:
        records = json.loads(LEARNING_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return records if isinstance(records, list) else []


def record_lotofacil_feedback(
    games: list[list[str]],
    actual_draw: list[str],
    draw_date: date,
    method: str,
) -> list[dict]:
    records = load_lotofacil_learning_records()
    signature = (draw_date.isoformat(), tuple(actual_draw))
    for index, game in enumerate(games, start=1):
        if any(
            record.get("draw_date") == signature[0]
            and record.get("actual_draw") == list(signature[1])
            and record.get("game_numbers") == game
            and record.get("game_index") == index
            for record in records
        ):
            continue
        hits = sorted(set(game) & set(actual_draw))
        records.append(
            {
                "draw_date": signature[0],
                "method": method,
                "game_index": index,
                "game_numbers": game,
                "actual_draw": list(signature[1]),
                "hits": hits,
                "hit_count": len(hits),
            }
        )
    LEARNING_FILE.parent.mkdir(parents=True, exist_ok=True)
    LEARNING_FILE.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    return records


def build_lotofacil_learning_profile(records: list[dict]) -> pd.DataFrame:
    stats = []
    for number in NUMBERS:
        chosen = drawn = hits = missed = 0
        for record in records:
            in_game = number in record.get("game_numbers", [])
            in_draw = number in record.get("actual_draw", [])
            chosen += int(in_game)
            drawn += int(in_draw)
            hits += int(in_game and in_draw)
            missed += int(in_draw and not in_game)
        hit_rate = (hits + 1) / (chosen + 2) if chosen else 15 / 25
        coverage_gap = (missed + 1) / (drawn + 2) if drawn else 0.0
        stats.append(
            {
                "dezena": number,
                "vezes_escolhida": chosen,
                "acertos_quando_escolhida": hits,
                "sorteada_fora_da_aposta": missed,
                "confianca": min(1.0, np.sqrt(chosen + drawn) / np.sqrt(max(len(records) * 2, 1))),
                "ai_learning_score": (hit_rate - 0.60) + (0.25 * coverage_gap),
            }
        )
    return pd.DataFrame(stats)


def _format_percent(value: float) -> str:
    return f"{value * 100:.4f}%"


def _render_overview(draws: pd.DataFrame, ranking: pd.DataFrame) -> None:
    latest = draws.iloc[-1]
    columns = st.columns(4)
    columns[0].metric("Concursos", f"{len(draws):,}".replace(",", "."))
    columns[1].metric("Último concurso", int(latest["Concurso"]))
    columns[2].metric("Data", latest["Data Sorteio"].strftime("%d/%m/%Y"))
    columns[3].metric("Dezenas por sorteio", "15 de 25")

    st.subheader("Último resultado oficial")
    st.write(" · ".join(latest[DRAW_COLUMNS].tolist()))
    odd_count = sum(int(number) % 2 for number in latest[DRAW_COLUMNS])
    st.caption(
        f"Pares: {15 - odd_count} · Ímpares: {odd_count} · "
        f"Soma: {sum(int(number) for number in latest[DRAW_COLUMNS])}"
    )
    st.dataframe(draws.tail(20), hide_index=True, width="stretch")
    st.subheader("Mais frequentes no histórico")
    frequency_columns = ranking.set_index("dezena")[["freq_total", "freq_25"]].rename(
        columns={"freq_total": "Histórico", "freq_25": "Últimos 25"}
    )
    st.bar_chart(frequency_columns)
    patterns = build_lotofacil_pattern_report(draws).tail(100)
    st.subheader("Padrões dos últimos 100 concursos")
    st.dataframe(patterns.describe().round(2).transpose(), width="stretch")
    st.caption("Frequências descrevem resultados passados; não garantem resultados futuros.")


def _render_ranking(ranking: pd.DataFrame) -> None:
    st.subheader("Frequência, recorrência e atraso")
    display = ranking.copy()
    display["saiu_ultimo"] = display["saiu_ultimo"].map({True: "Sim", False: "Não"})
    display = display.rename(
        columns={
            "dezena": "Dezena",
            "freq_total": "Concursos com a dezena",
            "freq_100": "Últimos 100 concursos",
            "freq_25": "Últimos 25 concursos",
            "atraso": "Atraso atual",
            "saiu_ultimo": "Saiu no último",
            "score_historico": "Score histórico",
        }
    )
    st.dataframe(display, hide_index=True, width="stretch")


def estimate_lotofacil_portfolio(games: list[list[str]], simulations: int = 3000, seed: int = 42) -> dict[str, object]:
    if not games:
        return {"tiers": {}, "average_best_hits": 0.0, "covered_numbers": 0, "average_overlap": 0.0}
    game_sets = [set(game) for game in games]
    ticket_size = len(games[0])
    rng = np.random.default_rng(seed)
    tier_counts = {tier: 0 for tier in PRIZE_TIERS}
    best_hits_total = 0
    for _ in range(simulations):
        draw = set(rng.choice(NUMBERS, size=15, replace=False).tolist())
        best_hits = max(len(game & draw) for game in game_sets)
        best_hits_total += best_hits
        for tier in tier_counts:
            if best_hits >= tier:
                tier_counts[tier] += 1
    overlaps = [
        len(game_sets[left] & game_sets[right])
        for left in range(len(game_sets))
        for right in range(left + 1, len(game_sets))
    ]
    return {
        "tiers": {tier: count / simulations for tier, count in tier_counts.items()},
        "single_ticket_tiers": lotofacil_ticket_odds(ticket_size),
        "average_best_hits": best_hits_total / simulations,
        "covered_numbers": len(set().union(*game_sets)),
        "average_overlap": float(np.mean(overlaps)) if overlaps else 0.0,
        "ticket_size": ticket_size,
    }


def _render_movement(draws: pd.DataFrame, ranking: pd.DataFrame) -> None:
    movement = build_lotofacil_movement(draws, ranking)
    st.subheader("Movimentação das dezenas")
    st.caption("A cor de cada dezena segue o status de força; as linhas finais mostram frequência, atraso, tendência e força.")
    control_columns = st.columns(2)
    with control_columns[0]:
        row_count = st.slider("Concursos na tabela", 5, 30, 12, key="lotofacil_movement_rows")
    with control_columns[1]:
        window = st.slider("Janela de frequência", 5, 50, 10, key="lotofacil_movement_window")
    counts = movement["status"].value_counts().reindex(["FORTE", "BOM", "OBS", "FRACO"], fill_value=0)
    metric_columns = st.columns(4)
    for column, label in zip(metric_columns, counts.index):
        column.metric(label.title(), int(counts[label]))
    movement_table = build_lotofacil_movement_table(draws, rows=row_count, strength_window=window)
    st.markdown("**Tabela colorida por concurso e dezena**")
    st.dataframe(style_lotofacil_movement_table(movement_table), width="stretch", height=470)
    st.markdown("**Legenda de força**")
    legend = pd.DataFrame(
        [
            {"Cor": "Verde", "Status": "FORTE", "Critério": "Score de movimento no grupo superior"},
            {"Cor": "Azul", "Status": "BOM", "Critério": "Score acima da faixa intermediária"},
            {"Cor": "Lilás", "Status": "OBS", "Critério": "Faixa intermediária"},
            {"Cor": "Vermelho", "Status": "FRACO", "Critério": "Score no grupo inferior"},
        ]
    )
    st.dataframe(legend, hide_index=True, width="stretch")
    st.markdown("**Ranking da movimentação**")
    display = movement[
        ["dezena", "freq_25", "freq_100", "tendencia_25_vs_100", "atraso", "status", "score_movimento"]
    ].rename(
        columns={
            "dezena": "Dezena",
            "freq_25": "Últimos 25",
            "freq_100": "Últimos 100",
            "tendencia_25_vs_100": "Tendência normalizada",
            "atraso": "Atraso",
            "status": "Status",
            "score_movimento": "Score de movimento",
        }
    )
    st.dataframe(display, hide_index=True, width="stretch")


def _render_predictions(ranking: pd.DataFrame, model_report: pd.DataFrame | None) -> None:
    predictions = ranking.copy()
    if model_report is not None:
        predictions = predictions.merge(model_report, on="dezena", how="left")
        model_rank = predictions["score_modelos"].rank(pct=True).fillna(0.5)
        predictions["score_final"] = 0.65 * predictions["score_historico"] + 0.35 * model_rank
        description = "Score combinado: 65% histórico e 35% posição relativa dos modelos."
    else:
        predictions["score_final"] = predictions["score_historico"]
        description = "Score histórico descritivo. Treine os modelos para incluir o sinal supervisionado."
    predictions = predictions.sort_values("score_final", ascending=False).reset_index(drop=True)
    st.subheader("Ranking para o próximo concurso")
    st.caption(f"{description} Não é uma probabilidade oficial nem uma garantia de acerto.")
    st.dataframe(
        predictions[[column for column in ("dezena", "score_final", "freq_100", "freq_25", "atraso", "saiu_ultimo") if column in predictions.columns]].head(25),
        hide_index=True,
        width="stretch",
    )
    st.markdown("**Primeiras 15 dezenas pelo score atual**")
    st.write(" · ".join(predictions.head(15)["dezena"].tolist()))
    st.download_button(
        "Baixar ranking de previsões",
        data=predictions.to_csv(index=False),
        file_name="ranking_lotofacil.csv",
        mime="text/csv",
        key="lotofacil_download_predictions",
    )


def _render_models(model_report: pd.DataFrame | None, model_metrics: pd.DataFrame | None) -> None:
    st.subheader("Modelos supervisionados")
    st.caption("Treino temporal com frequência móvel, presença anterior e atraso, usando apenas concursos anteriores.")
    if model_metrics is None or model_report is None:
        st.info("Use os botões de treino na lateral. O modo rápido usa um modelo; o avançado compara Extra Trees e Random Forest.")
        return
    st.dataframe(model_metrics, hide_index=True, width="stretch")
    st.markdown("**Scores do modelo para cada dezena**")
    st.bar_chart(model_report.set_index("dezena")["score_modelos"])
    st.dataframe(model_report, hide_index=True, width="stretch")


def _render_ai_learning(draws: pd.DataFrame, games: list[list[str]]) -> None:
    st.subheader("Aprendizado por avaliação real")
    latest = draws.iloc[-1]
    actual_draw = st.multiselect(
        "Resultado real (15 dezenas)",
        NUMBERS,
        default=latest[DRAW_COLUMNS].tolist(),
        max_selections=15,
        key="lotofacil_feedback_actual_draw",
    )
    draw_date = st.date_input("Data do resultado avaliado", value=latest["Data Sorteio"].date(), key="lotofacil_feedback_date")
    if not games:
        st.info("Gere apostas na aba Gerador para avaliá-las e registrar feedback.")
    elif len(actual_draw) != 15:
        st.warning("Selecione exatamente 15 dezenas do resultado real.")
    else:
        actual_set = set(actual_draw)
        evaluations = []
        for index, game in enumerate(games, start=1):
            hits = len(set(game) & actual_set)
            evaluations.append(
                {
                    "Aposta": index,
                    "Acertos": hits,
                    "Faixa": PRIZE_TIERS.get(hits, "Sem prêmio"),
                    "Dezenas": " · ".join(game),
                }
            )
        st.dataframe(pd.DataFrame(evaluations), hide_index=True, width="stretch")
        if st.button("Salvar avaliação e atualizar aprendizado", key="lotofacil_save_feedback"):
            records = record_lotofacil_feedback(
                games,
                actual_draw,
                draw_date,
                st.session_state.get("lotofacil_last_generation_method", "manual"),
            )
            st.success(f"Feedback Lotofácil salvo. Total de avaliações: {len(records)}.")
            st.rerun()

    records = load_lotofacil_learning_records()
    if records:
        profile = build_lotofacil_learning_profile(records).sort_values("ai_learning_score", ascending=False)
        st.markdown("**Memória aprendida para a Lotofácil**")
        st.caption(f"{len(records)} avaliações salvas em arquivo separado da Lotomania.")
        st.dataframe(profile, hide_index=True, width="stretch")
    else:
        st.info("Ainda não há feedback salvo para a Lotofácil.")


def _render_montecarlo(result: dict[str, object] | None) -> None:
    st.subheader("Simulação Monte Carlo")
    st.caption("A simulação pondera as dezenas pelas frequências históricas; as chances oficiais são combinatórias e mostradas separadamente.")
    if result is None:
        st.info("Inicie a simulação pelo painel lateral.")
        return
    st.metric("Simulações", f"{result['simulations']:,}".replace(",", "."))
    if result["portfolio_tiers"]:
        metrics = st.columns(5)
        for column, tier in zip(metrics, (15, 14, 13, 12, 11)):
            column.metric(f"Carteira {tier}+", _format_percent(result["portfolio_tiers"][tier]))
        st.metric("Média do melhor resultado", round(result["average_best_hits"], 2))
    rates = pd.DataFrame(result["number_rates"].items(), columns=["Dezena", "Frequência simulada"])
    st.bar_chart(rates.set_index("Dezena"))


def _render_backtest(backtest: pd.DataFrame | None) -> None:
    st.subheader("Backtest walk-forward")
    st.caption("Em cada corte, o ranking usa somente o histórico anterior; a referência aleatória esperada é 9 acertos em média.")
    if backtest is None or backtest.empty:
        st.info("Execute o backtest pela lateral para avaliar o ranking em concursos passados.")
        return
    columns = st.columns(4)
    columns[0].metric("Cortes", len(backtest))
    columns[1].metric("Média de acertos", round(float(backtest["Acertos no top-15"].mean()), 2))
    columns[2].metric("Mediana", round(float(backtest["Acertos no top-15"].median()), 2))
    columns[3].metric("Faixa 11+", _format_percent(float(backtest["Acertou 11+"].mean())))
    st.line_chart(backtest.set_index("Concurso")["Acertos no top-15"])
    st.dataframe(backtest, hide_index=True, width="stretch")


def _render_genetic(genetic_report: pd.DataFrame | None, ticket_size: int) -> None:
    st.subheader("Busca genética")
    st.caption(f"O algoritmo evolui apostas completas de {ticket_size} dezenas no universo 01–25.")
    if genetic_report is None or genetic_report.empty:
        st.info("Configure população e gerações na lateral e inicie a busca genética.")
        return
    display = genetic_report.copy()
    display["game"] = display["game"].map(lambda game: " · ".join(game))
    st.dataframe(display, hide_index=True, width="stretch")
    if st.button("Adicionar as melhores apostas à carteira", key="lotofacil_add_genetic_games"):
        current = st.session_state.get("lotofacil_games", [])
        new_games = genetic_report["game"].head(10).tolist()
        st.session_state.lotofacil_games = current + [game for game in new_games if game not in current]
        st.success("Apostas adicionadas à carteira.")


def _render_correlations(pairs: pd.DataFrame, lift: pd.DataFrame) -> None:
    st.subheader("Correlação e padrões de coocorrência")
    st.caption("Lift acima de 1 indica coocorrência acima da independência observada no histórico; não implica causalidade.")
    st.markdown("**Pares com maior lift**")
    st.dataframe(pairs.head(30), hide_index=True, width="stretch")
    st.markdown("**Matriz de lift (01–25)**")
    st.dataframe(lift.round(2), width="stretch")


def _render_generator(ranking: pd.DataFrame, draws: pd.DataFrame, ticket_size: int, game_count: int, patterns: list[str]) -> None:
    selected = st.session_state.get("lotofacil_selected_numbers", [])
    if st.button("Gerar novas apostas", type="primary", key="lotofacil_generate"):
        method = st.session_state.get("lotofacil_method", "Score equilibrado")
        previous_games = st.session_state.get("lotofacil_games", [])
        seed = int(st.session_state.get("lotofacil_seed", 42)) + 1
        st.session_state.lotofacil_games = generate_lotofacil_games(
            ranking,
            ticket_size,
            game_count,
            method,
            selected=selected,
            seed=seed,
            draws=draws,
            patterns=patterns,
            previous_games=previous_games if method == "Cobertura/diversidade" else None,
        )
        st.session_state.lotofacil_seed = seed
        st.session_state.lotofacil_last_generation_method = method
        st.session_state.lotofacil_games_ticket_size = ticket_size

    games = st.session_state.get("lotofacil_games", [])
    if not games:
        st.info("Escolha os parâmetros na lateral e gere suas apostas. Os filtros podem reduzir a quantidade de combinações possíveis.")
        return

    saved_ticket_size = len(games[0])
    if saved_ticket_size != ticket_size:
        st.warning(f"A carteira atual usa {saved_ticket_size} dezenas por aposta. Gere novamente para aplicar o tamanho selecionado ({ticket_size}).")
    odds = lotofacil_ticket_odds(saved_ticket_size)
    combinations = comb(saved_ticket_size, 15)
    portfolio = estimate_lotofacil_portfolio(games)
    if len(games) < game_count:
        st.warning(f"Os filtros produziram {len(games)} de {game_count} apostas. Desmarque algum padrão para ampliar a carteira.")
    metric_columns = st.columns(5)
    metric_columns[0].metric("Apostas", len(games))
    metric_columns[1].metric("Cobertura", portfolio["covered_numbers"])
    metric_columns[2].metric("Sobreposição média", round(portfolio["average_overlap"], 1))
    metric_columns[3].metric("Média do melhor acerto", round(portfolio["average_best_hits"], 2))
    metric_columns[4].metric("Carteira 11+", _format_percent(portfolio["tiers"][11]))
    st.caption(
        f"{saved_ticket_size} dezenas · {combinations:,} apostas simples combinadas · "
        f"custo oficial estimado por jogo: R$ {3.5 * combinations:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    )
    st.dataframe(
        pd.DataFrame(
            [
                {"Aposta": index, "Dezenas": " · ".join(game), "Quantidade": len(game)}
                for index, game in enumerate(games, start=1)
            ]
        ),
        hide_index=True,
        width="stretch",
    )
    st.caption(
        f"Chance teórica de 11 ou mais acertos para uma aposta: {_format_percent(odds[11])}. "
        "As chances são combinatórias e não representam previsão do próximo sorteio."
    )
    chance_rows = [
        {
            "Faixa": PRIZE_TIERS[tier],
            "A partir de": tier,
            "Uma aposta (exato)": _format_percent(odds[tier]),
            "Carteira (simulada)": _format_percent(portfolio["tiers"][tier]),
        }
        for tier in PRIZE_TIERS
    ]
    st.dataframe(pd.DataFrame(chance_rows), hide_index=True, width="stretch")
    if st.button("Limpar carteira Lotofácil", key="lotofacil_clear_games"):
        st.session_state.lotofacil_games = []
        st.rerun()
    st.download_button(
        "Baixar apostas",
        data="\n".join(";".join(game) for game in games),
        file_name="apostas_lotofacil.txt",
        mime="text/plain",
        key="lotofacil_download_games",
    )


def _render_evaluation(draws: pd.DataFrame, ticket_size: int) -> None:
    games = st.session_state.get("lotofacil_games", [])
    if not games:
        st.info("Gere apostas na aba Gerador para avaliá-las.")
        return

    latest = draws.iloc[-1]
    actual = set(latest[DRAW_COLUMNS])
    rows = []
    for index, game in enumerate(games, start=1):
        hits = len(set(game) & actual)
        rows.append(
            {
                "Aposta": index,
                "Acertos": hits,
                "Faixa": PRIZE_TIERS.get(hits, "Sem prêmio"),
                "Dezenas": " · ".join(game),
            }
        )
    st.subheader(f"Avaliação no concurso {int(latest['Concurso'])}")
    st.caption("Comparação retrospectiva com o último resultado oficial; não estima prêmio futuro.")
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    st.subheader("Chances combinatórias da aposta")
    ticket_size = len(games[0])
    odds = lotofacil_ticket_odds(ticket_size)
    denominator = comb(25, 15)
    odds_table = pd.DataFrame(
        [
            {
                "Faixa": PRIZE_TIERS[tier],
                "Acertos exatos": tier,
                "Probabilidade de pelo menos essa faixa": _format_percent(probability),
                "1 em": f"{(1 / probability):,.0f}".replace(",", ".") if probability else "—",
            }
            for tier, probability in odds.items()
        ]
    )
    st.dataframe(odds_table, hide_index=True, width="stretch")
    st.caption(f"Total de resultados possíveis: {denominator:,}".replace(",", "."))


def run_lotofacil_dashboard() -> None:
    st.title("Lotofácil · Análise e geração de apostas")
    st.sidebar.header("Controles da Lotofácil")
    ticket_size = st.sidebar.slider("Dezenas por aposta", 15, 20, 15, key="lotofacil_ticket_size")
    game_count = st.sidebar.slider("Quantidade de apostas", 1, 30, 5, key="lotofacil_game_count")
    st.sidebar.selectbox(
        "Critério de geração",
        [
            "Score equilibrado",
            "Frequência recente",
            "Mais atrasadas",
            "Pós-resultado",
            "Pontuação máxima",
            "Cobertura/diversidade",
            "Aprendizado",
            "Aleatório",
        ],
        key="lotofacil_method",
    )
    patterns = st.sidebar.multiselect(
        "Padrões de composição",
        PATTERN_OPTIONS,
        default=PATTERN_OPTIONS[:5],
        key="lotofacil_patterns",
    )
    st.sidebar.multiselect(
        "Dezenas fixas",
        NUMBERS,
        max_selections=ticket_size,
        key="lotofacil_selected_numbers",
    )
    simulations = st.sidebar.slider("Simulações Monte Carlo", 1000, 20000, 5000, 1000, key="lotofacil_simulations")
    population_size = st.sidebar.slider("População genética", 10, 100, 40, 10, key="lotofacil_population")
    generations = st.sidebar.slider("Gerações genéticas", 5, 100, 20, 5, key="lotofacil_generations")
    fitness_criteria = st.sidebar.selectbox(
        "Critério genético",
        ["score_total", "diversity", "coverage", "balanced"],
        key="lotofacil_fitness_criteria",
    )
    run_quick_models = st.sidebar.button("Treinar modelos rápido", key="lotofacil_train_quick")
    run_models = st.sidebar.button("Treinar modelos", key="lotofacil_train_models")
    run_montecarlo = st.sidebar.button("Executar Monte Carlo", key="lotofacil_run_montecarlo")
    run_backtest = st.sidebar.button("Executar backtest", key="lotofacil_run_backtest")
    run_genetic = st.sidebar.button("Executar busca genética", key="lotofacil_run_genetic")
    run_correlations = st.sidebar.button("Calcular correlações", key="lotofacil_run_correlations")
    st.sidebar.caption("Lotofácil: sorteio de 15 entre 25; apostas de 15 a 20 dezenas; prêmio de 11 a 15 acertos.")

    try:
        draws = load_lotofacil_history()
    except (OSError, ValueError) as exc:
        st.error(f"Não foi possível carregar o histórico oficial da Lotofácil: {exc}")
        st.info("Atualize a base com `python scripts/update_lotofacil_draws.py`.")
        return

    ranking = build_lotofacil_ranking(draws)
    learning_records = load_lotofacil_learning_records()
    learning_profile = build_lotofacil_learning_profile(learning_records)
    ranking = ranking.merge(learning_profile[["dezena", "ai_learning_score", "confianca"]], on="dezena", how="left")
    ranking[["ai_learning_score", "confianca"]] = ranking[["ai_learning_score", "confianca"]].fillna(0.0)
    if learning_records:
        learning_rank = ranking["ai_learning_score"].rank(pct=True)
        ranking["score_total"] = 0.85 * ranking["score_historico"] + 0.15 * learning_rank

    if run_quick_models or run_models:
        with st.spinner("Treinando modelos Lotofácil com validação temporal..."):
            model_report, model_metrics = train_lotofacil_models(draws, quick=run_quick_models)
        st.session_state.lotofacil_model_report = model_report
        st.session_state.lotofacil_model_metrics = model_metrics
    model_report = st.session_state.get("lotofacil_model_report")
    model_metrics = st.session_state.get("lotofacil_model_metrics")

    if run_montecarlo:
        with st.spinner("Executando simulações ponderadas da Lotofácil..."):
            st.session_state.lotofacil_montecarlo = run_lotofacil_montecarlo(
                ranking,
                games=st.session_state.get("lotofacil_games", []),
                simulations=simulations,
                seed=int(st.session_state.get("lotofacil_seed", 42)),
            )
    if run_backtest:
        with st.spinner("Executando backtest walk-forward..."):
            st.session_state.lotofacil_backtest = run_lotofacil_backtest(draws)
    if run_genetic:
        genetic_features = ranking[["dezena", "score_total"]].rename(columns={"dezena": "number"})
        with st.spinner("Evoluindo apostas Lotofácil de 15 dezenas..."):
            st.session_state.lotofacil_genetic_report = GeneticGameOptimizer(
                draws,
                genetic_features,
                numbers=NUMBERS,
                game_size=ticket_size,
            ).run_evolutionary_search(
                population_size=population_size,
                generations=generations,
                fitness_criteria=fitness_criteria,
            )
    if run_correlations or st.session_state.get("lotofacil_correlation_pairs") is None:
        pairs, lift = build_lotofacil_correlations(draws)
        st.session_state.lotofacil_correlation_pairs = pairs
        st.session_state.lotofacil_correlation_lift = lift

    st.caption(f"Histórico oficial carregado: {len(draws):,} concursos".replace(",", "."))
    st.sidebar.download_button(
        "Exportar histórico CSV",
        data=draws.to_csv(index=False),
        file_name="historico_lotofacil.csv",
        mime="text/csv",
        key="lotofacil_export_history",
    )
    overview_tab, ranking_tab, movement_tab, predictions_tab, models_tab, generator_tab, learning_tab, montecarlo_tab, backtest_tab, genetic_tab, correlations_tab = st.tabs(
        [
            "Resumo",
            "Ranking",
            "Movimentação",
            "Previsões",
            "Modelos",
            "Gerador",
            "IA Aprendizado",
            "Monte Carlo",
            "Backtest",
            "Genético",
            "Correlações",
        ]
    )
    with overview_tab:
        _render_overview(draws, ranking)
    with ranking_tab:
        _render_ranking(ranking)
    with movement_tab:
        _render_movement(draws, ranking)
    with predictions_tab:
        _render_predictions(ranking, model_report)
    with models_tab:
        _render_models(model_report, model_metrics)
    with generator_tab:
        _render_generator(ranking, draws, ticket_size, game_count, patterns)
    with learning_tab:
        _render_evaluation(draws, ticket_size)
        _render_ai_learning(draws, st.session_state.get("lotofacil_games", []))
    with montecarlo_tab:
        _render_montecarlo(st.session_state.get("lotofacil_montecarlo"))
    with backtest_tab:
        _render_backtest(st.session_state.get("lotofacil_backtest"))
    with genetic_tab:
        _render_genetic(st.session_state.get("lotofacil_genetic_report"), ticket_size)
    with correlations_tab:
        _render_correlations(
            st.session_state.lotofacil_correlation_pairs,
            st.session_state.lotofacil_correlation_lift,
        )