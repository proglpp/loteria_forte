"""Aggressive portfolio optimizer for high-hit Lotomania attempts.

This module deliberately optimizes for a high-hit portfolio target instead of
maximizing broad coverage. Lottery draws remain random, so this improves the
selection discipline without guaranteeing prizes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from ai_learning_engine import build_learning_profile, normalize_numbers
from config import DRAW_COLUMNS


NUMBERS = [f"{number:02d}" for number in range(100)]


@dataclass(frozen=True)
class EliteProfile:
    name: str
    weights: dict[str, float]
    avg_hits: float
    target_hits: int
    target_rate: float
    near_rate: float
    hit_19_rate: float
    hit_18_rate: float
    hit_15_rate: float
    hit_14_rate: float
    max_hits: int


WEIGHT_CANDIDATES = [
    {"freq_25": 0.26, "freq_50": 0.18, "freq_100": 0.12, "decay": 0.14, "delay_mid": 0.08, "last": 0.03, "pattern": 0.13, "learning": 0.06},
    {"freq_25": 0.18, "freq_50": 0.22, "freq_100": 0.16, "decay": 0.12, "delay_mid": 0.10, "last": 0.02, "pattern": 0.12, "learning": 0.08},
    {"freq_10": 0.10, "freq_25": 0.21, "freq_50": 0.18, "decay": 0.15, "delay_mid": 0.10, "last": 0.03, "pattern": 0.13, "learning": 0.10},
    {"freq_25": 0.15, "freq_50": 0.16, "freq_100": 0.16, "freq_250": 0.10, "decay": 0.12, "delay_long": 0.10, "pattern": 0.13, "learning": 0.08},
    {"freq_10": 0.07, "freq_25": 0.16, "freq_50": 0.22, "freq_100": 0.16, "decay": 0.10, "delay_mid": 0.08, "last": 0.02, "pattern": 0.11, "learning": 0.08},
]


def _minmax(values: pd.Series, neutral: float = 0.5) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").fillna(0.0).astype(float)
    spread = numeric.max() - numeric.min()
    if abs(float(spread)) < 1e-12:
        return pd.Series(np.full(len(numeric), neutral), index=numeric.index)
    return (numeric - numeric.min()) / (spread + 1e-12)


def _draw_set(row: pd.Series) -> set[str]:
    return {str(row[col]).zfill(2) for col in DRAW_COLUMNS}


def _decay_counts(draws: pd.DataFrame, decay: float = 0.975) -> pd.Series:
    counts = pd.Series(0.0, index=NUMBERS)
    for _, row in draws.iterrows():
        counts *= decay
        present = list(_draw_set(row))
        counts.loc[present] += 1.0
    return counts.reindex(NUMBERS, fill_value=0.0)


def build_cycle_pattern_features(draws: pd.DataFrame) -> pd.DataFrame:
    history = [_draw_set(row) for _, row in draws.reset_index(drop=True).iterrows()]
    rows = []
    for number in NUMBERS:
        flags = [number in draw_set for draw_set in history]
        current_hit_streak = 0
        current_miss_streak = 0
        for flag in reversed(flags):
            if flag:
                if current_miss_streak:
                    break
                current_hit_streak += 1
            else:
                if current_hit_streak:
                    break
                current_miss_streak += 1

        repeat_events = []
        return_events_by_delay: dict[int, list[int]] = {}
        miss_streak = 0
        for index in range(1, len(flags)):
            if flags[index - 1]:
                repeat_events.append(1 if flags[index] else 0)
            previous_delay = min(miss_streak, 12)
            return_events_by_delay.setdefault(previous_delay, []).append(1 if flags[index] else 0)
            miss_streak = 0 if flags[index] else miss_streak + 1

        delay_bucket = min(current_miss_streak, 12)
        bucket_events = return_events_by_delay.get(delay_bucket, [])
        nearby_events = []
        for bucket in range(max(0, delay_bucket - 1), min(12, delay_bucket + 1) + 1):
            nearby_events.extend(return_events_by_delay.get(bucket, []))

        repeat_rate = float(np.mean(repeat_events)) if repeat_events else 0.20
        return_rate = float(np.mean(bucket_events or nearby_events)) if (bucket_events or nearby_events) else 0.20
        miss_pressure = min(current_miss_streak / 12.0, 1.0)
        hit_exhaustion = min(max(current_hit_streak - 1, 0) / 4.0, 1.0)
        repeat_signal = repeat_rate if flags[-1:] == [True] else 0.0
        return_signal = return_rate * miss_pressure if current_miss_streak else 0.0
        exhaustion_penalty = hit_exhaustion * (1.0 - repeat_rate)
        pattern_score = (0.52 * return_signal) + (0.30 * repeat_signal) + (0.18 * miss_pressure) - (0.32 * exhaustion_penalty)

        rows.append(
            {
                "number": number,
                "current_hit_streak": int(current_hit_streak),
                "current_miss_streak": int(current_miss_streak),
                "repeat_next_rate": repeat_rate,
                "return_after_delay_rate": return_rate,
                "return_signal": return_signal,
                "repeat_signal": repeat_signal,
                "pattern_score": pattern_score,
            }
        )
    return pd.DataFrame(rows)


def build_elite_scores(
    draws: pd.DataFrame,
    learning_records: list[dict] | None = None,
    weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    weights = weights or WEIGHT_CANDIDATES[0]
    history = draws.reset_index(drop=True)
    rows = pd.DataFrame({"number": NUMBERS})

    for window in (10, 25, 50, 100, 250):
        values = history.tail(window)[DRAW_COLUMNS].values.flatten()
        counts = pd.Series(values).astype(str).str.zfill(2).value_counts().reindex(NUMBERS, fill_value=0)
        rows[f"freq_{window}"] = counts.values.astype(float)

    all_values = history[DRAW_COLUMNS].values.flatten()
    rows["freq_all"] = pd.Series(all_values).astype(str).str.zfill(2).value_counts().reindex(NUMBERS, fill_value=0).values.astype(float)
    rows["decay"] = _decay_counts(history).values

    last_seen = {number: -1 for number in NUMBERS}
    for idx, row in history.iterrows():
        for number in _draw_set(row):
            last_seen[number] = int(idx)
    rows["delay"] = rows["number"].map(lambda number: len(history) - last_seen[number] if last_seen[number] >= 0 else len(history))
    rows["delay_mid"] = 1.0 - ((_minmax((rows["delay"] - 7).abs(), neutral=0.0)).clip(0.0, 1.0))
    rows["delay_long"] = _minmax(rows["delay"])

    last_draw = _draw_set(history.iloc[-1]) if not history.empty else set()
    rows["last"] = rows["number"].isin(last_draw).astype(float)
    pattern_features = build_cycle_pattern_features(history)
    rows = rows.merge(pattern_features, on="number", how="left")

    if learning_records:
        profile = build_learning_profile(learning_records)[["number", "ai_learning_score", "pressao_de_resgate"]]
        rows = rows.merge(profile, on="number", how="left")
    else:
        rows["ai_learning_score"] = 0.0
        rows["pressao_de_resgate"] = 0.0
    rows[["ai_learning_score", "pressao_de_resgate"]] = rows[["ai_learning_score", "pressao_de_resgate"]].fillna(0.0)

    score = pd.Series(0.0, index=rows.index)
    for column, weight in weights.items():
        if column == "learning":
            score += float(weight) * _minmax(rows["ai_learning_score"] + (0.35 * rows["pressao_de_resgate"]))
        elif column == "pattern":
            score += float(weight) * _minmax(rows["pattern_score"])
        elif column in rows.columns:
            score += float(weight) * _minmax(rows[column])
    rows["elite_score"] = score
    rows["tier"] = pd.qcut(rows["elite_score"].rank(method="first"), q=5, labels=["E", "D", "C", "B", "A"])
    return rows.sort_values("elite_score", ascending=False).reset_index(drop=True)


def tune_elite_profile(
    draws: pd.DataFrame,
    learning_records: list[dict] | None = None,
    validation_window: int = 60,
    ticket_size: int = 50,
    target_hits: int = 15,
) -> EliteProfile:
    history = draws.reset_index(drop=True)
    start = max(80, len(history) - int(validation_window))
    best: EliteProfile | None = None

    for index, weights in enumerate(WEIGHT_CANDIDATES, start=1):
        train = history.iloc[:start]
        ranked = set(build_elite_scores(train, learning_records, weights)["number"].head(ticket_size))
        hits = [len(ranked & _draw_set(history.iloc[split])) for split in range(start, len(history))]
        if not hits:
            continue
        hit_series = pd.Series(hits)
        profile = EliteProfile(
            name=f"elite_{index}",
            weights=weights,
            avg_hits=float(hit_series.mean()),
            target_hits=int(target_hits),
            target_rate=float((hit_series >= target_hits).mean()),
            near_rate=float((hit_series >= max(target_hits - 1, 0)).mean()),
            hit_19_rate=float((hit_series >= 19).mean()),
            hit_18_rate=float((hit_series >= 18).mean()),
            hit_15_rate=float((hit_series >= 15).mean()),
            hit_14_rate=float((hit_series >= 14).mean()),
            max_hits=int(hit_series.max()),
        )
        current_score = (
            profile.target_rate * 2000.0
            + profile.near_rate * 420.0
            + profile.hit_18_rate * 180.0
            + profile.hit_15_rate * 28.0
            + profile.avg_hits
            + (profile.max_hits * 0.75)
        )
        best_score = -1.0
        if best is not None:
            best_score = (
                best.target_rate * 2000.0
                + best.near_rate * 420.0
                + best.hit_18_rate * 180.0
                + best.hit_15_rate * 28.0
                + best.avg_hits
                + (best.max_hits * 0.75)
            )
        if best is None or current_score > best_score:
            best = profile

    return best or EliteProfile("elite_default", WEIGHT_CANDIDATES[0], 0.0, target_hits, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)


def _ticket_quality(game: Iterable[str], scores: pd.DataFrame) -> float:
    score_map = scores.set_index("number")["elite_score"].to_dict()
    numbers = list(game)
    return float(sum(score_map.get(number, 0.0) for number in numbers))


def generate_elite_portfolio(
    draws: pd.DataFrame,
    learning_records: list[dict] | None = None,
    n_games: int = 8,
    ticket_size: int = 50,
    seed: int = 8609,
    target_hits: int = 15,
) -> tuple[list[list[str]], EliteProfile, pd.DataFrame]:
    profile = tune_elite_profile(draws, learning_records, target_hits=target_hits)
    scores = build_elite_scores(draws, learning_records, profile.weights)
    ranked = scores["number"].tolist()
    score_map = scores.set_index("number")["elite_score"].to_dict()

    if target_hits >= 19:
        core_size = 0
        fill_threshold = 34
        pool_size = 100
        usage_strength = 0.160
        noise_strength = 0.012
    else:
        core_size = 16
        fill_threshold = 42
        pool_size = 72
        usage_strength = 0.08
        noise_strength = 0.008

    core = ranked[:core_size]
    elite_pool = ranked[:pool_size]
    reserve_pool = ranked[pool_size:100]
    rng = np.random.default_rng(seed)
    usage = {number: 0 for number in NUMBERS}
    games: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()

    for game_index in range(int(n_games)):
        chosen = list(core)
        offset = (game_index * 7) % len(elite_pool)
        rotated = elite_pool[offset:] + elite_pool[:offset]
        for number in rotated:
            if len(chosen) >= fill_threshold:
                break
            if number not in chosen:
                chosen.append(number)

        candidates = [number for number in elite_pool + reserve_pool if number not in chosen]
        while len(chosen) < ticket_size and candidates:
            parity_even = sum(int(number) % 2 == 0 for number in chosen)
            decade_counts = pd.Series([int(number) // 10 for number in chosen]).value_counts().to_dict()

            def value(number: str) -> float:
                decade = int(number) // 10
                parity_penalty = 0.06 if (parity_even >= 27 and int(number) % 2 == 0) or (parity_even <= 23 and int(number) % 2 == 1) else 0.0
                decade_penalty = max(0, decade_counts.get(decade, 0) - 5) * 0.045
                usage_penalty = usage.get(number, 0) * usage_strength
                noise = rng.normal(0.0, noise_strength)
                return score_map.get(number, 0.0) - parity_penalty - decade_penalty - usage_penalty + noise

            best_number = max(candidates, key=value)
            chosen.append(best_number)
            candidates.remove(best_number)

        game = sorted(chosen[:ticket_size])
        attempts = 0
        while tuple(game) in seen and attempts < 30:
            attempts += 1
            removable = [number for number in game if number not in core]
            incoming = [number for number in ranked if number not in game]
            if not removable or not incoming:
                break
            out_number = min(removable, key=lambda number: score_map.get(number, 0.0) - (usage.get(number, 0) * 0.05))
            in_number = incoming[(attempts + game_index) % len(incoming)]
            game = sorted([number for number in game if number != out_number] + [in_number])

        seen.add(tuple(game))
        games.append(game)
        for number in game:
            usage[number] += 1

    if target_hits >= 19:
        missing_numbers = [number for number in NUMBERS if usage.get(number, 0) == 0]
        for missing in missing_numbers:
            best_game_index = None
            best_out_number = None
            best_value = -1e9
            for game_index, game in enumerate(games):
                game_set = set(game)
                if missing in game_set:
                    continue
                removable = [number for number in game if usage.get(number, 0) > 1]
                if not removable:
                    continue
                out_number = min(removable, key=lambda number: score_map.get(number, 0.0) - (usage.get(number, 0) * 0.06))
                value = usage.get(out_number, 0) - score_map.get(out_number, 0.0) + score_map.get(missing, 0.0)
                if value > best_value:
                    best_value = value
                    best_game_index = game_index
                    best_out_number = out_number
            if best_game_index is None or best_out_number is None:
                continue
            games[best_game_index] = sorted([number for number in games[best_game_index] if number != best_out_number] + [missing])
            usage[best_out_number] -= 1
            usage[missing] = usage.get(missing, 0) + 1

    games.sort(key=lambda game: _ticket_quality(game, scores), reverse=True)
    return games, profile, scores


def evaluate_portfolio_on_draw(games: Iterable[Iterable[str | int]], draw_numbers: Iterable[str | int] | str) -> dict[str, float]:
    actual = set(normalize_numbers(draw_numbers, limit=20))
    hits = [len(set(normalize_numbers(game, limit=50)) & actual) for game in games]
    if not hits:
        return {"best": 0, "avg": 0.0, "count_15_plus": 0}
    return {
        "best": int(max(hits)),
        "avg": float(np.mean(hits)),
        "count_15_plus": int(sum(hit >= 15 for hit in hits)),
    }
