"""Feedback learning helpers for generated Lotomania games."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from config import ROOT_DIR

LEARNING_FILE = ROOT_DIR / "database" / "ai_learning_feedback.json"


DEFAULT_EVAL_DRAW = [
    "00", "01", "05", "12", "14", "21", "27", "33", "47", "55",
    "63", "66", "67", "68", "69", "70", "73", "77", "82", "91",
]


def normalize_numbers(values: Iterable[str | int] | str, limit: int | None = None) -> list[str]:
    if isinstance(values, str):
        values = values.replace("-", " ").replace(",", " ").replace(";", " ").split()

    normalized = []
    for value in values or []:
        try:
            number = int(str(value).strip())
        except (TypeError, ValueError):
            continue
        if 0 <= number <= 99:
            normalized.append(f"{number:02d}")

    unique = list(dict.fromkeys(normalized))
    return unique[:limit] if limit is not None else unique


def load_learning_records(path: Path = LEARNING_FILE) -> list[dict]:
    if not path.exists():
        return []
    try:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        return payload if isinstance(payload, list) else []
    except (OSError, json.JSONDecodeError):
        return []


def save_learning_records(records: list[dict], path: Path = LEARNING_FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=False, indent=2)


def evaluate_games_against_draw(games: Iterable[Iterable[str | int]], draw_numbers: Iterable[str | int] | str) -> pd.DataFrame:
    actual = set(normalize_numbers(draw_numbers, limit=20))
    rows = []
    for index, game in enumerate(games or [], start=1):
        game_numbers = set(normalize_numbers(game, limit=50))
        hits = sorted(game_numbers & actual)
        missed_draw_numbers = sorted(actual - game_numbers)
        chosen_not_drawn = sorted(game_numbers - actual)
        rows.append(
            {
                "jogo": index,
                "acertos": len(hits),
                "erros_no_jogo": len(chosen_not_drawn),
                "sorteados_fora": len(missed_draw_numbers),
                "numeros_acertados": ", ".join(hits),
                "sorteados_fora_do_jogo": ", ".join(missed_draw_numbers),
            }
        )
    return pd.DataFrame(rows)


def record_learning_feedback(
    games: Iterable[Iterable[str | int]],
    draw_numbers: Iterable[str | int] | str,
    draw_date: str,
    method: str,
    path: Path = LEARNING_FILE,
) -> list[dict]:
    records = load_learning_records(path)
    actual = normalize_numbers(draw_numbers, limit=20)
    saved_at = datetime.now().isoformat(timespec="seconds")
    seen = {
        (
            record.get("draw_date"),
            record.get("method"),
            int(record.get("game_index", 0)),
            tuple(normalize_numbers(record.get("actual_draw", []), limit=20)),
            tuple(normalize_numbers(record.get("game_numbers", []), limit=50)),
        )
        for record in records
    }

    for index, game in enumerate(games or [], start=1):
        game_numbers = normalize_numbers(game, limit=50)
        signature = (draw_date, method, index, tuple(actual), tuple(game_numbers))
        if signature in seen:
            continue
        hits = sorted(set(game_numbers) & set(actual))
        records.append(
            {
                "saved_at": saved_at,
                "draw_date": draw_date,
                "method": method,
                "game_index": index,
                "actual_draw": actual,
                "game_numbers": game_numbers,
                "hits": hits,
                "hit_count": len(hits),
            }
        )
        seen.add(signature)

    save_learning_records(records, path)
    return records


def build_learning_profile(records: list[dict]) -> pd.DataFrame:
    numbers = [f"{number:02d}" for number in range(100)]
    if not records:
        records = []
    latest_draw_date = max((str(record.get("draw_date", "")) for record in records), default="")
    total_weight = 0.0
    stats = {
        number: {
            "number": number,
            "vezes_escolhido": 0.0,
            "acertos_quando_escolhido": 0.0,
            "erros_quando_escolhido": 0.0,
            "sorteado_fora_do_jogo": 0.0,
            "vezes_sorteado": 0.0,
            "sorteado_no_resultado_mais_recente": 0.0,
        }
        for number in numbers
    }

    for index, record in enumerate(records):
        game = set(normalize_numbers(record.get("game_numbers", []), limit=50))
        actual = set(normalize_numbers(record.get("actual_draw", []), limit=20))
        draw_date = str(record.get("draw_date", ""))
        chronological_weight = 0.55 + (0.45 * ((index + 1) / max(len(records), 1)))
        latest_weight = 1.85 if draw_date == latest_draw_date else 1.0
        record_weight = chronological_weight * latest_weight
        total_weight += record_weight
        for number in numbers:
            chosen = number in game
            drawn = number in actual
            if drawn:
                stats[number]["vezes_sorteado"] += record_weight
                if draw_date == latest_draw_date:
                    stats[number]["sorteado_no_resultado_mais_recente"] = 1.0
            if chosen:
                stats[number]["vezes_escolhido"] += record_weight
                if drawn:
                    stats[number]["acertos_quando_escolhido"] += record_weight
                else:
                    stats[number]["erros_quando_escolhido"] += record_weight
            elif drawn:
                stats[number]["sorteado_fora_do_jogo"] += record_weight

    df = pd.DataFrame(stats.values())
    total_records = max(float(total_weight), 1.0)
    chosen = df["vezes_escolhido"].astype(float)
    drawn = df["vezes_sorteado"].astype(float)
    hits = df["acertos_quando_escolhido"].astype(float)
    errors = df["erros_quando_escolhido"].astype(float)
    missed = df["sorteado_fora_do_jogo"].astype(float)

    baseline_hit_rate = 20 / 100
    baseline_pick_rate = 50 / 100
    chosen_rate = chosen / total_records
    df["taxa_acerto_escolhido"] = ((df["acertos_quando_escolhido"] + 1.0) / (chosen + 2.0)).where(
        chosen > 0,
        baseline_hit_rate,
    )
    df["taxa_erro_escolhido"] = ((df["erros_quando_escolhido"] + 1.0) / (chosen + 2.0)).where(
        chosen > 0,
        1 - baseline_hit_rate,
    )
    df["taxa_escolha"] = chosen_rate
    df["taxa_sorteio_feedback"] = drawn / total_records
    df["taxa_cobertura_quando_sorteado"] = ((hits + 1.0) / (drawn + 2.0)).where(drawn > 0, baseline_pick_rate)
    df["taxa_erro_repetido"] = ((errors + 1.0) / (chosen + 2.0)).where(chosen > 0, baseline_pick_rate)
    df["pressao_de_resgate"] = ((missed + 1.0) / (drawn + 2.0)).where(drawn > 0, 0.0)

    hit_lift = df["taxa_acerto_escolhido"] - baseline_hit_rate
    coverage_gap = (1.0 - df["taxa_cobertura_quando_sorteado"]) * (df["taxa_sorteio_feedback"] > 0).astype(float)
    false_positive_pressure = df["taxa_erro_repetido"] * (df["taxa_sorteio_feedback"] == 0).astype(float)
    overexposure = (df["taxa_escolha"] - baseline_pick_rate).clip(lower=0.0)
    recent_draw_boost = df["sorteado_no_resultado_mais_recente"].astype(float)
    sample_confidence = np.sqrt(np.maximum(chosen + drawn, 1.0)) / np.sqrt(total_records * 2.0)

    df["ai_learning_confidence"] = sample_confidence.clip(0.0, 1.0)
    df["ai_learning_score"] = (
        0.58 * hit_lift
        + 0.34 * df["pressao_de_resgate"]
        + 0.28 * coverage_gap
        + 0.22 * recent_draw_boost
        - 0.30 * false_positive_pressure
        - 0.14 * overexposure
    ) * (0.55 + 0.45 * df["ai_learning_confidence"])
    return df.sort_values("ai_learning_score", ascending=False).reset_index(drop=True)


def apply_learning_to_features(features: pd.DataFrame, records: list[dict]) -> pd.DataFrame:
    if features is None or features.empty or not records:
        return features

    profile_cols = [
        "number",
        "ai_learning_score",
        "ai_learning_confidence",
        "taxa_acerto_escolhido",
        "taxa_erro_repetido",
        "pressao_de_resgate",
        "sorteado_no_resultado_mais_recente",
    ]
    profile = build_learning_profile(records).reindex(columns=profile_cols, fill_value=0.0)
    profile["number"] = profile["number"].astype(str).str.zfill(2)
    adjusted = features.copy()
    adjusted["number"] = adjusted["number"].astype(str).str.zfill(2)

    learned_cols = [column for column in profile_cols if column != "number"]
    adjusted = adjusted.drop(columns=[column for column in learned_cols if column in adjusted.columns])
    adjusted = adjusted.merge(profile, on="number", how="left")
    for column in learned_cols:
        adjusted[column] = adjusted[column].fillna(0.0)
    return adjusted
