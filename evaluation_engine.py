import logging
from typing import Any, Dict, Optional

import pandas as pd

from config import DRAW_COLUMNS

logger = logging.getLogger(__name__)


def _normalize_actual(draw_row: pd.Series) -> set[str]:
    return {str(draw_row[col]).zfill(2) for col in DRAW_COLUMNS}


def evaluate_predictions(
    actual_numbers: set[str],
    report: pd.DataFrame,
    prob_column: str = "prob_final",
    top_ns: Optional[list[int]] = None,
) -> Dict[str, Any]:
    if top_ns is None:
        top_ns = [20, 50]

    df = report.copy()
    df["number"] = df["number"].astype(str).str.zfill(2)
    if prob_column not in df.columns:
        prob_column = "prob_ensemble" if "prob_ensemble" in df.columns else "prob_mean"
    ranked = df.sort_values(prob_column, ascending=False).reset_index(drop=True)
    ranked["rank"] = ranked.index + 1

    per_number = []
    for number in sorted(actual_numbers, key=int):
        row = ranked[ranked["number"] == number]
        if row.empty:
            continue
        rank = int(row["rank"].iloc[0])
        prob = float(row[prob_column].iloc[0])
        if rank <= 20:
            band = "strong"
        elif rank <= 50:
            band = "medium"
        else:
            band = "weak"
        per_number.append({"number": number, "rank": rank, "prob": prob, "band": band})

    top_slices = {}
    for n in top_ns:
        top = set(ranked.head(n)["number"])
        hits = sorted(top & actual_numbers, key=int)
        top_slices[str(n)] = {
            "hits": len(hits),
            "hit_numbers": hits,
            "false_positives": sorted(top - actual_numbers, key=int),
            "missed": sorted(actual_numbers - top, key=int),
        }

    ranks = [item["rank"] for item in per_number]
    summary = {
        "prob_column": prob_column,
        "actual_count": len(actual_numbers),
        "avg_rank": float(sum(ranks) / len(ranks)) if ranks else None,
        "median_rank": float(sorted(ranks)[len(ranks) // 2]) if ranks else None,
        "per_number": per_number,
        "top_slices": top_slices,
    }
    return summary


def evaluate_last_draw(
    draws: pd.DataFrame,
    report: pd.DataFrame,
    prob_column: str = "prob_final",
) -> Dict[str, Any]:
    if len(draws) < 1:
        raise ValueError("Need at least one draw to evaluate")
    last_row = draws.iloc[-1]
    concurso = int(last_row["Concurso"]) if "Concurso" in draws.columns else len(draws)
    actual = _normalize_actual(last_row)
    evaluation = evaluate_predictions(actual, report, prob_column=prob_column)
    evaluation["concurso"] = concurso
    evaluation["draw_date"] = str(last_row.get("Data Sorteio", ""))
    logger.info(
        "Last draw evaluation concurso %s: %s/%s hits in top-20 (%s)",
        concurso,
        evaluation["top_slices"].get("20", {}).get("hits", 0),
        len(actual),
        prob_column,
    )
    return evaluation
