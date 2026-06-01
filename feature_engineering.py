import logging
from collections import defaultdict
from typing import Dict, Optional

import numpy as np
import pandas as pd

from config import DRAW_COLUMNS, NUMBER_RANGE
from entropy_engine import compute_shannon_entropy

logger = logging.getLogger(__name__)

WINDOWS = [10, 25, 50, 100, 250, 500, 1000]


def build_feature_matrix(
    draws: pd.DataFrame,
    graph_metrics: Optional[pd.DataFrame] = None,
    markov_report: Optional[Dict[str, pd.DataFrame]] = None,
    hmm_report: Optional[Dict[str, pd.Series]] = None,
) -> pd.DataFrame:
    logger.info("Building feature matrix for Lotomania numbers")
    frequency = _calculate_frequency_windows(draws)
    delay_features = _calculate_delay_metrics(draws)
    position_features = _calculate_position_metrics(draws)
    entropy_features = _calculate_entropy_metrics(draws)
    trend_features = _calculate_frequency_trend(frequency)

    base = pd.DataFrame({"number": [f"{n:02d}" for n in NUMBER_RANGE]})
    for feature_df in (frequency, delay_features, position_features, entropy_features, trend_features):
        base = base.merge(feature_df, on="number", how="left")

    if graph_metrics is not None:
        base = base.merge(graph_metrics, on="number", how="left")

    if markov_report is not None:
        for order, df in markov_report.items():
            if "score" in df.columns:
                base = base.merge(
                    df[["number", "score"]].rename(columns={"score": f"markov_order_{order}_score"}),
                    on="number",
                    how="left",
                )

    if hmm_report is not None:
        for key, series in hmm_report.items():
            base[key] = base["number"].map(series.to_dict())

    base = base.fillna(0)
    numeric_cols = base.select_dtypes(include=["number"]).columns
    numeric_cols = numeric_cols.drop("number") if "number" in numeric_cols else numeric_cols
    base[numeric_cols] = base[numeric_cols].astype(float)
    base["score_total"] = base[numeric_cols].apply(
        lambda col: (col - col.min()) / (col.max() - col.min() + 1e-12)
    ).sum(axis=1)
    return base


def _calculate_frequency_windows(draws: pd.DataFrame) -> pd.DataFrame:
    counts = defaultdict(dict)
    for window in WINDOWS:
        recent = draws.tail(window)
        values = recent[DRAW_COLUMNS].values.flatten()
        freq = pd.Series(values).value_counts().reindex([f"{n:02d}" for n in NUMBER_RANGE], fill_value=0)
        counts[f"freq_{window}"] = freq.values

    global_freq = pd.Series(draws[DRAW_COLUMNS].values.flatten()).value_counts().reindex([f"{n:02d}" for n in NUMBER_RANGE], fill_value=0)
    df = pd.DataFrame({"number": [f"{n:02d}" for n in NUMBER_RANGE], "freq_global": global_freq.values})
    for window in WINDOWS:
        df[f"freq_{window}"] = counts[f"freq_{window}"]
    df[[f"freq_{window}_rel" for window in WINDOWS]] = df[[f"freq_{window}" for window in WINDOWS]].div(
        df[[f"freq_{window}" for window in WINDOWS]].sum(axis=0), axis=1
    ).fillna(0)
    df["freq_global_rel"] = df["freq_global"] / df["freq_global"].sum()
    return df


def _calculate_delay_metrics(draws: pd.DataFrame) -> pd.DataFrame:
    last_seen = {f"{n:02d}": None for n in NUMBER_RANGE}
    delays = defaultdict(list)
    for idx, row in draws.iterrows():
        present = set(row[DRAW_COLUMNS].tolist())
        for number in [f"{n:02d}" for n in NUMBER_RANGE]:
            if last_seen[number] is None:
                delays[number].append(idx)
            else:
                delays[number].append(idx - last_seen[number] - 1)
            if number in present:
                last_seen[number] = idx

    output = []
    for number, values in delays.items():
        output.append(
            {
                "number": number,
                "delay_mean": np.mean(values),
                "delay_median": np.median(values),
                "delay_std": np.std(values, ddof=1) if len(values) > 1 else 0.0,
                "delay_last": values[-1] if values else 0,
            }
        )
    return pd.DataFrame(output)


def _calculate_position_metrics(draws: pd.DataFrame) -> pd.DataFrame:
    positions = defaultdict(list)
    last_seen = {f"{n:02d}": -1 for n in NUMBER_RANGE}
    for idx, row in draws.iterrows():
        for pos, col in enumerate(DRAW_COLUMNS, start=1):
            number = row[col]
            positions[number].append(pos)
            last_seen[number] = idx

    output = []
    for number in [f"{n:02d}" for n in NUMBER_RANGE]:
        number_positions = positions[number]
        output.append(
            {
                "number": number,
                "position_mean": np.mean(number_positions) if number_positions else 0.0,
                "position_median": np.median(number_positions) if number_positions else 0.0,
                "position_std": np.std(number_positions, ddof=1) if len(number_positions) > 1 else 0.0,
                "position_entropy": compute_shannon_entropy([str(pos) for pos in number_positions]) if number_positions else 0.0,
                "appearance_rate": len(number_positions) / max(len(draws), 1),
                "recency": len(draws) - last_seen[number] if last_seen[number] >= 0 else len(draws),
            }
        )
    return pd.DataFrame(output)


def _calculate_entropy_metrics(draws: pd.DataFrame) -> pd.DataFrame:
    occurrences = defaultdict(list)
    for idx, row in draws.iterrows():
        for pos, col in enumerate(DRAW_COLUMNS, start=1):
            number = row[col]
            occurrences[number].append(idx)

    output = []
    for number in [f"{n:02d}" for n in NUMBER_RANGE]:
        indices = occurrences[number]
        if len(indices) > 1:
            distances = [indices[i] - indices[i - 1] for i in range(1, len(indices))]
            entropy = compute_shannon_entropy([str(distance) for distance in distances])
        else:
            entropy = 0.0
        output.append({"number": number, "appearance_entropy": entropy})
    return pd.DataFrame(output)


def _calculate_frequency_trend(frequency: pd.DataFrame) -> pd.DataFrame:
    df = frequency[["number", "freq_10", "freq_25", "freq_50", "freq_100"]].copy()
    df["freq_10_to_100"] = df["freq_10"] / (df["freq_100"] + 1e-12)
    df["freq_25_to_100"] = df["freq_25"] / (df["freq_100"] + 1e-12)
    return df[["number", "freq_10_to_100", "freq_25_to_100"]]
