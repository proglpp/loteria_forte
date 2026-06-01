import logging
from typing import Dict, List

import numpy as np
import pandas as pd

from config import DRAW_COLUMNS, NUMBER_RANGE

logger = logging.getLogger(__name__)
WINDOWS = [10, 25, 50, 100, 250, 500, 1000]


def build_statistics_report(draws: pd.DataFrame) -> pd.DataFrame:
    logger.info("Building statistics report")
    numbers = [f"{n:02d}" for n in NUMBER_RANGE]
    df = pd.DataFrame({"number": numbers})
    total_draws = len(draws)
    global_counts = draws[DRAW_COLUMNS].values.flatten()
    counts = pd.Series(global_counts).value_counts().reindex(numbers, fill_value=0)
    df["freq_abs"] = counts.values
    df["freq_rel"] = counts.values / counts.values.sum()
    df["freq_cum"] = np.cumsum(df["freq_abs"]).astype(int)
    df["freq_bayes"] = (df["freq_abs"] + 1) / (df["freq_abs"].sum() + len(numbers))
    df["adjusted_frequency"] = _adjust_by_decay(draws, numbers)
    df["freq_recent"] = _frequency_recent(draws, numbers)
    df = _categorize_numbers(df)
    return df


def _frequency_recent(draws: pd.DataFrame, numbers: List[str]) -> pd.Series:
    recent = draws.tail(25)
    values = recent[DRAW_COLUMNS].values.flatten()
    return pd.Series(values).value_counts().reindex(numbers, fill_value=0)


def _adjust_by_decay(draws: pd.DataFrame, numbers: List[str]) -> pd.Series:
    frequency = pd.Series(0.0, index=numbers)
    decay = 0.98
    for _, row in draws.iterrows():
        present = set(row[DRAW_COLUMNS].tolist())
        frequency.loc[list(present)] += 1.0
        frequency *= decay
    return frequency


def _categorize_numbers(df: pd.DataFrame) -> pd.DataFrame:
    percentiles = np.percentile(df["freq_abs"], [20, 40, 60, 80])
    bins = [0, percentiles[0], percentiles[1], percentiles[2], percentiles[3], df["freq_abs"].max() + 1]
    labels = ["FROZEN", "COLD", "NEUTRAL", "WARM", "HOT"]
    df["rank"] = pd.cut(df["freq_abs"], bins=bins, labels=labels, include_lowest=True)
    return df
