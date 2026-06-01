import logging
from typing import Dict, List

import pandas as pd

from config import DRAW_COLUMNS, NUMBER_RANGE

logger = logging.getLogger(__name__)
WINDOWS = [10, 25, 50, 100, 250, 500, 1000]


def compute_frequency_windows(draws: pd.DataFrame) -> pd.DataFrame:
    logger.info("Computing frequency windows")
    numbers = [f"{n:02d}" for n in NUMBER_RANGE]
    output = {"number": numbers}
    for window in WINDOWS:
        recent = draws.tail(window)
        counts = pd.Series(recent[DRAW_COLUMNS].values.flatten()).value_counts().reindex(numbers, fill_value=0)
        output[f"freq_{window}"] = counts.values
        output[f"freq_{window}_rel"] = counts.values / counts.values.sum()
    return pd.DataFrame(output)


def compute_adjusted_frequency(draws: pd.DataFrame) -> pd.Series:
    logger.info("Computing adjusted frequency with decay")
    decay = 0.985
    numbers = [f"{n:02d}" for n in NUMBER_RANGE]
    adjusted = pd.Series(0.0, index=numbers)
    for _, row in draws.iterrows():
        present = set(row[DRAW_COLUMNS].tolist())
        adjusted.loc[list(present)] += 1.0
        adjusted *= decay
    return adjusted
