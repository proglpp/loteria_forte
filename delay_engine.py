import logging
from collections import defaultdict
from typing import List

import numpy as np
import pandas as pd

from config import DRAW_COLUMNS, NUMBER_RANGE

logger = logging.getLogger(__name__)


def compute_delay_metrics(draws: pd.DataFrame) -> pd.DataFrame:
    logger.info("Computing draw delay metrics")
    last_seen = {f"{n:02d}": None for n in NUMBER_RANGE}
    metrics = []
    draws = draws.reset_index(drop=True)
    for number in [f"{n:02d}" for n in NUMBER_RANGE]:
        gaps = []
        for idx, row in draws.iterrows():
            if number in row[DRAW_COLUMNS].tolist():
                if last_seen[number] is not None:
                    gaps.append(idx - last_seen[number] - 1)
                last_seen[number] = idx
        metrics.append(
            {
                "number": number,
                "current_delay": (len(draws) - 1 - last_seen[number]) if last_seen[number] is not None else len(draws),
                "max_delay": int(max(gaps)) if gaps else 0,
                "median_delay": int(np.median(gaps)) if gaps else 0,
                "delay_std": float(np.std(gaps, ddof=1)) if len(gaps) > 1 else 0.0,
            }
        )
    return pd.DataFrame(metrics)
