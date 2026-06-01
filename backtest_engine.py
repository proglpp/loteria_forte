import logging
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from config import DRAW_COLUMNS, NUMBER_RANGE

logger = logging.getLogger(__name__)


def _binary_targets(draw: pd.Series) -> Dict[str, int]:
    return {f"{n:02d}": int(f"{n:02d}" in draw[DRAW_COLUMNS].tolist()) for n in NUMBER_RANGE}


class BacktestEngine:
    def __init__(self, draws: pd.DataFrame, features: pd.DataFrame):
        self.draws = draws.reset_index(drop=True)
        self.features = features

    def walk_forward_validation(self) -> pd.DataFrame:
        logger.info("Starting walk-forward backtest")
        records = []
        numbers = [f"{n:02d}" for n in NUMBER_RANGE]
        for split in range(10, len(self.draws) - 1):
            train = self.draws.iloc[:split]
            test = self.draws.iloc[split]
            freq = pd.Series(train[DRAW_COLUMNS].values.flatten()).value_counts().reindex(numbers, fill_value=0)
            predicted = list(freq.nlargest(20).index)
            actual = set(test[DRAW_COLUMNS].tolist())
            hits = len(set(predicted).intersection(actual))
            precision = hits / 20.0
            recall = hits / 20.0
            f1 = 2 * precision * recall / max(precision + recall, 1e-12)
            expose = np.array([1.0 if number in predicted else 0.0 for number in numbers])
            target = np.array([1.0 if number in actual else 0.0 for number in numbers])
            logloss = log_loss(target, np.clip(expose, 1e-12, 1 - 1e-12))
            expected_value = precision - (20 - hits) / 80.0
            records.append(
                {
                    "split": split,
                    "hits": hits,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "log_loss": logloss,
                    "roi_theoretical": expected_value,
                }
            )
        return pd.DataFrame(records)
