import logging
from collections import defaultdict
from typing import Dict, List

import pandas as pd

from config import DRAW_COLUMNS, NUMBER_RANGE

logger = logging.getLogger(__name__)


def _build_presence_matrix(draws: pd.DataFrame) -> pd.DataFrame:
    numbers = [f"{n:02d}" for n in NUMBER_RANGE]
    records = []
    for _, row in draws.iterrows():
        presence = {number: int(number in row[DRAW_COLUMNS].tolist()) for number in numbers}
        records.append(presence)
    return pd.DataFrame(records)


class MarkovEngine:
    def __init__(self, draws: pd.DataFrame):
        self.draws = draws.reset_index(drop=True)
        self.presence = _build_presence_matrix(self.draws)

    def _build_order(self, order: int) -> pd.DataFrame:
        logger.info("Building Markov order %d model", order)
        transitions = defaultdict(lambda: defaultdict(int))
        totals = defaultdict(int)
        n_rows = len(self.presence)
        for idx in range(order, n_rows):
            previous = self.presence.iloc[idx - order : idx].sum(axis=0)
            current = self.presence.iloc[idx]
            for number in self.presence.columns:
                key = tuple((previous > 0).astype(int).tolist())
                transitions[number][key] += int(current[number])
                totals[number] += 1
        scores = []
        for number in self.presence.columns:
            probability = sum(transitions[number].values()) / max(totals[number], 1)
            scores.append({"number": number, "score": probability})
        return pd.DataFrame(scores)

    def build_all_orders(self) -> Dict[str, pd.DataFrame]:
        return {
            "1": self._build_order(1),
            "2": self._build_order(2),
            "3": self._build_order(3),
        }
