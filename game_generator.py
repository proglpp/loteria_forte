import logging
from typing import List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _build_game(numbers: List[str]) -> dict:
    return {
        "numbers": sorted(numbers),
        "count": len(numbers),
        "sum": sum(int(n) for n in numbers),
        "evens": sum(1 for n in numbers if int(n) % 2 == 0),
        "odds": sum(1 for n in numbers if int(n) % 2 != 0),
        "primes": sum(1 for n in numbers if _is_prime(int(n))),
        "fibonacci": sum(1 for n in numbers if _is_fibonacci(int(n))),
    }


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    for divisor in range(2, int(value**0.5) + 1):
        if value % divisor == 0:
            return False
    return True


def _is_fibonacci(value: int) -> bool:
    return any(n == value for n in [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89])


class GameGenerator:
    def __init__(self, features: pd.DataFrame):
        self.features = features.set_index("number")

    def generate_games(self) -> pd.DataFrame:
        logger.info("Generating heuristic game portfolios")
        games = []
        hot = self._build_game(self.features.sort_values("freq_global", ascending=False).head(20).index.tolist())
        cold = self._build_game(self.features.sort_values("freq_global", ascending=True).head(20).index.tolist())
        balanced = self._build_game(self._mix_numbers(10, 10))
        games.append({**hot, "strategy": "HOT"})
        games.append({**cold, "strategy": "COLD"})
        games.append({**balanced, "strategy": "BALANCED"})
        return pd.DataFrame(games)

    def _mix_numbers(self, hot_count: int, cold_count: int) -> List[str]:
        hot = self.features.sort_values("freq_global", ascending=False).head(hot_count).index.tolist()
        cold = self.features.sort_values("freq_global", ascending=True).head(cold_count).index.tolist()
        return sorted(set(hot + cold))[:20]
