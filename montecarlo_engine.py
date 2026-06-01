import logging
from typing import Dict, List

import numpy as np
from joblib import Parallel, delayed

from config import DRAW_COLUMNS, NUMBER_RANGE, N_JOBS

logger = logging.getLogger(__name__)


def _sample_draw(weights: List[float]) -> List[str]:
    numbers = [f"{n:02d}" for n in NUMBER_RANGE]
    weights = np.array(weights, dtype=float)
    weights = weights / weights.sum()
    return list(np.random.choice(numbers, size=20, replace=False, p=weights))


class MonteCarloSimulator:
    def __init__(self, draws):
        self.draws = draws
        self.weights = self._compute_empirical_probabilities()

    def _compute_empirical_probabilities(self) -> List[float]:
        frequencies = self.draws[DRAW_COLUMNS].values.flatten()
        counts = {f"{n:02d}": 0 for n in NUMBER_RANGE}
        for number in frequencies:
            counts[number] += 1
        return [counts[f"{n:02d}"] + 1.0 for n in NUMBER_RANGE]

    def run_simulation(self, n_simulations: int = 10000) -> Dict[str, object]:
        logger.info("Running Monte Carlo with %d simulations", n_simulations)
        results = Parallel(n_jobs=N_JOBS)(delayed(_sample_draw)(self.weights) for _ in range(n_simulations))
        flattened = [number for draw in results for number in draw]
        counts = {f"{n:02d}": flattened.count(f"{n:02d}") for n in NUMBER_RANGE}
        sorted_counts = sorted(counts.items(), key=lambda item: item[1], reverse=True)
        return {
            "simulations": n_simulations,
            "probabilities": {number: count / (n_simulations * 20) for number, count in sorted_counts},
            "top_20": sorted_counts[:20],
        }
