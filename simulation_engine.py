import logging
from typing import List

import numpy as np

from config import DRAW_COLUMNS, NUMBER_RANGE

logger = logging.getLogger(__name__)


def simulate_random_draw() -> List[str]:
    numbers = [f"{n:02d}" for n in NUMBER_RANGE]
    return list(np.random.choice(numbers, size=20, replace=False))


def simulate_weighted_draw(weights: List[float]) -> List[str]:
    numbers = [f"{n:02d}" for n in NUMBER_RANGE]
    probabilities = np.array(weights, dtype=float)
    probabilities = probabilities / probabilities.sum()
    return list(np.random.choice(numbers, size=20, replace=False, p=probabilities))


def build_simulation_summary(draws: pd.DataFrame, simulations: List[List[str]]) -> dict:
    flattened = [number for sim in simulations for number in sim]
    counts = {number: flattened.count(number) for number in [f"{n:02d}" for n in NUMBER_RANGE]}
    return {
        "simulations": len(simulations),
        "top_10": sorted(counts.items(), key=lambda item: item[1], reverse=True)[:10],
    }
