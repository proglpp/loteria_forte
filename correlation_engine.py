import logging
from itertools import combinations
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from config import DRAW_COLUMNS, NUMBER_RANGE

logger = logging.getLogger(__name__)


def build_cooccurrence_matrix(draws: pd.DataFrame) -> pd.DataFrame:
    logger.info("Building cooccurrence matrix")
    numbers = [f"{n:02d}" for n in NUMBER_RANGE]
    matrix = pd.DataFrame(0, index=numbers, columns=numbers, dtype=int)
    for _, row in draws.iterrows():
        draw_numbers = [str(row[col]) for col in DRAW_COLUMNS]
        for a, b in combinations(draw_numbers, 2):
            matrix.at[a, b] += 1
            matrix.at[b, a] += 1
    return matrix


def build_correlation_report(draws: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    cooccurrence = build_cooccurrence_matrix(draws)
    total_draws = len(draws)
    probabilities = cooccurrence / total_draws
    marginals = np.diag(cooccurrence) / total_draws
    lift = pd.DataFrame(index=cooccurrence.index, columns=cooccurrence.columns, dtype=float)
    pmi = pd.DataFrame(index=cooccurrence.index, columns=cooccurrence.columns, dtype=float)
    jaccard = pd.DataFrame(index=cooccurrence.index, columns=cooccurrence.columns, dtype=float)
    cosine = pd.DataFrame(index=cooccurrence.index, columns=cooccurrence.columns, dtype=float)

    for i in cooccurrence.index:
        for j in cooccurrence.columns:
            if i == j:
                lift.at[i, j] = 0.0
                pmi.at[i, j] = 0.0
                jaccard.at[i, j] = 1.0
                cosine.at[i, j] = 1.0
                continue
            pij = probabilities.at[i, j]
            pi = marginals[int(i)] if i.isdigit() else 0
            pj = marginals[int(j)] if j.isdigit() else 0
            lift.at[i, j] = pij / (pi * pj + 1e-12)
            pmi.at[i, j] = np.log2((pij + 1e-12) / (pi * pj + 1e-12))
            denom = cooccurrence.at[i, i] + cooccurrence.at[j, j] - cooccurrence.at[i, j]
            jaccard.at[i, j] = cooccurrence.at[i, j] / (denom + 1e-12)
            cosine.at[i, j] = cooccurrence.at[i, j] / (np.sqrt(cooccurrence.at[i, i] * cooccurrence.at[j, j]) + 1e-12)

    return {
        "cooccurrence": cooccurrence,
        "pmi": pmi,
        "lift": lift,
        "jaccard": jaccard,
        "cosine": cosine,
    }


def strongest_pairs(cooccurrence: pd.DataFrame, top_n: int = 20) -> List[Tuple[str, str, int]]:
    results = []
    for i, j in combinations(cooccurrence.index, 2):
        results.append((i, j, int(cooccurrence.at[i, j])))
    results.sort(key=lambda item: item[2], reverse=True)
    return results[:top_n]
