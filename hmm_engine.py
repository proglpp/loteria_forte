import logging
from typing import Dict

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

from config import DRAW_COLUMNS, HMM_STATES, NUMBER_RANGE, SEED

logger = logging.getLogger(__name__)


def _build_draw_features(draws: pd.DataFrame) -> np.ndarray:
    metrics = []
    for _, row in draws.iterrows():
        numbers = [int(row[col]) for col in DRAW_COLUMNS]
        hist, _ = np.histogram(numbers, bins=10, range=(0, 100))
        metrics.append(hist.astype(float))
    return np.vstack(metrics)


class HMMEngine:
    def __init__(self, draws: pd.DataFrame):
        self.draws = draws.reset_index(drop=True)
        self.features = _build_draw_features(self.draws)

    def train_and_score(self) -> Dict[str, pd.Series]:
        logger.info("Training HMM with %d hidden states", HMM_STATES)
        model = GaussianHMM(n_components=HMM_STATES, covariance_type="diag", random_state=SEED, n_iter=200)
        model.fit(self.features)
        state_sequence = model.predict(self.features)
        probabilities = model.predict_proba(self.features)

        state_scores = pd.Series(state_sequence, index=self.draws.index, name="state")
        state_prob_summary = pd.Series(probabilities.max(axis=1), index=self.draws.index, name="state_confidence")

        number_scores = {f"{n:02d}": 0.0 for n in NUMBER_RANGE}
        for idx, row in self.draws.iterrows():
            for number in [f"{n:02d}" for n in NUMBER_RANGE]:
                if number in row[DRAW_COLUMNS].tolist():
                    number_scores[number] += float(state_prob_summary.loc[idx])
        number_scores = pd.Series(number_scores)
        number_scores = number_scores / (number_scores.max() + 1e-12)
        return {"hmm_score": number_scores}
