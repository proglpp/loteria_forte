import logging
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)


class EnsembleEngine:
    def __init__(self, features: pd.DataFrame, ml_report: pd.DataFrame):
        self.features = features
        self.ml_report = ml_report.copy()

    def build_stack(self) -> pd.DataFrame:
        logger.info("Building stacking ensemble")
        prob_columns = [col for col in self.ml_report.columns if col.startswith("prob_") and col != "prob_mean"]
        if not prob_columns:
            raise ValueError("No probability columns found for ensemble training")

        X = self.ml_report[prob_columns].fillna(0.0)
        y = self.ml_report["target"].astype(int)

        model = LogisticRegression(max_iter=1000)
        model.fit(X, y)
        self.ml_report["prob_ensemble"] = model.predict_proba(X)[:, 1]
        self.ml_report["prob_ensemble"] = self.ml_report["prob_ensemble"].clip(0.0, 1.0)
        self.ml_report = self.ml_report.sort_values("prob_ensemble", ascending=False).reset_index(drop=True)
        return self.ml_report
