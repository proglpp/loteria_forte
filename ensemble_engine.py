import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)

FINAL_BLEND = {
    "ensemble": 0.45,
    "score": 0.30,
    "delay": 0.15,
    "recent_gap": 0.10,
}


class EnsembleEngine:
    def __init__(
        self,
        features: pd.DataFrame,
        ml_report: pd.DataFrame,
        model_weights: Optional[Dict[str, float]] = None,
    ):
        self.features = features.set_index("number") if "number" in features.columns else features.copy()
        self.features.index = self.features.index.astype(str).str.zfill(2)
        self.ml_report = ml_report.copy()
        self.model_weights = model_weights or {}

    def _normalized_series(self, series: pd.Series) -> pd.Series:
        values = pd.to_numeric(series, errors="coerce").fillna(0.0)
        span = float(values.max() - values.min())
        if span <= 1e-12:
            return pd.Series(0.5, index=series.index)
        return (values - values.min()) / span

    def _feature_signals(self) -> pd.DataFrame:
        signals = pd.DataFrame(index=[f"{n:02d}" for n in range(100)])
        signals["score_norm"] = self._normalized_series(self.features.get("score_total", 0))
        signals["delay_norm"] = self._normalized_series(self.features.get("delay_last", 0))
        freq_25 = pd.to_numeric(self.features.get("freq_25", 0), errors="coerce").fillna(0.0)
        freq_100 = pd.to_numeric(self.features.get("freq_100", 1), errors="coerce").fillna(1.0)
        gap = 1.0 - (freq_25 / (freq_100 + 1.0))
        signals["recent_gap_norm"] = self._normalized_series(gap)
        return signals

    def build_stack(self, *, fit_targets: bool = True) -> pd.DataFrame:
        logger.info("Building ensemble and final ranking")
        prob_columns = [
            col
            for col in self.ml_report.columns
            if col.startswith("prob_") and col not in ("prob_mean", "prob_weighted", "prob_ensemble", "prob_final")
        ]
        if not prob_columns:
            raise ValueError("No probability columns found for ensemble training")

        if "prob_weighted" in self.ml_report.columns:
            base_prob = self.ml_report["prob_weighted"].to_numpy()
        else:
            base_prob = self.ml_report[prob_columns].mean(axis=1).to_numpy()

        has_targets = (
            fit_targets
            and "target" in self.ml_report.columns
            and self.ml_report["target"].notna().any()
            and int(self.ml_report["target"].fillna(0).sum()) > 0
        )

        if has_targets:
            X = self.ml_report[prob_columns].fillna(0.0)
            y = self.ml_report["target"].fillna(0).astype(int)
            model = LogisticRegression(max_iter=1000)
            model.fit(X, y)
            self.ml_report["prob_ensemble"] = model.predict_proba(X)[:, 1]
        else:
            self.ml_report["prob_ensemble"] = base_prob

        self.ml_report["prob_ensemble"] = self.ml_report["prob_ensemble"].clip(0.0, 1.0)

        self.ml_report["number"] = self.ml_report["number"].astype(str).str.zfill(2)
        signals = self._feature_signals()
        merged = self.ml_report.set_index("number").join(signals, how="left").fillna(0.0)

        merged["prob_final"] = (
            FINAL_BLEND["ensemble"] * merged["prob_ensemble"]
            + FINAL_BLEND["score"] * merged["score_norm"]
            + FINAL_BLEND["delay"] * merged["delay_norm"]
            + FINAL_BLEND["recent_gap"] * merged["recent_gap_norm"]
        ).clip(0.0, 1.0)

        self.ml_report = merged.reset_index()
        if self.ml_report.columns[0] != "number":
            self.ml_report = self.ml_report.rename(columns={self.ml_report.columns[0]: "number"})
        self.ml_report = self.ml_report.sort_values("prob_final", ascending=False).reset_index(drop=True)
        return self.ml_report
