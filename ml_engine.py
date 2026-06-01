import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from config import DRAW_COLUMNS, NUMBER_RANGE

logger = logging.getLogger(__name__)

BASE_MODELS = {
    "random_forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "extra_trees": ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "hist_gbm": HistGradientBoostingClassifier(random_state=42),
    "xgboost": XGBClassifier(eval_metric="logloss", random_state=42, n_jobs=1, verbosity=0),
    "lightgbm": LGBMClassifier(random_state=42, n_jobs=-1),
    "catboost": CatBoostClassifier(verbose=0, random_state=42),
}


class MachineLearningEngine:
    def __init__(self, draws: pd.DataFrame, features: Optional[pd.DataFrame] = None, min_history: int = 20):
        self.draws = draws.reset_index(drop=True)
        self.features = features.copy() if features is not None else None
        self.number_features = self.features.set_index("number") if self.features is not None and "number" in self.features.columns else None
        if self.number_features is not None:
            self.number_features.index = self.number_features.index.astype(str).str.zfill(2)
        self.min_history = min_history
        self.feature_importances_ = None
        self.model_metrics_ = None

    def _build_feature_rows(self, history: pd.DataFrame, target_numbers: set, draw_index: int) -> pd.DataFrame:
        numbers = [f"{n:02d}" for n in NUMBER_RANGE]
        positions = {number: [] for number in numbers}
        last_seen = {number: -1 for number in numbers}

        for idx, row in history.iterrows():
            for pos, col in enumerate(DRAW_COLUMNS, start=1):
                number = row[col]
                positions[number].append(pos)
                last_seen[number] = int(idx)

        global_counts = pd.Series(history[DRAW_COLUMNS].values.flatten()).value_counts().reindex(numbers, fill_value=0)
        recent_counts_50 = pd.Series(history.tail(50)[DRAW_COLUMNS].values.flatten()).value_counts().reindex(numbers, fill_value=0)
        recent_counts_25 = pd.Series(history.tail(25)[DRAW_COLUMNS].values.flatten()).value_counts().reindex(numbers, fill_value=0)

        rows = []
        for number in numbers:
            values = positions[number]
            count_all = int(global_counts[number])
            count_50 = int(recent_counts_50[number])
            count_25 = int(recent_counts_25[number])
            recency = draw_index - last_seen[number] if last_seen[number] >= 0 else draw_index
            row = {
                "draw_index": draw_index,
                "number": number,
                "freq_global_prev": count_all,
                "freq_last_50": count_50,
                "freq_last_25": count_25,
                "delay_since_last": float(recency),
                "last_position": float(values[-1]) if values else 0.0,
                "position_mean_prev": float(np.mean(values)) if values else 0.0,
                "position_std_prev": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "appearance_rate_prev": float(len(values)) / max(len(history), 1),
                "trend_50_to_global": float(count_50) / (count_all + 1.0),
                "trend_25_to_global": float(count_25) / (count_all + 1.0),
                "target": int(number in target_numbers),
            }
            if self.number_features is not None and number in self.number_features.index:
                static_values = self.number_features.loc[number]
                if isinstance(static_values, pd.DataFrame):
                    static_values = static_values.iloc[0]
                if isinstance(static_values, pd.Series):
                    static_items = static_values.items()
                else:
                    static_items = dict(static_values).items()
                for key, value in static_items:
                    if key == "community":
                        if "community_code" in self.number_features.columns:
                            try:
                                cc = self.number_features.loc[number, "community_code"]
                                row["feat_community_code"] = float(cc) if not pd.isna(cc) else 0.0
                            except Exception:
                                row["feat_community_code"] = 0.0
                        continue
                    if key == "community_code":
                        try:
                            row["feat_community_code"] = float(value) if not pd.isna(value) else 0.0
                        except Exception:
                            row["feat_community_code"] = 0.0
                        continue
                    coerced = pd.to_numeric(value, errors="coerce")
                    if not pd.isna(coerced):
                        row[f"feat_{key}"] = float(coerced)
            rows.append(row)
        return pd.DataFrame(rows)

    def _prepare_training_data(self) -> pd.DataFrame:
        training_frames = []
        start_index = max(1, min(self.min_history, len(self.draws) - 2))
        for draw_index in range(start_index, len(self.draws) - 1):
            history = self.draws.iloc[:draw_index]
            target_numbers = set(self.draws.iloc[draw_index][DRAW_COLUMNS].tolist())
            training_frames.append(self._build_feature_rows(history, target_numbers, draw_index))
        if not training_frames:
            raise ValueError("Not enough historical draws to build training data")
        return pd.concat(training_frames, ignore_index=True)

    def _prepare_target_data(self) -> pd.DataFrame:
        if len(self.draws) < 2:
            raise ValueError("Not enough draw history to build prediction features")
        history = self.draws.iloc[:-1]
        target_numbers = set(self.draws.iloc[-1][DRAW_COLUMNS].tolist())
        return self._build_feature_rows(history, target_numbers, len(self.draws) - 1)

    def train_models(self) -> pd.DataFrame:
        logger.info("Training machine learning models")
        training_data = self._prepare_training_data()
        target_data = self._prepare_target_data()

        X_train = training_data.drop(columns=["draw_index", "number", "target"])
        y_train = training_data["target"].astype(int).to_numpy()
        X_target = target_data.drop(columns=["draw_index", "number", "target"])

        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
        )
        X_target_scaled = pd.DataFrame(
            scaler.transform(X_target), columns=X_target.columns, index=X_target.index
        )

        self.feature_importances_ = pd.Series(dtype=float)
        importance_models = 0
        model_metrics = []

        predictions = {"number": target_data["number"].tolist()}
        for model_name, model in BASE_MODELS.items():
            prob_values = np.zeros(len(X_target), dtype=float)
            metrics = {"model": model_name, "status": "failed", "cv_log_loss": None}
            try:
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                if len(set(y_train)) > 1:
                    cv_pred = cross_val_predict(model, X_train_scaled, y_train, cv=cv, method="predict_proba")[:, 1]
                    metrics["cv_log_loss"] = float(log_loss(y_train, np.clip(cv_pred, 1e-12, 1 - 1e-12)))
                model.fit(X_train_scaled, y_train)
                prob_values = model.predict_proba(X_target_scaled)[:, 1]
                metrics["status"] = "ok"
                if hasattr(model, "feature_importances_"):
                    importances = pd.Series(model.feature_importances_, index=X_train.columns)
                    self.feature_importances_ = (
                        self.feature_importances_.add(importances, fill_value=0.0)
                        if not self.feature_importances_.empty
                        else importances.copy()
                    )
                    importance_models += 1
            except Exception as exc:
                logger.warning("Model %s failed during training: %s", model_name, exc)
            predictions[f"prob_{model_name}"] = prob_values
            model_metrics.append(metrics)

        if importance_models > 0:
            self.feature_importances_ = (self.feature_importances_ / importance_models).sort_values(ascending=False)
        else:
            self.feature_importances_ = None

        self.model_metrics_ = pd.DataFrame(model_metrics)
        df = pd.DataFrame(predictions)
        df["target"] = target_data["target"].astype(int).values
        df["prob_mean"] = df[[col for col in df.columns if col.startswith("prob_")]].mean(axis=1)
        df = df.sort_values("prob_mean", ascending=False).reset_index(drop=True)
        return df
