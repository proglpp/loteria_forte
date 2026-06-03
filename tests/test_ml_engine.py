import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier

from data_loader import LotomaniaDataLoader
from feature_engineering import build_feature_matrix
import ml_engine as ml_engine_module
from ml_engine import MachineLearningEngine
from preprocessing import preprocess_draws
from config import DATA_FILE, DATABASE_FILE


def test_machine_learning_engine_predicts(monkeypatch):
    fast_models = {
        "random_forest": RandomForestClassifier(n_estimators=5, random_state=42, n_jobs=1),
        "extra_trees": ExtraTreesClassifier(n_estimators=5, random_state=42, n_jobs=1),
    }
    monkeypatch.setattr(ml_engine_module, "BASE_MODELS", fast_models)

    loader = LotomaniaDataLoader(DATA_FILE, DATABASE_FILE)
    draws = loader.load_data()
    preprocessed = preprocess_draws(draws).tail(80).reset_index(drop=True)
    features = build_feature_matrix(preprocessed)

    ml_engine = MachineLearningEngine(preprocessed, features, min_history=20)
    report = ml_engine.train_models()

    assert report.shape[0] == 100
    assert "prob_mean" in report.columns
    assert report["prob_weighted"].between(0.0, 1.0).all()
    assert report["target"].isin([0, 1]).all()

    next_report = ml_engine.predict_next_draw()
    assert next_report.shape[0] == 100
    assert "prob_weighted" in next_report.columns
    assert next_report["target"].isna().all()


def test_machine_learning_engine_skips_textual_community():
    columns = [f"Bola{i}" for i in range(1, 21)]
    draws = pd.DataFrame([
        {col: f"{(start + offset) % 100:02d}" for offset, col in enumerate(columns)}
        for start in range(1, 8)
    ])
    features = pd.DataFrame([
        {"number": "01", "community": "C1", "community_code": 1},
        {"number": "02", "community": "C2", "community_code": 2},
    ])

    ml_engine = MachineLearningEngine(draws, features, min_history=1)
    training_data = ml_engine._prepare_training_data()

    assert "feat_community_code" in training_data.columns
    assert "feat_community" not in training_data.columns
    assert training_data["feat_community_code"].notna().any()
