import pandas as pd

from data_loader import LotomaniaDataLoader
from feature_engineering import build_feature_matrix
from ml_engine import MachineLearningEngine
from preprocessing import preprocess_draws
from config import DATA_FILE, DATABASE_FILE


def test_machine_learning_engine_predicts():
    loader = LotomaniaDataLoader(DATA_FILE, DATABASE_FILE)
    draws = loader.load_data()
    preprocessed = preprocess_draws(draws)
    features = build_feature_matrix(preprocessed)

    ml_engine = MachineLearningEngine(preprocessed, features, min_history=20)
    report = ml_engine.train_models()

    assert report.shape[0] == 100
    assert "prob_mean" in report.columns
    assert report["prob_mean"].between(0.0, 1.0).all()
    assert report["target"].isin([0, 1]).all()


def test_machine_learning_engine_skips_textual_community():
    columns = [f"draw_{i}" for i in range(1, 6)]
    draws = pd.DataFrame([
        {col: f"{i:02d}" for col, i in zip(columns, [1, 2, 3, 4, 5])},
        {col: f"{i:02d}" for col, i in zip(columns, [6, 7, 8, 9, 10])},
        {col: f"{i:02d}" for col, i in zip(columns, [11, 12, 13, 14, 15])},
        {col: f"{i:02d}" for col, i in zip(columns, [16, 17, 18, 19, 20])},
        {col: f"{i:02d}" for col, i in zip(columns, [1, 2, 3, 4, 5])},
        {col: f"{i:02d}" for col, i in zip(columns, [6, 7, 8, 9, 10])},
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
