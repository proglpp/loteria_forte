from data_loader import LotomaniaDataLoader
from feature_engineering import build_feature_matrix
from preprocessing import preprocess_draws
from config import DATA_FILE, DATABASE_FILE


def test_build_feature_matrix():
    loader = LotomaniaDataLoader(DATA_FILE, DATABASE_FILE)
    draws = loader.load_data()
    preprocessed = preprocess_draws(draws)
    features = build_feature_matrix(preprocessed)

    assert features.shape[0] == 100
    assert "freq_global" in features.columns
    assert "delay_last" in features.columns
    assert "position_entropy" in features.columns
    assert "freq_10_to_100" in features.columns
    assert features["score_total"].dtype != object
