import pandas as pd

from evaluation_engine import evaluate_predictions


def test_evaluate_predictions_top20_hits():
    actual = {f"{n:02d}" for n in [1, 16, 30, 32, 45]}
    report = pd.DataFrame(
        {
            "number": [f"{n:02d}" for n in range(100)],
            "prob_final": [1.0 if f"{n:02d}" in actual else 0.0 for n in range(100)],
        }
    )
    result = evaluate_predictions(actual, report, prob_column="prob_final")
    assert result["top_slices"]["20"]["hits"] == 5
