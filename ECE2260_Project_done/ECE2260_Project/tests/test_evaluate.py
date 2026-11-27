import numpy as np

from src.evaluate import (
    accuracy_by_period,
    evaluate_predictions,
    misclassified_examples,
)
from src.features import build_dataset


def test_metrics_on_known_predictions():
    y_true = [1, 1, 0, 0]
    y_pred = [1, 0, 0, 0]
    m = evaluate_predictions(y_true, y_pred)
    assert m["accuracy"] == 0.75
    assert m["balanced_accuracy"] == 0.75
    assert m["confusion_matrix"] == [[2, 0], [1, 1]]
    assert m["recall"][1] == 0.5 and m["precision"][1] == 1.0


def test_metrics_handle_single_class_prediction():
    m = evaluate_predictions([0, 1, 1], [1, 1, 1])
    assert m["precision"][0] == 0.0  # no crash when a class is never predicted


def test_period_accuracy_and_errors(prices):
    data = build_dataset(prices).reset_index(drop=True)
    pred = data["Target"].to_numpy().copy()
    pred[:3] = 1 - pred[:3]
    table = accuracy_by_period(data, pred, "Q")
    assert table["days"].sum() == len(data)
    wrong = misclassified_examples(data, pred, np.full(len(data), 0.7))
    assert len(wrong) == 3
