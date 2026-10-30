import numpy as np
import pandas as pd
import pytest

from src.evaluation import metrics_from_scores, predict_with_threshold, threshold_for_target_recall


@pytest.fixture
def scored():
    """10 cases: 4 positives. At threshold 0.5: TP=3, FN=1, FP=1, TN=5."""
    y_true = pd.Series([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    y_score = np.array([0.9, 0.8, 0.6, 0.3, 0.7, 0.4, 0.2, 0.1, 0.1, 0.05])
    return y_true, y_score


class TestMetricsFromScores:
    def test_confusion_matrix_counts(self, scored):
        metrics = metrics_from_scores(*scored, threshold=0.5)
        assert metrics["confusion_matrix"] == {"tn": 5, "fp": 1, "fn": 1, "tp": 3}

    def test_threshold_dependent_metrics(self, scored):
        metrics = metrics_from_scores(*scored, threshold=0.5)
        assert metrics["recall"] == pytest.approx(3 / 4)
        assert metrics["specificity"] == pytest.approx(5 / 6)
        assert metrics["precision"] == pytest.approx(3 / 4)
        assert metrics["f2"] == pytest.approx(5 * 0.75 * 0.75 / (4 * 0.75 + 0.75))
        assert metrics["accuracy"] == pytest.approx(0.8)

    def test_lower_threshold_raises_recall(self, scored):
        metrics = metrics_from_scores(*scored, threshold=0.25)
        assert metrics["recall"] == 1.0
        assert metrics["confusion_matrix"]["fn"] == 0

    def test_threshold_free_metrics_do_not_change(self, scored):
        at_half, at_low = metrics_from_scores(*scored, 0.5), metrics_from_scores(*scored, 0.25)
        for key in ("roc_auc", "average_precision", "brier_score"):
            assert at_half[key] == at_low[key]

    def test_brier_score(self, scored):
        y_true, y_score = scored
        expected = np.mean((y_score - y_true.values) ** 2)
        assert metrics_from_scores(y_true, y_score)["brier_score"] == pytest.approx(expected)


class TestThresholdForTargetRecall:
    def test_reaches_target_with_highest_threshold(self, scored):
        y_true, y_score = scored
        result = threshold_for_target_recall(y_true, y_score, target_recall=0.75)
        # Recall 3/4 is first reached at 0.6; any higher threshold gives recall 2/4.
        assert result["threshold"] == pytest.approx(0.6)
        assert result["recall"] == pytest.approx(0.75)
        assert result["precision"] == pytest.approx(0.75)

    def test_full_recall_target(self, scored):
        y_true, y_score = scored
        result = threshold_for_target_recall(y_true, y_score, target_recall=1.0)
        assert result["threshold"] == pytest.approx(0.3)
        assert metrics_from_scores(y_true, y_score, result["threshold"])["recall"] == 1.0

    def test_no_positives_raises(self):
        with pytest.raises(ValueError, match="without positive cases"):
            threshold_for_target_recall(pd.Series([0, 0, 0]), np.array([0.1, 0.5, 0.9]), 0.9)


def test_predict_with_threshold():
    class FixedModel:
        def predict_proba(self, X):
            p = np.array([0.2, 0.5, 0.8])
            return np.column_stack([1 - p, p])

    assert predict_with_threshold(FixedModel(), None, 0.5).tolist() == [0, 1, 1]
    assert predict_with_threshold(FixedModel(), None, 0.9).tolist() == [0, 0, 0]
