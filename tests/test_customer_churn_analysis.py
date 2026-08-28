"""Tests for churn-data preparation and model evaluation."""

import pandas as pd

from scripts.customer_churn_analysis import evaluate_models, prepare_features


def test_prepare_features_separates_target_and_identifier():
    """Preparation should remove the target and customer identifier."""
    data = pd.DataFrame(
        {
            "Customer ID": range(12),
            "Call Failure": range(12),
            "Complains": [0, 1] * 6,
            "Subscription Length": range(20, 32),
            "Churn": [0, 1] * 6,
        }
    )

    features, target = prepare_features(data)

    assert "churn" not in features
    assert "customer_id" not in features
    assert target.tolist() == [0, 1] * 6


def test_model_evaluation_returns_decision_metrics():
    """Evaluation should compare both models using churn-relevant metrics."""
    rows = 40
    features = pd.DataFrame(
        {
            "complains": [0, 1] * (rows // 2),
            "seconds_of_use": range(rows),
            "subscription_length": range(10, 10 + rows),
        }
    )
    target = pd.Series([0, 1] * (rows // 2))

    results = evaluate_models(features, target, folds=2)

    assert set(results["model"]) == {"logistic_regression", "random_forest"}
    assert {"roc_auc", "pr_auc", "f1", "precision", "recall"}.issubset(results.columns)
