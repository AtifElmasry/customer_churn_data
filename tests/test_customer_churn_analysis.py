"""Tests for the customer churn analysis workflow."""

import pandas as pd

from scripts.customer_churn_analysis import prepare_features


def test_prepare_features_encodes_target_and_removes_identifiers():
    """Feature preparation should produce numeric model inputs."""
    data = pd.DataFrame(
        {
            "CustomerID": ["C001", "C002"],
            "Churn": ["Yes", "No"],
            "Gender": ["Female", "Male"],
            "ContractType": ["Monthly", "Annual"],
            "InternetService": ["Fiber", "DSL"],
            "PaymentMethod": ["Card", "Bank transfer"],
            "MonthlyCharges": [80.0, 45.0],
            "TenureMonths": [4, 24],
        }
    )

    prepared, features, target = prepare_features(data)

    assert target.tolist() == [1, 0]
    assert "CustomerID" not in features.columns
    assert "Churn" not in features.columns
    assert prepared["Gender"].tolist() == [0, 1]
    assert not features.select_dtypes(include="object").columns.tolist()
