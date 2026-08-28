"""Model churn risk using the UCI Iranian telecom dataset."""

from pathlib import Path
import re

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import make_scorer, average_precision_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from ucimlrepo import fetch_ucirepo

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PLOTS_DIR = PROJECT_ROOT / "plots"


def to_snake_case(value):
    """Convert a column label to lowercase snake case."""
    value = re.sub(r"[^a-zA-Z0-9]+", "_", str(value))
    return value.strip("_").lower()


def load_uci_data():
    """Retrieve UCI dataset 563 and return one analysis-ready table."""
    dataset = fetch_ucirepo(id=563)
    features = dataset.data.features.copy()
    target = dataset.data.targets.copy()
    data = pd.concat([features, target], axis=1)
    data.columns = [to_snake_case(column) for column in data.columns]
    return data


def prepare_features(data):
    """Validate the churn target and separate features from labels."""
    prepared = data.copy()
    prepared.columns = [to_snake_case(column) for column in prepared.columns]

    if "churn" not in prepared:
        raise ValueError("Expected a 'churn' target column.")

    target = pd.to_numeric(prepared.pop("churn"), errors="raise").astype(int)
    identifiers = [
        column for column in prepared.columns
        if column in {"customer_id", "anonymous_customer_id"}
    ]
    features = prepared.drop(columns=identifiers, errors="ignore")
    return features, target


def build_models():
    """Return interpretable and nonlinear candidate models."""
    logistic = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=2_000, class_weight="balanced"),
    )
    forest = make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(
            n_estimators=500,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    )
    return {"logistic_regression": logistic, "random_forest": forest}


def evaluate_models(features, target, folds=5):
    """Evaluate candidate models with stratified cross-validation."""
    cross_validation = StratifiedKFold(
        n_splits=folds,
        shuffle=True,
        random_state=42,
    )
    scoring = {
        "roc_auc": "roc_auc",
        "pr_auc": make_scorer(
            average_precision_score,
            response_method="predict_proba",
        ),
        "f1": "f1",
        "precision": "precision",
        "recall": "recall",
    }
    rows = []

    for name, model in build_models().items():
        scores = cross_validate(
            model,
            features,
            target,
            cv=cross_validation,
            scoring=scoring,
            n_jobs=-1,
        )
        row = {"model": name}
        for metric in scoring:
            row[metric] = scores[f"test_{metric}"].mean()
            row[f"{metric}_std"] = scores[f"test_{metric}"].std()
        rows.append(row)

    return pd.DataFrame(rows).sort_values("pr_auc", ascending=False)


def save_feature_importance(features, target, output_dir=PLOTS_DIR):
    """Fit the Random Forest and save its feature-importance chart."""
    output_dir.mkdir(parents=True, exist_ok=True)
    forest_pipeline = build_models()["random_forest"]
    forest_pipeline.fit(features, target)
    forest = forest_pipeline.named_steps["randomforestclassifier"]

    importance = pd.Series(
        forest.feature_importances_,
        index=features.columns,
    ).sort_values(ascending=False)

    top_features = importance.head(10).sort_values()
    axis = top_features.plot.barh(figsize=(9, 6), color="#3264a8")
    axis.set_title("Top churn-risk features")
    axis.set_xlabel("Random Forest feature importance")
    axis.figure.tight_layout()
    axis.figure.savefig(output_dir / "feature_importance.png", dpi=160)
    plt.close(axis.figure)
    return importance


def main():
    """Run data retrieval, model comparison and interpretation output."""
    data = load_uci_data()
    features, target = prepare_features(data)
    results = evaluate_models(features, target)
    importance = save_feature_importance(features, target)

    print(f"Customers: {len(data):,}")
    print(f"Churn rate: {target.mean():.1%}")
    print("\nCross-validation results:")
    print(results.round(3).to_string(index=False))
    print("\nTop predictive features:")
    print(importance.head(10).round(4).to_string())


if __name__ == "__main__":
    main()
