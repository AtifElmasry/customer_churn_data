"""Analyze synthetic customer churn data and generate model diagnostics."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "customer_churn_data.csv"
DEFAULT_PLOTS_DIR = PROJECT_ROOT / "plots"


def load_data(data_path=DEFAULT_DATA_PATH):
    """Load the customer churn dataset from a CSV file."""
    return pd.read_csv(data_path)


def prepare_features(data):
    """Convert categorical fields and return model features and target."""
    prepared = data.copy()
    prepared["Churn"] = prepared["Churn"].map({"Yes": 1, "No": 0})
    prepared["Gender"] = prepared["Gender"].map({"Male": 1, "Female": 0})
    prepared = pd.get_dummies(
        prepared,
        columns=["ContractType", "InternetService", "PaymentMethod"],
        drop_first=True,
    )

    features = prepared.drop(["CustomerID", "Churn"], axis=1)
    target = prepared["Churn"]
    return prepared, features, target


def train_model(features, target):
    """Train a Random Forest model and return the model and test data."""
    features_train, features_test, target_train, target_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
    )
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(features_train, target_train)
    return model, features_test, target_test


def save_plots(data, plots_dir=DEFAULT_PLOTS_DIR):
    """Create and save the project's exploratory visualizations."""
    plots_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    sns.countplot(x="Churn", data=data)
    plt.title("Churn Count")
    plt.savefig(plots_dir / "churn_count.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.boxplot(x="Churn", y="MonthlyCharges", data=data)
    plt.title("Monthly Charges by Churn")
    plt.savefig(plots_dir / "monthly_charges_by_churn.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.histplot(data["TenureMonths"], kde=True, bins=30)
    plt.title("Tenure Distribution")
    plt.savefig(plots_dir / "tenure_distribution.png")
    plt.close()


def main():
    """Run the complete customer churn analysis."""
    data = load_data()
    print("Dataset Shape:", data.shape)
    print(data.head())

    prepared, features, target = prepare_features(data)
    model, features_test, target_test = train_model(features, target)
    predictions = model.predict(features_test)

    print("Classification Report:")
    print(classification_report(target_test, predictions, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(target_test, predictions))

    save_plots(prepared)


if __name__ == "__main__":
    main()
