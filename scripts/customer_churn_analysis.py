# Customer Churn Analysis Script

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load the dataset
data = pd.read_csv('C:/Users/atifa/OneDrive/Documents/GitHub/customer_churn_data/data/customer_churn_data.csv')

# Basic info
print("Dataset Shape:", data.shape)
print(data.head())

# Convert categorical columns
data['Churn'] = data['Churn'].map({'Yes': 1, 'No': 0})
data['Gender'] = data['Gender'].map({'Male': 1, 'Female': 0})
data = pd.get_dummies(data, columns=['ContractType', 'InternetService', 'PaymentMethod'], drop_first=True)

# Features and target
X = data.drop(['CustomerID', 'Churn'], axis=1)
y = data['Churn']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Plotting
plt.figure(figsize=(10, 6))
sns.countplot(x='Churn', data=data)
plt.title('Churn Count')
plt.savefig('plots/churn_count.png')
plt.close()

plt.figure(figsize=(10, 6))
sns.boxplot(x='Churn', y='MonthlyCharges', data=data)
plt.title('Monthly Charges by Churn')
plt.savefig('plots/monthly_charges_by_churn.png')
plt.close()

plt.figure(figsize=(10, 6))
sns.histplot(data['TenureMonths'], kde=True, bins=30)
plt.title('Tenure Distribution')
plt.savefig('plots/tenure_distribution.png')
plt.close()