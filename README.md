# Telecom Customer Churn: Risk Modeling and Retention Prioritization

A reproducible churn-analysis project using **3,150 customer records collected from an Iranian telecommunications company over 12 months**.

The project treats churn as a decision problem—not just a classification exercise. It compares an interpretable baseline with a nonlinear model, uses stratified cross-validation and reports metrics suited to an imbalanced retention use case.

## Business question

Which customers show elevated churn risk, which behaviors are associated with that risk, and how should a retention team evaluate a model before using it?

## Dataset

- **Source:** [UCI Iranian Churn Dataset](https://archive.ics.uci.edu/dataset/563/iranian+churn+dataset)
- **DOI:** [10.24432/C5JW3Z](https://doi.org/10.24432/C5JW3Z)
- **License:** CC BY 4.0
- **Rows:** 3,150 customers
- **Features:** usage, complaints, subscription length, tariff, status and customer value
- **Target:** churn state measured at the end of month 12

Features are aggregated over the first nine months and the following three months form the planning gap before the churn label. This makes the prediction framing more realistic than using information recorded at the moment of departure.

## Analytical approach

1. Retrieve the dataset from UCI with its official identifier.
2. Validate the target and remove non-predictive identifiers.
3. Compare logistic regression and Random Forest models.
4. Use stratified five-fold cross-validation.
5. Report ROC–AUC, PR–AUC and F1 rather than accuracy alone.
6. Inspect Random Forest feature importance for prioritization hypotheses.
7. Keep model output separate from causal claims.

## Why these metrics?

| Metric | Decision value |
|---|---|
| ROC–AUC | Overall ranking ability |
| PR–AUC | Performance on the churn class when it is uncommon |
| F1 | Balance between precision and recall |
| Recall | Share of churners reached by an intervention |
| Precision | Share of contacted customers who actually churn |

A production threshold should be chosen using campaign capacity, contact cost, retention value and the cost of missing a likely churner.

## Repository structure

```text
scripts/customer_churn_analysis.py   Data retrieval, validation and modeling
tests/                               Unit tests using local fixtures
data/README.md                       Source and licensing information
requirements.txt                     Reproducible dependencies
.github/workflows/                   Pylint and multi-version test checks
```

## Run the project

```bash
git clone https://github.com/AtifElmasry/customer_churn_data.git
cd customer_churn_data
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/customer_churn_analysis.py
```

The script downloads the data directly from UCI, prints cross-validation results and saves a feature-importance chart in `plots/`.

## Interpretation guardrails

- Feature importance indicates predictive association, not causation.
- Customer status may be operationally close to churn and should be reviewed for leakage.
- The dataset comes from one telecom market and requires validation before transfer elsewhere.
- Model thresholds should be evaluated against actual retention economics.
- Fairness and customer-treatment impacts require assessment before deployment.

## Skills demonstrated

Problem framing, source evaluation, leakage awareness, class-imbalance metrics, cross-validation, model comparison, interpretable baselines, feature importance, testing and CI.

## Author

[Atif Elmasry](https://github.com/AtifElmasry) · [LinkedIn](https://www.linkedin.com/in/tioatifelmasry/)
