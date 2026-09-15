```markdown
# Financial Fraud Detection & Anomaly Detection Pipeline

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Latest-orange.svg)](https://scikit-learn.org/)
[![Imbalanced-Learn](https://img.shields.io/badge/Imbalanced--Learn-SMOTE-green.svg)](https://imbalanced-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end, production-oriented machine learning and anomaly detection pipeline designed to detect fraudulent credit card transactions under extreme class imbalance (0.172% fraud prevalence).

This project implements leakage-controlled preprocessing, cyclical feature engineering, cost-sensitive learning, SMOTE inside `ImbPipeline`, and **leakage-free validation threshold calibration** using an operational business cost function.

---

## 📌 Executive Summary & Key Results

On an independent, untouched test set of **56,962 transactions** containing **98 actual fraud cases**:

| Metric | Baseline (Logistic Regression) | Random Forest (Class-Weighted) | Random Forest (SMOTE @ 0.50) | **Final Model (RF + SMOTE @ Threshold 0.17)** |
| :--- | :---: | :---: | :---: | :---: |
| **Recall (Fraud Caught)** | 65.31% | 74.49% | 80.61% | **89.80% (88 / 98)** |
| **Precision** | 81.01% | 96.05% | 91.86% | **56.77%** |
| **PR-AUC (Average Precision)** | 0.741 | 0.858 | 0.867 | **0.867** |
| **ROC-AUC** | 0.9605 | 0.9623 | 0.9671 | **0.9671** |
| **Alerts Flagged** | 79 | 76 | 86 | **155 (Triage Volume)** |
| **False Positive Count** | 15 | 3 | 7 | **67 (0.118% FPR)** |
| **Missed Frauds (FN)** | 34 | 25 | 19 | **10** |

> **Operational Impact:** By calibrating the decision threshold to `0.17` on a dedicated validation split, the model captures **almost 90% of all fraud** while restricting investigation alert volume to just 155 cases out of nearly 57,000 transactions. More than **1 in every 2 alerts** reviewed is verified fraud.

---

## 📊 Dataset Overview

* **Source:** [Kaggle - Credit Card Fraud Detection (ULB Machine Learning Group)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud?select=creditcard.csv)
* **Timeframe:** European cardholder transactions across two days in September 2013.
* **Total Transactions:** 284,807
* **Fraudulent Transactions:** 492 (**0.1727%** prevalence)
* **Legitimate Transactions:** 284,315 (**99.8273%** prevalence)
* **Missing Values:** 0
* **Duplicate Rows:** 1,081 (Reported and retained, as identical transactions can occur legitimately in batch processing).

### Feature Breakdown
| Feature Name | Type | Description |
| :--- | :--- | :--- |
| `Time` | Continuous (Float) | Seconds elapsed between the transaction and the first transaction in the dataset. |
| `V1` – `V28` | Continuous (Float) | Confidential principal components obtained via PCA (dimensionality reduction). |
| `Amount` | Continuous (Float) | Transaction dollar amount; heavily right-skewed with a maximum of $25,691.16. |
| `Class` | Categorical (Binary) | Target variable (`1` = Fraudulent, `0` = Legitimate). |

---

## 🛠️ Pipeline Architecture & Methodology

```text
Raw Transaction Data (creditcard.csv)
   │
   ├──> Data Quality Checks (Nulls, Duplicates, Schema Validation)
   │
   ├──> Exploratory Data Analysis (Amounts, Hourly Trends, Correlation Heatmap)
   │
   ├──> Feature Engineering
   │      ├── Log Transformation: LogAmount = log1p(Amount) [Raw Amount dropped]
   │      └── Cyclical Encoding: Hour_sin & Hour_cos via (2π * Hour / 24)
   │
   ├──> Leakage-Controlled 3-Way Stratified Split
   │      ├── Test Set (20% = 56,962 samples) ──> Strictly Held Out
   │      └── Train Full (80%)
   │            ├── Train Fold (85% = 193,668 samples) ──> Pipeline Model Training
   │            └── Validation Fold (15% = 34,177 samples) ──> Threshold & Cost Tuning
   │
   ├──> Model Training & Benchmarking
   │      ├── 1. Baseline: StandardScaler + Logistic Regression
   │      ├── 2. Cost-Sensitive: Random Forest (class_weight='balanced_subsample')
   │      ├── 3. Resampled: SMOTE inside ImbPipeline + Random Forest
   │      └── 4. Unsupervised: Isolation Forest (Contamination = 0.173%)
   │
   ├──> Threshold Calibration on Validation Set (Objective: Max Recall @ Precision ≥ 50%)
   │      └── Business Cost Optimization: ($15 FP Review Cost vs $122 FN Fraud Loss)
   │      └── Selected Optimal Threshold: 0.17
   │
   └──> Final Unbiased Evaluation on Test Set & Triage Output Generation
```

### 1. Leakage Prevention (3-Way Split)
A critical flaw in many fraud projects is tuning probability thresholds directly on the test set. Here, a **3-way stratified split** is applied:
* **Train Fold:** Model fitting and SMOTE interpolation.
* **Validation Fold:** Precision-Recall curve analysis and cost-optimal threshold search.
* **Test Fold:** Evaluated **only once** at the fixed threshold of `0.17`.

### 2. Feature Engineering
* **Collinearity Elimination:** Taking `np.log1p(df['Amount'])` eliminates extreme skewness. The raw `Amount` feature is explicitly dropped to prevent duplicate splits and collinear tree degradation.
* **Cyclical Time Transformation:** Elapsed seconds are converted to hour-of-day ($0-23$) and encoded using sine/cosine functions:
  $$\text{Hour\_sin} = \sin\left(\frac{2\pi \times \text{Hour}}{24}\right), \quad \text{Hour\_cos} = \cos\left(\frac{2\pi \times \text{Hour}}{24}\right)$$
  This guarantees numerical continuity between `23:00` (11 PM) and `00:00` (midnight).

### 3. Balanced Sampling via `ImbPipeline`
SMOTE is strictly encapsulated inside `imblearn.pipeline.Pipeline`. Oversampling occurs **exclusively during training batches**, completely insulating the validation and test distributions from synthetic contamination.

---

## 📈 Model Performance & Evaluation

### Precision-Recall Trade-Off
In extreme imbalance (0.17%), ROC-AUC can remain misleadingly high (~0.96) even with high false alarms. **Precision-Recall AUC (PR-AUC / Average Precision)** serves as the primary metric:

* **Random Forest (SMOTE):** `PR-AUC = 0.867` *(Top Performer)*
* **Random Forest (Class-Weighted):** `PR-AUC = 0.858`
* **Logistic Regression:** `PR-AUC = 0.741`
* **Isolation Forest (Unsupervised Benchmark):** `PR-AUC = 0.190`

### Confusion Matrix on Test Set (Threshold = 0.17)

```text
                  Predicted Legitimate    Predicted Fraud
Actual Legitimate        56,797                 67       (False Positives)
Actual Fraud               10                   88       (True Positives)
```

* **True Positives (TP):** 88 frauds detected
* **False Negatives (FN):** 10 frauds missed
* **False Positives (FP):** 67 benign transactions flagged (0.118% error rate)
* **True Negatives (TN):** 56,797 legitimate transactions cleared

### Top 5 Predictive Feature Drivers
Tree ensemble Gini importance shows four primary latent PCA components dominating detection:
1. `V14` (~18.6%)
2. `V10` (~11.5%)
3. `V12` (~11.4%)
4. `V4`  (~9.6%)
5. `V17` (~7.8%)

`LogAmount` ranks among the top 15 features, confirming the non-linear value signal without collinear distortion.

---

## 💼 Business Cost Model & Operations Strategy

Rather than adopting an arbitrary threshold of 0.50, the operational threshold was optimized using practical unit economics:
* **Cost of a False Positive ($C_{FP}$):** **$15.00** (Operational expense for an analyst to review or send an automated SMS/2FA challenge).
* **Cost of a False Negative ($C_{FN}$):** **$122.00** (Average monetary loss per undetected fraud transaction observed during EDA).

$$\text{Estimated Loss} = (FP \times \$15.00) + (FN \times \$122.00)$$

### Recommended Operational Workflow
Transactions with model scores $\ge 0.17$ are exported to a prioritized alert triage table:
* **Risk Score $\ge 0.80$:** Automatic real-time transaction decline / biometric challenge.
* **Risk Score $0.17 - 0.79$:** Step-up two-factor authentication (SMS/App prompt) or route to Level-1 human investigator queue.
* **Risk Score $< 0.17$:** Frictionless approval (99.88% accuracy on legitimate flow).

---

## 📁 Repository Structure

```text
credit-card-fraud-detection/
│
├── creditcard.csv                 # Kaggle dataset file (not tracked in git)
├── financial_fraud_detection.py   # Complete end-to-end Python pipeline script
├── financial_fraud_detection.ipynb# Jupyter/Colab notebook version
├── requirements.txt               # Dependencies
├── README.md                      # Project documentation and findings
└── assets/                        # Plots (PR curves, confusion matrix, feature importance)
```

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```

### 2. Set Up a Virtual Environment
```bash
python -m venv venv
# Linux / macOS
source venv/bin/activate
# Windows
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download the Dataset
1. Download `creditcard.csv` directly from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud?select=creditcard.csv).
2. Place `creditcard.csv` inside the root directory of the project.

### 5. Run the Pipeline
* **To run as a Python script:**
  ```bash
  python financial_fraud_detection.py
  ```
* **To run interactively in Jupyter Lab / Notebook:**
  ```bash
  jupyter notebook financial_fraud_detection.ipynb
  ```

---

## 📦 Dependencies (`requirements.txt`)

```text
numpy>=1.22.0
pandas>=1.4.0
scikit-learn>=1.1.0
imbalanced-learn>=0.9.0
matplotlib>=3.5.0
seaborn>=0.11.2
```

---

## 📜 Citation & Acknowledgements

* **Dataset Authors:** Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson, and Gianluca Bontempi. *Calibrating Probability with Undersampling for Unbalanced Classification.* IEEE SSCI 2015.
* **ULB MLG Group:** Machine Learning Group (http://mlg.ulb.ac.be) of Université Libre de Bruxelles.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
```
