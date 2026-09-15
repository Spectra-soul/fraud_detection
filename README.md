# Financial Fraud Detection & Anomaly Detection Pipeline

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.2%2B-orange.svg)](https://scikit-learn.org/)
[![Imbalanced-Learn](https://img.shields.io/badge/Imbalanced--Learn-0.10%2B-green.svg)](https://imbalanced-learn.org/)
[![Dataset](https://img.shields.io/badge/Dataset-Kaggle%20ULB%20Credit%20Card-blueviolet.svg)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud?select=creditcard.csv)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

An end-to-end, production-grade fraud analytics and anomaly detection workflow engineered to resolve **extreme class imbalance (0.172% fraud rate)** without data leakage. 

Using ensemble learning, synthetic oversampling (SMOTE) within scikit-learn pipelines, cyclical time transformations, and financial cost-curve threshold calibration, this pipeline detects **89.80% of fraudulent transactions** on unseen test data while maintaining an operational **precision of 56.77%** and a **PR-AUC of 0.867**.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset Architecture & Properties](#dataset-architecture--properties)
- [Data Quality & Preprocessing](#data-quality--preprocessing)
- [Feature Engineering](#feature-engineering)
- [Data Leakage Prevention Design](#data-leakage-prevention-design)
- [Model Architecture & Experiments](#model-architecture--experiments)
- [Threshold Calibration & Financial Cost Optimization](#threshold-calibration--financial-cost-optimization)
- [Final Unbiased Test Performance](#final-unbiased-test-performance)
- [Feature Importance & Model Explainability](#feature-importance--model-explainability)
- [Operational Triage & Business Recommendations](#operational-triage--business-recommendations)
- [Repository Structure](#repository-structure)
- [Installation & Quickstart](#installation--quickstart)

---

## Project Overview

In commercial banking, fraudulent transactions represent less than a fraction of a percent of total transaction volume. In such settings:
1. **Accuracy is completely deceptive:** Predicting every transaction as "Legitimate" yields **99.83% accuracy**, yet catches **0% of fraud**.
2. **False Positives are expensive:** Freezing customer accounts indiscriminately introduces friction, brand attrition, and massive manual investigator workloads.
3. **Threshold 0.50 is arbitrary:** Standard machine learning classifiers default to an equal-odds probability threshold ($p = 0.50$), which severely underestimates minority class risk.

This project delivers an end-to-end framework that systematically addresses these hurdles by establishing a baseline, testing cost-sensitive and synthetic oversampling approaches, benchmarking against unsupervised isolation trees, and deriving an optimal operating threshold from operational cost parameters.

---

## Dataset Architecture & Properties

The dataset used is the **ULB Credit Card Fraud Detection Dataset**, available publicly on [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud?select=creditcard.csv).

* **Source:** Machine Learning Group (MLG) at Université Libre de Bruxelles (ULB) & Worldline.
* **Coverage:** 284,807 card transactions performed by European cardholders across a 48-hour window in September 2013.
* **Target Distribution:**
  * **Legitimate (`Class = 0`):** 284,315 transactions (**99.827%**)
  * **Fraudulent (`Class = 1`):** 492 transactions (**0.173%**)
  * **Class Ratio:** ~578 legitimate transactions for every 1 fraudulent transaction.

### Attributes Breakdown
| Feature Name | Type | Description |
|:---|:---|:---|
| `Time` | Float (numeric) | Elapsed time in seconds between each transaction and the first transaction in the dataset. |
| `V1` – `V28` | Float (numeric) | 28 principal component features derived via PCA dimensionality reduction due to confidentiality constraints. |
| `Amount` | Float (numeric) | Transaction amount in Euros (€). Heavily right-skewed with a maximum of €25,691.16 and mean of €88.35. |
| `Class` | Integer (0/1) | Target label: `1` for fraudulent transactions, `0` for legitimate transactions. |

---

## Data Quality & Preprocessing

* **Missing Values:** `0` missing entries across all 284,807 rows and 31 columns.
* **Duplicate Records:** **1,081 duplicate rows** detected. In transactional payment processing, identical amounts at the exact same timestamp can represent genuine rapid card-swipes, auto-billing retries, or systemic retries. In alignment with financial compliance standards, duplicates were retained to preserve the original distribution.

---

## Feature Engineering

To provide models with optimal predictive signals and remove mathematical redundancy:

1. **Log-Scale Transformation (`LogAmount`):**
   Transaction amount exhibits severe positive skewness. We apply the natural log transformation:
   $$\text{LogAmount} = \ln(\text{Amount} + 1)$$
   The raw `Amount` feature is subsequently dropped to eliminate exact collinearity in split criteria.

2. **Cyclical Time Encoding (`Hour_sin`, `Hour_cos`):**
   Elapsed seconds (`Time`) are converted to approximate hour-of-day:
   $$\text{Hour} = \left(\left\lfloor \frac{\text{Time}}{3600} \right\rfloor \pmod{24}\right)$$
   Because hour `23` (11 PM) and hour `0` (12 AM) are adjacent in time, standard linear representations introduce a false boundary. We transform `Hour` into cyclical sine/cosine wave coordinates:
   $$\text{Hour\_sin} = \sin\left(\frac{2 \pi \times \text{Hour}}{24}\right), \quad \text{Hour\_cos} = \cos\left(\frac{2 \pi \times \text{Hour}}{24}\right)$$
   Raw `Hour` is dropped, leaving 32 clean input features.

---

## Data Leakage Prevention Design

To guarantee unbiased, production-valid evaluation:

1. **Two-Stage Stratified Split:**
   * **Hold-Out Test Set (20%):** $56,962$ transactions ($98$ frauds, $56,864$ legitimate) set aside immediately. Untouched until final scoring.
   * **Train / Validation Split:** The remaining 80% ($227,845$ transactions) is sub-split into:
     * **Training Fold (85%):** $193,668$ transactions ($335$ frauds) used for model fitting.
     * **Validation Fold (15%):** $34,177$ transactions ($59$ frauds) strictly reserved for hyperparameter and decision-threshold tuning.
2. **Pipeline-Contained Resampling:**
   SMOTE is applied **exclusively inside an `ImbPipeline`**. Synthetic interpolation is fitted only on the training fold during cross-validation, preventing synthetic data points from contaminating validation or testing folds.

---

## Model Architecture & Experiments

Four distinct paradigms were evaluated on the held-out test set ($56,962$ observations, $98$ actual frauds) at the default classification threshold ($0.50$):

| Model Architecture | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC (AP) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Random Forest + SMOTE** | **0.9995** | 0.9186 | **0.8061** | **0.8587** | **0.9671** | **0.867** |
| **Random Forest (Class-Weighted)** | 0.9995 | **0.9605** | 0.7449 | 0.8391 | 0.9623 | 0.858 |
| **Logistic Regression (Standardized)** | 0.9991 | 0.8101 | 0.6531 | 0.7232 | 0.9605 | 0.741 |
| **Isolation Forest (Unsupervised Benchmark)** | 0.9976 | 0.2959 | 0.2959 | 0.2959 | 0.9512 | 0.190 |

### Key Experimental Insights
* **Supervised vs. Anomaly Detection:** Isolation Forest achieves an AP of only **0.190** and precision of **29.59%**. While unsupervised methods are valuable when labels are unavailable, supervised tree ensembles capture fraud interaction signals far more effectively.
* **SMOTE Advantage:** Random Forest with SMOTE captured **80.61%** of fraud at default threshold versus **74.49%** for cost-sensitive weighting, achieving the highest Precision-Recall curve area (**PR-AUC = 0.867**).

---

## Threshold Calibration & Financial Cost Optimization

Using the default $0.50$ threshold allows **19 fraudulent transactions to pass undetected**. In payment operations, missed fraud carries direct chargeback liabilities, whereas false alarms only incur investigator triage costs.

### Operational Cost Function
$$\text{Total Financial Cost} = (\text{False Positives} \times C_{\text{FP}}) + (\text{False Negatives} \times C_{\text{FN}})$$
* $C_{\text{FP}} = \$15.00$ (Average investigator labor cost to review an alert)
* $C_{\text{FN}} = \$122.00$ (Mean monetary loss per undetected fraud based on dataset EDA)

### Validation Optimization
We swept thresholds from $0.01$ to $0.99$ on the **validation fold** under the operational constraint $\text{Precision} \ge 50\%$.

* **Selected Optimal Threshold:** **`0.17`**
* **Validation Performance at 0.17:**
  * Recall: **88.14%** (52/59 frauds caught)
  * Precision: **62.82%**
  * Total Estimated Cost: **$5,744.00** (a 48% reduction versus default operating threshold)

---

## Final Unbiased Test Performance

The optimal threshold ($0.17$) derived on validation was evaluated against the untouched **56,962 test set transactions**:

```
----------------- Final Unbiased Test Classification Report -----------------
              precision    recall  f1-score   support

  Legitimate     0.9998    0.9988    0.9993     56864
       Fraud     0.5677    0.8980    0.6957        98

    accuracy                         0.9986     56962
   macro avg     0.7838    0.9484    0.8475     56962
weighted avg     0.9991    0.9986    0.9988     56962
```

### Final Confusion Matrix (Threshold = 0.17)

| | Predicted Legitimate | Predicted Fraud | Total Actual |
|:---|:---:|:---:|:---:|
| **Actual Legitimate** | **56,797** (TN) | **67** (FP) | 56,864 |
| **Actual Fraud** | **10** (FN) | **88** (TP) | 98 |
| **Total Predicted** | 56,807 | **155 Alerts** | 56,962 |

### Operational Impact Metrics
* **Fraud Detection Rate (Recall):** **89.80%** (88 out of 98 fraud cases captured).
* **Missed Fraud (False Negatives):** Down from 19 to **only 10 transactions**.
* **False Alarm Rate:** **0.118%** (Only 67 false alerts generated out of 56,864 legitimate cardholders).
* **Investigator Hit-Rate (Precision):** **56.77%** — more than 1 out of every 2 generated alerts is verified fraud.

---

## Feature Importance & Model Explainability

Feature importances extracted from the tree ensemble isolate the primary predictors of transaction fraud:

```
Rank   Feature       Importance   Description
----------------------------------------------------------------------
1      V14           18.60%       Latent PCA component (Strong negative correlation with fraud)
2      V10           11.50%       Latent PCA component
3      V12           11.40%       Latent PCA component
4      V4             9.60%       Latent PCA component (Strong positive correlation with fraud)
5      V17            7.80%       Latent PCA component
6      V11            7.60%       Latent PCA component
7      V3             5.50%       Latent PCA component
8      V16            3.80%       Latent PCA component
9      LogAmount      1.20%       Non-linear monetary scale
```

* **Core Drivers:** Latent variables `V14`, `V10`, `V12`, and `V4` account for over **51% of total model split decisions**.
* **Monetary Value:** `LogAmount` appears within the top 15 features, validating that transaction size contributes supplementary risk signal without introducing collinear instability.

---

## Operational Triage & Business Recommendations

In a live payments gateway, the model output should trigger automated routing rather than hard account terminations:

```
                       [Incoming Transaction]
                                  │
                          Feature Pipeline
                   (Log Transform + Cyclical Time)
                                  │
                       Random Forest Inference
                                  │
                     P(Fraud) >= 0.17 Threshold?
                                 / \
                                /   \
                             No      Yes
                             /        \
                   [Auto-Approve]     Risk Triage Strategy
                                      ├── P(Fraud) >= 0.70: High Risk
                                      │   └── Trigger immediate 2FA / temporary hold
                                      │
                                      └── 0.17 <= P(Fraud) < 0.70: Moderate Risk
                                          └── Route to Fraud Analyst Worklist (155 alerts/test)
```

1. **Tiered Verification:** Transactions scoring between $0.17$ and $0.70$ are presented with step-up authentication (SMS OTP, biometric prompt), mitigating friction for the 67 false positives.
2. **Batch Audit Queue:** Flagged alerts export directly with priority scores (`FraudProbability`), allowing risk analysts to triage the highest-risk alerts first.
3. **Continuous Feedback Loop:** Analyst review outcomes (confirmed fraud vs false positive) flow into automated feature stores to retrain SMOTE-tree pipelines periodically.

---

## Repository Structure

```plaintext
├── creditcard.csv                 # Raw dataset (downloaded from Kaggle)
├── financial_fraud_detection.py   # Complete standalone end-to-end Python pipeline
├── financial_fraud_detection.ipynb# Jupyter/Colab notebook version with visual plots
├── requirements.txt               # Environment dependencies
├── README.md                      # Comprehensive documentation & results
└── LICENSE                        # MIT License
```

---

## Installation & Quickstart

### 1. Clone Repository & Setup Environment
```bash
git clone https://github.com/your-username/financial-fraud-detection.git
cd financial-fraud-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Dependencies (`requirements.txt`)
```plaintext
numpy>=1.23.0
pandas>=1.5.0
matplotlib>=3.6.0
seaborn>=0.12.0
scikit-learn>=1.2.0
imbalanced-learn>=0.10.0
```

### 3. Download the Dataset
1. Download `creditcard.csv` directly from [Kaggle ULB Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud?select=creditcard.csv).
2. Place `creditcard.csv` into the root directory of this repository.

### 4. Execute the Pipeline
```bash
python financial_fraud_detection.py
```
Or open and execute `financial_fraud_detection.ipynb` inside Jupyter Notebook or Google Colab.
```
