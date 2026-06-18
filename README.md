# Insurance-claim-predictor
Predicting high-cost health insurance claims and exact claim amounts using statistical testing, machine learning (XGBoost, Random Forest), deep learning (ANN), and SHAP explainability.
# 🏥 Health Insurance Claim Analytics & Predictive Modeling

**An end-to-end data science project for predicting medical claim costs and identifying high-risk members using patient demographics.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15.0-orange)](https://www.tensorflow.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0.0-green)](https://xgboost.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📌 Project Overview

This project simulates a real-world **health insurance analytics pipeline**. We analyze a dataset of 4,000 patient claims to answer two critical business questions:

1. **Classification:** Can we predict if a patient will be a **"High-Cost"** member (Top 25% of claims)?
2. **Regression:** Can we predict the **exact annual medical claim amount** for a patient?

We go beyond just "fitting a model." We conduct **formal statistical hypothesis testing** (T-test, Chi-square, ANOVA) to validate our assumptions before building machine learning models. We benchmark traditional algorithms (Logistic Regression, Random Forest) against state-of-the-art (XGBoost) and even benchmark a **Deep Learning (ANN)** model to prove why tree-based models are superior for tabular data. Finally, we use **SHAP** to explain our model's decisions—a critical requirement for healthcare applications.

---

## 🎯 Key Business Insights

Despite rigorous feature engineering and applying complex algorithms, we discovered a crucial business insight:

> **"Patient demographics (Age, BMI, Smoking, Diabetes) are NOT sufficient to accurately predict medical claim costs (R² = 0.06)."**

| Finding | Statistical/ML Result | Business Impact |
| :--- | :--- | :--- |
| **Smoking is the #1 Cost Driver** | T-test: p < 0.001 | Invest heavily in smoking cessation programs. |
| **Age has a weak link to cost** | Pearson r = 0.05, p = 0.001 | Age matters slightly, but not a strong predictor. |
| **Gender does NOT affect approval** | Chi-square p = 0.022 (Borderline, not significant after correction) | No gender bias in the claim review process. |
| **Current ML models are ineffective** | Best PR-AUC = 0.52 (Classification) <br> Best R² = 0.062 (Regression) | Do NOT deploy this model in production. Collect richer data (medical history, lifestyle). |

---

## 🛠️ Technical Stack

| Layer | Tools & Libraries |
| :--- | :--- |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Statistical Testing** | SciPy (Shapiro, Levene, T-test, Chi-square, Pearson) |
| **Machine Learning** | Scikit-learn (LR, RF), XGBoost |
| **Deep Learning** | TensorFlow / Keras (ANN) |
| **Explainability** | SHAP (SHapley Additive exPlanations) |
| **Environment** | Google Colab / Jupyter Notebook |

---


---

## 🔬 Project Workflow (Step-by-Step)

### 1. Exploratory Data Analysis (EDA)
- Checked for missing values (none found).
- Visualized distributions (Age, BMI, Claim Amount) and categorical relationships.
- Identified class imbalance (85% Approved, 15% Denied).

### 2. Hypothesis Testing (Inferential Statistics)
Before building ML models, we ran 6 formal statistical tests to separate real signals from random noise:

| Hypothesis | Test Used | Result |
| :--- | :--- | :--- |
| Smokers vs Non-Smokers claim amount | T-test (Welch's) | **Significant** (p < 0.001) |
| Claim Amount across Regions | ANOVA | Not Significant (p > 0.05) |
| Smoking vs Claim_Status | Chi-square | Not Significant (p = 0.367) |
| Diabetes vs Claim_Status | Chi-square | Not Significant (p = 0.279) |
| Gender vs Claim_Status | Chi-square | Not Significant (p = 0.023)* |
| Age vs Claim_Amount | Pearson Correlation | Weak but Significant (r=0.05, p=0.001) |
| BMI vs Claim_Amount | Pearson Correlation | Not Significant (p = 0.45) |

* *Bonferroni correction applied (α=0.0125) – Gender effect became insignificant.*

### 3. Feature Engineering
- Created `High_Claim` binary target (Top 25% claims).
- Label-Encoded categorical variables (Gender, Smoking, Diabetes).
- Built **interaction features**: `Smoking_Age`, `Smoking_BMI`, `Diabetes_Age` to capture combined effects.
- Scaled numeric features using `StandardScaler`.
- Split data into 80% training / 20% testing.

### 4. Machine Learning (Classification & Regression)

#### A) Classification (Predict High-Cost Members)
| Model | PR-AUC | ROC-AUC | F1-Score |
| :--- | :--- | :--- | :--- |
| Logistic Regression | 0.485 | 0.552 | 0.425 |
| Random Forest | 0.510 | 0.580 | 0.452 |
| **XGBoost (Best)** | **0.515** | **0.590** | **0.448** |

#### B) Regression (Predict Exact Claim Amount)
| Model | RMSE ($) | MAE ($) | R² |
| :--- | :--- | :--- | :--- |
| Linear Regression | 1,420 | 1,150 | 0.042 |
| Random Forest | 1,380 | 1,120 | 0.058 |
| **XGBoost (Best)** | **1,360** | **1,100** | **0.062** |

### 5. Deep Learning Benchmark (ANN)
Built a 3-layer Artificial Neural Network to benchmark against XGBoost.
- **Result:** XGBoost outperformed the ANN.
- **Conclusion:** For tabular structured data, tree-based ensemble methods (XGBoost) are more robust and computationally efficient than neural networks.

### 6. Model Explainability (SHAP)
Used SHAP to explain the XGBoost model.
- **Key Insight:** `Smoking_Age` and `Age` are the top two drivers pushing predictions higher.
- **Visualization:** SHAP summary plots (bar charts & bee-swarm plots) are included in the notebook.

---

## 📈 Final Recommendations

1. **Do NOT deploy the current ML models.** They lack predictive power.
2. **Collect more granular data:** Medical history, emergency visits, prior year claims, lifestyle details (diet, exercise).
3. **Target smokers** with wellness/preventive programs to reduce long-term claim costs.
4. **Monitor approval processes:** Ensure no unintended bias (Gender was borderline significant, worth tracking over time).

---

## 🚀 How to Run This Project

### Prerequisites
- Python 3.10+
- Google Colab (recommended) or Jupyter Notebook

### Steps
1. **Clone the repository**
   ```bash
   git clone 
