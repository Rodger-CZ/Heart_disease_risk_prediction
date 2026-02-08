# 🫀 Heart Disease Risk Prediction (Healthcare ML Project)

## 📌 Project Overview

This project builds an end-to-end machine learning pipeline to predict the probability of heart disease using structured clinical data.

The objective is to develop a reliable predictive model that can support **early risk identification**, enabling preventive interventions and improved healthcare decision-making.

Although the dataset originates from a Kaggle Playground competition, the project is structured as a real-world healthcare analytics case study, emphasizing model reliability, validation, and interpretability.

---

## 🎯 Objectives

* Analyze clinical risk factors associated with heart disease
* Build a robust preprocessing pipeline for mixed data types
* Develop baseline and advanced predictive models
* Evaluate performance using appropriate medical-risk metrics (ROC AUC)
* Ensure model stability through stratified cross-validation
* Generate probabilistic risk predictions for unseen patients

---

## 🛠️ Tools & Technologies

* **Python**
* **Jupyter Notebook**
* pandas, numpy
* matplotlib, seaborn
* scikit-learn

---

## 📂 Project Structure

```
heart-disease-risk-prediction/
│
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
│
├── notebooks/
│   └── heart_disease_prediction_s6e2.ipynb
│
├── submission.csv
├── README.md
└── requirements.txt
```

---

## 🔍 Methodology

### 1. Data Preparation

* Automatic target identification based on train/test schema differences
* Missing value handling using median (numeric) and most-frequent (categorical) imputation
* One-hot encoding for categorical clinical variables
* Feature scaling for linear model stability

A unified **ColumnTransformer pipeline** ensures reproducibility and prevents data leakage.

---

### 2. Model Development

| Model                            | Purpose                                                    |
| -------------------------------- | ---------------------------------------------------------- |
| Logistic Regression              | Baseline interpretable clinical risk model                 |
| HistGradientBoostingClassifier   | Captures non-linear relationships and feature interactions |
| Ensemble (Average Probabilities) | Improves prediction stability                              |

Stratified 5-fold cross-validation was used to preserve class distribution.

---

## 📊 Results

| Model                | CV ROC AUC |
| -------------------- | ---------- |
| Logistic Regression  | ~0.950     |
| HistGradientBoosting | ~0.955     |

### Kaggle Leaderboard Performance

* **Public ROC AUC:** 0.95284
* Rank: ~Top 1,000 (first submissions)

The close alignment between cross-validation and leaderboard scores indicates strong generalization and a reliable validation strategy.

---

## 💡 Healthcare Insights

* Tree-based models capture complex interactions between clinical risk factors that linear models may miss.
* High ROC AUC suggests strong ability to distinguish high-risk from low-risk patients.
* Such models could support:

  * Preventive screening programs
  * Risk stratification in clinical settings
  * Population health management

---

## ⚠️ Disclaimer

This project is for educational and portfolio purposes only. The model is trained on synthetic/competition data and is **not intended for clinical use**.

---

## 🚀 Future Improvements

* Model interpretability (SHAP feature importance)
* Threshold optimization for clinical decision support
* Deployment as a lightweight risk scoring API
* External validation on real-world clinical datasets

---

## 👤 Author

**Faustine Rodgers**
Data Analyst | Machine Learning | Healthcare Analytics

