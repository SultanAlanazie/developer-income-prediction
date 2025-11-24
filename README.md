

# **Predicting Developer Income – CRISP-DM Machine Learning Project**

**by Sultan Alanazi**

This repository contains a complete data science project following the **CRISP-DM framework**, using the **Stack Overflow Developer Survey 2023** dataset.
The goal is to explore developer demographics and build a machine learning model that predicts whether a respondent belongs to a **High Income** or **Low Income** category.

---

# 📌 **Project Motivation**

Developer salaries vary widely across countries, age groups, and education levels.
For organizations, understanding these factors helps with:

* Workforce salary planning
* Recruitment and HR analytics
* Understanding global developer compensation trends

This project aims to:

1. Analyze how country, age, and education impact income
2. Build a predictive model for income level
3. Compare multiple ML models
4. Demonstrate a real deployment-use prediction scenario

---

# 📚 **Libraries Used**

The project uses the following Python libraries:

```python
import pandas as pd
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import tree
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
%matplotlib inline
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score, roc_curve
)
```

---

# 📁 **Repository Structure**

```
📦 developer-income-prediction
 ┣ 📜 README.md
 ┣ 📜 survey_income_prediction.ipynb   → Main project notebook (CRISP-DM)
 ┣ 📜 survey_results_public.csv        → Source dataset
 ┣ 📜 requirements.txt                 → List of Python dependencies
```

---

# 🔍 **Summary of the Analysis (CRISP-DM)**

### **1. Business Understanding**

We investigate how demographics and education affect income and build a model to predict High vs. Low income.

---

### **2. Data Understanding**

* Dataset shape: **65,437 rows**, **114 columns**
* Key features selected:

  * `CountryGroup`
  * `Age`
  * `EdLevel`
  * `ConvertedCompYearly` (income)

---

### **3. Data Preparation**

Steps included:

* Removing salary outliers
* Creating High/Low income categories
* Mapping age ranges to numeric values
* Label-encoding education
* One-hot encoding country
* Splitting into training/testing data

---

### **4. Modeling**

The following models were trained:

* Decision Tree
* Random Forest
* Gradient Boosting
* Logistic Regression

---

### **5. Evaluation**

| Model                 | Accuracy   | Precision  | Recall     | F1 Score   | ROC AUC    |
| --------------------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| Decision Tree         | 0.7113     | 0.7285     | 0.6914     | 0.7095     | 0.7890     |
| Random Forest         | 0.7364     | 0.7489     | 0.7266     | 0.7376     | 0.8148     |
| **Gradient Boosting** | **0.7376** | 0.7452     | **0.7374** | **0.7413** | **0.8175** |
| Logistic Regression   | 0.7289     | **0.7682** | 0.6705     | 0.7160     | 0.8067     |

📌 **Best Overall Model: Gradient Boosting**

---

### **6. Deployment Scenario**

A company can use the Gradient Boosting model to predict expected income level of a new survey respondent based on:

* Country
* Age
* Education Level

This can support HR planning, salary benchmarking, or targeted recruitment.

---

# 🧠 **Key Insights**

### ✔ Countries like the US, UK, and Germany show higher income levels

### ✔ Older developers generally earn more

### ✔ Higher education significantly increases the chance of High Income

### ✔ Ensemble models outperform simple models

### ✔ Gradient Boosting provides the strongest predictive performance

---

# ⚠️ **Model Limitations**

* Only 3 demographic features used — income depends on many more factors
* Self-reported data → may contain bias
* Salary outliers removed → may lose extreme but valid cases
* Gradient Boosting is harder to interpret

---

# 🚀 **Future Improvements**

* Add more features (`YearsCodePro`, job type, industry, company size)
* Hyperparameter tuning with GridSearchCV
* Test advanced models like XGBoost or LightGBM
* Build a small API or Streamlit deployment

---

# 🙏 **Acknowledgements**

* Dataset: **Stack Overflow Developer Survey**
* Tools & Libraries: Python, Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn
* References: Documentation, StackOverflow, Kaggle tutorials


