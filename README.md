# Loan-Prediction-Analysis-using-ML
The aim of this project was to develop a predictive model for loan approval based on applicant characteristics. The focus is on determining the probability of loan approval using a broad set of features such as income, credit history, and employment status.

# About Dataset

The loan approval dataset is a collection of financial records and associated information used to determine the eligibility of individuals or organizations for obtaining loans from a lending institution. It includes various factors such as cibil score, income, employment status, loan term, loan amount, assets value, and loan status. This dataset is commonly used in machine learning and data analysis to develop models and algorithms that predict the likelihood of loan approval based on the given features.

We can also find the dataset over kaggle.com: https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset?sort=published


About columns (Information provided by the owner)

    loan_id
    no_of_dependents: Number of Dependents of the Applicant
    education: Education of the Applicant (Graduate/Not Graduate)
    self_employed: Employment Status of the Applicant
    income_annum: Annual Income of the Applicant
    loan_amount: Loan Amount
    loan_term: Loan Term in Years
    cibil_score: Credit Score
    residential_assets_value
    commercial_assets_value
    luxury_assets_value
    bank_asset_value
    loan_status: Loan Approval Status (Approved/Rejected)

# Installation and Setup

    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    from scipy.stats import chi2_contingency
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from xgboost import XGBClassifier
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.feature_selection import RFE

# Predictive Model for Loan Approval Outcomes

## Overview

The project aims to revolutionize and optimize the student loan approval process in the Indian banking sector through machine learning. By leveraging predictive models, the project seeks to objectively assess loan applications based on applicant characteristics, addressing inefficiencies, delays, and subjective decision-making.

---

## 🚀 Project Summary

### Report Summary

...

### Motivation

...

---

## 📊 Data Visualization

#### Model Evaluation Metrics

| Model                   | F1 Score | Accuracy | Precision | Recall   |
|-------------------------|----------|----------|-----------|----------|
| Logistic Regression     | 91%      | 91.5%    | 88.42%    | 86.48%   |
| Decision Trees          | 97.13%   | 97.47%   | 95.7%     | 95.5%    |
| Random Forest Classifier| 97.29%   | 97.18%   | 97.48%    | 96.58%   |
| Gradient Boost          | 98.18%   | 97.89%   | 98.125%   | 96.3%    |

#### Model Iteration

...

---

## 🌟 Conclusion

The project successfully addresses inefficiencies in the student loan approval process through machine learning. The Gradient Boost model emerged as the top-performing model, achieving training and testing accuracies of 97.40% and 98.18%, respectively. The project holds significance not only in improving the loan approval process but also in positively impacting individuals seeking educational financing.

---

## 🛠️ Project Description

### Dataset

...

### Intended Discoveries (Aims)

...

---

## 📈 Methods

### Data Preparation

...

### Exploratory Data Analysis (EDA)

...

### Model Development

...

### Performance Evaluation

...

---

## 🌐 [Live Demo](#) | 📄 [Download Full Report](#)

---

## 📌 How to Contribute

If you'd like to contribute to this project, please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature-name`).
3. Commit your changes (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature/your-feature-name`).
5. Open a pull request.

---

## 📫 Contact Information

For any inquiries or feedback, feel free to contact me:

- **Email:** your.email@example.com
- **LinkedIn:** [Your Name](https://www.linkedin.com/in/yourname/)
- **Twitter:** [@yourtwitter](https://twitter.com/yourtwitter)

---

## 📝 License

This project is licensed under the [MIT License](LICENSE).

---
