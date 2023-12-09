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

# Data Pre-Processing
    data = pd.read_csv('loan_approval_dataset.csv')
