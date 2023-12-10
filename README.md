# Loan-Prediction-Analysis-using-ML
The project aims to revolutionize and optimize the student loan approval process in the Indian banking sector through machine learning. By leveraging predictive models, the project seeks to objectively assess loan applications based on applicant characteristics, addressing inefficiencies, delays, and subjective decision-making.

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
---

# 🚀 Abstract

This project aims to address the inefficiencies in the student loan approval process within the Indian banking sector by harnessing machine learning techniques. The primary objective is to create predictive models capable of assessing loan applications based on applicant characteristics, thereby increasing efficiency, objectivity, and the speed of loan decisions.The process begins with data preparation, where the dataset is loaded, preprocessed,performed some visualization and statistical tests and then split into training and testing sets. Extensive data preprocessing and exploratory data analysis have been performed to understand the relationships between various features and loan approval outcomes. I have used different ML models, including Logistic Regression, Random Forest Classifier, Gradient Boost and Decision Trees. RFE estimator was used to select top 4 features for model development. After hyper tuning, Gradient Boost showed the most promising results with a training and testing accuracy of 98.13% and 97.4% respectively. The motivation stems from the inefficiencies in the current manual approach, and the project holds personal significance, grounded in a familial connection to the banking sector. Ultimately, the goal is to not only improve the loan approval process but also positively impact individuals seeking educational financing.

---

# 📊 Data Visualization


---

## 📈 Methods

### Data Preparation

...

### 📊 Data Visualization

Exploratory Data Analysis (EDA) involved a comprehensive set of visualizations to understand the distribution, relationships, and patterns within the dataset. The following visualizations highlight key aspects of the data:

#####  Distribution of Applicant Features:

Utilized histograms to visualize the distribution of key applicant features, including credit score, annual income, loan amount, and loan term.
Identified the central tendencies and spread of each feature to gain insights into the dataset's characteristics.
![1](https://github.com/prarthanasigedar/Loan-Prediction-Analysis-using-ML/blob/main/1.png)

##### Correlation Heatmap:

Generated a heatmap to visualize the correlation matrix of numerical features.
Identified strong correlations between certain features, providing insights into potential multicollinearity.
![4](https://github.com/prarthanasigedar/Loan-Prediction-Analysis-using-ML/blob/main/4.png)

##### Loan Approval Trends :

Explored seasonality or trends that may impact loan approval outcomes
![2](https://github.com/prarthanasigedar/Loan-Prediction-Analysis-using-ML/blob/main/2.png)
![3](https://github.com/prarthanasigedar/Loan-Prediction-Analysis-using-ML/blob/main/3.png)
    

### Model Development


#### Performance Evaluation

To assess the performance of the developed predictive models, various metrics were employed, providing a nuanced understanding of their effectiveness in automating and enhancing the loan approval process. The following metrics were chosen for their relevance in the context of loan approval scenarios:

F1 Score: The F1 score represents the harmonic mean of precision and recall, offering a balanced measure of a model's ability to make accurate positive predictions while correctly identifying actual positive cases. In scenarios like loan approval, where both precision and recall are crucial, the F1 score provides a comprehensive evaluation.

Accuracy: Accuracy is a measure of the overall correctness of the model's predictions. High accuracy indicates that the model generally makes correct loan approval predictions.

Precision: Precision is the ratio of correctly predicted positive observations to the total predicted positives. In the context of loan approval, it indicates how often the model correctly predicted "Approved" when it made a positive prediction.

Recall: Recall is the ratio of correctly predicted positive observations to the all observations in actual class. In the context of loan approval, it indicates how well the model identifies loan applications that should be approved.

| Model                   | F1 Score | Accuracy | Precision | Recall   |
|-------------------------|----------|----------|-----------|----------|
| Logistic Regression     | 91%      | 91.5%    | 88.42%    | 86.48%   |
| Decision Trees          | 97.13%   | 97.47%   | 95.7%     | 95.5%    |
| Random Forest Classifier| 97.29%   | 97.18%   | 97.48%    | 96.58%   |
| Gradient Boost          | 98.18%   | 97.89%   | 98.125%   | 96.3%    |


#### HyperParameter Tuning
Hyperparameter tuning is a crucial step in optimizing the performance of machine learning models. It involves systematically searching the hyperparameter space to find the set of hyperparameters that result in the best model performance. The following details outline the hyperparameter tuning process for each algorithm:

| **Model**               | **Before Tuning F1 Score** | **After Tuning F1 Score** | **Improvement** | **Hyperparameter Details**                                   |
|-------------------------|-----------------------------|----------------------------|------------------|------------------------------------------------------------|
| Logistic Regression     | 88%                         | 91%                        | 3%               | C = 0.1, Solver = 'lbfgs', Max Iterations = 100             |
| Decision Trees          | 97.13%                      | 97.62%                     | 0.50%            | Max Depth = 10, Min Samples Split = 2, Min Samples Leaf = 1 |
| Random Forest Classifier| 97.29%                      | 97.40%                     | 0.11%            | N Estimators = 100, Max Depth = 12, Min Samples Split = 2   |
| Gradient Boost          | 97.40%                      | 98.18%                     | 0.94%            | Learning Rate = 0.1, N Estimators = 200, Max Depth = 5     |

#### Confusion Metrix
A confusion matrix is a valuable tool for understanding the performance of a classification model. It breaks down the model's predictions into four categories: true positive (TP), true negative (TN), false positive (FP), and false negative (FN).

|                  | Predicted: No Approval | Predicted: Approval |
|------------------|------------------------|----------------------|
| Actual: No Approval | 261                   | 5                   |
| Actual: Approval    | 3                   | 158                   |


True Positive (TP): The model correctly identified 158 approved loan applications.
True Negative (TN): The model correctly identified 261 denied loan applications.
False Positive (FP): The model incorrectly predicted that 5 loan applications would be approved, but they were actually denied.
False Negative (FN): The model incorrectly predicted that 3 loan applications would be denied, but they were actually approved.


---
## 🌟 Conclusion

The project successfully addresses inefficiencies in the student loan approval process through machine learning. The Gradient Boost model emerged as the top-performing model, achieving training and testing accuracies of 97.40% and 98.18%, respectively. The project holds significance not only in improving the loan approval process but also in positively impacting individuals seeking educational financing.

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

- **Email:** prarthana.sigedar@gmail.com
- **LinkedIn:** [Prarthana Sigedar](https://www.linkedin.com/in/psigedar/)
---

## 📝 License

This project is licensed under the [MIT License](LICENSE).

---
