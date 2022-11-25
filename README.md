# Lending_club_loans

# PURPOSE

The aim of this project was to create a Machine Learning tool for Lending club, which could predict the acceptance/rejection of the loan application, grade, subgrade and the interest rate of the loan.


# STRUCTURE:

- **EDA_lending_club.ipynb**
Deep analysis of the given data sets, data cleaning, feature engineering to prepare the final, clean data set for modeling. Inferential statistical analysis was proceeded.Trying to find patterns, which could indicate first of all:
 1. What are the most important features when deciding to approve or to reject a loan for the applicant;
 2. When it is decided to approve the loan, what features could impact the decision on the loan's: grade, sub_garde and the interest rate.

- **Modeling_lending_club.ipynb**

The success measure of the models - ROC AUC. Different models were explored to see, which could do best on this data. The two best models by ROC AUC were selected and developed further. The main idea was to check does Tree based models need feature pre-processing (scaling, one hot encoding) or you can easily throw raw features and tree based model will perform the same. So this idea was implemented, by testing models with different types of data (pre-processed and not). Under sampling was used. Circle meaning values (like months, time) were encoded by sin/cos values.

Also, SHAP values, permutation importance, feature importance by XGBoost was checked, was tried to find optimal probability threshold for the certain model, hyperparameter tuning was proceeded.

- **helpers[folder]**

Contain .py file with all classes and functions, used in the EDA and Modeling.

- **api[folder]**

Contains code of deploying model on localhost with fastapi, Dockerfile (app is 'dockerized')
and deployed to Google Cloud storage.
