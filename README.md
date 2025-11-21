# Approval Predictor - A Predictive Classification Model for Determining Loan Approval Outcomes

Approval Predictor is a machine-learning (ML) project using a publicly available dataset to determine whether a ML pipeline could be built to predict whether a loan application will be approved or rejected. This was achieved by using a classification task, using the `loan_approved` attribute from the dataset as the target and the remaining attributes as features.

## Table of Contents

- [Dataset Content](#dataset-content)
- [Business Requirements](#business-requirements)
- [Hypothesis](#hypothesis)
- [Mapping Business Requirements to Data Visualisation and ML Tasks](#mapping-business-requirements-to-data-visualisation-and-ml-tasks)
- [ML Business Case](#ml-business-case)
- [Epics and User Stories](#epics-and-user-stories)
- [Dashboard Design](#dashboard-design)
- [Technologies Used](#technologies-used)
- [Testing](#testing)
- [Unfixed Bugs](#unfixed-bugs)
- [Deployment](#deployment)
- [Credits](#credits)
- [Acknowledgements](#acknowledgements)

## Dataset Content

The dataset is sourced from [Kaggle](https://www.kaggle.com/datasets/anishdevedward/loan-approval-dataset/data?select=loan_approval.csv). We created then a fictitious user story where predictive analytics can be applied in a real project in the workplace.

This dataset shows loan applications and approval outcomes for 2,000 individuals. It contains demographic, financial, and employment-related attributes that can be used to predict whether a loan application will be approved or rejected. The data has 8 columns and 2000 rows, this dataset size will be useful as its large enough to hopefully demonstrate patterns in the data and small enough to run quickly in codespace.

```python
df = pd.read_csv('/kaggle/input/loan-approval-dataset/loan_approval.csv')
```

| Variable          | Meaning                                                        | Units                                                         |
|-------------------|----------------------------------------------------------------|---------------------------------------------------------------|
| name              | Applicant name                                                 | Text for name                                            |
| city              | Applicant's city/location                                      | Text for location (City)                     |
| income            | Applicant's annual income                                      | Dollar amount (30,000 - 150,000)                              |
| credit_score      | Applicant's credit score                                       | Numeric score (300 - 850)                                     |
| loan_amount       | Amount of loan requested                                       | Dollar amount (1,037 - 49,999)                                |
| years_employed    | Number of years with current employer                         | Years (0 - 40)                                                |
| points            | Mortgage discount points paid upfront                         | Points (10 - 100)                                             |
| loan_approved     | Whether loan was approved or not                               | True or False (Target variable)                               |


## Project Terms & Jargon

- An **applicant**:  Person who has submitted a loan application to the financial institution.
- An **approved loan**: Loan application that has been accepted and will receive funding.
- A **rejected loan**: Loan application that has been rejected and has been denied for funding.
- **Credit score**: Numerical representation of an applicant's creditworthiness.
- **Points**: Mortgage points are fees paid upfront to reduce the interest rate on a loan. One point equals 1% of the loan amount.
- **Years employed**: The length of time an applicant has been with their current employer, which is used to assess employment stability.
- **Loan-to-income ratio(LTI)**: Compares the loan amount to the applicant's income. This helps assess risk and affordability.



## Business Requirements

The client is a fictitious financial institution that processes a large number of loan applications daily. Manual review of loan applications is time-consuming and resource-intensive. The institution wants to understand what factors contribute most to loan approval decisions and whether loan approvals can be accurately predicted to streamline operations and improve risk assessment.

**Business Requirement 1** - The client is interested in determining which applicant variables are most strongly correlated with the loan approval outcome. They want a ranked list of variables to be provided based on their relevance and impact.

**Business Requirement 2** - The client aims to offer a guide for potential applicants by identifying the influential factors that contribute to loan approval. These insights will be used to recommend specific improvements for applicants and guide them to increase their chances of having a loan approved.

## Hypothesis and how to validate?

The client wants to know whether the data supports the following hypotheses:

- **Hypothesis 1**: Applicants with higher credit scores tend to have higher approval rates.
- Validation: A correlation analysis that indicates a strong relationship between credit score and the target loan_approved.
- **Hypothesis 2**: Applicants who have been employed longer have higher approval rates than those with shorter employment history.
- Validation: A correlation study that measures the relationship between years employed and loan approval outcomes.
- **Hypothesis 3**: Applications with higher loan amounts relative to income (LTI ratio) have lower approval rates.
- Validation: Analysis of the loan-to-income ratio feature, created as a new feature (loan_amount / income), and its correlation with approval outcomes.

- **Null Hypothesis**
-There is no significant relationship between applicant features and loan approval outcomes. Approval decisions cannot be predicted from these features due to insufficient data.

[Back to top](#approval-predictor---a-predictive-classification-model-for-determining-loan-approval-outcomes)

## Mapping Business Requirements to Data Visualisation and ML Tasks

### Business Requirement 1: Data Visualisation and Correlation study

- We will inspect the data related to the applicant base.
- We will conduct a correlation study (Pearson and Spearman) to understand better how the variables are correlated to loan approval.
- We will plot the main variables against loan approval to visualise insights.
- We will analyse approval patterns by income ranges, point ranges, credit score ranges, loan amounts, employment history, and loan-to-income ratios.
- This will be carried out during the Data Visualisation, Cleaning, and Preparation.

### Business Requirement 2: Classification Model

- We need to predict whether a loan application will be approved or not.
- Therefore we need to build a binary classification model.
- A machine learning pipeline will be able to map the relationships between the features and target.
- Hyperparameter optimisation will give us the best chance at a highly accurate prediction.
- The model will provide actionable guidance to applicants on how to improve their approval odds.
- This will be carried out during the Model Training and Optimisation.

[Back to top](#approval-predictor---a-predictive-classification-model-for-determining-loan-approval-outcomes)

## ML Business Case

We want an ML model to predict if a loan application will be approved based on historical data from the loan application database. The target variable is categorical and contains 2-classes. We consider a classification model. It is a supervised model, a 2-class, single-label, classification model output: 0 (not approved), 1 (approved).

Our ideal outcome is to provide our loan officers with reliable insight into processing applications with higher accuracy and efficiency.

**The model success metrics are:**
- At least 80% Recall for Approved, on train and test set
- At least 80% Precision for Approved, on train and test set




## Dashboard Design

Page 1: Quick Project Summary
Quick summary of the Approval Prediction project.
Description of the dataset.
State the business requirements:
Build a model to predict loan approval with 80%+ precision and recall.
Provide actionable insights to help applicants improve their approval odds.

Page 2: Feature Impact Study (Legacy Model with Points)
Business requirement addressed: Analyse the legacy model that uses the points feature.
Before analysis we expected this page to highlight the perfect accuracy achieved with points.
After analysis, the page shows:
Business requirement 1 results using the points-driven model.
Checkbox: Explore dataset shape and preview records.
Highlight that points has 100% feature importance and explain why this is problematic.
Checkbox: Plots showing approval split by points threshold.
Checkbox: Parallel plot comparing points vs other features.

Page 3: Approval Predictor (Without Points)
Business requirement addressed: Provide a reliable prediction tool using applicant features only.
Input widgets for applicant profile:
Credit score, income, loan amount, years employed and loan_to_income.
“Run Predictive Analysis” button:
Sends values to the optimised LogisticRegression pipeline.
Predicts approval outcome and associated probability.
Provides tailored guidance based on feature contributions (e.g., “Increase credit score above X”).

Page 4: Project Hypotheses and Findings
Before analysis, we listed hypotheses to validate. After analysis, we can report:
Applicants with higher points always get approved.
Confirmed: points feature dominates approval decisions.
Approval probability is predictable using others features and removing points.
Confirmed in Notebook 06: credit_score, loan_to_income, and income drive final predictions.
Confirmed: Final model gives 87% recall / 86% precision with actionable insights.

Page 5: Final ML Model (Without Points)
Key takeaways after training the final pipeline (Notebook 06).
ML pipeline steps:
BoxCox transformation (loan_to_income only).
Robust scaling.
LogisticRegression (C=0.1, penalty=l2, class_weight=None).
Feature importance chart showing:
credit_score, loan_to_income, income, years_employed, loan_amount.
Pipeline performance metrics:
Train: ~92% precision/recall.
Test: 87% recall, 86% precision.


## Unfixed Bugs
* You will need to mention unfixed bugs and why they were not fixed. This section should include shortcomings of the frameworks or technologies used. Although time can be a significant variable to consider, paucity of time and difficulty understanding implementation is not a valid reason to leave bugs unfixed.
Added the code ! unzip {DestinationFolder}/*.zip -d {DestinationFolder} \
    && rm {DestinationFolder}/*.zip \
    && rm kaggle.json
    this required - o to overwrite the zip command as there as already a csv file uploaded

	ValueError: Could not interpret value `loan_to_income` for `x`. An entry with this name does not appear in `data`.
	Fixed the plot numerical function by adding df['loan_to_income'] = df['loan_amount'] / df['income'] as when plot_numerical was called it was unable to find the loan_to_income variable. This code had already been noted prior to the function. 




## Deployment
### Heroku

* The App live link is: https://YOUR_APP_NAME.herokuapp.com/ 
* Set the runtime.txt Python version to a [Heroku-24](https://devcenter.heroku.com/articles/python-support#supported-runtimes) stack currently supported version.
* The project was deployed to Heroku using the following steps.

1. Log in to Heroku and create an App
2. At the Deploy tab, select GitHub as the deployment method.
3. Select your repository name and click Search. Once it is found, click Connect.
4. Select the branch you want to deploy, then click Deploy Branch.
5. The deployment process should happen smoothly if all deployment files are fully functional. Click now the button Open App on the top of the page to access your App.
6. If the slug size is too large then add large files not required for the app to the .slugignore file.


## Main Data Analysis and Machine Learning Libraries
* Here you should list the libraries you used in the project and provide an example(s) of how you used these libraries.


## Credits 

* In this section, you need to reference where you got your content, media and extra help from. It is common practice to use code from other repositories and tutorials, however, it is important to be very specific about these sources to avoid plagiarism. 
* You can break the credits section up into Content and Media, depending on what you have included in your project. 

### Content 

- The text for the Home page was taken from Wikipedia Article A
- Instructions on how to implement form validation on the Sign-Up page were taken from [Specific YouTube Tutorial](https://www.youtube.com/)
- The icons in the footer were taken from [Font Awesome](https://fontawesome.com/)

### Media

- The photos used on the home and sign-up page are from This Open-Source site
- The images used for the gallery page were taken from this other open-source site



## Acknowledgements (optional)
* Thank the people who provided support through this project.



Hyperparameter testing:




https://docs.streamlit.io/develop/api-reference/status for markdown for error handling.
https://docs.streamlit.io/develop/api-reference/data/st.column_config/st.column_config.progresscolumn?utm_source=streamlit. for progress column.

