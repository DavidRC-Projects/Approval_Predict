## Cloud IDE Reminders

To log into the Heroku toolbelt CLI:

1. Log in to your Heroku account and go to _Account Settings_ in the menu under your avatar.
2. Scroll down to the _API Key_ and click _Reveal_
3. Copy the key
4. In the terminal, run `heroku_config`
5. Paste in your API key when asked


You can now use the `heroku` CLI program - try running `heroku apps` to confirm it works. This API key is unique and private to you so do not share it. If you accidentally make it public then you can create a new one with _Regenerate API Key_.


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

- An **applicant** is a person who has submitted a loan application to the financial institution.
- An **approved loan** is a loan application that has been accepted by the institution and will receive funding.
- A **rejected loan** (or rejected application) is a loan application that has been denied by the institution.
- **Credit score** is a numerical representation of an applicant's creditworthiness.
- **Points** (mortgage points or discount points) are fees paid upfront to reduce the interest rate on a loan. One point equals 1% of the loan amount.
- **Years employed** refers to the length of time an applicant has been with their current employer, which is used to assess employment stability.
- **Loan-to-income ratio** (LTI) compares the loan amount to the applicant's income.



## Business Requirements

The client is a fictitious financial institution that processes a large number of loan applications daily. Manual review of loan applications is time-consuming and resource-intensive. The institution wants to understand what factors contribute most to loan approval decisions and whether loan approvals can be accurately predicted to streamline operations and improve risk assessment.

1. The client is interested in determining which applicant variables are most strongly correlated with the loan approval outcome. They want a ranked list of variables to be provided based on their relevance and impact.
2. The client aims to offer an evidence-based guide for potential applicants by identifying the top three influential factors that contribute to loan approval. These insights will be used to recommend specific improvements applicants can make to increase their chances of having a loan approved.



## Hypothesis and how to validate?

The client wants to know whether the data supports the following hypotheses:

- **Hypothesis 1**: Applicants with higher credit scores tend to have higher approval rates.
  - A correlation study can help in this investigation
- **Hypothesis 2**: Applicants who have been employed longer have higher approval rates than those with shorter employment history.
  - A correlation study can help in this investigation
- **Hypothesis 3**: Applications with higher loan amounts relative to income (LTI ratio) have lower approval rates.
  - A correlation study can help in this investigation
  - Create loan-to-income ratio (LTI) = loan_amount / income as a derived feature


## The rationale to map the business requirements to the Data Visualizations and ML tasks
- **Business Requirement 1:** Data Visualization and Correlation study
	- We will inspect the data related to the applicant base.
	- We will conduct a correlation study (Pearson and Spearman) to understand better how the variables are correlated to loan approval.
	- We will plot the main variables against loan approval to visualize insights.
	- We will analyse approval patterns by income ranges, credit score ranges, loan amounts, employment history, and loan-to-income ratios.

- **Business Requirement 2:** Testing and Validating Hypotheses
	- We want to test whether higher credit scores are associated with higher approval rates (Hypothesis 1).
	- We want to test whether longer employment history is associated with higher approval rates (Hypothesis 2).
	- We want to test whether higher loan-to-income ratios are associated with lower approval rates (Hypothesis 3).
	- We will use correlation studies and statistical tests to validate these hypotheses.


- **Business Requirement 3:** Classification and Risk Assessment
	- We want to predict if an applicant will be approved or not. We want to build a binary classifier.
	- We want to predict the probability of approval for an applicant to assess risk level.
	- We want to identify which features are most important for approval decisions.
	- We want to understand approval patterns to guide applicants on improving their chances of approval.


## ML Business Case

We want an ML model to predict if a loan application will be approved based on historical data from the loan application database. The target variable is categorical and contains 2-classes. We consider a classification model. It is a supervised model, a 2-class, single-label, classification model output: 0 (not approved), 1 (approved).

Our ideal outcome is to provide our loan officers with reliable insight into processing applications with higher accuracy and efficiency.

**The model success metrics are:**
- At least 80% Recall for Approved, on train and test set
- At least 80% Precision for Approved, on train and test set




## Dashboard Design
* List all dashboard pages and their content, either blocks of information or widgets, like buttons, checkboxes, images, or any other item that your dashboard library supports.
* Later, during the project development, you may revisit your dashboard plan to update a given feature (for example, at the beginning of the project you were confident you would use a given plot to display an insight but subsequently you used another plot type).



## Unfixed Bugs
* You will need to mention unfixed bugs and why they were not fixed. This section should include shortcomings of the frameworks or technologies used. Although time can be a significant variable to consider, paucity of time and difficulty understanding implementation is not a valid reason to leave bugs unfixed.
Added the code ! unzip {DestinationFolder}/*.zip -d {DestinationFolder} \
    && rm {DestinationFolder}/*.zip \
    && rm kaggle.json
    this required - o to overwrite the zip command as there as already a csv file uploaded




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

