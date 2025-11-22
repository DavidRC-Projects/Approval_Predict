import pandas as pd
import streamlit as st

from src.data_management import load_loan_approval_data, load_pkl_file
from src.machine_learning.approval_predictor import predict_loan_approval
from src.approval_guidance import guidance


def page_approval_predictor_body():
    """
    Streamlit page for the Approval Predictor.
    Loads the trained classification pipeline.
    Loads dataset and displays value ranges for the user to follow.
    Creates a single row dataframe containing applicants inputs.
    Has button to run prediction and display results.
    """
    st.title("ML Pipeline: Approval Predictor")

    version = "v2"
    pipeline = load_pkl_file(
        f"outputs/ml_pipeline/approval_prediction/"
        f"{version}/classification_pipeline.pkl"
    )

    st.info(
        "#### **Business Requirement 2**: Applicant Guidance\n\n"
        "* Predict if an applicant will be approved using a Logistic "
        "Regression pipeline trained.\n"
        "* Success metrics: ≥80% recall and ≥80% precision on the "
        "approved class (train & test).\n"
        "* Provide transparent feedback to help applicants improve "
        "their approval odds."
    )

    dataset = load_dataset()
    ranges = {
        "income": (
            int(dataset["income"].min()),
            int(dataset["income"].max()),
        ),
        "loan_amount": (
            int(dataset["loan_amount"].min()),
            int(dataset["loan_amount"].max()),
        ),
        "credit_score": (
            int(dataset["credit_score"].min()),
            int(dataset["credit_score"].max()),
        ),
        "years_employed": (
            int(dataset["years_employed"].min()),
            int(dataset["years_employed"].max()),
        ),
    }
    st.markdown(
        (
            "Please use the input ranges (based on training data). Adjust the "
            "values below to see how they impact loan approval.\n"
            f"- **Income:** `${ranges['income'][0]:,} — "
            f"${ranges['income'][1]:,}`\n"
            f"- **Loan amount:** `${ranges['loan_amount'][0]:,} — "
            f"${ranges['loan_amount'][1]:,}`\n"
            f"- **Credit score:** `{ranges['credit_score'][0]} — "
            f"{ranges['credit_score'][1]}`\n"
            f"- **Years employed:** `{ranges['years_employed'][0]} — "
            f"{ranges['years_employed'][1]} years`"
        )
    )
    X_live = draw_input_widgets(dataset)

    if st.button("Run Predictive Analysis"):
        prediction = predict_loan_approval(X_live, pipeline)
        guidance(X_live, dataset, prediction)
        display_loan_to_income(X_live)
        display_loan_to_income_summary(X_live, dataset)


def load_dataset():
    """
    Loads and prepares the loan approval dataset.
    Ensures that loan_to_income feature is present for analysis.
    """
    df = load_loan_approval_data()
    if "loan_to_income" not in df.columns:
        df["loan_to_income"] = df["loan_amount"] / df["income"]
    return df


def draw_input_widgets(df):
    """
    Builds Streamlit input widgets.
    Returns a single-row DataFrame containing all user inputs
    and calculated LTI.
    """
    X_live = pd.DataFrame(index=[0])

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        feature = "income"
        widget = st.number_input(
            label="Annual income ($)",
            min_value=int(df[feature].min()),
            max_value=int(df[feature].max()),
            value=int(df[feature].median()),
            step=1000,
            format="%d",
        )
        X_live[feature] = widget

    with col2:
        feature = "loan_amount"
        widget = st.number_input(
            label="Requested loan amount ($)",
            min_value=int(df[feature].min()),
            max_value=int(df[feature].max()),
            value=int(df[feature].median()),
            step=500,
            format="%d",
        )
        X_live[feature] = widget

    with col3:
        feature = "credit_score"
        widget = st.number_input(
            label="Credit score",
            min_value=int(df[feature].min()),
            max_value=int(df[feature].max()),
            value=int(df[feature].median()),
            step=5,
        )
        X_live[feature] = widget

    with col4:
        feature = "years_employed"
        widget = st.number_input(
            label="Years with current employer",
            min_value=int(df[feature].min()),
            max_value=int(df[feature].max()),
            value=int(df[feature].median()),
            step=1,
        )
        X_live[feature] = widget

    X_live["loan_to_income"] = X_live["loan_amount"] / X_live["income"]
    return X_live


def display_loan_to_income(X_live):
    """
    Display the applicant's calculated loan-to-income (LTI) ratio.
    The X_live parameter has the calculated LTI value.
    """
    lti_value = float(X_live.at[0, "loan_to_income"])

    st.metric(
        label="Estimated loan-to-income ratio",
        value=f"{lti_value:.2f}",
    )


def display_loan_to_income_summary(X_live, dataset):
    """
    Evaluate the applicant's loan-to-income (LTI) ratio
    using quartile-based risk bands.
    The Q1 (0-25%) gives a low-risk affordability band.
    The Q2 (25-50%) gives a below-average risk band.
    The Q3 (50-75%) gives a moderate risk band.
    Above Q3 (75%) gives a high risk band.
    This then provies an explanation of affordability
    risk and improvement tips.
    """
    lti = float(X_live.at[0, "loan_to_income"])

    q1 = dataset["loan_to_income"].quantile(0.25)
    q2 = dataset["loan_to_income"].quantile(0.50)
    q3 = dataset["loan_to_income"].quantile(0.75)

    if lti <= q1:
        band = "Low Risk"
        msg = (
            "Your loan-to-income ratio is in the **lowest 25%** "
            "of applicants.\n"
            "This indicates **excellent affordability** and strongly "
            "supports approval."
        )

    elif lti <= q2:
        band = "Below Average Risk"
        msg = (
            "Your loan-to-income ratio is below the **median "
            "applicant**.\n"
            "This is a **favourable affordability range**, "
            "improving approval likelihood."
        )

    elif lti <= q3:
        band = "Moderate Risk"
        msg = (
            "Your ratio is in the **upper-middle range**.\n"
            "You may still be approved, but I'd recommend lowering "
            "your loan amount or increasing income."
        )

    else:
        band = "High Risk"
        msg = (
            "Your loan-to-income ratio is in the **highest 25%** "
            "of applicants.\n"
            "This signals **potential affordability concerns** and "
            "may significantly reduce approval chances.\n"
            "Reducing the loan amount or increasing income would "
            "improve this."
        )

    st.metric(
        label="Loan-to-Income Risk Band",
        value=f"{band}"
    )

    st.info(
        f"### Loan-to-Income (LTI) Evaluation\n"
        f"**Your LTI:** {lti:.2f}\n\n"
        f"{msg}"
    )
