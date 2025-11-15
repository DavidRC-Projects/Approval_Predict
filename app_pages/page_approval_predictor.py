import pandas as pd
import streamlit as st

from src.data_management import load_loan_approval_data, load_pkl_file
from src.machine_learning.approval_predictor import predict_loan_approval
from src.approval_guidance import guidance


def page_approval_predictor_body():
    st.title("ML Pipeline: Approval Predictor")

    version = "v2"
    pipeline = load_pkl_file(
        f"outputs/ml_pipeline/approval_prediction/{version}/classification_pipeline.pkl"
    )

    st.info(
        "#### **Business Requirement 2**: Applicant Guidance & Classification\n\n"
        "* Predict if an applicant will be approved using a Logistic Regression pipeline trained \n"
        "* Success metrics: ≥80% recall and ≥80% precision on the approved class (train & test).\n"
        "* Provide transparent feedback to help applicants improve their approval odds."
    )

    ##Loads the cleaned loan-approval dataset (calling load_approval_data() and making sure loan_to_income exists
    dataset = _load_dataset()
    ranges = {
        "income": (int(dataset["income"].min()), int(dataset["income"].max())),
        "loan_amount": (int(dataset["loan_amount"].min()), int(dataset["loan_amount"].max())),
        "credit_score": (int(dataset["credit_score"].min()), int(dataset["credit_score"].max())),
        "years_employed": (int(dataset["years_employed"].min()), int(dataset["years_employed"].max())),
    }
    st.markdown(
    f"""
        Please use the input ranges (based on training data). Adjust the values below to see how they impact loan approval.
        - **Income:** `${ranges['income'][0]:,} — ${ranges['income'][1]:,}` 
        - **Loan amount:** `${ranges['loan_amount'][0]:,} — ${ranges['loan_amount'][1]:,}` 
        - **Credit score:** `{ranges['credit_score'][0]} — {ranges['credit_score'][1]}` 
        - **Years employed:** `{ranges['years_employed'][0]} — {ranges['years_employed'][1]} years`
    """
)
    X_live = _draw_input_widgets(dataset)


    if st.button("Run Predictive Analysis"):
        prediction = predict_loan_approval(X_live, pipeline)
        guidance(X_live, dataset, prediction)


def _load_dataset():
    df = load_loan_approval_data()
    if "loan_to_income" not in df.columns:
        df["loan_to_income"] = df["loan_amount"] / df["income"]
    return df


def _draw_input_widgets(df):
    
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

    
