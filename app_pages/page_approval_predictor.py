import pandas as pd
import streamlit as st

from src.data_management import load_loan_approval_data, load_pkl_file

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

    



