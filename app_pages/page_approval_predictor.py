import pandas as pd
import streamlit as st

def page_approval_predictor_body():
    st.title("Feature Impact Study (Legacy Model)")


def DrawInputWidgets():
    df = load_approval_data()

## APPROVAL_FEATURES = ["loan_to_income", "credit_score", "years_employed", "loan_amount", "income"]