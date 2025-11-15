import pandas as pd
import streamlit as st


APPROVAL_FEATURES = ["loan_to_income", "credit_score", "years_employed", "loan_amount", "income"]

def predict_loan_approval(X_live, pipeline):
    """
    Filter live applicant data to the production features and return the approval prediction.
    """
    X_live = X_live.filter(APPROVAL_FEATURES)
    prediction = pipeline.predict(X_live)
    probabilities = pipeline.predict_proba(X_live)

    approval_chance = probabilities[0, prediction][0] * 100
    if prediction == 1:
        result_text = "will"
    else:
        result_text = "will not"

    statement = (
        f"### There is a {approval_chance.round(1)}% probability "
        f"that this applicant **{result_text}** be approved."
    )
    st.write(statement)

    return prediction