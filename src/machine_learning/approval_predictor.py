import streamlit as st


APPROVAL_FEATURES = [
    "loan_to_income",
    "credit_score",
    "years_employed",
    "loan_amount",
    "income",
]


def predict_loan_approval(X_live, pipeline):
    """
    Filter live applicant data to the production features and
    return the approval prediction.
    """
    X_live = X_live[APPROVAL_FEATURES]
    prediction = pipeline.predict(X_live)
    probabilities = pipeline.predict_proba(X_live)

    predicted_class = prediction[0]
    approval_chance = probabilities[0, predicted_class] * 100
    if predicted_class == 1:
        result_text = "will"
    else:
        result_text = "will not"

    statement = (
        f"### There is a {approval_chance.round(1)}% probability "
        f"that this applicant **{result_text}** be approved."
    )
    st.write(statement)

    return predicted_class
