import streamlit as st


def guidance(X_live, dataset, prediction):
    """Display tailored guidance messages based on model prediction."""
    medians = {
        "credit_score": dataset["credit_score"].median(),
        "income": dataset["income"].median(),
        "loan_amount": dataset["loan_amount"].median(),
        "years_employed": dataset["years_employed"].median(),
    }

    if prediction == 1:
        st.success(
            "### Your details aligns with most approved applicants.\n"
            "This is **not financial advice**, but your current inputs "
            "indicate **strong approval likelihood**."
        )
        return

    tips = []

    if X_live.at[0, "credit_score"] < medians["credit_score"]:
        tips.append(
            f"Increase credit score above ~{medians['credit_score']:.0f}."
        )

    if X_live.at[0, "income"] < medians["income"]:
        tips.append(
            f"Raise annual income above ~${medians['income']:,.0f}."
        )

    if X_live.at[0, "loan_amount"] > medians["loan_amount"]:
        tips.append(
            f"Request less than ~${medians['loan_amount']:,.0f}."
        )

    if X_live.at[0, "years_employed"] < medians["years_employed"]:
        tips.append(
            f"Stay in your current job for "
            f"{medians['years_employed']:.0f}+ years."
        )

    if tips:
        st.info(
            "### Ways to improve your approval odds:\n- "
            + "\n- ".join(tips)
        )
    else:
        st.info(
            "### Your profile is close to approval.\n"
            "Small improvements across each metric could raise your "
            "approval odds."
        )
