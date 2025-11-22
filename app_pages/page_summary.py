import streamlit as st


def page_summary_body():
    """
    Render the Quick Project Summary page.
    Gives an overview of the dataset and project context.
    Definitions of key terminology used throughout the analysis.
    A link to the full project README for additional documentation.
    A summary of the business requirements.
    """
    st.write("### Quick Project Summary")

    st.info(
        "**Dataset Snapshot**\n"
        "* Source: Kaggle loan approval dataset with "
        "2,000 applications.\n"
        "* Each record includes applicant demographics, credit score, "
        "points, employment history, loan amount, income and approval "
        "outcome.\n"
        "* We created a workplace scenario where loan officers need "
        "decision support with an aim to process applications faster.\n\n"
        "**Key Terminology**\n"
        "* **Applicant** is a person requesting the loan.\n"
        "* **Approved loan** is an application accepted for funding.\n"
        "* **Rejected loan** is an application denied.\n"
        "* **Loan-to-income ratio** is the loan amount divided by annual "
        "income."
    )

    st.write(
        f"* For full context, refer to the "
        f"[Project README]"
        "(https://github.com/DavidRC-Projects/Approval_Predict)."
    )

    st.success(
        "**Business Requirements**\n"
        "* **Requirement 1 Feature insight:** Identify which applicant "
        "variables drive approval outcomes and explain the patterns "
        "we observe.\n"
        "* **Requirement 2 Applicant guidance:** Provide an interactive "
        "predictor that estimates approval probability and offers tips "
        "to improve approval."
    )

