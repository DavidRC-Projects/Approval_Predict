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
        f"**Dataset Snapshot**\n"
        f"* Source: Kaggle loan approval dataset with 2,000 applications.\n"
        f"* Each record includes applicant demographics, credit score, points, "
        f"employment history, loan amount, income and approval outcome.\n"
        f"* We created a workplace scenario where loan officers need"
        f"decision support with an aim to process applications faster.\n\n"
        f"**Key Terminology**\n"
        f"* **Applicant** is a person requesting the loan.\n"
        f"* **Approved loan** is an application accepted for funding.\n"
        f"* **Rejected loan** is an application denied.\n"
        f"* **Loan-to-income ratio** is the loan amount divided by annual income."
    )

    st.write(
        f"* For full context, refer to the "
        f"[Project README](https://github.com/DavidRC-Projects/Approval_Predict)."
    )

    st.success(
        f"**Business Requirements**\n"
        f"* **Requirement 1 Feature insight:** Identify which applicant variables "
        f"drive approval outcomes and explain the patterns we observe.\n"
        f"* **Requirement 2 Applicant guidance:** Provide an interactive predictor "
        f"that estimates approval probability and offers tips to improve approval"
    )

