import streamlit as st


def page_summary_body():
    st.write("### Quick Project Summary")

    st.info(
        f"**Dataset Snapshot**\n"
        f"* Source: Kaggle loan approval dataset with 2,000 applications.\n"
        f"* Each record includes applicant demographics, financial indicators, "
        f"employment history, loan request details, and approval outcome.\n"
        f"* We created a realistic workplace scenario where loan officers need "
        f"decision support to process applications faster and more consistently.\n\n"
        f"**Key Terminology**\n"
        f"* **Applicant** – person requesting the loan.\n"
        f"* **Approved loan** – application accepted for funding.\n"
        f"* **Rejected loan** – application denied.\n"
        f"* **Credit score** – numeric proxy for creditworthiness (300–850).\n"
        f"* **Loan-to-income ratio** – loan amount divided by annual income."
    )

    st.write(
        f"* For full context, refer to the "
        f"[Project README](https://github.com/DavidRC-Projects/Approval_Predict)."
    )

    st.success(
        f"**Business Requirements**\n"
        f"* **Requirement1 Feature insight:** Identify which applicant variables "
        f"drive approval outcomes and explain the patterns we observe.\n"
        f"* **BR2 Applicant guidance:** Provide an interactive predictor "
        f"that estimates approval probability (without relying on the legacy "
        f"points feature) and offers actionable tips to improve approval odds."
    )

