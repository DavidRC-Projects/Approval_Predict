import streamlit as st


def page_project_findings_body():
    st.write("### Project Hypotheses and Findings")

    st.info(
        f"**Business Requirements**\n"
        f"* **Requirement 1 Feature insight:** Identify which applicant variables "
        f"drive approval outcomes and explain the patterns we observe.\n"
        f"* **Requirement 2 Applicant guidance:** Provide an interactive predictor "
        f"that estimates approval probability and offers tips to improve approval"
    )

    st.info(
        "Before modelling we believed that:\n\n"
        "(1) higher credit scores lead to more approvals.\n"
        "(2) longer employment history improves approval odds.\n"
        "(3) high loan-to-income ratios reduce approval chances."
    )

    st.subheader("Model Success Metrics highlights")
    col1, col2 = st.columns(2)
    col1.metric("Recall (approved class)", "87%")
    col2.metric("Precision (approved class)", "86%")


    st.success(
        "- Recall metric highlights that out of all actual approved loans, the model correctly identified 87%\n"
        "- Precision metric highlights that out of all loans predicted as approved, 86% were actually approved.\n"
        "- Overall, the model demonstrates a strong performance in identifying approved loans.\n"
    )

    st.write(
        "- Removing points produces a reliable, fairer predictor with actionable guidance.\n"
        "- Credit score, loan to income, income, years employed, and loan amount formed the final feature set.\n"
        "- Feature contributions align with helping loan officers explain decisions."
    )