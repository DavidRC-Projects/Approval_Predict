import streamlit as st


def page_project_findings_body():
    """
    Render the Project Hypotheses and Findings page.
    This page explains the business requirements, outlines the initial project
    hypotheses, validates these hypotheses, and presents key model performance
    metrics (precision and recall) for the approved class.
    """
    st.write("### Project Hypotheses and Findings")

    st.info(
        "**Business Requirements**\n"
        "* **Requirement 1 - Feature insight:** Identify which applicant "
        "variables drive approval outcomes and explain the patterns "
        "observed.\n"
        "* **Requirement 2 - Applicant guidance:** Provide an interactive "
        "predictor that estimates approval probability and offers tips to "
        "improve approval chances."
    )

    st.info(
        "Before modelling we believed that:\n\n"
        "- (1) Higher credit scores lead to more approvals.\n"
        "- (2) Longer employment history improves approval odds.\n"
        "- (3) High loan-to-income ratios reduce approval chances."
    )

    st.subheader("Hypothesis Validation")

    st.info(
        """
        ### Hypothesis Validation Summary

        **Hypothesis 1 — “Higher credit scores lead to more approvals.”**
        **Validated.**
        Credit score shows one of the strongest correlations with approval
        (high Pearson and Spearman values) and is a top predictor in the ML
        model.
        This provides strong evidence supporting the hypothesis.

        **Hypothesis 2 — “Longer employment history improves approval odds.”**
        **Not Validated.**
        Years employed displays very weak correlation with approval
        in both Pearson and Spearman metrics.
        It adds minimal predictive value and does **not** support the
        hypothesis.

        **Hypothesis 3 — “High loan-to-income ratios reduce approval
        chances.”**
        **Validated.**
        Loan-to-income ratio shows a moderate negative correlation with
        approval in both Pearson and Spearman analyses.
        Applicants with higher ratios are more likely to be rejected,
        confirming this hypothesis.
        """
    )

    st.subheader("Model Success Metrics Highlights")
    col1, col2 = st.columns(2)
    col1.metric("Recall (approved class)", "87%")
    col2.metric("Precision (approved class)", "86%")

    st.success(
        "- Recall metric: The model correctly identifies 87% of all truly "
        "approved applicants.\n"
        "- Precision metric: Out of all applications predicted as approved, "
        "86% are genuinely approved.\n"
        "- Together, these show the model is reliable at identifying approved "
        "candidates while keeping false approvals low.\n"
        )

    st.write(
        "- Removing points produced a fairer, more generalisable model.\n"
        "- Final features include credit score, loan-to-income, income, "
        "years employed, and loan amount.\n"
        "- Feature contributions align with helping loan officers provide "
        "transparent explanations to applicants."
    )
