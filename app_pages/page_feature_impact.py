from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import streamlit as st

from src.data_management import load_loan_approval_data
from src.utils import engineer_features
from src.utils import discretize_for_parallel


def plot_numerical(df, vars_to_study, target_var):
    """
    Create histograms and boxplots to analyse numerical feature behaviour
    and compare distributions between approved and rejected applicants.
    """
    sns.set_style("whitegrid")

    for col in vars_to_study:
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))

        sns.histplot(
            data=df, x=col, hue=target_var, kde=True, element="step",
            palette="Set1", ax=axes[0]
        )
        axes[0].set_title(f"{col} distribution by {target_var}")

        sns.boxplot(
            data=df, x=target_var, y=col,
            palette="Set2", ax=axes[1]
        )
        axes[1].set_title(f"{col} vs {target_var}")

        plt.tight_layout()
        st.pyplot(fig)


def page_feature_impact_body():
    """
    Display dataset overview, feature insights, and visual analyses
    (numerical plots, heatmap, and parallel categories) for loan approval patterns.
    """
    st.title("Feature Impact Analysis")

    raw_dataset = load_loan_approval_data()
    
    st.subheader("Dataset Overview")
    df_preview = engineer_features(raw_dataset)
    st.dataframe(df_preview.head(5))
    st.write(f"* The dataset has {df_preview.shape[0]} rows and {df_preview.shape[1]} columns.")

    vars_to_study = [
        'points', 'credit_score', 'income',
        'loan_amount', 'years_employed', 'loan_to_income'
    ]

    st.write("### Loan Applicant Feature Study")
    st.info(
        "The goal of this analysis is to understand how different applicant features "
        "such as income, credit score, employment history, points and loan-to-income ratio "
        "influence loan approval patterns. This helps identify the most relevant "
        "predictors contributing to approval decisions."
    )

    st.subheader("Correlation Study Summary")
    st.write(
        "A correlation analysis was performed to explore how different applicant features "
        "relate to loan approval outcomes. Understanding these relationships helps identify "
        "the most influential factors in the approval process.\n\n"
        f"**Features included in the study:** {', '.join(vars_to_study)}"
    )

    st.subheader("Numerical Feature Analysis Influencing Loan Approval")

    if st.checkbox("Summary of numerical feature insights from the plots below"):

        st.write(
    """
    - **Points** and **Credit Score** are the strongest predictors.  
    Approved applicants consistently have *much higher* values, with clear separation between approved and rejected groups.

    - **Income** shows a moderate positive effect.  
    Higher-income applicants are more likely to be approved, but the impact is less pronounced than points and credit score.

    - **Loan Amount** has a mild negative relationship.  
    Applicants requesting larger loans tend to be rejected more often, though the difference is not as strong as other variables.

    - **Years Employed** shows only a weak relationship.  
    Longer employment history helps slightly, but there is substantial overlap between the two groups.

    - The **Loan-to-Income Ratio** is a significant affordability indicator.  
    Lower ratios are associated with approvals, while higher ratios and outliers indicate financial risk and more frequent rejections.
    """
    )

    df_eda = df_preview.filter(vars_to_study + ['loan_approved'])
    target_var = 'loan_approved'


    with st.expander("Show Numerical Feature Distributions"):
        plot_numerical(df_eda, vars_to_study, target_var)

    st.subheader("Correlation Heatmap Analysis")

    if st.checkbox("Show summary of correlation heatmap insights"):
        st.write(
        """
        - **Points** and **Credit Score** show the strongest positive correlation with loan approval.  
        - **Income** has a weak positive correlation, indicaating a higher income slightly improves approval.  
        - **Loan Amount** shows a slight negative relationship, suggesting larger requests slightly reduce approval odds.  
        - **Years Employed** has very small correlation with approval outcomes.  
        """
        )

    with st.expander("Show Correlation Heatmap"):
        corr = df_eda.corr(numeric_only=True)

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    st.subheader("Parallel Categories Plot Analysis")

    if st.checkbox("Show summary of parallel categories insights"):
        st.write(
        """
        - Applicants with **high points**, **high credit scores**, and **low loan-to-income ratios** cluster strongly toward approval.  
        - Applicants with **low points**, **low credit scores**, or **high loan-to-income ratios** frequently fall into the rejected category.  
        - The plot reinforces that **creditworthiness and affordability** jointly drive approval outcomes.
        """
        )

    with st.expander("Show Parallel Categories Plot"):
        df_disc = discretize_for_parallel(df_eda)

        fig = px.parallel_categories(
            df_disc[['credit_score', 'points', 'loan_to_income', 'loan_approved']],
            color="loan_approved",
            color_continuous_scale=px.colors.sequential.Plasma
        )
        st.plotly_chart(fig, use_container_width=True)


    corr_spearman = df_preview.corr(method="spearman")["loan_approved"] \
        .sort_values(key=abs, ascending=False)[1:].head(5)

    corr_pearson = df_preview.corr(method="pearson")["loan_approved"] \
        .sort_values(key=abs, ascending=False)[1:].head(5)

    correlation_sets = {
        "Top 5 Pearson Correlations": corr_pearson,
        "Top 5 Spearman Correlations": corr_spearman
    }

    for title, corr_series in correlation_sets.items():
    
        df_plot = pd.DataFrame({
            "Feature": corr_series.index,
            "Correlation": corr_series.values
        })

        df_plot["abs_correlation"] = df_plot["Correlation"].abs()

        st.write(f"### {title}")

        st.data_editor(
            df_plot[["Feature", "abs_correlation"]],
            column_config={
                "abs_correlation": st.column_config.ProgressColumn(
                    "Correlation Strength",
                    help=f"Absolute {title.split()[2]} correlation with loan approval",
                    format="%.2f",
                    min_value=0.0,
                    max_value=1.0,
                ),
            },
            hide_index=True,
            disabled=True,
        )

    st.subheader("Conclusions")
    st.info(
    """
    - Applicants with **higher points** are more likely to be approved.  
    - Applicants with **higher credit scores** are more likely to be approved.  
    - Applicants with **low points** and **low credit scores** show a strong likelihood of rejection.  
    - **Income**, **loan amount**, and **years employed** show weaker relationships with approval outcomes.   
    - The parallel categories analysis highlights a strong link between **loan-to-income ratio** and loan approval decisions.  
    """
    )









