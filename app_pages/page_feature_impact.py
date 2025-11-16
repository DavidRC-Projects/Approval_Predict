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
    sns.set_style("whitegrid")

    for col in vars_to_study:
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))

        # Histogram
        sns.histplot(
            data=df, x=col, hue=target_var, kde=True, element="step",
            palette="Set1", ax=axes[0]
        )
        axes[0].set_title(f"{col} distribution by {target_var}")

        # Boxplot
        sns.boxplot(
            data=df, x=target_var, y=col,
            palette="Set2", ax=axes[1]
        )
        axes[1].set_title(f"{col} vs {target_var}")

        plt.tight_layout()
        st.pyplot(fig)


def page_feature_impact_body():
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


    df_eda = df_preview.filter(vars_to_study + ['loan_approved'])
    target_var = 'loan_approved'


    with st.expander("Show Numerical Feature Distributions"):
        plot_numerical(df_eda, vars_to_study, target_var)

    with st.expander("Show Correlation Heatmap"):
        corr = df_eda.corr(numeric_only=True)

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    with st.expander("Show Parallel Categories Plot"):
        df_disc = discretize_for_parallel(df_eda)

        fig = px.parallel_categories(
            df_disc[['credit_score', 'points', 'loan_to_income', 'loan_approved']],
            color="loan_approved",
            color_continuous_scale=px.colors.sequential.Plasma
        )
        st.plotly_chart(fig, use_container_width=True)


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









