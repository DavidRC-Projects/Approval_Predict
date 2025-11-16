from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import streamlit as st

from src.data_management import load_loan_approval_data
from src.utils import engineer_features
from src.utils import discretize_for_parallel

def page_feature_impact_body():
    st.title("Feature Impact Analysis")
    raw_dataset = load_loan_approval_data()
    
    st.subheader("Dataset Overview")
    df_preview = engineer_features(raw_dataset)
    st.dataframe(df_preview.head(5))

    vars_to_study = [
        'points', 'credit_score', 'income',
        'loan_amount', 'years_employed', 'loan_to_income'
    ]

