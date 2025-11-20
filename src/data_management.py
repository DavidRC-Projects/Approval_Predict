import streamlit as st
import pandas as pd
import numpy as np
import joblib


@st.cache_data
def load_loan_approval_data():
    """Return the raw loan-approval dataset from the outputs directory."""
    df = pd.read_csv("outputs/datasets/collection/loan_approval.csv")
    return df


def load_pkl_file(file_path):
    """Load a Joblib object."""
    return joblib.load(filename=file_path)