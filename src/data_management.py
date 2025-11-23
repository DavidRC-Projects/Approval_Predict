import joblib
import pandas as pd
import streamlit as st


@st.cache_data
def load_loan_approval_data():
    """Load the loan approval dataset with error handling."""
    file_path = "outputs/datasets/collection/loan_approval.csv"

    try:
        df = pd.read_csv(file_path)

        if df.empty:
            "Dataset file is empty. Please check the data collection "
            "process."
        st.stop()

        return df

    except FileNotFoundError:
        st.error(
            f"Dataset file not found.\n\n"
            f"Expected at: `{file_path}`\n\n"
            f"Please ensure the data collection notebook has been run."
        )
        st.stop()

    except Exception as e:
        st.error(f"Unexpected error while loading dataset:\n\n{str(e)}")
        st.stop()


def load_pkl_file(file_path):
    """
    Load ML pipeline with error handling.
    """
    try:
        return joblib.load(file_path)

    except FileNotFoundError:
        st.error(
            f"Model file not found.\n\n"
            f"Expected at: `{file_path}`\n\n"
            f"Please ensure the model training notebook has been run."
        )
        st.stop()

    except Exception as e:
        st.error(f"Unexpected error while loading model:\n\n{str(e)}")
        st.stop()
