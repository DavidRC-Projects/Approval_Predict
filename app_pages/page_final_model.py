import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

from src.data_management import load_pkl_file
from src.machine_learning.evaluate_clf import clf_performance

def get_base_path():
    version = "v2"
    return f"outputs/ml_pipeline/approval_prediction/{version}"


def load_pipeline():
    return load_pkl_file(
        "outputs/ml_pipeline/approval_prediction/v2/classification_pipeline.pkl"
    )


def load_feature_importance():
    return plt.imread(f"{get_base_path()}/features_importance.png")


def load_data_splits():
    base = get_base_path()
    X_train = pd.read_csv(f"{base}/X_train.csv")
    X_test  = pd.read_csv(f"{base}/X_test.csv")
    y_train = pd.read_csv(f"{base}/y_train.csv")
    y_test  = pd.read_csv(f"{base}/y_test.csv")
    return X_train, X_test, y_train, y_test


def page_final_model_body():
    st.title("Final ML Model")

    pipeline = load_pipeline()
    feat_importance = load_feature_importance()
    X_train, X_test, y_train, y_test = load_data_splits()

    st.subheader("Feature Importance Plot")
    st.image(feat_importance, caption="Feature Importance")

    st.subheader("Model Performance")
    label_map = ["Rejected", "Approved"]

    clf_performance(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        pipeline=pipeline,
        label_map=label_map
    )
