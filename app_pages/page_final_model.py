import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

from src.data_management import load_pkl_file
from src.machine_learning.evaluate_clf import clf_performance

def get_base_path():
    """
    Return the base directory path for loading model outputs for version (v2).
    """
    version = "v2"
    return f"outputs/ml_pipeline/approval_prediction/{version}"


def load_pipeline():
    """
    Load and return the trained classification pipeline.
    """
    return load_pkl_file(
        "outputs/ml_pipeline/approval_prediction/v2/classification_pipeline.pkl"
    )


def load_feature_importance():
    """
    Load and return the saved feature importance image for display.
    """
    return plt.imread(f"{get_base_path()}/features_importance.png")


def load_data_splits():
    """
    Load the stored train/test feature matrices for evaluation.
    """
    base = get_base_path()
    X_train = pd.read_csv(f"{base}/X_train.csv")
    X_test  = pd.read_csv(f"{base}/X_test.csv")
    y_train = pd.read_csv(f"{base}/y_train.csv")
    y_test  = pd.read_csv(f"{base}/y_test.csv")
    return X_train, X_test, y_train, y_test


def page_final_model_body():
    """
    Render the Final Model page, including:
    - Feature importance explanation
    - Feature importance plot
    - Train vs test evaluation results
    - Confusion matrices and classification reports
    """
    st.title("Final ML Model")

    st.subheader("Feature Importance Summary")

    st.info(
    """
    ### What the Feature Importance Plot Tells Us

    - **Credit Score** is the dominant predictor in the model.  

    - **Loan-to-Income Ratio** is the second strongest influence.  
      Applicants with more affordable loans relative to income are more likely to be approved.

    - **Income** and **Years Employed** add moderate predictive value.  
      They improve the model but do not drive decisions on their own.

    - **Loan Amount** has minimal impact, especially when credit score is high or low.

    **In summary:**  
    Loan approval is primarily driven by **credit score** and **affordability (loan-to-income)**.
    Other features help fine-tune predictions but do not strongly shift outcomes.
    """
    )

    pipeline = load_pipeline()
    feat_importance = load_feature_importance()
    X_train, X_test, y_train, y_test = load_data_splits()

    st.subheader("Feature Importance Plot")
    st.image(feat_importance, caption="Feature Importance")

    st.subheader("Model Performance")

    st.subheader("Model Performance Overview")

    st.info(
    """
    ### Train vs Test Performance

    **Train Accuracy:** 92%  
    **Test Accuracy:** 88%

    The model performs very strongly on both datasets, indicating:
    - No signs of overfitting (train and test scores are close)
    - Good generalisation to unseen applicants

    ### Train Set Observations
    - Very high precision (91%) and recall (93%) for the *approved* class  
    - This indicates the model will most of the time identify approved applicants correctly 

    ### Test Set Observations
    - Precision (86%) and recall (88%) for *approved* remain strong  
    - Only a small drop from the train set 
    - Confirms the model is reliable and not overly tuned to the training data

    **Conclusion:**  
    The model provides robust predictions with strong precision and recall.
    """
    )

    label_map = ["Rejected", "Approved"]

    clf_performance(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        pipeline=pipeline,
        label_map=label_map
    )
