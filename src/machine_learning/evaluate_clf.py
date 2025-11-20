import streamlit as st
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix


def confusion_matrix_and_report(X, y, pipeline, label_map):
    """Render confusion matrix and classification report for a dataset split."""
    prediction = pipeline.predict(X)

    st.write('#### Confusion Matrix')
    st.code(pd.DataFrame(confusion_matrix(y_true=y, y_pred=prediction),
                         columns=[["Actual " + sub for sub in label_map]],
                         index=[["Prediction " + sub for sub in label_map]]
                         ))

    st.write('#### Classification Report')
    st.code(classification_report(y, prediction, target_names=label_map), "\n")


def clf_performance(X_train, y_train, X_test, y_test, pipeline, label_map):
    """Compare model performance on train/test splits."""
    st.info("Train Set")
    confusion_matrix_and_report(X_train, y_train, pipeline, label_map)

    st.info("Test Set")
    confusion_matrix_and_report(X_test, y_test, pipeline, label_map)