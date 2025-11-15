import pandas as pd
from imblearn.over_sampling import SMOTE
from feature_engine.transformation import BoxCoxTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler


APPROVAL_FEATURES = ["loan_to_income", "credit_score", "years_employed", "loan_amount", "income"]
TARGET_COLUMN = "loan_approved"
DROP_COLUMNS = ["name", "city"]


def engineer_features(df):
    """
    Feature engineering for the loan approval dataset:
      - remove identifier columns (name, city)
      - ensure target is numeric
      - create the loan_to_income ratio
    """
    engineered = df.drop(columns=[col for col in DROP_COLUMNS if col in df.columns]).copy()
    if TARGET_COLUMN in engineered.columns:
        engineered[TARGET_COLUMN] = engineered[TARGET_COLUMN].astype(int)
    engineered["loan_to_income"] = engineered["loan_amount"] / engineered["income"]
    return engineered


def get_features_and_target(df: pd.DataFrame):
    """
    Split the engineered dataframe into feature matrix X and target vector y.
    """
    engineered = engineer_features(df)
    X = engineered[APPROVAL_FEATURES].copy()
    y = engineered[TARGET_COLUMN].copy()
    return X, y


def balance_with_smote(X: pd.DataFrame, y: pd.Series, random_state: int = 42):
    """
    SMOTE balancing to address the class imbalance in the target(loan_approved).
    """
    smote = SMOTE(sampling_strategy="minority", random_state=random_state)
    return smote.fit_resample(X, y)


def build_feature_pipeline():
    """
    Construct the transformation stack used before modeling.
    """
    return Pipeline(
        steps=[
            ("boxcox", BoxCoxTransformer(variables=["loan_to_income"])),
            ("scaler", RobustScaler()),
        ]
    )


def build_classification_pipeline(model):
    """
    Create the classification pipeline with the following steps:
    BoxCoxTransformer, RobustScaler and the model
    """
    return Pipeline(
        steps=[
            ("boxcox", BoxCoxTransformer(variables=["loan_to_income"])),
            ("scaler", RobustScaler()),
            ("model", model),
        ]
    )
