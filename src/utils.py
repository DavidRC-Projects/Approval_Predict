from feature_engine.discretisation import ArbitraryDiscretiser
import numpy as np
from pathlib import Path

RAW_DATASET = Path("outputs/datasets/collection/loan_approval.csv")
DROP_COLUMNS = ["name", "city"]
TARGET_COLUMN = "loan_approved"
APPROVAL_FEATURES = [
    "loan_to_income",
    "credit_score",
    "years_employed",
    "loan_amount",
    "income",
]


def engineer_features(df):
    """
    Feature engineering for the loan approval dataset:
      - remove identifier columns (name, city)
      - ensure target is numeric
      - create the loan_to_income ratio
    """
    engineered = df.drop(
        columns=[col for col in DROP_COLUMNS if col in df.columns]
    ).copy()
    if TARGET_COLUMN in engineered.columns:
        engineered[TARGET_COLUMN] = engineered[TARGET_COLUMN].astype(int)
    engineered["loan_to_income"] = (
        engineered["loan_amount"] / engineered["income"]
    )
    return engineered


def discretize_for_parallel(df):
    """
    Discretise variables for the parallel categories plot using the same bins
    defined in Notebook 02, and map bin indices to readable labels.
    """
    credit_map = [-np.inf, 580, 670, 740, 800, np.inf]
    points_map = [-np.inf, 40, 60, 80, np.inf]
    lti_map = [-np.inf, 0.3, 0.5, 0.7, 1.0, np.inf]

    disc = ArbitraryDiscretiser(binning_dict={
        "credit_score": credit_map,
        "points": points_map,
        "loan_to_income": lti_map
        }
    )
    df_disc = disc.fit_transform(df.copy())

    def make_label_map(binner_dict, variable):
        bins = binner_dict[variable]
        n_classes = len(bins) - 1
        classes_ranges = bins[1:-1]
        labels_map = {}
        for n in range(n_classes):
            if n == 0:
                labels_map[n] = f"<{classes_ranges[0]}"
            elif n == n_classes - 1:
                labels_map[n] = f"+{classes_ranges[-1]}"
            else:
                labels_map[n] = f"{classes_ranges[n-1]} to {classes_ranges[n]}"
        return labels_map

    for var in ["credit_score", "points", "loan_to_income"]:
        if var in disc.binner_dict_:
            df_disc[var] = df_disc[var].replace(
                make_label_map(disc.binner_dict_, var)
            )

    for col in ["credit_score", "points", "loan_to_income"]:
        df_disc[col] = df_disc[col].astype(str)

    if (
        "loan_approved" in df_disc.columns
        and df_disc["loan_approved"].dtype != int
    ):
        try:
            df_disc["loan_approved"] = df_disc["loan_approved"].astype(int)
        except Exception:
            pass

    return df_disc
