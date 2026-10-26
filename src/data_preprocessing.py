import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

RANDOM_STATE = 42
TARGET_COL = "Target"
CATEGORICAL_NUNIQUE_THRESHOLD = 10


def load_dataset(path):
    return pd.read_csv(path)


def clean_dataset(df):
    df = df.copy()

    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        df = df.drop_duplicates().reset_index(drop=True)

    numeric_cols = df.drop(columns=[TARGET_COL]).select_dtypes(include=[np.number]).columns
    categorical_cols = df.drop(columns=[TARGET_COL]).select_dtypes(exclude=[np.number]).columns

    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0])

    return df


def prepare_features(df):
    df = df.copy()

    target = df[TARGET_COL]

    if not pd.api.types.is_numeric_dtype(target):
        positive_tokens = {"yes", "true", "1", "disease", "positive", "y"}
        df[TARGET_COL] = (
            target.astype(str)
            .str.strip()
            .str.lower()
            .apply(lambda value: 1 if value in positive_tokens else 0)
        )
    else:
        unique_values = sorted(target.dropna().unique().tolist())
        if set(unique_values).issubset({0, 1}):
            df[TARGET_COL] = target.astype(int)
        else:
            df[TARGET_COL] = (target > 0).astype(int)

    X = df.drop(columns=[TARGET_COL]).copy()
    y = df[TARGET_COL].copy()

    numeric_features = []
    categorical_features = []

    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            categorical_features.append(col)
            continue

        n_unique = X[col].nunique()
        if 2 < n_unique <= CATEGORICAL_NUNIQUE_THRESHOLD:
            categorical_features.append(col)
        else:
            numeric_features.append(col)

    binary_numeric_features = [
        col for col in numeric_features if X[col].nunique() == 2
    ]

    for col in categorical_features:
        X[col] = X[col].astype(str)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    return (
        X,
        y,
        preprocessor,
        numeric_features,
        categorical_features,
        binary_numeric_features,
    )


def split_data(X, y):
    return train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )
