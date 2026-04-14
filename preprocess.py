import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import IsolationForest
from scipy import sparse


def remove_outliers_func(df):
    df = df.copy()
    num_df = df.select_dtypes(include=np.number)

    iso = IsolationForest(contamination=0.05, random_state=42)
    preds = iso.fit_predict(num_df)

    df = df[preds == 1]
    return df


def preprocess_pipeline(df, target, remove_out=False):
    df = df.copy()

    y = df[target]
    X = df.drop(columns=[target])

    # Target encoding safe
    if y.dtype == "object":
        y = y.map({"yes":1, "no":0, "Y":1, "N":0}).fillna(0)
    y = y.astype(int)

    # Outlier removal
    if remove_out and len(df) > 50:
        temp = pd.concat([X, y], axis=1)
        temp = remove_outliers_func(temp)
        y = temp[target]
        X = temp.drop(columns=[target])

    num_cols = X.select_dtypes(include=np.number).columns
    cat_cols = X.select_dtypes(include="object").columns

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, cat_cols)
    ])

    X_processed = preprocessor.fit_transform(X)

    if sparse.issparse(X_processed):
        X_processed = X_processed.toarray()

    selector = VarianceThreshold(0.01)
    X_selected = selector.fit_transform(X_processed)

    return X_selected, y, preprocessor, selector