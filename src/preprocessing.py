import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from typing import *


def scale_features(df: pd.DataFrame, cols: Union[str, List[str]], scaler: MinMaxScaler = None) -> tuple[pd.DataFrame, MinMaxScaler]:
    """
    Scale column with values between 0 and 1
    
    Parameters
    ----------
    df : pd.DataFrame
        Input Dataframe.
    col : Union[str, List[str]]
        Column to be scaled
    Return
    -------
    df : pd.DataFrame
        DataFrame with columns standardized
    scaler : MinMaxScaler
        MinMaxScaler object
    """
    if isinstance(cols, str):
        cols = [cols]
    if scaler is None:
        scaler = MinMaxScaler()
        df[cols] = scaler.fit_transform(df[cols])
    else:
        df[cols] = scaler.transform(df[cols])
    return df, scaler


def add_outlier_flags(df: pd.DataFrame, cols: List[str] = None, k: int = 1.5) -> tuple[pd.DataFrame, dict]:
    """
    Add flag columns when outliers are detected using IQR method.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input Dataframe.
    cols : list (default=None)
        List of numerical columns to be evalueted. If None, uses all numerical columns.
    k : float (default=1.5)
        IQR multiplicator to define outlier limits.
    
    Return
    -------
    df : pd.DataFrame
        DataFrame with extra columns is_outlier_<col>.
    thresholds: dict
        Dictionary with lower and upper bounds of IQR
    """
    thresholds = {}
    
    if cols is None:
        cols = df.select_dtypes(include="number").columns
    
    for col in cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - k * IQR
        upper = Q3 + k * IQR

        thresholds[col] = (lower, upper)
        flag_col = f"is_outlier_{col}"
        df[flag_col] = ((df[col] < lower) | (df[col] > upper)).astype(int)
    
    return df, thresholds


def target_feature_split(df: pd.DataFrame, target: str = None, exclude_cols: List[str] = None ) -> tuple[pd.DataFrame, pd.Series]:
    """
    Splits target, features, and columns that will not be used
    
    Parameters
    ----------
    df : pd.DataFrame
        Input Dataframe.
    target : str
        Target Variable.
    exclude_cols : list (default=None)
        List of columns to be excluded from the feature
    
    Return
    -------
    features_df : pd.DataFrame
        DataFrame with features.
    target_series : pd.Series
        Series with target variable.
    """
    target_series = None
    if target is not None:
        target_series = df[target]
        exclude_cols.append(target)

    for col in exclude_cols:
        del df[col]
    features_df = df
    return features_df, target_series

def one_hot_encoding(df: pd.DataFrame, col: str, encoder: OneHotEncoder = None) -> tuple[pd.DataFrame, OneHotEncoder]:
    """
    Wrapper for OneHotEncoder Class from Scikit-learn
    
    Parameters
    ----------
    df : pd.DataFrame
        Input Dataframe.
    col : str
        Target Variable for encoding.

    Return
    -------
    df : pd.DataFrame
        DataFrame with encoded columns.
    encoder : OneHotEncoder
        Encoder.
    """
    if not isinstance(encoder, OneHotEncoder):
        encoder = OneHotEncoder(drop='first', sparse_output=False).fit(df[col].values.reshape(-1, 1))
    
    encoded_array = encoder.transform(df[col].values.reshape(-1, 1))
    feature_names = encoder.get_feature_names_out([col])
    encoded_df = pd.DataFrame(encoded_array, columns=feature_names, index=df.index)
    
    df = pd.concat([df, encoded_df], axis=1)
    df = df.drop(columns=[col])
    
    return df, encoder