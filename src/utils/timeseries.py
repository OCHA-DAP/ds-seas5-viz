import numpy as np
import pandas as pd


def detrend_column(
    df: pd.DataFrame,
    col: str,
    index_col: str = "valid_date",
    min_index=None,
    max_index=None,
) -> pd.DataFrame:
    """
    Detrend a column in a DataFrame using linear regression (via NumPy).

    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame. Must contain a datetime column.
    col : str
        The name of the column to detrend.
    time_col : str
        The name of the datetime column. Default is "valid_date".

    Returns:
    --------
    pd.DataFrame
        Copy of the input DataFrame with a new column: <col>_detrended
    """
    if min_index is None:
        min_index = df[index_col].min()
    if max_index is None:
        max_index = df[index_col].max()

    df_sorted = df.sort_values(index_col).copy()
    df_model = df_sorted[
        (df_sorted[index_col] >= min_index)
        & (df_sorted[index_col] <= max_index)
    ]

    x = df_model[index_col]
    y = df_model[col].values

    # Linear regression fit
    A = np.vstack([x, np.ones_like(x)]).T
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]

    trend = a * df_sorted[index_col] + b
    detrended = df_sorted[col] - trend
    detrended += y.mean()  # Shift to preserve original mean

    df_sorted[f"{col}_detrended"] = detrended

    return df_sorted
