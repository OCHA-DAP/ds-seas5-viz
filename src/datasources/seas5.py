from typing import List

import ocha_stratus as stratus
import pandas as pd

from src.utils.timeseries import detrend_column


def load_seas5(
    pcode: str,
    issued_months: List[int] = None,
    valid_months: List[int] = None,
):
    if issued_months is None:
        issued_months = range(1, 13)  # Default to all months
    if valid_months is None:
        valid_months = range(1, 13)

    query = """
    SELECT *
    FROM public.seas5
    WHERE pcode = %s
      AND EXTRACT(MONTH FROM issued_date) IN %s
      AND EXTRACT(MONTH FROM valid_date) IN %s
    """
    engine = stratus.get_engine("prod")
    with engine.connect() as conn:
        df = pd.read_sql(
            query,
            conn,
            params=(pcode, tuple(issued_months), tuple(valid_months)),
            parse_dates=["valid_date", "issued_date"],
        )
    return df


def aggregate_seas5_yearly(
    df: pd.DataFrame,
    issued_month: int,
    valid_months: List[int],
):
    df_monthly = df[
        (df["issued_date"].dt.month == issued_month)
        & (df["valid_date"].dt.month.isin(valid_months))
    ]
    df_yearly = (
        df_monthly.groupby(df_monthly["valid_date"].dt.year)["mean"]
        .mean()
        .reset_index()
    )
    df_yearly = df_yearly.rename(columns={"valid_date": "year"})
    max_year = df_yearly["year"].max()
    df_yearly = detrend_column(
        df_yearly, "mean", index_col="year", max_index=max_year - 1
    )
    return df_yearly
