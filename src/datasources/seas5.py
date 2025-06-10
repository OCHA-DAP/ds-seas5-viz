from typing import List

import ocha_stratus as stratus
import pandas as pd

from src.utils.timeseries import detrend_column


def load_seas5(
    pcode: str,
    issued_month: int,
    valid_months: List[int],
):
    if issued_month < 1 or issued_month > 12:
        raise ValueError(
            f"Invalid issued month: {issued_month}. Must be between 1 and 12."
        )
    for valid_month in valid_months:
        if valid_month < 1 or valid_month > 12:
            raise ValueError(
                f"Invalid month: {valid_month}. Must be between 1 and 12."
            )
        lt = (valid_month - issued_month) % 12
        if lt > 6:
            raise ValueError(
                f"Invalid lead time: {lt} (from valid month {valid_month}). "
                "Must be 6 months or less."
            )

    query = """
    SELECT *
    FROM public.seas5
    WHERE pcode = %s
      AND EXTRACT(MONTH FROM issued_date) = %s
      AND EXTRACT(MONTH FROM valid_date) IN %s
    """
    engine = stratus.get_engine("prod")
    with engine.connect() as conn:
        df = pd.read_sql(
            query,
            conn,
            params=(pcode, issued_month, tuple(valid_months)),
            parse_dates=["valid_date", "issued_date"],
        )
    return df


def load_seas5_yearly(
    pcode: str,
    issued_month: int,
    valid_months: List[int],
):
    df_monthly = load_seas5(
        pcode=pcode, issued_month=issued_month, valid_months=valid_months
    )
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
