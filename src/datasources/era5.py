from typing import List

import ocha_stratus as stratus
import pandas as pd

from src.utils.timeseries import detrend_column


def load_era5(
    pcode: str,
    valid_months: List[int] = None,
):
    if valid_months is None:
        valid_months = range(1, 13)

    query = """
    SELECT *
    FROM public.era5
    WHERE pcode = %s
      AND EXTRACT(MONTH FROM valid_date) IN %s
    """
    engine = stratus.get_engine("prod")
    with engine.connect() as conn:
        df = pd.read_sql(
            query,
            conn,
            params=(pcode, tuple(valid_months)),
            parse_dates=["valid_date"],
        )
    return df


def aggregate_era5_yearly(
    df: pd.DataFrame,
    valid_months: List[int],
):
    df_monthly = df[df["valid_date"].dt.month.isin(valid_months)]
    df_yearly = (
        df_monthly.groupby(df_monthly["valid_date"].dt.year)["mean"]
        .mean()
        .reset_index()
    )
    df_yearly = df_yearly.rename(columns={"valid_date": "year"})
    df_yearly = detrend_column(df_yearly, "mean", index_col="year")
    return df_yearly
