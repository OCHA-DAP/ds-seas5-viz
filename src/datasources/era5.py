from typing import List

import ocha_stratus as stratus
import pandas as pd

from src.utils.timeseries import detrend_column


def load_era5(
    pcode: str,
    valid_months: List[int],
):
    for valid_month in valid_months:
        if valid_month < 1 or valid_month > 12:
            raise ValueError(
                f"Invalid month: {valid_month}. Must be between 1 and 12."
            )

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


def load_era5_yearly(
    pcode: str,
    valid_months: List[int],
):
    df_monthly = load_era5(pcode=pcode, valid_months=valid_months)
    df_yearly = (
        df_monthly.groupby(df_monthly["valid_date"].dt.year)["mean"]
        .mean()
        .reset_index()
    )
    df_yearly = df_yearly.rename(columns={"valid_date": "year"})
    df_yearly = detrend_column(df_yearly, "mean", index_col="year")
    return df_yearly
