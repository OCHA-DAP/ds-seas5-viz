import marimo

__generated_with = "0.16.0"
app = marimo.App(app_title="SEAS5-ERA5-EMDAT")


@app.cell
def _(mo):
    mo.image(src="exploration/assets/centre_banner.png", height=100)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # SEAS5-ERA5-EMDAT explorer

    This app compares SEAS5 seasonal forecasts with ERA5 reanalysis and EM-DAT impact.
    """
    )
    return


@app.cell
def _():
    import calendar
    from typing import List

    import duckdb
    import marimo as mo
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    import numpy as np
    import ocha_stratus as stratus
    import pandas as pd

    return List, calendar, duckdb, mo, mpatches, np, pd, plt, stratus


@app.cell(hide_code=True)
def _(np, pd):
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

    return (detrend_column,)


@app.cell(hide_code=True)
def _(List, detrend_column, pd, stratus):
    # ERA5

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
        df_monthly = df[df["valid_date"].dt.month.isin(valid_months)].copy()
        df_monthly["year"] = df_monthly["valid_date"].dt.year
        df_monthly["month"] = df_monthly["valid_date"].dt.month

        # Ensure each year has *all* valid months
        complete_years = (
            df_monthly.groupby("year")["month"]
            .nunique()
            .loc[lambda x: x == len(valid_months)]
            .index
        )
        df_complete = df_monthly[df_monthly["year"].isin(complete_years)]

        if 1 in valid_months and 12 in valid_months:

            def shift_valid_year(row):
                year = row["year"]
                return year if row["month"] >= 7 else year - 1

            df_complete["season_year"] = df_complete.apply(
                shift_valid_year, axis=1
            )
        df_yearly = df_complete.groupby("year")["mean"].mean().reset_index()
        df_yearly = detrend_column(df_yearly, "mean", index_col="year")
        return df_yearly

    return aggregate_era5_yearly, load_era5


@app.cell(hide_code=True)
def _(List, pd, stratus):
    # SEAS5

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
            df_monthly.groupby(df_monthly["issued_date"].dt.year)["mean"]
            .mean()
            .reset_index()
        )
        df_yearly = df_yearly.rename(columns={"issued_date": "year"})
        # max_year = df_yearly["year"].max()
        # df_yearly = detrend_column(
        #     df_yearly, "mean", index_col="year", max_index=max_index
        # )
        return df_yearly

    return aggregate_seas5_yearly, load_seas5


@app.cell(hide_code=True)
def _(duckdb, stratus):
    # EM-DAT

    EMDAT_PROC_BLOB_NAME = "emdat/processed/emdat_all.parquet"

    def load_emdat(
        iso3: str = None, disaster_type: str = None, historic: bool = False
    ):
        if iso3 is None and disaster_type is None and historic:
            return stratus.load_parquet_from_blob(
                EMDAT_PROC_BLOB_NAME, container_name="global"
            )

        url = (
            stratus.get_container_client(container_name="global")
            .get_blob_client(EMDAT_PROC_BLOB_NAME)
            .url
        )

        filters = []
        if iso3 is not None:
            filters.append(f"ISO = '{iso3.upper()}'")
        if disaster_type is not None:
            filters.append(f"\"Disaster Type\" = '{disaster_type}'")
        if not historic:
            filters.append("Historic = 'No'")

        where_clause = f"WHERE {' AND '.join(filters)}" if filters else ""

        query = f"""
        SELECT *
        FROM read_parquet('{url}')
        {where_clause}
        """

        with duckdb.connect() as conn:
            return conn.execute(query).df()

    def load_emdat_yearly(
        iso3: str = None,
        disaster_type: str = None,
        historic: bool = False,
        col: str = "Total Affected",
    ):
        df = load_emdat(
            iso3=iso3, disaster_type=disaster_type, historic=historic
        )

        df_emdat_yearly = df.groupby("Start Year")[col].sum().reset_index()
        df_emdat_yearly = df_emdat_yearly.set_index("Start Year")
        df_emdat_yearly = df_emdat_yearly.reindex(
            range(2000, 2025), fill_value=0
        )
        df_emdat_yearly = df_emdat_yearly.reset_index().rename(
            columns={"Start Year": "year"}
        )
        if col == "Total Affected":
            df_emdat_yearly[col] = df_emdat_yearly[col].astype(int)

        return df_emdat_yearly

    return (load_emdat_yearly,)


@app.cell(hide_code=True)
def _(pd):
    # CERF
    # just a dummy function hard coding values until we load the actual thing
    def load_cerf_raw():
        columns = [
            "iso3",
            "Allocation date",
            "Amount in US$",
        ]
        data = [["ETH", f"{x}-01-01", 1] for x in [2023, 2020, 2018, 2006]] + [
            ["SSD", f"{x}-01-01", 1] for x in [2019, 2020, 2021, 2022, 2024]
        ]

        df = pd.DataFrame(data, columns=columns)
        df["Allocation date"] = pd.to_datetime(df["Allocation date"])
        df["Window"] = "Rapid Response"
        df["Emergency"] = "Flood"
        return df

    def load_cerf_yearly(
        emergency: str, iso3: str, window: str = "Rapid Response"
    ):
        df_raw = load_cerf_raw()
        df = df_raw[
            (df_raw["Emergency"] == emergency)
            & (df_raw["Window"] == window)
            & (df_raw["iso3"] == iso3)
        ].copy()
        df["year"] = pd.to_datetime(df["Allocation date"]).dt.year
        df_yearly = df.groupby("year")["Amount in US$"].sum().reset_index()
        df_yearly = df_yearly.set_index("year")
        df_yearly = df_yearly.reindex(
            range(2006, 2025), fill_value=0
        ).reset_index()
        df_yearly["allocation"] = df_yearly["Amount in US$"].apply(
            lambda x: "Yes" if x > 0 else "No"
        )
        # just set all to pre-CERF if haven't been filled in
        if df.empty:
            df_yearly["allocation"] = "pre-CERF"
        return df_yearly

    return (load_cerf_yearly,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Set parameters""")
    return


@app.cell
def _():
    disaster_type = "Flood"
    impact_col = "Total Affected"
    return disaster_type, impact_col


@app.cell
def _(pd, stratus):
    df_adms = pd.read_sql(
        "SELECT pcode, name, iso3, adm_level FROM public.polygon ORDER BY name ASC",
        stratus.get_engine(stage="prod"),
    )
    df_adm0 = df_adms.set_index("adm_level").loc[0]
    df_adm1 = df_adms.set_index("adm_level").loc[1]
    df_adm2 = df_adms.set_index("adm_level").loc[2]
    adm0_options = {
        row["name"]: row["pcode"]
        for _, row in df_adm0.iterrows()
        if row["name"] is not None
    }
    return adm0_options, df_adm1, df_adm2, df_adms


@app.cell
def _(adm0_options, mo):
    adm0_dropdown = mo.ui.dropdown(
        options=adm0_options, label="Select country:", value="Ethiopia"
    )
    return (adm0_dropdown,)


@app.cell
def _(mo):
    mo.md(r"""### Administrative division""")
    return


@app.cell
def _(adm0_dropdown):
    adm0_dropdown
    return


@app.cell
def _(adm0_dropdown, df_adms):
    adm0_pcode = adm0_dropdown.value
    adm0_name = adm0_dropdown.selected_key
    iso3 = df_adms[df_adms["pcode"] == adm0_pcode].iloc[0]["iso3"]
    return adm0_name, adm0_pcode, iso3


@app.cell
def _(mo):
    adm_level_dropdown = mo.ui.dropdown(
        options=[0, 1, 2], label="Select admin level:", value=0
    )
    return (adm_level_dropdown,)


@app.cell
def _(adm_level_dropdown):
    adm_level_dropdown
    return


@app.cell
def _(adm_level_dropdown):
    adm_level = adm_level_dropdown.value
    return (adm_level,)


@app.cell
def _(adm0_pcode, adm_level, df_adm1, iso3, mo):
    if adm_level > 0 and adm0_pcode is not None:
        adm1_options = {
            row["name"]: row["pcode"]
            for _, row in df_adm1[df_adm1["iso3"] == iso3].iterrows()
        }
    else:
        adm1_options = []

    adm1_dropdown = mo.ui.dropdown(
        options=adm1_options, label="Select admin1:", value=None
    )
    adm1_dropdown
    return (adm1_dropdown,)


@app.cell
def _(adm1_dropdown):
    adm1_pcode = adm1_dropdown.value
    adm1_name = adm1_dropdown.selected_key
    return adm1_name, adm1_pcode


@app.cell
def _(adm1_pcode, adm_level, df_adm2, mo):
    if adm_level > 1 and adm1_pcode is not None:
        adm2_options = {
            row["name"]: row["pcode"]
            for _, row in df_adm2[
                df_adm2["pcode"].str.startswith(adm1_pcode)
            ].iterrows()
        }
        adm2_dropdown = mo.ui.dropdown(
            options=adm2_options, label="Select admin2:", value=None
        )
    else:
        adm2_dropdown = mo.ui.dropdown(
            options=[], label="Select admin2:", value=None
        )
    return (adm2_dropdown,)


@app.cell
def _(adm2_dropdown):
    adm2_dropdown
    return


@app.cell
def _(adm2_dropdown):
    adm2_pcode = adm2_dropdown.value
    adm2_name = adm2_dropdown.selected_key
    return adm2_name, adm2_pcode


@app.cell
def _(
    adm0_name,
    adm0_pcode,
    adm1_name,
    adm1_pcode,
    adm2_name,
    adm2_pcode,
    adm_level,
):
    if adm_level == 0:
        pcode = adm0_pcode
        adm_name_str = adm0_name
    elif adm_level == 1:
        if adm1_pcode is None:
            raise ValueError("adm1 not set")
        pcode = adm1_pcode
        adm_name_str = f"{adm1_name}, {adm0_name}"
    elif adm_level == 2:
        if adm2_pcode is None:
            raise ValueError("adm2 not set")
        pcode = adm2_pcode
        adm_name_str = f"{adm2_name}, {adm1_name}, {adm0_name}"
    return adm_name_str, pcode


@app.cell
def _(mo):
    mo.md(r"""### Months""")
    return


@app.cell
def _(calendar, mo):
    issued_month_dropdown = mo.ui.dropdown(
        options={calendar.month_abbr[x]: x for x in range(1, 13)},
        label="Issued month:",
        value="May",
    )
    return (issued_month_dropdown,)


@app.cell
def _(issued_month_dropdown):
    issued_month_dropdown
    return


@app.cell
def _(issued_month_dropdown):
    issued_month = issued_month_dropdown.value
    return (issued_month,)


@app.cell
def _(mo):
    valid_months_slider = mo.ui.range_slider(
        steps=range(7), label="Leadtimes", value=(1, 3)
    )
    return (valid_months_slider,)


@app.cell
def _(mo, valid_mo_str, valid_months_note, valid_months_slider):
    mo.hstack(
        [
            valid_months_slider,
            mo.md(f"**{valid_mo_str}**"),
            mo.md(valid_months_note),
        ],
        justify="start",
    )
    return


@app.cell
def _(issued_month, valid_months_slider):
    # valid_months = valid_month_widget.value
    valid_months = [
        (issued_month + x - 1) % 12 + 1
        for x in range(
            valid_months_slider.value[0], valid_months_slider.value[1] + 1
        )
    ]
    if 1 in valid_months and 12 in valid_months:
        valid_months_shift = [(x - 7) % 12 + 1 for x in valid_months]
        valid_months_shift = sorted(valid_months_shift)
        valid_months = [(x + 5) % 12 + 1 for x in valid_months_shift]
    else:
        valid_months = sorted(valid_months)
    return (valid_months,)


@app.cell
def _(calendar, issued_month, valid_months):
    if len(valid_months) < 3:
        valid_mo_str = "-".join([calendar.month_abbr[x] for x in valid_months])
    else:
        valid_mo_str = "".join(
            [calendar.month_abbr[x][0] for x in valid_months]
        )

    issued_mo_str = calendar.month_abbr[issued_month]
    return issued_mo_str, valid_mo_str


@app.cell
def _(load_seas5, pcode):
    df_seas5_all = load_seas5(pcode=pcode)
    return (df_seas5_all,)


@app.cell
def _(aggregate_seas5_yearly, df_seas5_all, issued_month, valid_months):
    df_seas5_season = aggregate_seas5_yearly(
        df_seas5_all,
        issued_month=issued_month,
        valid_months=valid_months,
    )
    if min(valid_months) < issued_month and 12 not in valid_months:
        df_seas5_season["year"] += 1

    return (df_seas5_season,)


@app.cell
def _(detrend_column, df_seas5_season, show_current_forecast):
    max_year = df_seas5_season["year"].max()
    max_index = max_year - 1 if show_current_forecast else max_year
    df_seas5 = detrend_column(
        df_seas5_season, "mean", index_col="year", max_index=max_index
    )
    return (df_seas5,)


@app.cell
def _(df_seas5_season):
    forecast_issued_year = df_seas5_season["year"].max()
    return (forecast_issued_year,)


@app.cell
def _(df_era5, forecast_issued_year):
    show_current_forecast = forecast_issued_year not in df_era5["year"].values
    valid_months_note = (
        ""
        if show_current_forecast
        else "_reanalysis available; current forecast line will not be shown_"
    )
    return show_current_forecast, valid_months_note


@app.cell
def _(load_era5, pcode):
    df_era5_all = load_era5(pcode=pcode)
    return (df_era5_all,)


@app.cell
def _(aggregate_era5_yearly, df_era5_all, valid_months):
    df_era5 = aggregate_era5_yearly(df_era5_all, valid_months=valid_months)
    return (df_era5,)


@app.cell
def _(calendar, df_era5_all):
    # just for the seasonality plot
    max_full_year = df_era5_all["valid_date"].dt.year.max() - 1
    df_era5_monthly = (
        df_era5_all[df_era5_all["valid_date"].dt.year <= max_full_year]
        .groupby(df_era5_all["valid_date"].dt.month)["mean"]
        .mean()
        .reset_index()
    )
    df_era5_monthly["valid_month_str"] = df_era5_monthly["valid_date"].apply(
        lambda x: calendar.month_abbr[x]
    )
    return df_era5_monthly, max_full_year


@app.cell
def _(disaster_type, impact_col, iso3, load_emdat_yearly):
    df_emdat = load_emdat_yearly(
        iso3=iso3, disaster_type=disaster_type, col=impact_col
    )
    return (df_emdat,)


@app.cell
def _(disaster_type, iso3, load_cerf_yearly):
    df_cerf = load_cerf_yearly(emergency=disaster_type, iso3=iso3)
    return (df_cerf,)


@app.cell
def _(df_cerf, df_emdat, df_era5, df_seas5):
    df_compare = (
        df_seas5.merge(
            df_era5, on="year", how="outer", suffixes=("_seas5", "_era5")
        )
        .merge(df_emdat, how="outer")
        .merge(df_cerf, how="outer")
    )
    return (df_compare,)


@app.cell
def _(df_compare):
    df_compare.loc[df_compare["year"] < 2006, "allocation"] = "pre-CERF"
    return


@app.cell
def _(mo):
    mo.md(r"""### Plot options""")
    return


@app.cell
def _(mo):
    high_tercile_selector = mo.ui.checkbox(label="Upper tercile")
    low_tercile_selector = mo.ui.checkbox(label="Lower tercile")
    return high_tercile_selector, low_tercile_selector


@app.cell
def _(high_tercile_selector, low_tercile_selector, mo):
    mo.hstack(
        [
            mo.md("Show tercile boundaries:"),
            mo.vstack([high_tercile_selector, low_tercile_selector], gap=0),
        ],
        align="center",
    )
    return


@app.cell
def _(high_tercile_selector, low_tercile_selector):
    show_high_tercile = high_tercile_selector.value
    show_low_tercile = low_tercile_selector.value
    return show_high_tercile, show_low_tercile


@app.cell
def _(adm_level, mo):
    allow_impact = adm_level == 0
    options = ["Flood"] if allow_impact else []

    hazard_dropdown = mo.ui.dropdown(
        options=options,
        label="Display impact data: ",
    )
    hazard_note = (
        "" if allow_impact else "_impact data only available for ADM0_"
    )
    return hazard_dropdown, hazard_note


@app.cell
def _(hazard_dropdown, hazard_note, mo):
    mo.hstack(
        [
            hazard_dropdown,
            mo.md(
                hazard_note,
            ),
        ],
        justify="start",
    )
    return


@app.cell
def _(hazard_dropdown):
    hazard = hazard_dropdown.value
    return (hazard,)


@app.cell
def _(mo):
    min_year_selector = mo.ui.dropdown(
        options=range(1981, 2011),
        allow_select_none=False,
        value=2000,
        label="Start year: ",
    )
    return (min_year_selector,)


@app.cell
def _(min_year_note, min_year_selector, mo):
    mo.hstack([min_year_selector, mo.md(min_year_note)], justify="start")
    return


@app.cell
def _(min_year_selector):
    min_year = min_year_selector.value
    return (min_year,)


@app.cell
def _(min_year):
    min_year_note = (
        "_note that impact data before 2000 is not shown_"
        if min_year < 2000
        else ""
    )
    return (min_year_note,)


@app.cell
def _(mo):
    mo.md(r"""## Plot""")
    return


@app.cell
def _():
    col_to_label = {
        "mean_detrended_seas5": "Forecasted mean daily rainfall (mm) [SEAS5]",
        "mean_detrended_era5": "Observed mean daily rainfall (mm) [ERA5]",
    }
    return (col_to_label,)


@app.cell
def _(np):
    tercile_colors = {"upper": "royalblue", "lower": "chocolate"}
    current_color = "mediumorchid"
    cerf_color_mapping = {
        "Yes": "crimson",
        "No": "k",
        "pre-CERF": "#595959",
        np.nan: "k",
    }
    return cerf_color_mapping, current_color, tercile_colors


@app.cell(hide_code=True)
def _(
    cerf_color_mapping,
    col_to_label,
    current_color,
    mpatches,
    np,
    plt,
    tercile_colors,
):
    def plot_comparison(
        df,
        xcol: str,
        ycol: str,
        colorcol: str = None,
        sizecol: str = None,
        rotation: int = 0,
        min_year: int = None,
        title: str = None,
        show_high_tercile: bool = False,
        show_low_tercile: bool = False,
        show_current_forecast: bool = True,
    ):
        _fig, _ax = plt.subplots(dpi=200, figsize=(7, 7))
        if min_year is not None:
            df = df[df["year"] >= min_year]
        df = df.copy()
        xmax, ymax = df[[xcol, ycol]].max()
        xmin, ymin = df[[xcol, ycol]].min()
        padding = 0.1
        xrange = xmax - xmin
        yrange = ymax - ymin
        xlim = (xmin - padding * xrange, xmax + padding * xrange)
        ylim = (ymin - padding * yrange, ymax + padding * yrange)
        if show_high_tercile and show_low_tercile:
            tercile_alpha = 0.05
        else:
            tercile_alpha = 0.1

        def show_tercile(level):
            df_ref = df.dropna(subset=[xcol, ycol])
            q = 2 / 3 if level == "upper" else 1 / 3
            x_thresh, y_thresh = df_ref[[xcol, ycol]].quantile(q)
            color = tercile_colors[level]
            _ax.axvspan(
                xmin=x_thresh if level == "upper" else xlim[0],
                xmax=xlim[1] if level == "upper" else x_thresh,
                facecolor=color,
                alpha=tercile_alpha,
                zorder=-2,
            )
            _ax.annotate(
                f"  {level} tercile",
                (x_thresh, ylim[0]),
                color=color,
                zorder=-1,
                fontsize=8,
                rotation=90,
                fontstyle="italic",
                alpha=0.5,
                ha="left" if level == "upper" else "right",
            )
            _ax.axhspan(
                ymin=y_thresh if level == "upper" else ylim[0],
                ymax=ylim[1] if level == "upper" else y_thresh,
                facecolor=color,
                alpha=tercile_alpha,
                zorder=-2,
            )
            _ax.annotate(
                f"  {level} tercile",
                (xlim[0], y_thresh),
                color=color,
                zorder=-1,
                fontsize=8,
                fontstyle="italic",
                alpha=0.5,
                va="bottom" if level == "upper" else "top",
            )

        if show_high_tercile:
            show_tercile("upper")
        if show_low_tercile:
            show_tercile("lower")

        max_bubble_size = 2000
        if sizecol is None:
            sizes = np.full(len(df), 0)
            max_size_value = None
        else:
            sizes = df[sizecol].fillna(0) / df[sizecol].max() * max_bubble_size
            max_size_value = df[sizecol].max()
        if colorcol is None:
            df["color"] = "k"
        else:
            df["color"] = df[colorcol].map(cerf_color_mapping)
        _ax.scatter(
            df[xcol],
            df[ycol],
            s=sizes,
            c=df["color"],
            alpha=0.3,
            edgecolor="none",
            zorder=2,
        )
        for year, row in df.set_index("year").iterrows():
            _ax.annotate(
                str(year),
                (row[xcol], row[ycol]),
                fontsize=8,
                ha="center",
                va="center",
                color=row["color"],
                rotation=rotation,
                zorder=3,
            )
        # if show_current_forecast and "seas5" in xcol:
        if show_current_forecast:
            # if 2025 in df["year"].to_list():
            forecast_year = df["year"].max()
            current_val = df.set_index("year").loc[forecast_year][xcol]
            _ax.axvline(
                current_val, color=current_color, linestyle="--", zorder=-1
            )
            _ax.annotate(
                f" {forecast_year} forecast",
                (current_val, ylim[0]),
                rotation=90,
                va="bottom",
                ha="right",
                color=current_color,
                zorder=-1,
                fontstyle="italic",
            )
        _ax.set_xlabel(col_to_label.get(xcol, xcol))
        _ax.set_ylabel(col_to_label.get(ycol, ycol))
        if title is not None:
            _ax.set_title(title)
        _ax.spines["top"].set_visible(False)
        _ax.spines["right"].set_visible(False)
        _ax.set_xlim(xlim)
        _ax.set_ylim(ylim)

        if sizecol is not None or colorcol is not None:

            def get_legend_y(row_num):
                return ylim[1] - yrange * 0.04 - yrange * row_num * 0.03

            legend_x = xlim[0] + xrange * 0.18

            def plot_legend_box(xstart, xwidth):
                rect = mpatches.Rectangle(
                    (xstart, get_legend_y(4.7)),
                    xwidth,
                    yrange * 0.16,
                    linewidth=0.5,
                    color="white",
                    zorder=0,
                    alpha=0.5,
                )
                _ax.add_patch(rect)

        if colorcol is not None:
            _ax.annotate(
                "CERF allocation:",
                (legend_x, get_legend_y(0)),
                va="top",
                fontstyle="italic",
                fontsize=6,
            )
            for i, (label, color) in enumerate(cerf_color_mapping.items()):
                if str(label) == "nan":
                    continue
                _ax.annotate(
                    label,
                    (legend_x, get_legend_y(i + 1)),
                    va="top",
                    color=color,
                    fontsize=6,
                )
            plot_legend_box(legend_x, xrange * 0.16)
        if sizecol is not None:
            x_legend_bubble = legend_x - xrange * 0.08
            y_legend_bubble = get_legend_y(2)
            _ax.scatter(
                [x_legend_bubble],
                [y_legend_bubble],
                s=[max_bubble_size],
                facecolor="none",
                edgecolor="k",
                linewidth=0.5,
            )
            _ax.annotate(
                f"{sizecol}:\n{max_size_value:,.0f}",
                (x_legend_bubble, y_legend_bubble),
                ha="center",
                va="center",
                fontstyle="italic",
                fontsize=6,
            )
            plot_legend_box(legend_x - xrange * 0.16, xrange * 0.16)
            # rect = mpatches.Rectangle(
            #     (legend_x - xrange * 0.16, get_legend_y(4.7)),
            #     xrange * 0.32,
            #     yrange * 0.16,
            #     linewidth=0.5,
            #     color="white",
            #     zorder=0,
            #     alpha=0.5,
            # )
            # _ax.add_patch(rect)
        return (_fig, _ax)

    return (plot_comparison,)


@app.cell
def _(
    adm_name_str,
    df_compare,
    hazard,
    iso3,
    issued_mo_str,
    min_year,
    plot_comparison,
    show_current_forecast,
    show_high_tercile,
    show_low_tercile,
    valid_mo_str,
):
    title = f"{adm_name_str} — $\\bf{{{valid_mo_str}}}$ observed vs. forecasted rainfall\nIssue month: $\\bf{{{issued_mo_str}}}$"

    CERF_ISO3S = ["SSD", "ETH"]

    if hazard == "Flood":
        sizecol = "Total Affected"
        if iso3 in CERF_ISO3S:
            colorcol = "allocation"
        else:
            colorcol = None
    else:
        sizecol, colorcol = None, None

    _fig, _ax = plot_comparison(
        df_compare,
        xcol="mean_detrended_seas5",
        ycol="mean_detrended_era5",
        sizecol=sizecol,
        colorcol=colorcol,
        title=title,
        min_year=min_year,
        show_high_tercile=show_high_tercile,
        show_low_tercile=show_low_tercile,
        show_current_forecast=show_current_forecast,
    )

    _fig
    return


@app.cell
def _(df_compare, min_year):
    if min_year is not None:
        df_ref = df_compare[df_compare["year"] >= min_year]
    else:
        df_ref = df_compare

    df_ref = df_ref.dropna(
        subset=["mean_detrended_seas5", "mean_detrended_era5"]
    )

    metrics = {}
    metrics.update(
        {
            "corr": df_ref[["mean_detrended_seas5", "mean_detrended_era5"]]
            .corr()
            .iloc[0, 1]
        }
    )

    for _tercile in ["upper", "lower"]:
        q = 2 / 3 if _tercile == "upper" else 1 / 3
        seas5_thresh, era5_thresh = df_ref[
            ["mean_detrended_seas5", "mean_detrended_era5"]
        ].quantile(q)
        if _tercile == "upper":
            pp = df_ref["mean_detrended_seas5"] > seas5_thresh
            p = df_ref["mean_detrended_era5"] > era5_thresh
        else:
            pp = df_ref["mean_detrended_seas5"] < seas5_thresh
            p = df_ref["mean_detrended_era5"] < era5_thresh
        tp = pp & p
        tpr = tp.sum() / p.sum()
        metrics.update({f"{_tercile}_tpr": tpr})
    return df_ref, metrics


@app.function
def calculate_one_group_rp(group, col_name: str = "q", ascending: bool = True):
    """Calculate the empirical RP for a single group.

    Parameters
    ----------
    group : pd.DataFrame
        The group for which to calculate the RP.
    col_name : str, optional
        The name of the column for which to calculate the RP, by default "q".
    ascending : bool, optional
        Whether to rank the column in ascending order, by default True.
        Should be False for cases where a high number is severe
        (e.g. precipitation for flooding), and True for cases where a low
        number is severe (e.g. precipitation for drought).

    Returns
    -------
    pd.DataFrame
        The input group with the RP columns added.
    """
    group[f"{col_name}_rank"] = group[col_name].rank(ascending=ascending)
    group[f"{col_name}_rp"] = (len(group) + 1) / group[f"{col_name}_rank"]
    return group


@app.cell
def _(df_compare, df_ref, np, show_current_forecast):
    rps = {}

    if show_current_forecast:
        df_rp_calc = df_ref.copy()
        forecast_year = df_compare["year"].max()
        current_val = df_compare.set_index("year").loc[forecast_year][
            "mean_detrended_seas5"
        ]
        for _tercile in ["upper", "lower"]:
            df_rp_calc = calculate_one_group_rp(
                df_rp_calc,
                col_name="mean_detrended_seas5",
                ascending=_tercile == "lower",
            )
            df_rp_calc = df_rp_calc.sort_values("mean_detrended_seas5")
            rp = np.interp(
                current_val,
                df_rp_calc["mean_detrended_seas5"],
                df_rp_calc["mean_detrended_seas5_rp"],
            )
            rps.update({_tercile: rp})
    return (rps,)


@app.cell
def _(rps, show_current_forecast):
    if show_current_forecast:
        rp_table_str = f"""
        | Upper | Lower |
        |-|-|
        | {rps["upper"]:.1f} years | {rps["lower"]:.1f} years|
        """
    else:
        rp_table_str = "No RP shown as forecast no longer relevant"
    return (rp_table_str,)


@app.cell
def _(metrics, mo, rp_table_str):
    mo.md(
        f"""
    ### Return Period

    {rp_table_str}

    ### Accuracy Metrics

    | Correlation | Upper tercile F1 | Lower tercile F1 |
    |-|-|-|
    | {metrics["corr"]:.2f} | {metrics["upper_tpr"]:.2f} | {metrics["lower_tpr"]:.2f} |

    Recall that values less than 0 and 0.33 are _worse than random_ for correlation and F1, respectively.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    Notes on reading the plot:

    - The year shown is the year of the _first valid_ month. For example, a forecast issued in Nov 2025 would appear as the year:
        - 2026 if it is for JFM
        - 2025 it if is for DJF
    - If the full reanalysis for the relevant valid months is available, the most recent forecast will not be shown as a vertical line.
    - The shaded zones at the top, bottom, left, or right of the plots correspond to the upper or lower terciles of the distribution for the reanalysis and reforecast respectively. The tercile boundaries are calculated empricially using only the years shown on the plot (i.e. the years since "Start Year").
    - Both the reanalysis and reforecast have been de-trended (based on the full reference period since 1981), using a linear curve fit.
    - _[Flood only]_ The size of the bubbles corresponds to the total impact from "Flood" events in the EM-DAT database during that year. The legend shows the size of the largest bubble, and the corresponding maximum impact.
    - _[Flood only]_ Bubbles in red denote years with at least one "Rapid Response" CERF allocation for a "Flood" during that year. **Note that this has only been added for Ethiopia and South Sudan so far, all other countries will just show "pre-CERF".**
    """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""## Reference""")
    return


@app.cell
def _(mo):
    mo.md(r"""### Seasonal rainfall""")
    return


@app.cell
def _(adm_name_str, df_era5_all, df_era5_monthly, max_full_year, plt):
    _fig, _ax = plt.subplots(dpi=200)
    df_era5_monthly.plot.bar(
        x="valid_month_str", y="mean", legend=False, ax=_ax, color="royalblue"
    )
    _ax.set_xlabel("Month")
    _ax.set_ylabel("Mean daily rainfall per month (mm) [ERA5]")
    _ax.set_title(
        f"{adm_name_str}: precipitation seasonality\n(reference period: {df_era5_all['valid_date'].dt.year.min()}-{max_full_year})"
    )
    _ax.spines["top"].set_visible(False)
    _ax.spines["right"].set_visible(False)
    _ax
    return


if __name__ == "__main__":
    app.run()
