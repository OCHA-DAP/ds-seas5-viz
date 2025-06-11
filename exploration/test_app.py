import marimo

__generated_with = "0.13.15"
app = marimo.App()

# flake8: noqa: E501


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # SEAS5-ERA5-EM-DAT explorer

    This app compares SEAS5 seasonal forecasts with ERA5 reanalysis and EM-DAT impact. Currently it is set only to "Flood" mode, meaning it shows the historical impact from flooding and the above-normal rainfall tercile.
    """
    )
    return


@app.cell
def _():
    import calendar

    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    import numpy as np
    import ocha_stratus as stratus
    import pandas as pd

    from src.datasources import cerf, emdat, era5, seas5

    return calendar, cerf, emdat, era5, mpatches, np, pd, plt, seas5, stratus


@app.cell
def _():
    import marimo as mo

    return (mo,)


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
    df_polygons = pd.read_sql(
        "SELECT pcode, name, iso3, adm_level FROM public.polygon WHERE adm_level = 0",
        stratus.get_engine(stage="prod"),
    )
    options = {
        f"{row['name']} ({row['iso3']})": row["pcode"]
        for _, row in df_polygons.iterrows()
    }
    return (options,)


@app.cell
def _(calendar, mo, options):
    pcode_dropdown = mo.ui.dropdown(
        options=options, label="Select Country", value="Ethiopia (ETH)"
    )
    issued_month_dropdown = mo.ui.dropdown(
        options={calendar.month_abbr[x]: x for x in range(1, 13)},
        label="Select issued month",
        value="May",
    )
    return issued_month_dropdown, pcode_dropdown


@app.cell
def _(issued_month_dropdown):
    issued_month = issued_month_dropdown.value
    return (issued_month,)


@app.cell
def _(pcode_dropdown):
    pcode_dropdown
    return


@app.cell
def _(pcode_dropdown):
    pcode = pcode_dropdown.value
    return (pcode,)


@app.cell
def _(issued_month_dropdown):
    issued_month_dropdown
    return


@app.cell
def _(calendar, issued_month, mo):
    valid_month_options = [issued_month + x for x in range(7)]
    valid_month_widget = mo.ui.multiselect(
        options={calendar.month_abbr[x]: x for x in valid_month_options},
        label="Select valid months",
        value=["Jul", "Aug", "Sep"],
    )
    valid_month_widget
    return (valid_month_widget,)


@app.cell
def _(valid_month_widget):
    valid_months = valid_month_widget.value
    return (valid_months,)


@app.cell
def _(calendar, issued_month, valid_months):
    if len(valid_months) < 3:
        valid_mo_str = "-".join(
            [calendar.month_abbr[x] for x in sorted(valid_months)]
        )
    else:
        valid_mo_str = "".join(
            [calendar.month_abbr[x][0] for x in sorted(valid_months)]
        )

    issued_mo_str = calendar.month_abbr[issued_month]
    return issued_mo_str, valid_mo_str


@app.cell
def _(pcode, pd, stratus):
    query = f"SELECT * FROM public.polygon WHERE pcode = '{pcode}'"
    df_adm = pd.read_sql(query, stratus.get_engine(stage="prod"))
    adm_name, iso3, adm_level = df_adm.iloc[0][["name", "iso3", "adm_level"]]
    if adm_level > 0:
        query = f"SELECT * FROM public.polygon WHERE iso3 = '{iso3}' AND adm_level = 0"
        df_adm0 = pd.read_sql(query, stratus.get_engine(stage="prod"))
        adm0_name = df_adm0.iloc[0]["name"]
        adm_name_str = f"{adm_name} ({adm0_name})"
    else:
        adm_name_str = adm_name
    return adm_name, adm_name_str, iso3


@app.cell
def _(pcode, seas5):
    df_seas5_all = seas5.load_seas5(pcode=pcode)
    return (df_seas5_all,)


@app.cell
def _(df_seas5_all, issued_month, seas5, valid_months):
    df_seas5 = seas5.aggregate_seas5_yearly(
        df_seas5_all, issued_month=issued_month, valid_months=valid_months
    )
    return (df_seas5,)


@app.cell
def _(era5, pcode):
    df_era5_all = era5.load_era5(pcode=pcode)
    return (df_era5_all,)


@app.cell
def _(df_era5_all, era5, valid_months):
    df_era5 = era5.aggregate_era5_yearly(
        df_era5_all, valid_months=valid_months
    )
    return (df_era5,)


@app.cell
def _(df_era5_all):
    max_full_year = df_era5_all["valid_date"].dt.year.max() - 1
    df_era5_monthly = (
        df_era5_all[df_era5_all["valid_date"].dt.year <= max_full_year]
        .groupby(df_era5_all["valid_date"].dt.month)["mean"]
        .mean()
        .reset_index()
    )
    return df_era5_monthly, max_full_year


@app.cell
def _(calendar, df_era5_monthly):
    df_era5_monthly["valid_month_str"] = df_era5_monthly["valid_date"].apply(
        lambda x: calendar.month_abbr[x]
    )
    return


@app.cell
def _(disaster_type, emdat, impact_col, iso3):
    df_emdat = emdat.load_emdat_yearly(
        iso3=iso3, disaster_type=disaster_type, col=impact_col
    )
    return (df_emdat,)


@app.cell
def _(cerf, disaster_type, iso3):
    df_cerf = cerf.load_cerf_yearly(emergency=disaster_type, iso3=iso3)
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
def _(issued_mo_str):
    col_to_label = {
        "mean_detrended_seas5": "Forecasted mean daily rainfall, detrended, "
        f"issued {issued_mo_str} (mm) [SEAS5]",
        "mean_detrended_era5": "Observed mean daily rainfall, detrended (mm) [ERA5]",
    }
    return (col_to_label,)


@app.cell
def _(np):
    high_color = "royalblue"
    current_color = "mediumorchid"
    cerf_color_mapping = {
        "Yes": "crimson",
        "No": "k",
        "pre-CERF": "gray",
        np.nan: "k",
    }
    return cerf_color_mapping, current_color, high_color


@app.cell
def _(
    cerf_color_mapping,
    col_to_label,
    current_color,
    high_color,
    mpatches,
    np,
    plt,
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
        show_terciles: bool = True,
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
        if show_terciles:
            df_ref = df.dropna(subset=[xcol, ycol])
            # ref_min_year = df_ref["year"].min()
            # ref_max_year = df_ref["year"].max()
            x_thresh, y_thresh = df_ref[[xcol, ycol]].quantile(2 / 3)
            _ax.axvspan(
                xmin=x_thresh,
                xmax=xlim[1],
                facecolor=high_color,
                alpha=0.1,
                zorder=-1,
            )
            _ax.axhspan(
                ymin=y_thresh,
                ymax=ylim[1],
                facecolor=high_color,
                alpha=0.1,
                zorder=-1,
            )
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
        if "seas5" in xcol:
            current_val = df.set_index("year").loc[2025][xcol]
            _ax.axvline(
                current_val, color=current_color, linestyle="--", zorder=-1
            )
            _ax.annotate(
                " 2025 forecast",
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
        legend_x = xlim[0] + xrange * 0.18

        def get_legend_y(row_num):
            return ylim[1] - yrange * 0.04 - yrange * row_num * 0.03

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
        rect = mpatches.Rectangle(
            (legend_x - xrange * 0.16, get_legend_y(4.7)),
            xrange * 0.32,
            yrange * 0.16,
            linewidth=0.5,
            color="white",
            zorder=0,
            alpha=0.5,
        )
        _ax.add_patch(rect)
        return (_fig, _ax)

    return (plot_comparison,)


@app.cell
def _(adm_name, df_compare, plot_comparison, valid_mo_str):
    min_year = 2000
    _fig, _ax = plot_comparison(
        df_compare,
        xcol="mean_detrended_seas5",
        ycol="mean_detrended_era5",
        sizecol="Total Affected",
        colorcol="allocation",
        title=f"{adm_name}: {valid_mo_str} observed vs. forecasted rainfall,\nsince {min_year}",
        min_year=min_year,
    )
    _ax
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    Notes on reading the plot:

    - The size of the bubbles corresponds to the total impact from "Flood" events in the EM-DAT database during that year.
    - Bubbles in red denote years with at least one "Rapid Response" CERF allocation for a "Flood" during that year. **Note that this has only been added for Ethiopia and South Sudan so far, all other countries will just show "pre-CERF".**
    - The blue zones at the top and to the right correspond to the upper tercile of the distribution for the reanalysis and reforecast respectively.
    - A stronger correlation between the reanalysis and the reforecast denotes a better forecast skill for this issue month / valid months / geography combination.
    - Both the reanalysis and reforecast have been de-trended.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""## Reference""")
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
