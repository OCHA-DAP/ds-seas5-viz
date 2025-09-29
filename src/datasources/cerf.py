import pandas as pd


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
