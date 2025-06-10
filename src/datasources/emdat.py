import duckdb
import ocha_stratus as stratus

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
    df = load_emdat(iso3=iso3, disaster_type=disaster_type, historic=historic)

    df_emdat_yearly = df.groupby("Start Year")[col].sum().reset_index()
    df_emdat_yearly = df_emdat_yearly.set_index("Start Year")
    df_emdat_yearly = df_emdat_yearly.reindex(range(2000, 2025), fill_value=0)
    df_emdat_yearly = df_emdat_yearly.reset_index().rename(
        columns={"Start Year": "year"}
    )
    if col == "Total Affected":
        df_emdat_yearly[col] = df_emdat_yearly[col].astype(int)

    return df_emdat_yearly
