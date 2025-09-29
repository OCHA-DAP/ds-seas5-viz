---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.1
  kernelspec:
    display_name: ds-seas5-viz
    language: python
    name: ds-seas5-viz
---

# Sandbox

```python
%load_ext jupyter_black
%load_ext autoreload
%autoreload 2
```

```python
import ocha_stratus as stratus
import pandas as pd

from src.utils.timeseries import detrend_column
from src.utils.rp_calc import calculate_one_group_rp
```

```python
query = """
SELECT *
FROM public.seas5
WHERE pcode = 'SO'
"""
```

```python
df = pd.read_sql(
    query,
    stratus.get_engine(stage="prod"),
    parse_dates=["valid_date", "issued_date"],
)
```

```python
df
```

```python
issued_month = 8
# valid_months = [10, 11, 12]
valid_months = [11, 12, 1]
```

```python
df_issue = (
    df[
        df["valid_date"].dt.month.isin(valid_months)
        & (df["issued_date"].dt.month == issued_month)
    ]
    .groupby(df["issued_date"].dt.year)["mean"]
    .mean()
    .rename_axis("year")
    .reset_index()
)
```

```python
df_issue = detrend_column(df_issue, col="mean", index_col="year")
```

```python
df_issue
```

```python
df_issue = calculate_one_group_rp(df_issue, col_name="mean", ascending=True)
df_issue = calculate_one_group_rp(
    df_issue, col_name="mean_detrended", ascending=True
)
```

```python
df_issue.sort_values("mean_rank")
```

```python

```
