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

# Data loading

Testing data loading

```python
%load_ext jupyter_black
%load_ext autoreload
%autoreload 2
```

```python
from src.datasources import seas5, era5, emdat
```

```python
pcode = "SS"
issued_month = 6
valid_months = [7, 8, 9]
```

## SEAS5

```python
df_test = seas5.load_seas5(
    pcode=pcode, issued_month=issued_month, valid_months=valid_months
)
```

```python
df_test
```

```python
df_test_yearly = seas5.load_seas5_yearly(
    pcode=pcode, issued_month=issued_month, valid_months=valid_months
)
```

```python
df_test_yearly.set_index("year").plot()
```

## ERA5

```python
df_test_era5 = era5.load_era5(pcode=pcode, valid_months=valid_months)
```

```python
df_test_era5["valid_date"].max()
```

```python
df_test_era5_yearly = era5.load_era5_yearly(
    pcode=pcode, valid_months=valid_months
)
```

```python
df_test_era5_yearly.set_index("year").plot()
```

## EM-DAT

```python
df_test_emdat = emdat.load_emdat(iso3="ssd", disaster_type="Flood")
```

```python
df_test_emdat
```
