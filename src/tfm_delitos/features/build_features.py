import pandas as pd

def prepare_timeseries(df: pd.DataFrame, date_col: str, freq: str) -> pd.DataFrame:
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)
    df = df.set_index(date_col).asfreq(freq)
    return df.reset_index()
