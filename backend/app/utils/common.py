# backend/app/utils/common.py

import pandas as pd

def filter_dataframe(df: pd.DataFrame, column: str, value) -> pd.DataFrame:
    """
    Return a filtered copy of df where df[column] == value.
    If column not in df, returns df unmodified (or raise error).
    """
    if column not in df.columns:
        raise ValueError(f"Filter column '{column}' not found in DataFrame.")
    return df[df[column] == value].copy()

# Common utility functions for the backend app 