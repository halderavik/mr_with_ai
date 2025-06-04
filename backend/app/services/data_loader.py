# backend/app/services/data_loader.py

import pandas as pd
import pyreadstat
from pathlib import Path
from ..services.file_handler import get_file_path
from fastapi import HTTPException

def load_dataset(user_id: str, dataset_id: str) -> pd.DataFrame:
    """
    Inspect the file extension; load using pyreadstat (for .sav) or pandas (for .csv/.xlsx).
    Return a pandas DataFrame.
    """
    filepath = get_file_path(user_id, dataset_id)
    suffix = filepath.suffix.lower()

    if suffix == ".sav":
        try:
            df, meta = pyreadstat.read_sav(str(filepath))
            return df
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error reading SPSS file: {e}")
    elif suffix in {".csv"}:
        try:
            df = pd.read_csv(str(filepath))
            return df
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error reading CSV file: {e}")
    elif suffix in {".xls", ".xlsx"}:
        try:
            df = pd.read_excel(str(filepath))
            return df
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error reading Excel file: {e}")
    else:
        raise HTTPException(status_code=400, detail="Unsupported file extension.")

# Logic for loading and parsing data files (CSV, Excel, SPSS, etc.) 