# backend/app/services/data_loader.py

import pandas as pd
import pyreadstat
from pathlib import Path
from ..services.file_handler import get_file_path
from fastapi import HTTPException
import os
import json
from typing import Any, Dict

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploaded_files")

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

def load_metadata(user_id: str, dataset_id: str) -> Dict[str, Any]:
    """
    Load SPSS metadata from a JSON file for the given user and dataset.
    Assumes metadata is stored as 'uploaded_files/{user_id}/{dataset_id}_meta.json'.
    """
    meta_path = os.path.join(UPLOAD_DIR, user_id, f"{dataset_id}_meta.json")
    if not os.path.exists(meta_path):
        return {}
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)

# Logic for loading and parsing data files (CSV, Excel, SPSS, etc.) 