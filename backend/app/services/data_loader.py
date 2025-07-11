# backend/app/services/data_loader.py

import pandas as pd
import pyreadstat
from pathlib import Path
from ..services.file_handler import get_file_path
from fastapi import HTTPException
import os
import json
from typing import Any, Dict, Optional
import pickle
from datetime import datetime, timedelta
import time
import numpy as np

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploaded_files")
DATA_DIR = os.getenv("DATA_DIR", "data")

# Cache for metadata with 5-minute expiry
_metadata_cache = {}
CACHE_EXPIRY = 300  # 5 minutes in seconds

def ensure_directory_exists(path: str) -> None:
    """
    Ensure that a directory exists, creating it if necessary.
    
    Args:
        path (str): Directory path to ensure exists
    """
    os.makedirs(path, exist_ok=True)

def get_metadata_path(user_id: str, dataset_id: str) -> str:
    """
    Get the path to the metadata file for a dataset.
    
    Args:
        user_id (str): User ID
        dataset_id (str): Dataset ID
        
    Returns:
        str: Path to metadata file
    """
    user_dir = os.path.join(DATA_DIR, user_id)
    ensure_directory_exists(user_dir)
    return os.path.join(user_dir, f"{dataset_id}_metadata.json")

def get_data_path(user_id: str, dataset_id: str) -> str:
    """
    Get the path to the data file for a dataset.
    
    Args:
        user_id (str): User ID
        dataset_id (str): Dataset ID
        
    Returns:
        str: Path to data file
    """
    user_dir = os.path.join(DATA_DIR, user_id)
    ensure_directory_exists(user_dir)
    return os.path.join(user_dir, f"{dataset_id}.sav")

def _get_cache_key(user_id: str, dataset_id: str) -> str:
    """Generate a cache key for metadata."""
    return f"{user_id}_{dataset_id}"

def _is_cache_valid(cache_entry: Dict[str, Any]) -> bool:
    """Check if a cache entry is still valid."""
    return time.time() - cache_entry['timestamp'] < CACHE_EXPIRY

def _extract_csv_metadata(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Extract metadata from a CSV DataFrame.
    
    Args:
        df (pd.DataFrame): The DataFrame to extract metadata from
        
    Returns:
        Dict[str, Any]: Extracted metadata
    """
    print(f"[DEBUG] Extracting metadata from CSV DataFrame with shape: {df.shape}")
    
    metadata = {
        'columns': df.columns.tolist(),
        'column_labels': {},  # CSV files don't have column labels, use column names
        'value_labels': {},   # CSV files don't have value labels
        'variable_labels': {}, # CSV files don't have variable labels
        'variable_values': {},
        'variable_formats': {},
        'variable_measure': {},
        'variable_roles': {},
        'variable_alignment': {},
        'variable_width': {},
        'variable_decimals': {},
        'row_count': len(df),
        'column_count': len(df.columns),
        'data_types': {},
        'missing_values': {},
        'unique_values': {},
        'basic_stats': {},
        'file_type': 'csv'
    }
    
    # Extract column information
    for col in df.columns:
        col_series = df[col]
        
        # Data type
        metadata['data_types'][col] = str(col_series.dtype)
        
        # Missing values
        missing_count = col_series.isnull().sum()
        missing_pct = (missing_count / len(col_series)) * 100
        metadata['missing_values'][col] = {
            'count': int(missing_count),
            'percentage': float(missing_pct)
        }
        
        # Unique values (for categorical/nominal data)
        unique_vals = col_series.dropna().unique()
        if len(unique_vals) <= 50:  # Only store if reasonable number of unique values
            metadata['unique_values'][col] = {
                'count': len(unique_vals),
                'values': unique_vals.tolist() if len(unique_vals) <= 20 else unique_vals[:20].tolist()
            }
        
        # Basic statistics based on data type
        if pd.api.types.is_numeric_dtype(col_series):
            # Numeric columns
            numeric_stats = col_series.describe()
            metadata['basic_stats'][col] = {
                'mean': float(numeric_stats.get('mean', 0)),
                'std': float(numeric_stats.get('std', 0)),
                'min': float(numeric_stats.get('min', 0)),
                'max': float(numeric_stats.get('max', 0)),
                'median': float(col_series.median()),
                'q25': float(numeric_stats.get('25%', 0)),
                'q75': float(numeric_stats.get('75%', 0))
            }
        else:
            # Categorical/text columns
            value_counts = col_series.value_counts()
            metadata['basic_stats'][col] = {
                'mode': value_counts.index[0] if len(value_counts) > 0 else None,
                'mode_count': int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                'unique_count': len(value_counts)
            }
    
    # Try to identify conjoint structure in CSV data
    conjoint_info = _identify_conjoint_structure(df)
    if conjoint_info:
        metadata['conjoint_structure'] = conjoint_info
    
    print(f"[DEBUG] Extracted metadata for {len(df.columns)} columns")
    return metadata

def _identify_conjoint_structure(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """
    Identify if the CSV data has a conjoint structure.
    
    Args:
        df (pd.DataFrame): The DataFrame to analyze
        
    Returns:
        Optional[Dict[str, Any]]: Conjoint structure information if found
    """
    print("[DEBUG] Analyzing CSV data for conjoint structure...")
    
    columns = df.columns.tolist()
    conjoint_info = {
        'detected': False,
        'structure_type': None,
        'identified_columns': {}
    }
    
    # Look for wide-format conjoint data (Respondent_ID, Task_1, Task_2, ...)
    respondent_patterns = ['respondent', 'subject', 'participant', 'id']
    task_patterns = ['task', 'question', 'scenario', 'set']
    choice_patterns = ['choice', 'selected', 'preference', 'pick']
    
    # Check for respondent ID column
    respondent_col = None
    for col in columns:
        if any(pattern in col.lower() for pattern in respondent_patterns):
            respondent_col = col
            break
    
    # Check for task columns
    task_cols = []
    for col in columns:
        if any(pattern in col.lower() for pattern in task_patterns):
            task_cols.append(col)
    
    # Check for choice columns (if not task columns)
    choice_cols = []
    if not task_cols:
        for col in columns:
            if any(pattern in col.lower() for pattern in choice_patterns):
                choice_cols.append(col)
    
    # Determine structure type
    if respondent_col and task_cols:
        conjoint_info['detected'] = True
        conjoint_info['structure_type'] = 'wide_format'
        conjoint_info['identified_columns'] = {
            'respondent_id': respondent_col,
            'task_columns': task_cols,
            'num_respondents': df[respondent_col].nunique(),
            'num_tasks': len(task_cols)
        }
        print(f"[DEBUG] Detected wide-format conjoint structure: {len(task_cols)} tasks, {df[respondent_col].nunique()} respondents")
        
    elif respondent_col and choice_cols:
        conjoint_info['detected'] = True
        conjoint_info['structure_type'] = 'choice_format'
        conjoint_info['identified_columns'] = {
            'respondent_id': respondent_col,
            'choice_columns': choice_cols
        }
        print(f"[DEBUG] Detected choice-format structure")
        
    else:
        # Check for long-format conjoint data
        required_cols = ['respondent_id', 'task_id', 'alternative_id', 'choice']
        found_cols = [col for col in columns if col.lower() in [req.lower() for req in required_cols]]
        
        if len(found_cols) >= 3:  # At least 3 of the 4 required columns
            conjoint_info['detected'] = True
            conjoint_info['structure_type'] = 'long_format'
            conjoint_info['identified_columns'] = {
                'found_columns': found_cols,
                'missing_columns': [col for col in required_cols if col.lower() not in [fc.lower() for fc in found_cols]]
            }
            print(f"[DEBUG] Detected long-format conjoint structure with columns: {found_cols}")
    
    return conjoint_info if conjoint_info['detected'] else None

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
    Load metadata for a dataset, creating it if it doesn't exist.
    Now supports both SPSS (.sav) and CSV files.
    
    Args:
        user_id (str): User ID
        dataset_id (str): Dataset ID
        
    Returns:
        Dict[str, Any]: Dataset metadata
    """
    try:
        # Check cache first
        cache_key = _get_cache_key(user_id, dataset_id)
        if cache_key in _metadata_cache and _is_cache_valid(_metadata_cache[cache_key]):
            print(f"[DEBUG] Using cached metadata for {dataset_id}")
            return _metadata_cache[cache_key]['metadata']
        
        # Try to load from file
        metadata_path = get_metadata_path(user_id, dataset_id)
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
                print(f"[DEBUG] Loaded metadata from file: {metadata_path}")
                
                # Check if column_labels is empty and regenerate if needed
                if not metadata.get('column_labels') or len(metadata.get('column_labels', {})) == 0:
                    print(f"[DEBUG] Column labels are empty, regenerating metadata")
                    # Clear the cache entry to force regeneration
                    if cache_key in _metadata_cache:
                        del _metadata_cache[cache_key]
                    # This will trigger the regeneration logic below
                    raise FileNotFoundError("Regenerating metadata due to empty column_labels")
                    
        except FileNotFoundError:
            print(f"[DEBUG] Metadata file not found or needs regeneration, creating from data: {metadata_path}")
            
            # Get the actual file path to determine file type
            filepath = get_file_path(user_id, dataset_id)
            suffix = filepath.suffix.lower()
            
            if suffix == ".sav":
                # Handle SPSS files
                metadata = _load_spss_metadata(user_id, dataset_id)
            elif suffix in {".csv", ".xls", ".xlsx"}:
                # Handle CSV/Excel files
                metadata = _load_csv_metadata(user_id, dataset_id)
            else:
                raise ValueError(f"Unsupported file type: {suffix}")
            
            # Save the metadata
            save_metadata(user_id, dataset_id, metadata)
            
        # Cache the metadata
        _metadata_cache[cache_key] = {
            'metadata': metadata,
            'timestamp': time.time()
        }
        
        return metadata
        
    except Exception as e:
        print(f"[ERROR] Failed to load metadata: {e}")
        raise

def _load_spss_metadata(user_id: str, dataset_id: str) -> Dict[str, Any]:
    """Load metadata from SPSS file."""
    # Load SPSS data to create metadata
    data_path = get_data_path(user_id, dataset_id)
    
    # Check if data file exists in upload directory first
    upload_data_path = os.path.join(UPLOAD_DIR, user_id, f"{dataset_id}.sav")
    if os.path.exists(upload_data_path):
        print(f"[DEBUG] Found data file in upload directory: {upload_data_path}")
        data_path = upload_data_path
    elif not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Data file not found in either location:\n"
            f"1. {data_path}\n"
            f"2. {upload_data_path}\n"
            "Please ensure the data file is uploaded first."
        )
        
    # Read SPSS data using pyreadstat to get proper metadata
    print(f"[DEBUG] Reading SPSS file: {data_path}")
    df, meta = pyreadstat.read_sav(data_path)
    
    # Create metadata with all available information from pyreadstat
    # Convert column_labels from list to dictionary format
    column_labels_dict = {}
    if hasattr(meta, 'column_labels') and meta.column_labels:
        # If column_labels is a list, convert to dict
        if isinstance(meta.column_labels, list):
            for i, label in enumerate(meta.column_labels):
                if i < len(df.columns):
                    column_labels_dict[df.columns[i]] = label
        # If it's already a dict, use as is
        elif isinstance(meta.column_labels, dict):
            column_labels_dict = meta.column_labels
    
    metadata = {
        'columns': df.columns.tolist(),
        'column_labels': column_labels_dict,
        'value_labels': meta.variable_value_labels if hasattr(meta, 'variable_value_labels') else {},
        'variable_labels': meta.variable_labels if hasattr(meta, 'variable_labels') else {},
        'variable_values': meta.variable_values if hasattr(meta, 'variable_values') else {},
        'variable_formats': meta.variable_formats if hasattr(meta, 'variable_formats') else {},
        'variable_measure': meta.variable_measure if hasattr(meta, 'variable_measure') else {},
        'variable_roles': meta.variable_roles if hasattr(meta, 'variable_roles') else {},
        'variable_alignment': meta.variable_alignment if hasattr(meta, 'variable_alignment') else {},
        'variable_width': meta.variable_width if hasattr(meta, 'variable_width') else {},
        'variable_decimals': meta.variable_decimals if hasattr(meta, 'variable_decimals') else {},
        'row_count': len(df),
        'column_count': len(df.columns),
        'spss_attributes': meta.__dict__ if hasattr(meta, '__dict__') else {},  # Store all SPSS attributes
        'file_type': 'spss'
    }
    
    return metadata

def _load_csv_metadata(user_id: str, dataset_id: str) -> Dict[str, Any]:
    """Load metadata from CSV file."""
    # Get the file path
    filepath = get_file_path(user_id, dataset_id)
    
    if not filepath.exists():
        raise FileNotFoundError(f"CSV file not found: {filepath}")
    
    # Load the CSV data
    print(f"[DEBUG] Reading CSV file: {filepath}")
    df = pd.read_csv(str(filepath))
    
    # Extract metadata from the DataFrame
    metadata = _extract_csv_metadata(df)
    
    return metadata

def save_metadata(user_id: str, dataset_id: str, metadata: Dict[str, Any]) -> None:
    """
    Save metadata to a JSON file.
    
    Args:
        user_id (str): User ID
        dataset_id (str): Dataset ID
        metadata (Dict[str, Any]): Metadata to save
    """
    metadata_path = get_metadata_path(user_id, dataset_id)
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
    
    # Save metadata to JSON file
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"[DEBUG] Saved metadata to: {metadata_path}")

def clear_metadata_cache(user_id: Optional[str] = None, dataset_id: Optional[str] = None) -> None:
    """
    Clear metadata cache for specific user/dataset or all cache.
    
    Args:
        user_id (Optional[str]): User ID to clear cache for
        dataset_id (Optional[str]): Dataset ID to clear cache for
    """
    if user_id and dataset_id:
        # Clear specific cache entry
        cache_key = _get_cache_key(user_id, dataset_id)
        if cache_key in _metadata_cache:
            del _metadata_cache[cache_key]
            print(f"[DEBUG] Cleared cache for {user_id}/{dataset_id}")
    else:
        # Clear all cache
        _metadata_cache.clear()
        print("[DEBUG] Cleared all metadata cache")

def force_regenerate_metadata(user_id: str, dataset_id: str) -> Dict[str, Any]:
    """
    Force regeneration of metadata by clearing cache and reloading.
    
    Args:
        user_id (str): User ID
        dataset_id (str): Dataset ID
        
    Returns:
        Dict[str, Any]: Regenerated metadata
    """
    print(f"[DEBUG] Force regenerating metadata for {user_id}/{dataset_id}")
    
    # Clear cache
    clear_metadata_cache(user_id, dataset_id)
    
    # Reload metadata (this will regenerate it)
    return load_metadata(user_id, dataset_id)

def test_metadata_loading(user_id: str, dataset_id: str) -> Dict[str, Any]:
    """
    Test function to load metadata and print debug information.
    
    Args:
        user_id (str): User ID
        dataset_id (str): Dataset ID
        
    Returns:
        Dict[str, Any]: Loaded metadata
    """
    print(f"[DEBUG] Testing metadata loading for {user_id}/{dataset_id}")
    
    try:
        metadata = load_metadata(user_id, dataset_id)
        
        # Print metadata summary
        print(f"[DEBUG] Metadata loaded successfully:")
        print(f"  - File type: {metadata.get('file_type', 'unknown')}")
        print(f"  - Columns: {len(metadata.get('columns', []))}")
        print(f"  - Rows: {metadata.get('row_count', 0)}")
        print(f"  - Column labels: {len(metadata.get('column_labels', {}))}")
        print(f"  - Value labels: {len(metadata.get('value_labels', {}))}")
        
        # Check for conjoint structure
        if 'conjoint_structure' in metadata:
            conjoint = metadata['conjoint_structure']
            print(f"  - Conjoint structure detected: {conjoint['detected']}")
            if conjoint['detected']:
                print(f"  - Structure type: {conjoint['structure_type']}")
                print(f"  - Identified columns: {conjoint['identified_columns']}")
        
        return metadata
        
    except Exception as e:
        print(f"[ERROR] Failed to regenerate metadata: {e}")
        raise 