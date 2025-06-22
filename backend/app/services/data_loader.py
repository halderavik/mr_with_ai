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
                    print(f"[DEBUG] Column labels are empty, regenerating metadata from SPSS data")
                    # Clear the cache entry to force regeneration
                    if cache_key in _metadata_cache:
                        del _metadata_cache[cache_key]
                    # Instead of deleting the file, just regenerate and overwrite
                    print(f"[DEBUG] Regenerating metadata without deleting file")
                    # This will trigger the regeneration logic below
                    raise FileNotFoundError("Regenerating metadata due to empty column_labels")
                    
        except FileNotFoundError:
            print(f"[DEBUG] Metadata file not found or needs regeneration, creating from SPSS data: {metadata_path}")
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
                'spss_attributes': meta.__dict__ if hasattr(meta, '__dict__') else {}  # Store all SPSS attributes
            }
            
            # Print all metadata fields for debugging
            print("\n[DEBUG] ===== Complete Metadata Information =====")
            print(f"Number of columns: {len(metadata['columns'])}")
            print("\nColumn Names:")
            for col in metadata['columns']:
                print(f"- {col}")
            
            print("\nColumn Labels:")
            for col, label in metadata['column_labels'].items():
                print(f"- {col}: {label}")
            
            print("\nValue Labels:")
            for col, labels in metadata['value_labels'].items():
                print(f"- {col}:")
                for val, label in labels.items():
                    print(f"  {val}: {label}")
            
            print("\nVariable Labels:")
            for var, label in metadata['variable_labels'].items():
                print(f"- {var}: {label}")
            
            print("\nVariable Values:")
            for var, values in metadata['variable_values'].items():
                print(f"- {var}: {values}")
            
            print("\nVariable Formats:")
            for var, fmt in metadata['variable_formats'].items():
                print(f"- {var}: {fmt}")
            
            print("\nVariable Measures:")
            for var, measure in metadata['variable_measure'].items():
                print(f"- {var}: {measure}")
            
            print("\nVariable Roles:")
            for var, role in metadata['variable_roles'].items():
                print(f"- {var}: {role}")
            
            print("\nSPSS Attributes:")
            for key, value in metadata['spss_attributes'].items():
                print(f"- {key}: {value}")
            
            print("=============================================\n")
            
            # Save metadata to file
            save_metadata(user_id, dataset_id, metadata)
            
            # Update cache
            _metadata_cache[cache_key] = {
                'metadata': metadata,
                'timestamp': time.time()
            }
            
            print(f"[DEBUG] Created and cached metadata for {dataset_id}")
            return metadata
            
        # Update cache
        _metadata_cache[cache_key] = {
            'metadata': metadata,
            'timestamp': time.time()
        }
        
        return metadata
        
    except Exception as e:
        print(f"[ERROR] Failed to load metadata: {str(e)}")
        raise

def save_metadata(user_id: str, dataset_id: str, metadata: Dict[str, Any]) -> None:
    """
    Save metadata for a dataset.
    
    Args:
        user_id (str): User ID
        dataset_id (str): Dataset ID
        metadata (Dict[str, Any]): Metadata to save
    """
    try:
        metadata_path = get_metadata_path(user_id, dataset_id)
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        print(f"[DEBUG] Saved metadata to: {metadata_path}")
    except Exception as e:
        print(f"[ERROR] Failed to save metadata: {str(e)}")
        raise

def clear_metadata_cache(user_id: Optional[str] = None, dataset_id: Optional[str] = None) -> None:
    """
    Clear metadata cache for specific user/dataset or all cache.
    
    Args:
        user_id (Optional[str]): User ID to clear cache for
        dataset_id (Optional[str]): Dataset ID to clear cache for
    """
    global _metadata_cache
    
    if user_id and dataset_id:
        # Clear specific cache entry
        cache_key = _get_cache_key(user_id, dataset_id)
        if cache_key in _metadata_cache:
            del _metadata_cache[cache_key]
            print(f"[DEBUG] Cleared metadata cache for {user_id}/{dataset_id}")
    else:
        # Clear all cache
        _metadata_cache.clear()
        print("[DEBUG] Cleared all metadata cache")

def force_regenerate_metadata(user_id: str, dataset_id: str) -> Dict[str, Any]:
    """
    Force regeneration of metadata by clearing cache and regenerating from SPSS data.
    
    Args:
        user_id (str): User ID
        dataset_id (str): Dataset ID
        
    Returns:
        Dict[str, Any]: Regenerated metadata
    """
    print(f"[DEBUG] Force regenerating metadata for {user_id}/{dataset_id}")
    
    # Clear cache
    clear_metadata_cache(user_id, dataset_id)
    
    # Remove metadata file if it exists
    metadata_path = get_metadata_path(user_id, dataset_id)
    try:
        if os.path.exists(metadata_path):
            os.remove(metadata_path)
            print(f"[DEBUG] Removed existing metadata file: {metadata_path}")
    except PermissionError:
        print(f"[DEBUG] Could not remove metadata file (in use), will overwrite: {metadata_path}")
    
    # Load metadata (this will regenerate it)
    return load_metadata(user_id, dataset_id)

def test_metadata_loading(user_id: str, dataset_id: str) -> Dict[str, Any]:
    """
    Test function to load and display complete metadata structure.
    
    Args:
        user_id (str): User ID
        dataset_id (str): Dataset ID
        
    Returns:
        Dict[str, Any]: Complete metadata
    """
    print(f"[DEBUG] Testing metadata loading for {user_id}/{dataset_id}")
    
    # Get the data path
    data_path = get_data_path(user_id, dataset_id)
    upload_data_path = os.path.join(UPLOAD_DIR, user_id, f"{dataset_id}.sav")
    
    if os.path.exists(upload_data_path):
        print(f"[DEBUG] Found data file in upload directory: {upload_data_path}")
        data_path = upload_data_path
    elif not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    # Read SPSS data using pyreadstat
    print(f"[DEBUG] Reading SPSS file: {data_path}")
    df, meta = pyreadstat.read_sav(data_path)
    
    print(f"\n[DEBUG] ===== Raw pyreadstat Metadata =====")
    print(f"DataFrame shape: {df.shape}")
    print(f"DataFrame columns: {list(df.columns)}")
    
    print(f"\nMeta object attributes:")
    for attr in dir(meta):
        if not attr.startswith('_'):
            value = getattr(meta, attr)
            print(f"- {attr}: {type(value)} = {value}")
    
    print(f"\n[DEBUG] ===== Processed Metadata =====")
    
    # Convert column_labels from list to dictionary format
    column_labels_dict = {}
    if hasattr(meta, 'column_labels') and meta.column_labels:
        print(f"Raw column_labels type: {type(meta.column_labels)}")
        print(f"Raw column_labels: {meta.column_labels}")
        
        # If column_labels is a list, convert to dict
        if isinstance(meta.column_labels, list):
            for i, label in enumerate(meta.column_labels):
                if i < len(df.columns):
                    column_labels_dict[df.columns[i]] = label
                    print(f"  {df.columns[i]} -> {label}")
        # If it's already a dict, use as is
        elif isinstance(meta.column_labels, dict):
            column_labels_dict = meta.column_labels
            print(f"  Using as dictionary: {column_labels_dict}")
    
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
        'spss_attributes': meta.__dict__ if hasattr(meta, '__dict__') else {}
    }
    
    print(f"\nFinal metadata structure:")
    for key, value in metadata.items():
        if key == 'columns':
            print(f"- {key}: {len(value)} items - {value[:5]}...")
        elif key == 'column_labels':
            print(f"- {key}: {len(value)} items")
            for col, label in list(value.items())[:3]:
                print(f"  {col}: {label}")
        elif key == 'value_labels':
            print(f"- {key}: {len(value)} items")
            for col, labels in list(value.items())[:3]:
                print(f"  {col}: {len(labels)} value labels")
        else:
            print(f"- {key}: {len(value) if isinstance(value, dict) else value}")
    
    return metadata

# Logic for loading and parsing data files (CSV, Excel, SPSS, etc.) 