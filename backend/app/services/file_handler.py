# backend/app/services/file_handler.py

import uuid
import shutil
import logging
from pathlib import Path
from fastapi import UploadFile, HTTPException
from ..config import UPLOAD_DIR, MAX_FILE_SIZE, ALLOWED_EXTENSIONS

logger = logging.getLogger(__name__)

def save_uploaded_file(user_id: str, file: UploadFile) -> dict:
    """
    Save an uploaded file under UPLOAD_DIR/<user_id>/
    Return a dict with dataset_id and filepath.
    Dataset IDs are only created during file upload and must be unique.
    """
    try:
        suffix = Path(file.filename).suffix.lower()
        if suffix not in ALLOWED_EXTENSIONS:
            raise HTTPException(status_code=400, detail=f"Unsupported file type. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}")

        # Generate a unique dataset_id (UUID4) only during file upload:
        dataset_id = str(uuid.uuid4())
        user_folder = UPLOAD_DIR / user_id
        user_folder.mkdir(parents=True, exist_ok=True)

        # Check if dataset_id already exists (should never happen with UUID4, but just in case)
        dest_path = user_folder / f"{dataset_id}{suffix}"
        if dest_path.exists():
            raise HTTPException(status_code=500, detail="Dataset ID collision detected. Please try uploading again.")

        logger.info(f"Saving file to: {dest_path}")

        # Read & write to disk:
        contents = file.file.read()
        if len(contents) > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail=f"File too large. Maximum size is {MAX_FILE_SIZE/1024/1024}MB")
        
        with open(dest_path, "wb") as f:
            f.write(contents)

        logger.info(f"File saved successfully with dataset_id: {dataset_id}")
        return {"dataset_id": dataset_id, "filepath": str(dest_path)}
    except HTTPException as he:
        logger.error(f"HTTP Exception in save_uploaded_file: {str(he)}")
        raise he
    except Exception as e:
        logger.error(f"Error saving uploaded file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error saving file: {str(e)}")

def get_file_path(user_id: str, dataset_id: str) -> Path:
    """
    Find the file on disk by scanning user folder for dataset_id.*
    This function only retrieves existing files and does not create new dataset IDs.
    """
    try:
        user_folder = UPLOAD_DIR / user_id
        if not user_folder.exists():
            raise HTTPException(status_code=404, detail="User folder not found.")

        # Find any file that starts with the dataset_id:
        for filepath in user_folder.iterdir():
            if filepath.name.startswith(dataset_id):
                return filepath
        raise HTTPException(status_code=404, detail="Dataset not found.")
    except HTTPException as he:
        logger.error(f"HTTP Exception in get_file_path: {str(he)}")
        raise he
    except Exception as e:
        logger.error(f"Error finding file path: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error finding file: {str(e)}")

# Logic for handling file uploads, saving, and validation 