# backend/app/services/file_handler.py

import uuid
import shutil
from pathlib import Path
from fastapi import UploadFile, HTTPException
from ..config import UPLOAD_DIR, MAX_FILE_SIZE, ALLOWED_EXTENSIONS

def save_uploaded_file(user_id: str, file: UploadFile) -> dict:
    """
    Save an uploaded file under UPLOAD_DIR/<user_id>/
    Return a dict with dataset_id and filepath.
    """
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported file type.")

    # Generate a unique dataset_id (UUID4):
    dataset_id = str(uuid.uuid4())
    user_folder = UPLOAD_DIR / user_id
    user_folder.mkdir(parents=True, exist_ok=True)

    # Destination path:
    dest_path = user_folder / f"{dataset_id}{suffix}"

    # Read & write to disk:
    contents = file.file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large.")
    with open(dest_path, "wb") as f:
        f.write(contents)

    return {"dataset_id": dataset_id, "filepath": str(dest_path)}

def get_file_path(user_id: str, dataset_id: str) -> Path:
    """
    Find the file on disk by scanning user folder for dataset_id.*
    """
    user_folder = UPLOAD_DIR / user_id
    if not user_folder.exists():
        raise HTTPException(status_code=404, detail="User not found.")

    # Find any file that starts with the dataset_id:
    for filepath in user_folder.iterdir():
        if filepath.name.startswith(dataset_id):
            return filepath
    raise HTTPException(status_code=404, detail="Dataset not found.")

# Logic for handling file uploads, saving, and validation 