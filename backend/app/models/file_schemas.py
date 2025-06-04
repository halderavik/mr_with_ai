# file_schemas.py
# Pydantic models for file upload, preview, and metadata 

# backend/app/models/file_schemas.py

from typing import List, Dict, Any
from pydantic import BaseModel

class UploadResponse(BaseModel):
    dataset_id: str
    filename: str
    preview_rows: List[dict]            # list of row‐dicts for preview
    metadata: Dict[str, Any] = None     # SPSS metadata JSON, or null for CSV/Excel


class PreviewRequest(BaseModel):
    dataset_id: str


class PreviewResponse(BaseModel):
    filename: str
    columns: List[str]
    preview_rows: List[dict]
    metadata: Dict[str, Any] = None     # also include metadata on preview if SPSS
