# main.py
# Entry point for FastAPI app (add routes, CORS, etc. here) 
# backend/app/main.py

import uvicorn
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from starlette.staticfiles import StaticFiles
from typing import List, Dict, Any
import logging
import traceback
import json
import math
import numpy as np

from .config import UPLOAD_DIR, ALLOWED_ORIGINS, MAX_FILE_SIZE
from .models.file_schemas import UploadResponse, PreviewRequest, PreviewResponse
from .models.chat_schemas import ChatRequest, ChatResponse
from .models.analysis_schemas import AnalysisRequest, AnalysisResponse

from .services.file_handler import save_uploaded_file, get_file_path
from .services.data_loader import load_dataset
from .agent_controller import AgentController

import pandas as pd
import pyreadstat
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle special float values and numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            if math.isnan(obj) or math.isinf(obj):
                return None
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

def safe_json_serialize(obj: Any) -> Any:
    """
    Safely serialize an object to JSON-compatible format.
    Handles numpy types, NaN, and infinity values.
    """
    if isinstance(obj, dict):
        return {k: safe_json_serialize(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [safe_json_serialize(item) for item in obj]
    elif isinstance(obj, (np.integer, np.floating)):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return float(obj) if isinstance(obj, np.floating) else int(obj)
    elif isinstance(obj, np.ndarray):
        return safe_json_serialize(obj.tolist())
    return obj

# Initialize FastAPI app with metadata
app = FastAPI(
    title="Market Pro API",
    description="AI-Powered Market Research Analysis Platform",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Add CORS middleware with proper configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS.split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition"],  # For file downloads
    max_age=3600,  # Cache preflight requests for 1 hour
)

# Mount static files
app.mount("/static", StaticFiles(directory=str(UPLOAD_DIR.parent / "static")), name="static")

agent_controller = AgentController()

def get_current_user():
    return "user_demo"  # your real auth should replace this stub

@app.get("/")
async def root():
    """
    Root endpoint that provides basic API information and available endpoints.
    """
    return {
        "name": "Market Pro API",
        "version": "1.0.0",
        "description": "AI-Powered Market Research Analysis Platform",
        "endpoints": {
            "documentation": "/docs",
            "upload": "/api/upload",
            "preview": "/api/preview",
            "chat": "/api/chat",
            "analyze": "/api/analyze"
        }
    }

@app.get("/health")
async def health_check():
    """
    Health check endpoint for monitoring.
    """
    return {
        "status": "healthy",
        "version": "1.0.0"
    }

def extract_spss_metadata(meta: pyreadstat.metadata_container) -> dict:
    """
    Return a JSON-serializable dict of SPSS metadata.
    Grabs named fields if present, plus a full_meta_dict of everything.
    """
    try:
        # 1) Copy raw __dict__ and sanitize it
        raw_meta = meta.__dict__.copy()
        safe_full = jsonable_encoder(raw_meta)

        # 2) Build result from only the attributes that exist
        named = {}
        for attr in (
            "column_names",
            "column_labels",
            "variable_value_labels",
            "value_labels",
            "variable_measure",
            "missing_ranges",
            "missing_user_ranges",
            "file_label",
            "file_encoding",
            "number_rows",
            "number_columns",
        ):
            if hasattr(meta, attr):
                named[attr] = jsonable_encoder(getattr(meta, attr))

        # 3) Merge
        named["full_meta_dict"] = safe_full
        return named
    except Exception as e:
        logger.error(f"Error in extract_spss_metadata: {str(e)}")
        # Fall back to just the full metadata dict
        return {"full_meta_dict": jsonable_encoder(meta.__dict__.copy())}

@app.post("/api/upload")
async def upload_data(
    file: UploadFile = File(...),
    current_user: str = Depends(get_current_user)
):
    """
    Upload a data file (SPSS, CSV, Excel) and return preview + metadata.
    """
    try:
        logger.info(f"Received file upload request: {file.filename}")
        
        # Validate file size
        file_size = 0
        chunk_size = 1024 * 1024  # 1MB chunks
        while chunk := await file.read(chunk_size):
            file_size += len(chunk)
            if file_size > MAX_FILE_SIZE:
                raise HTTPException(
                    status_code=413,
                    detail=f"File too large. Maximum size is {MAX_FILE_SIZE/1024/1024}MB"
                )
        await file.seek(0)  # Reset file pointer
        
        # Validate file extension
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in {".sav", ".csv", ".xlsx", ".xls"}:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type. Allowed types: .sav, .csv, .xlsx, .xls"
            )

        # Save the file and get dataset_id
        logger.info("Saving uploaded file...")
        result = save_uploaded_file(current_user, file)
        dataset_id = result["dataset_id"]
        
        # Load the data for preview
        logger.info("Loading dataset for preview...")
        df = load_dataset(current_user, dataset_id)
        
        # Get preview rows and sanitize them
        preview_rows = df.head(10).fillna("").to_dict(orient="records")
        safe_preview = jsonable_encoder(preview_rows)
        
        # For SPSS files, get metadata
        metadata_json = None
        if file_ext == ".sav":
            try:
                logger.info("Extracting SPSS metadata...")
                _, meta = pyreadstat.read_sav(str(result["filepath"]))
                metadata_json = extract_spss_metadata(meta)
                logger.info("SPSS metadata extracted successfully")
            except Exception as e:
                logger.error(f"Error reading SPSS metadata: {str(e)}")
                logger.error(traceback.format_exc())
                # Don't raise an exception, just log the error and continue
                metadata_json = {"error": str(e)}
        
        # Return via JSONResponse to bypass Pydantic model serialization
        logger.info("Upload completed successfully")
        return JSONResponse(
            content={
                "dataset_id": dataset_id,
                "filename": file.filename,
                "preview_rows": safe_preview,
                "metadata": metadata_json
            }
        )
    except HTTPException as he:
        logger.error(f"HTTP Exception during upload: {str(he)}")
        raise he
    except Exception as e:
        logger.error(f"Unexpected error during upload: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/api/preview")
async def preview_data(
    req: PreviewRequest,
    current_user: str = Depends(get_current_user)
):
    """
    Get preview of dataset with metadata.
    """
    try:
        # Load the dataset
        df = load_dataset(current_user, req.dataset_id)
        
        # Get preview rows and sanitize them
        preview_rows = df.head(10).fillna("").to_dict(orient="records")
        safe_preview = jsonable_encoder(preview_rows)
        
        # Get metadata if available
        metadata_json = None
        file_path = get_file_path(current_user, req.dataset_id)
        if file_path.suffix.lower() == ".sav":
            try:
                _, meta = pyreadstat.read_sav(str(file_path), metadataonly=True)
                metadata_json = extract_spss_metadata(meta)
            except Exception as e:
                logger.warning(f"Could not read SPSS metadata: {str(e)}")
                metadata_json = {"error": str(e)}
        
        # Return via JSONResponse to bypass Pydantic model serialization
        return JSONResponse(
            content={
                "filename": req.dataset_id,
                "columns": list(df.columns),
                "preview_rows": safe_preview,
                "metadata": metadata_json
            }
        )
    except Exception as e:
        logger.error(f"Error in preview: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(
    chat_req: ChatRequest,
    user_id: str = Depends(get_current_user)
):
    return agent_controller.handle_chat(user_id, chat_req)

@app.post("/api/analyze", response_model=AnalysisResponse)
async def analyze_endpoint(
    analysis_req: AnalysisRequest,
    user_id: str = Depends(get_current_user)
):
    df = load_dataset(user_id, analysis_req.dataset_id)
    analysis_key = analysis_req.analysis_type

    if analysis_key not in agent_controller.mcp_mapping:
        return AnalysisResponse(
            success=False,
            message=f"Analysis '{analysis_key}' not implemented."
        )

    try:
        mcp = agent_controller.mcp_mapping[analysis_key]
        result = mcp.run(df, analysis_req.params or {})
    except Exception as e:
        return AnalysisResponse(
            success=False,
            message=str(e)
        )

    return AnalysisResponse(
        success=True,
        tables=result.get("tables", {}),
        charts=result.get("charts", {}),
        insights=result.get("insights", ""),
        message="Analysis completed."
    )

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
