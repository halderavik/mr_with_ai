# main.py
# Entry point for FastAPI app (add routes, CORS, etc. here) 
# backend/app/main.py

import uvicorn
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
from starlette.staticfiles import StaticFiles
from typing import List, Dict, Any

from .config import UPLOAD_DIR
from .models.file_schemas import UploadResponse, PreviewRequest, PreviewResponse
from .models.chat_schemas import ChatRequest, ChatResponse
from .models.analysis_schemas import AnalysisRequest, AnalysisResponse

from .services.file_handler import save_uploaded_file, get_file_path
from .services.data_loader import load_dataset
from .agent_controller import AgentController

import pandas as pd
import pyreadstat
from pathlib import Path

app = FastAPI(title="MarketPro Backend")

# Mount /static so that chart PNGs (and future assets) are served:
app.mount("/static", StaticFiles(directory=str(UPLOAD_DIR.parent / "static")), name="static")

agent_controller = AgentController()

def get_current_user():
    return "user_demo"  # your real auth should replace this stub

@app.post("/api/upload", response_model=UploadResponse)
async def upload_data(
    file: UploadFile = File(...),
    user_id: str = Depends(get_current_user)
):
    """
    Upload a .sav, .csv, or .xlsx file.
    If .sav: read via pyreadstat to capture ALL metadata.
    Return: dataset_id, filename, preview_rows (first 10), AND metadata.
    """
    # 1. Save the raw file on disk under UPLOAD_DIR/<user_id>/
    saved = save_uploaded_file(user_id, file)
    dataset_id = saved["dataset_id"]
    filepath = Path(saved["filepath"])
    suffix = filepath.suffix.lower()

    preview_rows = []
    metadata_json: Dict[str, Any] = None

    # 2. If SPSS (.sav), capture metadata + preview via pyreadstat
    if suffix == ".sav":
        try:
            df, meta = pyreadstat.read_sav(str(filepath))  # load both data & metadata
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error reading SPSS file: {e}")

        # 2a. Build preview (first 10 rows)
        preview_rows = df.head(10).fillna("").to_dict(orient="records")

        # 2b. Convert **all** metadata fields to JSON
        # We pull out public attributes of meta.__dict__ that are JSON‐serializable
        # Typically: column_names, column_labels, variable_value_labels, value_labels, variable_measure, formats, missing_ranges, etc.
        # So we do a safe __dict__ grab.
        raw_meta_dict = meta.__dict__.copy()

        # Some attributes inside __dict__ might still be not JSON‐serializable (e.g. numpy arrays),
        # so let's convert them where needed. But in practice pyreadstat metadata attributes are simple Python dicts/lists.
        # We'll trust that meta.__dict__ is JSON serializable.

        # Build a “metadata_json” that includes these items plus a few named convenience fields:
        metadata_json = {
            "column_names": meta.column_names,
            "column_labels": meta.column_labels,
            "variable_value_labels": meta.variable_value_labels,
            "value_labels": meta.value_labels,
            "variable_measure": meta.variable_measure,
            "formats": meta.formats,
            "number_rows": meta.number_rows,
            "file_label": meta.file_label,
            "file_encoding": meta.file_encoding,
            # If you want literally everything: just include raw_meta_dict verbatim:
            "full_meta_dict": raw_meta_dict
        }

    else:
        # 3. If CSV/XLSX: load via pandas, preview only; metadata remains None
        try:
            if suffix == ".csv":
                df = pd.read_csv(str(filepath))
            else:
                df = pd.read_excel(str(filepath))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error reading file: {e}")

        preview_rows = df.head(10).fillna("").to_dict(orient="records")
        metadata_json = None

    return UploadResponse(
        dataset_id=dataset_id,
        filename=file.filename,
        preview_rows=preview_rows,
        metadata=metadata_json
    )


@app.post("/api/preview", response_model=PreviewResponse)
async def get_preview(
    req: PreviewRequest,
    user_id: str = Depends(get_current_user)
):
    """
    After upload, if user wants to re‐preview the data, return
    first 10 rows and (if SPSS) the stored metadata. 
    """
    df = load_dataset(user_id, req.dataset_id)
    preview_rows = df.head(10).fillna("").to_dict(orient="records")

    # Attempt to re‐extract metadata if needed:
    # If it was a SPSS upload, load metadata again
    from .services.file_handler import get_file_path
    from pathlib import Path
    file_path = Path(get_file_path(user_id, req.dataset_id))
    suffix = file_path.suffix.lower()
    metadata_json = None

    if suffix == ".sav":
        # Read JUST the metadata (no need to re‐load df again, but pyreadstat can't read metadata‐only out of the box)
        # So we do a quick read for meta:
        _, meta = pyreadstat.read_sav(str(file_path), metadataonly=True)
        raw_meta_dict = meta.__dict__.copy()
        metadata_json = {
            "column_names": meta.column_names,
            "column_labels": meta.column_labels,
            "variable_value_labels": meta.variable_value_labels,
            "value_labels": meta.value_labels,
            "variable_measure": meta.variable_measure,
            "formats": meta.formats,
            "number_rows": meta.number_rows,
            "file_label": meta.file_label,
            "file_encoding": meta.file_encoding,
            "full_meta_dict": raw_meta_dict
        }

    return PreviewResponse(
        filename=req.dataset_id,
        columns=list(df.columns),
        preview_rows=preview_rows,
        metadata=metadata_json
    )


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
