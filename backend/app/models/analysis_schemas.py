# backend/app/models/analysis_schemas.py

from pydantic import BaseModel
from typing import Optional, Dict, Any

class AnalysisRequest(BaseModel):
    dataset_id: str
    analysis_type: str                 # e.g., "van_westendorp"
    params: Optional[Dict[str, Any]]   # e.g. {"filter_column": "Gender", "filter_value": "Male"}


class AnalysisResponse(BaseModel):
    success: bool
    tables: Optional[Dict[str, Any]]
    charts: Optional[Dict[str, str]]   # e.g. {"vw_curve": "/static/charts/vw_curve_abc.png"}
    insights: Optional[str]
    message: Optional[str]

# Pydantic models for analysis-related data 