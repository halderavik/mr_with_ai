# backend/app/models/chat_schemas.py

from pydantic import BaseModel
from typing import Optional, Dict, Any, List

class ChatRequest(BaseModel):
    dataset_id: str
    message: str
    filter_column: Optional[str] = None
    filter_value: Optional[Any] = None


class ChatResponse(BaseModel):
    reply: str                                        # The agent’s textual answer
    analysis_tables: Optional[Dict[str, Any]] = None  # Key: table_name, Value: table data (list of rows)
    chart_paths: Optional[Dict[str, str]] = None      # Key: chart_name, Value: file path or URL
    insights: Optional[str] = None

# Pydantic models for chat-related data 