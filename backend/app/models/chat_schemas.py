# backend/app/models/chat_schemas.py

from pydantic import BaseModel
from typing import Optional, Dict, Any, List

class ChatRequest(BaseModel):
    dataset_id: str
    message: str
    filter_column: Optional[str] = None
    filter_value: Optional[Any] = None
    conversation_context: Optional[Dict[str, Any]] = None


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    reply: str                                        # The agent's textual answer
    analysis_tables: Optional[Dict[str, Any]] = None  # Key: table_name, Value: table data (list of rows)
    analysis_charts: Optional[Dict[str, Any]] = None
    insights: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    visualizations: Optional[Dict[str, Any]] = None  # Add this field for MCP visualizations

# Pydantic models for chat-related data 