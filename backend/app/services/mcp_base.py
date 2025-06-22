# backend/app/services/mcp_base.py

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import pandas as pd
import json

class MCPBase(ABC):
    """
    Abstract Base Class for all MCP (Micro‐service) analyses.
    Each subclass must implement:
      - `run()` which takes a DataFrame and optional params and returns a dict with:
          - tables (dict of table‐name: List[Dict])
          - charts (dict of chart‐name: str (either base64 PNG or static URL))
          - insights (string)
    """
    @abstractmethod
    def run(
        self,
        df: pd.DataFrame,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        :param df: pandas DataFrame containing the uploaded data (filtered already if needed)
        :param params: dictionary of parameters (e.g. filter, column mappings, etc.)
        :return: {
            "tables": { "table_name": [ {col: val, …}, … ] },
            "charts": { "chart_name": "data:image/png;base64,…" or "/static/…png" },
            "insights": "Human readable interpretation…"
        }
        """
        pass

# Base logic for market research analysis (MCP) 

class MCPBase:
    """
    Base class for Market Research Control Protocol (MCP) servers.
    Each MCP implements a specific type of market research analysis.
    """
    
    def __init__(self):
        self.name = "base"  # Override in subclasses
        self.required_columns = []  # Override in subclasses
        self.description = "Base MCP class"  # Override in subclasses
        
    def run(self, data: pd.DataFrame, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process the data using the MCP's analysis methodology.
        
        Args:
            data (pd.DataFrame): Input data
            params (Dict[str, Any]): Parameters including:
                - column_map: Optional mapping of question types to column names
                - chat_model: LLM model for column mapping
                - question: Optional follow-up question about the analysis
                - conversation_context: Optional previous conversation context
                - auto_map: Boolean indicating whether to handle mapping automatically
                - filters: Dict of filters to apply (e.g. {"gender": "male"})
                - segmentation: Optional segmentation parameter
                
        Returns:
            Dict[str, Any]: Analysis results including:
                - reply: Text response for the user
                - visualizations: Dict of charts and tables
                - insights: Business insights from the analysis
                - context: Updated conversation context
        """
        raise NotImplementedError("Subclasses must implement run()")
        
    def _apply_filters(self, data: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
        """
        Apply filters to the dataset.
        
        Args:
            data (pd.DataFrame): Input data
            filters (Dict[str, Any]): Dictionary of column:value filters
            
        Returns:
            pd.DataFrame: Filtered dataset
        """
        if not filters:
            return data
            
        filtered_data = data.copy()
        for column, value in filters.items():
            if column in filtered_data.columns:
                filtered_data = filtered_data[filtered_data[column] == value]
                print(f"[DEBUG] Applied filter: {column} = {value}")
                
        return filtered_data
        
    def _handle_segmentation(self, data: pd.DataFrame, segmentation: Optional[str]) -> Dict[str, pd.DataFrame]:
        """
        Handle segmentation of the dataset.
        
        Args:
            data (pd.DataFrame): Input data
            segmentation (Optional[str]): Column to segment by
            
        Returns:
            Dict[str, pd.DataFrame]: Dictionary of segment_name:filtered_data
        """
        if not segmentation or segmentation not in data.columns:
            return {"all": data}
            
        segments = {}
        for value in data[segmentation].unique():
            segment_data = data[data[segmentation] == value]
            segments[str(value)] = segment_data
            
        return segments
        
    def _validate_required_columns(self, data: pd.DataFrame, column_map: Dict[str, str]) -> bool:
        """
        Validate that all required columns are present in the data.
        
        Args:
            data (pd.DataFrame): Input data
            column_map (Dict[str, str]): Mapping of required columns to actual column names
            
        Returns:
            bool: True if all required columns are present
        """
        for col in self.required_columns:
            if col not in column_map or column_map[col] not in data.columns:
                return False
        return True
        
    def _generate_insights(self, results: Dict[str, Any], chat_model: Any) -> str:
        """
        Generate business insights from the analysis results.
        
        Args:
            results (Dict[str, Any]): Analysis results
            chat_model: LLM model for generating insights
            
        Returns:
            str: Generated insights
        """
        if not chat_model:
            return ""
            
        prompt = (
            "Based on the following analysis results, provide clear, business-focused insights:\n"
            f"{json.dumps(results, indent=2)}\n\n"
            "Focus on actionable recommendations and key findings."
        )
        
        return chat_model.generate_reply(prompt) 