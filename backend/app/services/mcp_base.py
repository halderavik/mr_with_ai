# backend/app/services/mcp_base.py

from abc import ABC, abstractmethod
from typing import Dict, Any
import pandas as pd

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