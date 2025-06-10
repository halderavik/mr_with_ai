# backend/app/agent_controller.py

import os
from typing import Dict, Any
import importlib
import inspect
from .config import DEEPSEEK_API_KEY, DEEPSEEK_MODEL
from .models.chat_schemas import ChatRequest, ChatResponse
from .services.data_loader import load_dataset, load_metadata
from .services.mcp_base import MCPBase
import pandas as pd

# === STEP 1: Real DeepSeek ChatModel using OpenAI SDK ===
try:
    from openai import OpenAI
except ImportError:
    raise ImportError("Please install openai: pip install openai")

class ChatModel:
    def __init__(self, api_key: str, model: str = "deepseek-chat"):
        self.api_key = api_key
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    def generate_reply(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant for market research analysis."},
                {"role": "user", "content": prompt},
            ],
            stream=False
        )
        return response.choices[0].message.content

class AgentController:
    """
    Core orchestrator that:
      1. Receives a ChatRequest
      2. Uses Deepseek to "understand" which analysis is requested
      3. Delegates to the appropriate MCP server (e.g. VanWestendorpMCP)
      4. Returns ChatResponse containing:
         - textual reply
         - any tables
         - any charts
         - human insights
    """

    def __init__(self):
        # Initialize DeepSeek chat model
        self.chat_model = ChatModel(api_key=DEEPSEEK_API_KEY, model=DEEPSEEK_MODEL)
        
        # Dynamically discover all MCP classes in app/mcp/
        self.mcp_mapping: Dict[str, MCPBase] = {}
        self._discover_mcp_servers()

    def _discover_mcp_servers(self):
        """
        Imports all modules under app/mcp/ and finds classes inheriting MCPBase.
        Uses the lowercase class name (without "MCP") as the key. e.g.
        class VanWestendorpMCP → key = "van_westendorp"
        """
        print("[DEBUG] Starting MCP discovery...")
        mcp_folder = os.path.join(os.path.dirname(__file__), "mcp")
        print(f"[DEBUG] MCP folder path: {mcp_folder}")
        
        if not os.path.exists(mcp_folder):
            print(f"[ERROR] MCP folder does not exist: {mcp_folder}")
            return
            
        print(f"[DEBUG] Contents of MCP folder: {os.listdir(mcp_folder)}")
        
        for filename in os.listdir(mcp_folder):
            if filename.endswith(".py") and not filename.startswith("__"):
                mod_name = filename[:-3]
                print(f"[DEBUG] Processing MCP module: {mod_name}")
                try:
                    # Use absolute import
                    module = importlib.import_module(f"app.mcp.{mod_name}")
                    print(f"[DEBUG] Successfully imported module: {module}")
                    
                    for name, obj in inspect.getmembers(module, inspect.isclass):
                        print(f"[DEBUG] Found class: {name}")
                        if issubclass(obj, MCPBase) and obj is not MCPBase:
                            # Convert CamelCase to snake_case and ensure consistent naming
                            key = name.replace("MCP", "").lower()
                            # Add underscore between words (e.g., VanWestendorp -> van_westendorp)
                            key = ''.join(['_' + c.lower() if c.isupper() else c for c in key]).lstrip('_')
                            self.mcp_mapping[key] = obj()
                            print(f"[DEBUG] Registered MCP: {key} -> {obj}")
                except Exception as e:
                    print(f"[ERROR] Failed to import MCP module '{mod_name}': {e}")
                    import traceback
                    print(traceback.format_exc())
        
        print("[DEBUG] Final MCP mapping:", self.mcp_mapping)
        print("[DEBUG] MCP mapping keys after discovery:", list(self.mcp_mapping.keys()))
        if not self.mcp_mapping:
            print("[WARNING] No MCPs were registered in mcp_mapping! Check your MCP directory and class definitions.")

    def handle_chat(self, user_id: str, chat_req: ChatRequest) -> ChatResponse:
        text = chat_req.message.lower()
        print("[DEBUG] AgentController received message:", text)

        # Use Deepseek to classify the analysis type
        analysis_key = self._extract_analysis_type_with_deepseek(chat_req.message)
        print(f"[DEBUG] Deepseek classified analysis as: {analysis_key}")

        if not analysis_key or analysis_key not in self.mcp_mapping:
            print(f"[DEBUG] MCP '{analysis_key}' not found in mcp_mapping. Using fallback.")
            reply = self.chat_model.generate_reply(chat_req.message)
            return ChatResponse(reply=reply)

        print(f"[DEBUG] Calling MCP '{analysis_key}'...")
        df = load_dataset(user_id, chat_req.dataset_id)
        metadata = load_metadata(user_id, chat_req.dataset_id)
        params: Dict[str, Any] = {}
        if chat_req.filter_column and chat_req.filter_value is not None:
            params["filter_column"] = chat_req.filter_column
            params["filter_value"] = chat_req.filter_value
        if metadata:
            params["metadata"] = metadata
        # Pass chat_model to MCP
        params["chat_model"] = self.chat_model

        try:
            mcp: MCPBase = self.mcp_mapping[analysis_key]
            result = mcp.run(df, params)
            print(f"[DEBUG] MCP '{analysis_key}' run completed.")
        except Exception as e:
            print(f"[DEBUG] Error running MCP '{analysis_key}':", e)
            return ChatResponse(
                reply=f"Error running {analysis_key} analysis: {e}"
            )

        insights = result.get("insights", "")
        polished_insights = self.chat_model.generate_reply(insights)
        reply_text = (
            f"Here are the results for your {analysis_key.replace('_',' ').title()} analysis:\n\n"
            f"{polished_insights}"
        )
        return ChatResponse(
            reply=reply_text,
            analysis_tables=result.get("tables", {}),
            chart_paths=result.get("charts", {}),
            insights=insights
        )

    def _extract_analysis_type_with_deepseek(self, message: str) -> str:
        """
        Use Deepseek to classify the analysis type from the user's message.
        """
        prompt = (
            "Given the following user request, which analysis type is being requested? "
            "Reply with EXACTLY one of these options (no underscores): vanwestendorp, gaborgranger, segmentation, satisfaction, or 'unknown'.\n"
            f"User request: {message}\n"
            "Analysis type:"
        )
        reply = self.chat_model.generate_reply(prompt)
        print(f"[DEBUG] Deepseek raw reply: {reply}")
        
        # Clean up the reply to match our key format
        reply = reply.strip().lower().replace('_', '')
        print(f"[DEBUG] Cleaned reply: {reply}")
        print(f"[DEBUG] Available MCP keys: {list(self.mcp_mapping.keys())}")
        
        # Check if the cleaned reply matches any MCP key
        if reply in self.mcp_mapping:
            print(f"[DEBUG] Found matching MCP key: {reply}")
            return reply
            
        # Fallback: try to match common typos or synonyms
        lower_msg = message.lower()
        if "van" in lower_msg or "westendorp" in lower_msg or "westerndrop" in lower_msg or "price sensitivity" in lower_msg:
            return "vanwestendorp"
        if "gabor" in lower_msg or "granger" in lower_msg:
            return "gaborgranger"
        if "segment" in lower_msg:
            return "segmentation"
        if "satisfaction" in lower_msg or "nps" in lower_msg:
            return "satisfaction"
        return "unknown"

    async def _run_mcp(self, mcp_name: str, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Run an MCP with the given data and parameters."""
        print(f"[DEBUG] Calling MCP '{mcp_name}'...")
        
        # Get the MCP class
        mcp_class = self.mcp_mapping.get(mcp_name)
        if not mcp_class:
            raise ValueError(f"MCP '{mcp_name}' not found in mapping. Available MCPs: {list(self.mcp_mapping.keys())}")
            
        # Create MCP instance
        mcp = mcp_class()
        
        # Add chat model to params
        params["chat_model"] = self.chat_model
        
        # Add metadata to params if available
        if hasattr(data, 'metadata'):
            params["metadata"] = data.metadata
        else:
            # Try to load metadata from file
            try:
                from .services.data_loader import load_metadata
                user_id = params.get("user_id")
                dataset_id = params.get("dataset_id")
                if user_id and dataset_id:
                    metadata = load_metadata(user_id, dataset_id)
                    if metadata:
                        params["metadata"] = metadata
                        print("[DEBUG] Loaded metadata for MCP:", metadata)
            except Exception as e:
                print(f"[WARNING] Failed to load metadata: {e}")
        
        # Run the MCP
        try:
            result = mcp.run(data, params)
            print(f"[DEBUG] MCP '{mcp_name}' completed successfully")
            return result
        except Exception as e:
            print(f"[ERROR] MCP '{mcp_name}' failed: {str(e)}")
            raise

# Logic for handling agent-related API endpoints 