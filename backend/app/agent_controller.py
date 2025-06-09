# backend/app/agent_controller.py

import os
from typing import Dict, Any
import importlib
import inspect

from .config import DEEPSEEK_API_KEY
from .models.chat_schemas import ChatRequest, ChatResponse
from .services.data_loader import load_dataset
from .services.mcp_base import MCPBase

# === STEP 1: Import Deepseek chat model ===
# (Replace this stub with your actual Deepseek Chat import)
#
# from deepseek import ChatModel
# chat_model = ChatModel(api_key=DEEPSEEK_API_KEY)
#
# For illustration, we'll mock a ChatModel interface:
class ChatModel:
    def __init__(self, api_key: str):
        self.api_key = api_key

    def generate_reply(self, prompt: str) -> str:
        # In practice, you'd call deepseek's API here. For now, echo:
        return "Deepseek‐generated reply to: " + prompt

chat_model = ChatModel(api_key=DEEPSEEK_API_KEY)


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
        # Dynamically discover all MCP classes in app/mcp/
        self.mcp_mapping: Dict[str, MCPBase] = {}
        self._discover_mcp_servers()

    def _discover_mcp_servers(self):
        """
        Imports all modules under app/mcp/ and finds classes inheriting MCPBase.
        Uses the lowercase class name (without "MCP") as the key. e.g.
        class VanWestendorpMCP → key = "van_westendorp"
        """
        mcp_folder = os.path.join(os.path.dirname(__file__), "mcp")
        for filename in os.listdir(mcp_folder):
            if filename.endswith(".py") and not filename.startswith("__"):
                mod_name = filename[:-3]
                try:
                    module = importlib.import_module(f"app.mcp.{mod_name}")
                    for name, obj in inspect.getmembers(module, inspect.isclass):
                        if issubclass(obj, MCPBase) and obj is not MCPBase:
                            key = name.replace("MCP", "").lower()
                            self.mcp_mapping[key] = obj()
                            print(f"[DEBUG] Registered MCP: {key} -> {obj}")
                except Exception as e:
                    print(f"[ERROR] Failed to import MCP module '{mod_name}': {e}")
        # Manual fallback registration for VanWestendorpMCP if not found
        if "van_westendorp" not in self.mcp_mapping:
            try:
                from app.mcp.van_westendorp import VanWestendorpMCP
                self.mcp_mapping["van_westendorp"] = VanWestendorpMCP()
                print("[DEBUG] Manually registered VanWestendorpMCP as fallback.")
            except Exception as e:
                print(f"[ERROR] Manual fallback for VanWestendorpMCP failed: {e}")
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
            reply = chat_model.generate_reply(chat_req.message)
            return ChatResponse(reply=reply)

        print(f"[DEBUG] Calling MCP '{analysis_key}'...")
        df = load_dataset(user_id, chat_req.dataset_id)
        params: Dict[str, Any] = {}
        if chat_req.filter_column and chat_req.filter_value is not None:
            params["filter_column"] = chat_req.filter_column
            params["filter_value"] = chat_req.filter_value
        # Pass metadata if present
        if hasattr(chat_req, "metadata") and chat_req.metadata:
            params["metadata"] = chat_req.metadata

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
        polished_insights = chat_model.generate_reply(insights)
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
        This is a simulated version. Replace with a real Deepseek API call for production.
        """
        prompt = (
            "Given the following user request, which analysis type is being requested? "
            "Reply with one of: van_westendorp, gabor_granger, segmentation, satisfaction, or 'unknown'.\n"
            f"User request: {message}\n"
            "Analysis type:"
        )
        reply = chat_model.generate_reply(prompt)
        print(f"[DEBUG] Deepseek raw reply: {reply}")
        # Simulate: extract the keyword from the reply
        for key in self.mcp_mapping.keys():
            if key in reply.lower():
                return key
        # Fallback: try to match common typos or synonyms
        lower_msg = message.lower()
        if "van" in lower_msg or "westendorp" in lower_msg or "westerndrop" in lower_msg or "price sensitivity" in lower_msg:
            return "van_westendorp"
        if "gabor" in lower_msg or "granger" in lower_msg:
            return "gabor_granger"
        if "segment" in lower_msg:
            return "segmentation"
        if "satisfaction" in lower_msg or "nps" in lower_msg:
            return "satisfaction"
        return "unknown"

# Logic for handling agent-related API endpoints 