# backend/app/agent_controller.py

import os
from typing import Dict, Any
import importlib
import inspect

from .config import DEESEEK_API_KEY
from .models.chat_schemas import ChatRequest, ChatResponse
from .services.data_loader import load_dataset
from .services.mcp_base import MCPBase

# === STEP 1: Import Deepseek chat model ===
# (Replace this stub with your actual Deepseek Chat import)
#
# from deepseek import ChatModel
# chat_model = ChatModel(api_key=DEESEEK_API_KEY)
#
# For illustration, we’ll mock a ChatModel interface:
class ChatModel:
    def __init__(self, api_key: str):
        self.api_key = api_key

    def generate_reply(self, prompt: str) -> str:
        # In practice, you’d call deepseek’s API here. For now, echo:
        return "Deepseek‐generated reply to: " + prompt

chat_model = ChatModel(api_key=DEESEEK_API_KEY)


class AgentController:
    """
    Core orchestrator that:
      1. Receives a ChatRequest
      2. Uses Deepseek to “understand” which analysis is requested
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
        Uses the lowercase class name (without “MCP”) as the key. e.g.
        class VanWestendorpMCP → key = "van_westendorp"
        """
        mcp_folder = os.path.join(os.path.dirname(__file__), "mcp")
        for filename in os.listdir(mcp_folder):
            if filename.endswith(".py") and not filename.startswith("__"):
                mod_name = filename[:-3]
                module = importlib.import_module(f"app.mcp.{mod_name}")
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if issubclass(obj, MCPBase) and obj is not MCPBase:
                        key = name.replace("MCP", "").lower()
                        self.mcp_mapping[key] = obj()

    def handle_chat(self, user_id: str, chat_req: ChatRequest) -> ChatResponse:
        """
        1. Identify which analysis_type: We’ll do a very simple keyword check.
           In production, you’d send `chat_req.message` to Deepseek, ask “Which analysis does the user want?” 
           For now, we do a naive check: if “van” in message.lower() → Van Westendorp. 
        2. Extract optional filters for gender, etc. (already part of chat_req).
        3. Load the dataset.
        4. Run the MCP server.
        5. Compose reply (text + tables + charts + insights).
        """
        text = chat_req.message.lower()
        if "van" in text:
            analysis_key = "van_westendorp"
        elif "gabor" in text:
            analysis_key = "gabor_granger"
        else:
            # Fallback: let Deepseek answer generally
            reply = chat_model.generate_reply(chat_req.message)
            return ChatResponse(reply=reply)

        # Ensure MCP server is available:
        if analysis_key not in self.mcp_mapping:
            return ChatResponse(
                reply=f"Sorry, I do not have the '{analysis_key}' analysis implemented yet."
            )

        # 3. Load dataset into pandas:
        df = load_dataset(user_id, chat_req.dataset_id)

        # 4. Prepare params for MCP:
        params: Dict[str, Any] = {}
        if chat_req.filter_column and chat_req.filter_value is not None:
            params["filter_column"] = chat_req.filter_column
            params["filter_value"] = chat_req.filter_value

        # (You could parse column mapping from the user in future.)
        # 5. Run MCP:
        try:
            mcp: MCPBase = self.mcp_mapping[analysis_key]
            result = mcp.run(df, params)
        except Exception as e:
            return ChatResponse(
                reply=f"Error running {analysis_key} analysis: {e}"
            )

        # 6. Compose final reply text:
        #    We can call Deepseek again, e.g. “Summarize these insights.”
        insights = result.get("insights", "")
        # Optionally, ask Deepseek to polish the insights:
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

# Logic for handling agent-related API endpoints 