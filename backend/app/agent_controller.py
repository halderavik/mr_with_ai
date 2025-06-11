# backend/app/agent_controller.py

import os
from typing import Dict, Any, Optional
import importlib
import inspect
from .config import DEEPSEEK_API_KEY, DEEPSEEK_MODEL
from .models.chat_schemas import ChatRequest, ChatResponse
from .services.data_loader import load_dataset, load_metadata
from .services.mcp_base import MCPBase
import pandas as pd
import json
import re

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
                {"role": "system", "content": "You are a helpful assistant for market research analysis. Never ask for confirmation or mapping - directly proceed with analysis. Always assume the MCP will handle variable mapping automatically."},
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

    def _parse_mapping_from_text(self, text: str) -> dict:
        """Parse a mapping from user text in the format 'too_cheap: Q1\nbargain: Q2...'"""
        mapping = {}
        for line in text.splitlines():
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().lower()
                value = value.strip()
                if key in ["too_cheap", "bargain", "getting_expensive", "too_expensive"]:
                    mapping[key] = value
        return mapping if mapping else None

    def handle_chat(self, user_id: str, chat_req: ChatRequest) -> ChatResponse:
        """
        Handle chat requests using existing dataset IDs.
        Dataset IDs must be created during file upload and cannot be created during chat.
        """
        text = chat_req.message.lower().strip()
        print("[DEBUG] AgentController received message:", text)

        # Validate dataset_id exists
        if not chat_req.dataset_id:
            return ChatResponse(reply="Error: No dataset ID provided. Please upload a file first.")

        try:
            # Verify dataset exists by attempting to load it
            df = load_dataset(user_id, chat_req.dataset_id)
        except Exception as e:
            return ChatResponse(reply=f"Error: Dataset not found. Please upload a file first. Details: {str(e)}")

        # Initialize or get conversation context
        context = chat_req.conversation_context or {}
        print(f"[DEBUG] Current conversation context: {json.dumps(context, indent=2)}")

        # Check if this is a new dataset by comparing with last dataset_id in context
        last_dataset_id = context.get('last_dataset_id')
        if last_dataset_id != chat_req.dataset_id:
            print(f"[DEBUG] New dataset detected. Clearing conversation context.")
            context = {
                'last_dataset_id': chat_req.dataset_id,
                'analysis_type': None,
                'previous_result': None
            }
            print(f"[DEBUG] Reset context for new dataset: {json.dumps(context, indent=2)}")

        # Check if we're in an active MCP conversation
        active_analysis = context.get('analysis_type')
        print(f"[DEBUG] Active analysis type: {active_analysis}")
        
        # If we have a last_answer about confirming mapping but no active_analysis, set it
        if not active_analysis and context.get('last_answer'):
            last_a = context['last_answer'].lower()
            if 'before running the' in last_a and 'analysis' in last_a:
                # Extract analysis type from the last answer
                if 'van westendorp' in last_a:
                    active_analysis = 'van_westendorp'
                    context['analysis_type'] = active_analysis
                    print(f"[DEBUG] Setting active analysis type from last answer: {active_analysis}")
        
        if active_analysis and active_analysis in self.mcp_mapping:
            print(f"[DEBUG] Continuing conversation with MCP: {active_analysis}")
            
            # Handle follow-up questions
            print(f"[DEBUG] Handling follow-up question for {active_analysis}")
            df = load_dataset(user_id, chat_req.dataset_id)
            metadata = load_metadata(user_id, chat_req.dataset_id)
            params = {
                "question": text,
                "previous_result": context.get('previous_result'),
                "conversation_context": context,
                "metadata": metadata,
                "chat_model": self.chat_model
            }
            try:
                mcp: MCPBase = self.mcp_mapping[active_analysis]
                result = mcp.run(df, params)
                print(f"[DEBUG] MCP '{active_analysis}' handled follow-up question")
                return self._create_chat_response(result, active_analysis)
            except Exception as e:
                print(f"[DEBUG] Error handling follow-up question for '{active_analysis}':", e)
                return ChatResponse(reply=f"Error handling follow-up question: {e}")

        # Start new analysis if user requests it or if no active analysis
        if "run" in text or "start" in text or "analyze" in text or not active_analysis:
            print("[DEBUG] Starting new analysis")
            analysis_key = self._extract_analysis_type_with_deepseek(text, context)
            print(f"[DEBUG] Determined analysis type: {analysis_key}")

            if not analysis_key or analysis_key not in self.mcp_mapping:
                print(f"[DEBUG] No valid analysis type found. Using fallback.")
                reply = self.chat_model.generate_reply(text)
                return ChatResponse(reply=reply)

            print(f"[DEBUG] Initializing MCP: {analysis_key}")
            df = load_dataset(user_id, chat_req.dataset_id)
            metadata = load_metadata(user_id, chat_req.dataset_id)
            params = {
                "metadata": metadata,
                "chat_model": self.chat_model,
                "conversation_context": context,
                "auto_map": True  # Tell MCP to handle mapping automatically
            }
            if chat_req.filter_column and chat_req.filter_value is not None:
                params["filter_column"] = chat_req.filter_column
                params["filter_value"] = chat_req.filter_value

            try:
                mcp: MCPBase = self.mcp_mapping[analysis_key]
                result = mcp.run(df, params)
                print(f"[DEBUG] MCP '{analysis_key}' initialized")
                # Set the analysis type in the context
                context['analysis_type'] = analysis_key
                return self._create_chat_response(result, analysis_key)
            except Exception as e:
                print(f"[DEBUG] Error initializing MCP '{analysis_key}':", e)
                return ChatResponse(reply=f"Error starting {analysis_key} analysis: {e}")
        
        # If we get here, treat it as a follow-up question to the last active analysis
        if active_analysis:
            print(f"[DEBUG] Treating as follow-up question for {active_analysis}")
            df = load_dataset(user_id, chat_req.dataset_id)
            metadata = load_metadata(user_id, chat_req.dataset_id)
            params = {
                "question": text,
                "previous_result": context.get('previous_result'),
                "conversation_context": context,
                "metadata": metadata,
                "chat_model": self.chat_model
            }
            try:
                mcp: MCPBase = self.mcp_mapping[active_analysis]
                result = mcp.run(df, params)
                print(f"[DEBUG] MCP '{active_analysis}' handled follow-up question")
                return self._create_chat_response(result, active_analysis)
            except Exception as e:
                print(f"[DEBUG] Error handling follow-up question for '{active_analysis}':", e)
                return ChatResponse(reply=f"Error handling follow-up question: {e}")
        
        # If no active analysis and no explicit request, use fallback
        print("[DEBUG] No active analysis and no explicit request. Using fallback.")
        reply = self.chat_model.generate_reply(text)
        return ChatResponse(reply=reply)

    def _create_chat_response(self, result: Dict[str, Any], analysis_key: str) -> ChatResponse:
        """Helper method to create a consistent ChatResponse from MCP results."""
        # Handle insights
        insights = result.get("insights", "")
        if insights and self.chat_model:
            polished_insights = self.chat_model.generate_reply(
                f"Please provide clear, business-focused insights based on this analysis: {insights}"
            )
        else:
            polished_insights = insights

        # Create response
        chat_response = ChatResponse(
            reply=result.get("reply", "I apologize, but I couldn't generate a proper response."),
            analysis_tables=result.get("analysis_tables"),
            analysis_charts=result.get("analysis_charts"),
            insights=polished_insights,
            context=result.get("context"),
            visualizations=result.get("visualizations")
        )

        # Debug logging
        print(f"[DEBUG] Created ChatResponse for {analysis_key}")
        print(f"[DEBUG] Reply: {chat_response.reply}")
        print(f"[DEBUG] Context: {json.dumps(chat_response.context, indent=2)}")
        print(f"[DEBUG] Visualizations: {json.dumps(chat_response.visualizations, indent=2) if chat_response.visualizations else 'None'}")

        return chat_response

    def _extract_analysis_type_with_deepseek(self, message: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Use Deepseek to classify the analysis type from the user's message.
        If context is provided, use it to better understand the user's intent.
        """
        # If this is a confirmation or simple response, return the existing analysis type
        if context and context.get('analysis_type'):
            if message.lower() in ["yes", "confirm", "ok", "sure", "proceed"]:
                return context['analysis_type']
            if message.lower() in ["no", "cancel", "stop"]:
                return "unknown"

        # For new analysis requests, use the LLM to classify
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