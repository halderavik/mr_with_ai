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
        Handle chat requests by routing them to the appropriate MCP.
        Once an MCP is determined, all subsequent messages are passed directly to that MCP
        unless a new analysis type is explicitly requested.
        
        Args:
            user_id (str): User identifier
            chat_req (ChatRequest): Chat request containing message and context
            
        Returns:
            ChatResponse: Response from the MCP
        """
        text = chat_req.message.lower().strip()
        print("\n[DEBUG] ===== Agent Controller Processing =====")
        print(f"[DEBUG] User Message: {text}")

        # Check if user wants to clear chat history
        if text in ["clear", "clear chat", "reset", "reset chat", "start over", "new chat"]:
            return self._clear_chat_history()

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

        # Check if we're in an active MCP conversation
        active_analysis = context.get('analysis_type')

        # Only check for new analysis type if:
        # 1. No active MCP, or
        # 2. User explicitly mentions a different analysis type
        should_check_new_analysis = (
            not active_analysis or 
            any(keyword in text for keyword in [
                "run", "perform", "do", "analyze", "analysis", "show",
                "van westendorp", "gabor granger", "segmentation", "satisfaction"
            ])
        )

        if should_check_new_analysis:
            # Check if user is requesting a new analysis type
            analysis_info = self._extract_analysis_type_with_deepseek(text, context)
            new_analysis_type = analysis_info["analysis_type"]
            
            # Only switch if user explicitly requested a different analysis
            if new_analysis_type != "unknown" and new_analysis_type != active_analysis:
                print(f"[DEBUG] User requested new analysis type: {new_analysis_type}")
                active_analysis = new_analysis_type
                context['analysis_type'] = new_analysis_type
                context['filters'] = analysis_info.get('filters', {})
                context['segmentation'] = analysis_info.get('segmentation')
        
        # If we have an active MCP, pass the message directly to it
        if active_analysis and active_analysis in self.mcp_mapping:
            print(f"[DEBUG] Routing to MCP: {active_analysis}")
            df = load_dataset(user_id, chat_req.dataset_id)
            
            # Load metadata with error handling
            try:
                metadata = load_metadata(user_id, chat_req.dataset_id)
            except Exception as e:
                print(f"[DEBUG] Failed to load metadata: {e}")
                print(f"[DEBUG] Attempting to force regenerate metadata")
                try:
                    from .services.data_loader import force_regenerate_metadata, test_metadata_loading
                    print(f"[DEBUG] Testing metadata loading first...")
                    test_metadata = test_metadata_loading(user_id, chat_req.dataset_id)
                    metadata = test_metadata
                    print(f"[DEBUG] Successfully loaded metadata using test function")
                except Exception as e2:
                    print(f"[ERROR] Failed to regenerate metadata: {e2}")
                    return ChatResponse(reply=f"Error loading dataset metadata: {e2}")
            
            print(f"[DEBUG] Loaded metadata keys: {list(metadata.keys()) if metadata else 'None'}")
            print(f"[DEBUG] Complete metadata content:")
            if metadata:
                for key, value in metadata.items():
                    if key == 'column_labels':
                        print(f"  {key}: {value}")
                        if isinstance(value, dict):
                            print(f"    Number of items: {len(value)}")
                            if len(value) == 0:
                                print(f"    WARNING: column_labels is empty!")
                            else:
                                print(f"    First few items: {list(value.items())[:3]}")
                    elif key == 'columns':
                        print(f"  {key}: {value[:10] if isinstance(value, list) and len(value) > 10 else value}")
                    elif key == 'value_labels':
                        print(f"  {key}: {len(value) if isinstance(value, dict) else value} items")
                    else:
                        print(f"  {key}: {value}")
            else:
                print("  No metadata loaded")
            
            # Apply any filters from context
            if context.get('filters'):
                for column, value in context['filters'].items():
                    if column in df.columns:
                        df = df[df[column] == value]
            
            params = {
                "question": text,
                "previous_result": context.get('previous_result'),
                "conversation_context": context,
                "metadata": metadata,
                "chat_model": self.chat_model,
                "filters": context.get('filters', {}),
                "segmentation": context.get('segmentation'),
                "column_map": context.get('column_map'),
                "column_map_confirmed": context.get('column_map_confirmed', False),
                "user_id": user_id,
                "dataset_id": chat_req.dataset_id
            }
            
            print("\n[DEBUG] ===== Agent Controller -> MCP Communication =====")
            print(f"[DEBUG] MCP: {active_analysis}")
            print(f"[DEBUG] Question: {text}")
            print(f"[DEBUG] Metadata passed: {list(metadata.keys()) if metadata else 'None'}")
            print(f"[DEBUG] Context: {json.dumps(context, indent=2)}")
            print("[DEBUG] ==============================================\n")
            
            try:
                mcp: MCPBase = self.mcp_mapping[active_analysis]
                result = mcp.run(df, params)
                
                print("\n[DEBUG] ===== MCP -> Agent Controller Response =====")
                print(f"[DEBUG] MCP: {active_analysis}")
                print(f"[DEBUG] Reply: {result.get('reply', 'No reply')}")
                print(f"[DEBUG] Context: {json.dumps(result.get('context', {}), indent=2)}")
                print("[DEBUG] ==============================================\n")
                
                # Update context with new information from MCP
                if result.get("context"):
                    context.update(result["context"])
                
                return self._create_chat_response(result, active_analysis)
            except Exception as e:
                print(f"[ERROR] MCP '{active_analysis}' failed: {str(e)}")
                return ChatResponse(reply=f"Error processing your request: {e}")
        
        # If no active MCP and no new analysis requested, use fallback
        print("[DEBUG] No active MCP and no new analysis requested. Using fallback.")
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

    def _extract_analysis_type_with_deepseek(self, message: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Use Deepseek to classify the analysis type and extract parameters from the user's message.
        If context is provided, use it to better understand the user's intent.
        
        Returns:
            Dict containing:
            - analysis_type: str - The type of analysis requested
            - filters: Dict[str, Any] - Any filters specified (e.g. {"gender": "male"})
            - segmentation: Optional[str] - Any segmentation requested
        """
        # If this is a confirmation or simple response, return the existing analysis type
        if context and context.get('analysis_type'):
            if message.lower() in ["yes", "confirm", "ok", "sure", "proceed"]:
                return {
                    "analysis_type": context['analysis_type'],
                    "filters": context.get('filters', {}),
                    "segmentation": context.get('segmentation')
                }
            if message.lower() in ["no", "cancel", "stop"]:
                return {
                    "analysis_type": "unknown",
                    "filters": {},
                    "segmentation": None
                }

        # For new analysis requests, use the LLM to classify and extract parameters
        prompt = (
            "Given the following user request, extract the analysis type and any filters/segmentation requested.\n"
            "Reply in JSON format with these fields:\n"
            "- analysis_type: one of [vanwestendorp, gaborgranger, segmentation, satisfaction, unknown]\n"
            "- filters: dict of any filters specified (e.g. {\"gender\": \"male\"})\n"
            "- segmentation: string if segmentation is requested, null otherwise\n\n"
            f"User request: {message}\n"
            "Response:"
        )
        reply = self.chat_model.generate_reply(prompt)
        print(f"[DEBUG] Raw LLM Response: {reply}")
        
        try:
            # Clean up the response by removing markdown formatting if present
            cleaned_reply = reply
            if "```json" in reply:
                # Extract content between ```json and ```
                json_match = re.search(r'```json\n(.*?)\n```', reply, re.DOTALL)
                if json_match:
                    cleaned_reply = json_match.group(1)
            elif "```" in reply:
                # Extract content between ``` and ```
                json_match = re.search(r'```\n(.*?)\n```', reply, re.DOTALL)
                if json_match:
                    cleaned_reply = json_match.group(1)
            
            # Parse the cleaned JSON response
            result = json.loads(cleaned_reply)
            print(f"[DEBUG] Parsed JSON: {json.dumps(result, indent=2)}")
            
            # Clean up the analysis type to match our key format
            analysis_type = result.get("analysis_type", "").strip().lower().replace('_', '')
            
            # Check if the cleaned analysis type matches any MCP key
            if analysis_type in self.mcp_mapping:
                return {
                    "analysis_type": analysis_type,
                    "filters": result.get("filters", {}),
                    "segmentation": result.get("segmentation")
                }
                
            # Fallback: try to match common typos or synonyms
            lower_msg = message.lower()
            if "van" in lower_msg or "westendorp" in lower_msg or "westerndrop" in lower_msg or "price sensitivity" in lower_msg:
                return {
                    "analysis_type": "vanwestendorp",
                    "filters": result.get("filters", {}),
                    "segmentation": result.get("segmentation")
                }
            if "gabor" in lower_msg or "granger" in lower_msg:
                return {
                    "analysis_type": "gaborgranger",
                    "filters": result.get("filters", {}),
                    "segmentation": result.get("segmentation")
                }
            if "segment" in lower_msg:
                return {
                    "analysis_type": "segmentation",
                    "filters": result.get("filters", {}),
                    "segmentation": result.get("segmentation")
                }
            if "satisfaction" in lower_msg or "nps" in lower_msg:
                return {
                    "analysis_type": "satisfaction",
                    "filters": result.get("filters", {}),
                    "segmentation": result.get("segmentation")
                }
                
            return {
                "analysis_type": "unknown",
                "filters": {},
                "segmentation": None
            }
            
        except json.JSONDecodeError as e:
            print(f"[ERROR] Failed to parse LLM response as JSON: {reply}")
            print(f"[ERROR] JSON Parse Error: {str(e)}")
            # Fallback to simple analysis type extraction
            return {
                "analysis_type": self._extract_analysis_type_simple(message),
                "filters": {},
                "segmentation": None
            }
            
    def _extract_analysis_type_simple(self, message: str) -> str:
        """Simple fallback method to extract analysis type without parameters."""
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

    def _clear_chat_history(self) -> ChatResponse:
        """
        Clear the chat history by returning a response with an empty context.
        
        Returns:
            ChatResponse: Response indicating chat history has been cleared
        """
        print("[DEBUG] Clearing chat history")
        return ChatResponse(
            reply="Chat history has been cleared. You can start a new conversation.",
            context={}  # Empty context will clear the history
        )

# Logic for handling agent-related API endpoints 