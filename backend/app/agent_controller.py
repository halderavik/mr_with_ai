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

class ConversationManager:
    """
    Manages conversation state and flow between user and MCPs.
    Handles clarification requests, follow-up questions, and conversation continuity.
    """
    
    def __init__(self):
        self.conversation_states = {}  # user_id -> conversation_state
        
    def get_conversation_state(self, user_id: str) -> Dict[str, Any]:
        """Get or create conversation state for a user."""
        if user_id not in self.conversation_states:
            self.conversation_states[user_id] = {
                'active_analysis': None,
                'pending_clarification': False,
                'pending_plan': None,
                'last_mcp_response': None,
                'conversation_history': [],
                'column_map': None,
                'column_map_confirmed': False
            }
        return self.conversation_states[user_id]
    
    def update_conversation_state(self, user_id: str, updates: Dict[str, Any]):
        """Update conversation state for a user."""
        state = self.get_conversation_state(user_id)
        state.update(updates)
        
    def is_clarification_response(self, message: str) -> bool:
        """Check if user message is responding to a clarification request."""
        clarification_indicators = [
            "yes", "confirm", "ok", "sure", "proceed", "continue",
            "no", "cancel", "stop", "different", "change"
        ]
        return any(indicator in message.lower() for indicator in clarification_indicators)
    
    def is_followup_question(self, message: str) -> bool:
        """Check if user message is a follow-up question about analysis results."""
        followup_indicators = [
            "what does this mean", "explain", "how", "why", "what about",
            "can you", "could you", "tell me more", "elaborate"
        ]
        return any(indicator in message.lower() for indicator in followup_indicators)
    
    def should_continue_conversation(self, user_id: str, message: str) -> bool:
        """Determine if we should continue with the current MCP conversation."""
        state = self.get_conversation_state(user_id)
        
        # If we have an active analysis and the message isn't requesting a new analysis
        if state['active_analysis']:
            # Check if user is requesting a new analysis type
            new_analysis_indicators = [
                "run", "perform", "do", "analyze", "analysis", "show",
                "van westendorp", "gabor granger", "segmentation", "satisfaction"
            ]
            is_new_analysis = any(indicator in message.lower() for indicator in new_analysis_indicators)
            
            # Continue conversation if:
            # 1. Not requesting new analysis, OR
            # 2. Responding to clarification, OR
            # 3. Asking follow-up questions
            return (not is_new_analysis or 
                   self.is_clarification_response(message) or 
                   self.is_followup_question(message))
        
        return False

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
        
        # Initialize conversation manager
        self.conversation_manager = ConversationManager()
        
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
                    # Try different import approaches
                    module = None
                    try:
                        # Try relative import first
                        module = importlib.import_module(f".mcp.{mod_name}", package="app")
                    except Exception as e1:
                        print(f"[DEBUG] Relative import failed: {e1}")
                        try:
                            # Try absolute import
                            module = importlib.import_module(f"app.mcp.{mod_name}")
                        except Exception as e2:
                            print(f"[DEBUG] Absolute import failed: {e2}")
                            # Try direct import using exec
                            mod_path = os.path.join(mcp_folder, filename)
                            with open(mod_path, 'r', encoding='utf-8') as f:
                                mod_code = f.read()
                            mod_globals = {}
                            exec(mod_code, mod_globals)
                            module = type('module', (), mod_globals)
                    
                    if module:
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
                    else:
                        print(f"[ERROR] Could not import module {mod_name} using any method")
                        
                except Exception as e:
                    print(f"[ERROR] Failed to import MCP module '{mod_name}': {e}")
                    import traceback
                    print(traceback.format_exc())
        
        print("[DEBUG] Final MCP mapping:", self.mcp_mapping)
        print("[DEBUG] MCP mapping keys after discovery:", list(self.mcp_mapping.keys()))
        if not self.mcp_mapping:
            print("[WARNING] No MCPs were registered in mcp_mapping! Check your MCP directory and class definitions.")
            
        # Fallback: manually import van_westendorp if not found
        if "van_westendorp" not in self.mcp_mapping and "van_westendorp" not in self.mcp_mapping:
            print("[DEBUG] Van Westendorp MCP not found in discovery, trying manual import...")
            try:
                from .mcp.van_westendorp import VanWestendorpMCP
                self.mcp_mapping["van_westendorp"] = VanWestendorpMCP()
                print("[DEBUG] Manually registered van_westendorp MCP")
            except Exception as e:
                print(f"[ERROR] Failed to manually import van_westendorp: {e}")
        
        # Fallback: manually import choice_based_conjoint if not found
        if "choice_based_conjoint" not in self.mcp_mapping:
            print("[DEBUG] Choice-Based Conjoint MCP not found in discovery, trying manual import...")
            try:
                from .mcp.choice_based_conjoint import ChoiceBasedConjointMCP
                self.mcp_mapping["choice_based_conjoint"] = ChoiceBasedConjointMCP()
                print("[DEBUG] Manually registered choice_based_conjoint MCP")
            except Exception as e:
                print(f"[ERROR] Failed to manually import choice_based_conjoint: {e}")
                
        print("[DEBUG] Final MCP mapping after fallback:", list(self.mcp_mapping.keys()))

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
        Manages conversation flow and handles clarification requests.
        
        Args:
            user_id (str): User identifier
            chat_req (ChatRequest): Chat request containing message and context
            
        Returns:
            ChatResponse: Response from the MCP or conversation manager
        """
        text = chat_req.message.lower().strip()
        print("\n[DEBUG] ===== Agent Controller Processing =====")
        print(f"[DEBUG] User Message: {text}")

        # Check if user wants to clear chat history
        if text in ["clear", "clear chat", "reset", "reset chat", "start over", "new chat"]:
            return self._clear_chat_history(user_id)

        # Validate dataset_id exists
        if not chat_req.dataset_id:
            return ChatResponse(reply="Error: No dataset ID provided. Please upload a file first.")

        try:
            # Verify dataset exists by attempting to load it
            df = load_dataset(user_id, chat_req.dataset_id)
        except Exception as e:
            return ChatResponse(reply=f"Error: Dataset not found. Please upload a file first. Details: {str(e)}")

        # Get conversation state
        conversation_state = self.conversation_manager.get_conversation_state(user_id)
        
        # Check if we should continue with current conversation
        if self.conversation_manager.should_continue_conversation(user_id, text):
            print(f"[DEBUG] Continuing conversation with active analysis: {conversation_state['active_analysis']}")
            return self._handle_continued_conversation(user_id, chat_req, text, conversation_state)
        
        # Check for new analysis request
        analysis_info = self._extract_analysis_type_with_deepseek(text, conversation_state)
        new_analysis_type = analysis_info["analysis_type"]
        
        if new_analysis_type != "unknown":
            print(f"[DEBUG] New analysis requested: {new_analysis_type}")
            # Update conversation state
            self.conversation_manager.update_conversation_state(user_id, {
                'active_analysis': new_analysis_type,
                'pending_clarification': False,
                'pending_plan': None,
                'last_mcp_response': None
            })
            return self._handle_new_analysis(user_id, chat_req, text, new_analysis_type, analysis_info)
        
        # If no analysis requested and no active conversation, use fallback
        print("[DEBUG] No analysis requested and no active conversation. Using fallback.")
        reply = self.chat_model.generate_reply(text)
        return ChatResponse(reply=reply)

    def _handle_continued_conversation(self, user_id: str, chat_req: ChatRequest, text: str, conversation_state: Dict[str, Any]) -> ChatResponse:
        """Handle continued conversation with an active MCP."""
        active_analysis = conversation_state['active_analysis']
        
        if not active_analysis or active_analysis not in self.mcp_mapping:
            return ChatResponse(reply="I'm sorry, but I lost track of our conversation. Could you please start over?")
        
        # Load data and metadata
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
        
        # Check if we're responding to a clarification request
        if conversation_state.get('pending_clarification'):
            return self._handle_clarification_response(user_id, chat_req, text, conversation_state, df, metadata)
        
        # Check if this is a follow-up question about previous results
        if conversation_state.get('last_mcp_response') and self.conversation_manager.is_followup_question(text):
            return self._handle_followup_question(user_id, chat_req, text, conversation_state, df, metadata)
        
        # Create a safe copy of conversation state to avoid circular references
        safe_conversation_state = {}
        for key, value in conversation_state.items():
            if key in ['last_mcp_response', 'conversation_history']:
                # Skip complex objects that might have circular references
                safe_conversation_state[key] = f"<{type(value).__name__} object>"
            else:
                safe_conversation_state[key] = value
        
        # Default: pass to MCP for processing
        return self._call_mcp(active_analysis, df, metadata, {
            "question": text,
            "conversation_context": safe_conversation_state,
            "chat_model": self.chat_model,
            "user_id": user_id,
            "dataset_id": chat_req.dataset_id
        })

    def _handle_clarification_response(self, user_id: str, chat_req: ChatRequest, text: str, 
                                     conversation_state: Dict[str, Any], df: pd.DataFrame, metadata: Dict[str, Any]) -> ChatResponse:
        """Handle user response to a clarification request."""
        pending_plan = conversation_state.get('pending_plan', {})
        
        # Check if user confirmed or provided new information
        if self.conversation_manager.is_clarification_response(text):
            # User confirmed, proceed with the pending plan
            print("[DEBUG] User confirmed clarification, proceeding with analysis")
            
            # Update conversation state
            self.conversation_manager.update_conversation_state(user_id, {
                'pending_clarification': False,
                'pending_plan': None
            })
            
            # Create a safe copy of conversation state to avoid circular references
            safe_conversation_state = {}
            for key, value in conversation_state.items():
                if key in ['last_mcp_response', 'conversation_history']:
                    # Skip complex objects that might have circular references
                    safe_conversation_state[key] = f"<{type(value).__name__} object>"
                else:
                    safe_conversation_state[key] = value
            
            # Call MCP with the confirmed plan
            return self._call_mcp(conversation_state['active_analysis'], df, metadata, {
                "question": "Proceed with confirmed analysis plan",
                "conversation_context": safe_conversation_state,
                "chat_model": self.chat_model,
                "filters": pending_plan.get('filters', {}),
                "segmentation": pending_plan.get('segmentation'),
                "user_id": user_id,
                "dataset_id": chat_req.dataset_id
            })
        else:
            # User provided new information, update the plan
            print("[DEBUG] User provided new information, updating analysis plan")
            
            # Extract new analysis parameters from user response
            analysis_info = self._extract_analysis_type_with_deepseek(text, conversation_state)
            
            # Update conversation state
            self.conversation_manager.update_conversation_state(user_id, {
                'pending_clarification': False,
                'pending_plan': None
            })
            
            # Call MCP with updated parameters
            return self._call_mcp(conversation_state['active_analysis'], df, metadata, {
                "question": text,
                "conversation_context": safe_conversation_state,
                "chat_model": self.chat_model,
                "filters": analysis_info.get('filters', {}),
                "segmentation": analysis_info.get('segmentation'),
                "user_id": user_id,
                "dataset_id": chat_req.dataset_id
            })

    def _handle_followup_question(self, user_id: str, chat_req: ChatRequest, text: str,
                                 conversation_state: Dict[str, Any], df: pd.DataFrame, metadata: Dict[str, Any]) -> ChatResponse:
        """Handle follow-up questions about previous analysis results."""
        last_response = conversation_state.get('last_mcp_response', {})
        
        # Check if last_response is valid for follow-up questions
        if not last_response or not isinstance(last_response, dict):
            print(f"[DEBUG] last_mcp_response is not valid for follow-up: {type(last_response)}")
            last_response = None
        
        # Create a safe copy of conversation state to avoid circular references
        safe_conversation_state = {}
        for key, value in conversation_state.items():
            if key in ['last_mcp_response', 'conversation_history']:
                # Skip complex objects that might have circular references
                safe_conversation_state[key] = f"<{type(value).__name__} object>"
            else:
                safe_conversation_state[key] = value
        
        return self._call_mcp(conversation_state['active_analysis'], df, metadata, {
            "question": text,
            "previous_result": last_response,
            "conversation_context": safe_conversation_state,
            "chat_model": self.chat_model,
            "user_id": user_id,
            "dataset_id": chat_req.dataset_id
        })

    def _handle_new_analysis(self, user_id: str, chat_req: ChatRequest, text: str, 
                           analysis_type: str, analysis_info: Dict[str, Any]) -> ChatResponse:
        """Handle a new analysis request."""
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
        
        # Get conversation state and create a safe copy
        conversation_state = self.conversation_manager.get_conversation_state(user_id)
        safe_conversation_state = {}
        for key, value in conversation_state.items():
            if key in ['last_mcp_response', 'conversation_history']:
                # Skip complex objects that might have circular references
                safe_conversation_state[key] = f"<{type(value).__name__} object>"
            else:
                safe_conversation_state[key] = value
        
        return self._call_mcp(analysis_type, df, metadata, {
            "question": text,
            "conversation_context": safe_conversation_state,
            "chat_model": self.chat_model,
            "filters": analysis_info.get('filters', {}),
            "segmentation": analysis_info.get('segmentation'),
            "user_id": user_id,
            "dataset_id": chat_req.dataset_id
        })

    def _call_mcp(self, analysis_type: str, df: pd.DataFrame, metadata: Dict[str, Any], params: Dict[str, Any]) -> ChatResponse:
        """Call the appropriate MCP and handle its response."""
        print(f"[DEBUG] Calling MCP: {analysis_type}")
        
        # Add metadata to params
        params["metadata"] = metadata
        
        print("\n[DEBUG] ===== Agent Controller -> MCP Communication =====")
        print(f"[DEBUG] MCP: {analysis_type}")
        print(f"[DEBUG] Question: {params.get('question', 'No question')}")
        print(f"[DEBUG] Metadata passed: {list(metadata.keys()) if metadata else 'None'}")
        
        # Safely serialize context to avoid circular references
        try:
            context_to_log = params.get('conversation_context', {})
            # Remove any potential circular references
            safe_context = {}
            for key, value in context_to_log.items():
                if key in ['last_mcp_response', 'conversation_history']:
                    # Skip complex objects that might have circular references
                    safe_context[key] = f"<{type(value).__name__} object>"
                else:
                    safe_context[key] = value
            print(f"[DEBUG] Context: {json.dumps(safe_context, indent=2)}")
        except Exception as e:
            print(f"[DEBUG] Context: <Error serializing context: {e}>")
        
        print("[DEBUG] ==============================================\n")
        
        try:
            mcp: MCPBase = self.mcp_mapping[analysis_type]
            result = mcp.run(df, params)
            
            print("\n[DEBUG] ===== MCP -> Agent Controller Response =====")
            print(f"[DEBUG] MCP: {analysis_type}")
            print(f"[DEBUG] Reply: {result.get('reply', 'No reply')}")
            
            # Safely serialize result context to avoid circular references
            try:
                result_context = result.get('context', {})
                safe_result_context = {}
                for key, value in result_context.items():
                    if key in ['results', 'visualizations']:
                        # Skip complex objects that might have circular references
                        safe_result_context[key] = f"<{type(value).__name__} object>"
                    else:
                        safe_result_context[key] = value
                print(f"[DEBUG] Context: {json.dumps(safe_result_context, indent=2)}")
            except Exception as e:
                print(f"[DEBUG] Context: <Error serializing result context: {e}>")
            
            print("[DEBUG] ==============================================\n")
            
            # Check if MCP is asking for clarification
            if result.get("followup_questions"):
                print("[DEBUG] MCP requested clarification")
                # Update conversation state to indicate pending clarification
                user_id = params.get("user_id")
                self.conversation_manager.update_conversation_state(user_id, {
                    'pending_clarification': True,
                    'pending_plan': result.get("context", {}).get("pending_plan")
                })
                
                return ChatResponse(
                    reply=result.get("reply", "I need some clarification before proceeding."),
                    context=result.get("context", {})
                )
            
            # Update conversation state with successful response
            user_id = params.get("user_id")
            
            # Store the actual result for follow-up questions, but create a copy to avoid circular references
            # We'll store the essential parts needed for follow-up questions
            followup_result = {
                'reply': result.get('reply'),
                'insights': result.get('insights'),
                'context': result.get('context'),
                'visualizations': result.get('visualizations')
            }
            
            self.conversation_manager.update_conversation_state(user_id, {
                'last_mcp_response': followup_result,
                'pending_clarification': False,
                'pending_plan': None
            })
            
            return self._create_chat_response(result, analysis_type)
            
        except Exception as e:
            print(f"[ERROR] MCP '{analysis_type}' failed: {str(e)}")
            return ChatResponse(reply=f"Error processing your request: {e}")

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
        if context and context.get('active_analysis'):
            if message.lower() in ["yes", "confirm", "ok", "sure", "proceed"]:
                return {
                    "analysis_type": context['active_analysis'],
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
            "- analysis_type: one of [vanwestendorp, choice_based_conjoint, gaborgranger, segmentation, satisfaction, unknown]\n"
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
            
            # Handle specific mappings for common analysis types
            if analysis_type == "vanwestendorp":
                # Check if van_westendorp exists in mapping
                if "van_westendorp" in self.mcp_mapping:
                    return {
                        "analysis_type": "van_westendorp",
                        "filters": result.get("filters", {}),
                        "segmentation": result.get("segmentation")
                    }
                elif "vanwestendorp" in self.mcp_mapping:
                    return {
                        "analysis_type": "van_westendorp",
                        "filters": result.get("filters", {}),
                        "segmentation": result.get("segmentation")
                    }
                
            # Fallback: try to match common typos or synonyms
            lower_msg = message.lower()
            if "van" in lower_msg or "westendorp" in lower_msg or "westerndrop" in lower_msg or "price sensitivity" in lower_msg:
                # Try both possible keys
                if "van_westendorp" in self.mcp_mapping:
                    return {
                        "analysis_type": "van_westendorp",
                        "filters": result.get("filters", {}),
                        "segmentation": result.get("segmentation")
                    }
                elif "vanwestendorp" in self.mcp_mapping:
                    return {
                        "analysis_type": "van_westendorp",
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
            # Check which key exists in the mapping
            if "van_westendorp" in self.mcp_mapping:
                return "van_westendorp"
            elif "vanwestendorp" in self.mcp_mapping:
                return "van_westendorp"
        if "conjoint" in lower_msg or "cbc" in lower_msg or "choice" in lower_msg:
            return "choice_based_conjoint"
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

    def _clear_chat_history(self, user_id: str) -> ChatResponse:
        """
        Clear the chat history by resetting conversation state.
        
        Returns:
            ChatResponse: Response indicating chat history has been cleared
        """
        print("[DEBUG] Clearing chat history")
        # Reset conversation state for this user
        self.conversation_manager.conversation_states[user_id] = {
            'active_analysis': None,
            'pending_clarification': False,
            'pending_plan': None,
            'last_mcp_response': None,
            'conversation_history': [],
            'column_map': None,
            'column_map_confirmed': False
        }
        return ChatResponse(
            reply="Chat history has been cleared. You can start a new conversation.",
            context={}  # Empty context will clear the history
        )

# Logic for handling agent-related API endpoints 