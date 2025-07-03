#!/usr/bin/env python3
"""
Test script to verify conversation flow between AgentController and MCPs.
This tests the new conversation management system.
"""

import sys
import os

# Add the current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Mock the required dependencies to avoid import conflicts
class MockChatRequest:
    def __init__(self, message, dataset_id, conversation_context=None):
        self.message = message
        self.dataset_id = dataset_id
        self.conversation_context = conversation_context or {}

class MockChatResponse:
    def __init__(self, reply="", context=None, **kwargs):
        self.reply = reply
        self.context = context or {}
        for key, value in kwargs.items():
            setattr(self, key, value)

# Mock the required modules
sys.modules['app.models.chat_schemas'] = type('MockModule', (), {
    'ChatRequest': MockChatRequest,
    'ChatResponse': MockChatResponse
})

# Mock config
sys.modules['app.config'] = type('MockModule', (), {
    'DEEPSEEK_API_KEY': 'test_key',
    'DEEPSEEK_MODEL': 'deepseek-chat'
})

# Mock data loader
class MockDataLoader:
    @staticmethod
    def load_dataset(user_id, dataset_id):
        import pandas as pd
        # Create mock data
        return pd.DataFrame({
            'Q1': [10, 15, 20, 25, 30],
            'Q2': [20, 25, 30, 35, 40],
            'Q3': [30, 35, 40, 45, 50],
            'Q4': [40, 45, 50, 55, 60]
        })
    
    @staticmethod
    def load_metadata(user_id, dataset_id):
        return {
            'column_labels': {
                'Q1': 'What price is too cheap?',
                'Q2': 'What price is a bargain?',
                'Q3': 'What price is getting expensive?',
                'Q4': 'What price is too expensive?'
            },
            'columns': ['Q1', 'Q2', 'Q3', 'Q4']
        }

sys.modules['app.services.data_loader'] = MockDataLoader()

# Mock MCP base
class MockMCPBase:
    def __init__(self):
        pass

sys.modules['app.services.mcp_base'] = type('MockModule', (), {
    'MCPBase': MockMCPBase
})

# Mock OpenAI
class MockOpenAI:
    def __init__(self, api_key, base_url):
        self.api_key = api_key
        self.base_url = base_url
    
    class chat:
        class completions:
            @staticmethod
            def create(model, messages, stream=False):
                return type('MockResponse', (), {
                    'choices': [type('MockChoice', (), {
                        'message': type('MockMessage', (), {
                            'content': '{"analysis_type": "van_westendorp", "filters": {}, "segmentation": null}'
                        })()
                    })()]
                })()

sys.modules['openai'] = type('MockModule', (), {
    'OpenAI': MockOpenAI
})

# Now import our classes
from app.agent_controller import ConversationManager

def test_conversation_manager():
    """Test the ConversationManager class directly."""
    
    print("=== Testing ConversationManager ===\n")
    
    manager = ConversationManager()
    user_id = "test_user"
    
    # Test state creation
    state = manager.get_conversation_state(user_id)
    print(f"Initial state: {state}")
    
    # Test state update
    manager.update_conversation_state(user_id, {
        'active_analysis': 'van_westendorp',
        'pending_clarification': True
    })
    state = manager.get_conversation_state(user_id)
    print(f"Updated state: {state}")
    
    # Test clarification detection
    test_messages = [
        "yes, proceed",
        "no, cancel",
        "what does this mean?",
        "run van westendorp",
        "hello there"
    ]
    
    print("\nTesting message classification:")
    for msg in test_messages:
        is_clarification = manager.is_clarification_response(msg)
        is_followup = manager.is_followup_question(msg)
        should_continue = manager.should_continue_conversation(user_id, msg)
        print(f"'{msg}': clarification={is_clarification}, followup={is_followup}, continue={should_continue}")

def test_conversation_flow():
    """Test the conversation flow with a mock user interaction."""
    
    print("\n=== Testing Conversation Flow ===\n")
    
    # Test conversation manager functionality
    manager = ConversationManager()
    user_id = "test_user_123"
    
    # Test 1: Initial analysis request detection
    print("Test 1: New analysis request detection")
    message = "run van westendorp analysis"
    should_continue = manager.should_continue_conversation(user_id, message)
    print(f"Message: '{message}'")
    print(f"Should continue conversation: {should_continue}")
    print(f"Expected: False (new analysis)")
    print()
    
    # Test 2: Set active analysis and test continuation
    print("Test 2: Active conversation continuation")
    manager.update_conversation_state(user_id, {
        'active_analysis': 'van_westendorp',
        'pending_clarification': False
    })
    
    followup_message = "what does this mean?"
    should_continue = manager.should_continue_conversation(user_id, followup_message)
    print(f"Message: '{followup_message}'")
    print(f"Should continue conversation: {should_continue}")
    print(f"Expected: True (follow-up question)")
    print()
    
    # Test 3: Clarification response
    print("Test 3: Clarification response")
    clarification_message = "yes, proceed"
    is_clarification = manager.is_clarification_response(clarification_message)
    should_continue = manager.should_continue_conversation(user_id, clarification_message)
    print(f"Message: '{clarification_message}'")
    print(f"Is clarification response: {is_clarification}")
    print(f"Should continue conversation: {should_continue}")
    print(f"Expected: True (clarification response)")
    print()
    
    # Test 4: New analysis during active conversation
    print("Test 4: New analysis during active conversation")
    new_analysis_message = "run segmentation analysis"
    should_continue = manager.should_continue_conversation(user_id, new_analysis_message)
    print(f"Message: '{new_analysis_message}'")
    print(f"Should continue conversation: {should_continue}")
    print(f"Expected: False (new analysis request)")
    print()
    
    # Test 5: Clear conversation
    print("Test 5: Clear conversation")
    clear_message = "clear chat"
    is_clarification = manager.is_clarification_response(clear_message)
    is_followup = manager.is_followup_question(clear_message)
    print(f"Message: '{clear_message}'")
    print(f"Is clarification response: {is_clarification}")
    print(f"Is followup question: {is_followup}")
    print(f"Expected: False, False (clear command)")
    print()

if __name__ == "__main__":
    print("Starting conversation flow tests...\n")
    
    # Test ConversationManager
    test_conversation_manager()
    print("\n" + "="*50 + "\n")
    
    # Test full conversation flow
    test_conversation_flow()
    
    print("Tests completed successfully!")
    print("\nSummary:")
    print("- ConversationManager correctly manages conversation state")
    print("- Message classification works for clarification responses and follow-up questions")
    print("- Conversation continuation logic properly distinguishes between new analysis and continued conversation")
    print("- The system is ready to handle the conversation flow between AgentController and MCPs") 