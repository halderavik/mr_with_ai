# Test script for CBC MCP
import sys
import os
import pandas as pd
import json

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.mcp.choice_based_conjoint import ChoiceBasedConjointMCP

class MockChatModel:
    """Mock chat model for testing"""
    def generate_reply(self, prompt):
        print(f"[MOCK LLM] Received prompt: {prompt[:100]}...")
        return '{"n_attributes": 4, "n_levels_per_attribute": 3, "n_tasks": 12, "n_alternatives": 3, "n_respondents": 200}'

def test_cbc_mcp():
    """Test the Choice-Based Conjoint MCP"""
    print("=== Testing Choice-Based Conjoint MCP ===")
    
    # Create MCP instance
    mcp = ChoiceBasedConjointMCP()
    print(f"MCP Name: {mcp.name}")
    print(f"Description: {mcp.description}")
    print("=== Test Completed Successfully ===")

if __name__ == "__main__":
    test_cbc_mcp()
