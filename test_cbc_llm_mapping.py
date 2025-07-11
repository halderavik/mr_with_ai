#!/usr/bin/env python3
"""
Test script to verify LLM-based column mapping for CBC analysis.
"""

import sys
import os
import pandas as pd
import json

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from app.mcp.choice_based_conjoint import ChoiceBasedConjointMCP

# Mock chat model for testing
class MockChatModel:
    def generate_reply(self, prompt):
        """Mock LLM response for testing."""
        print(f"[DEBUG] Mock LLM received prompt: {prompt[:200]}...")
        
        # Return a mock response based on the data structure
        return json.dumps({
            "structure_type": "long_format",
            "respondent_id": "RespondentID",
            "task_id": "Task",
            "alternative_id": "Alternative",
            "choice": "Chosen",
            "task_columns": [],
            "attributes": ["Brand", "Price", "Camera", "Storage", "AgeGroup", "Gender", "Income"]
        })

def test_llm_based_mapping():
    """Test the LLM-based column mapping with the actual data file."""
    
    print("=== Testing LLM-Based Column Mapping for CBC Analysis ===")
    
    # Initialize the CBC MCP
    cbc_mcp = ChoiceBasedConjointMCP()
    
    # Load the data file
    data_file = "uploaded_files/user_demo/d4b30626-22e0-428b-a3e1-b9db6c30fc4c.csv"
    
    if not os.path.exists(data_file):
        print(f"ERROR: Data file not found: {data_file}")
        return False
    
    print(f"Loading data from: {data_file}")
    data = pd.read_csv(data_file)
    print(f"Data shape: {data.shape}")
    print(f"Columns: {list(data.columns)}")
    
    # Create comprehensive metadata (simulate what the data loader would create)
    metadata = {
        'columns': list(data.columns),
        'column_labels': {
            'RespondentID': 'Respondent ID',
            'AgeGroup': 'Age Group',
            'Gender': 'Gender',
            'Income': 'Income Level',
            'Task': 'Choice Task',
            'Alternative': 'Alternative/Profile',
            'Chosen': 'Chosen Alternative',
            'Brand': 'Brand',
            'Price': 'Price',
            'Camera': 'Camera Quality',
            'Storage': 'Storage Capacity'
        },
        'value_labels': {
            'AgeGroup': {'18-24': '18-24', '25-34': '25-34', '35-44': '35-44', '45-54': '45-54'},
            'Gender': {'Male': 'Male', 'Female': 'Female', 'Other': 'Other'},
            'Income': {'20,000-50,000': '20,000-50,000', '50,001-100,000': '50,001-100,000', '100,001-150,000': '100,001-150,000', '150,001+': '150,001+'},
            'Brand': {'Apple': 'Apple', 'Samsung': 'Samsung'},
            'Price': {'$699': '$699', '$899': '$899'},
            'Camera': {'Dual': 'Dual Camera', 'Triple': 'Triple Camera'},
            'Storage': {'128GB': '128GB', '256GB': '256GB'}
        },
        'variable_labels': {
            'RespondentID': 'Unique identifier for each respondent',
            'Task': 'Choice task number',
            'Alternative': 'Alternative/profile number within task',
            'Chosen': 'Whether this alternative was chosen (1) or not (0)'
        },
        'data_types': {col: str(dtype) for col, dtype in data.dtypes.items()},
        'unique_values': {col: {'count': data[col].nunique(), 'values': sorted(data[col].unique().tolist())} for col in data.columns},
        'row_count': len(data),
        'column_count': len(data.columns),
        'file_type': 'csv'
    }
    
    # Create mock chat model
    mock_chat_model = MockChatModel()
    
    print("\n=== Testing LLM-Based Column Structure Identification ===")
    
    # Test the LLM-based column structure identification
    try:
        column_map = cbc_mcp._identify_conjoint_structure_llm(data, metadata, mock_chat_model)
        print(f"LLM identified column map: {column_map}")
        
        # Check if all required columns are mapped
        required_keys = ['structure_type', 'respondent_id', 'task_id', 'choice']
        missing_keys = [key for key in required_keys if not column_map.get(key)]
        
        if missing_keys:
            print(f"ERROR: Missing required column mappings: {missing_keys}")
            return False
        
        print("✅ All required columns successfully mapped by LLM!")
        
        # Test data preparation
        print("\n=== Testing Data Preparation with LLM Mapping ===")
        try:
            X_tensor, choice_mask_tensor = cbc_mcp._prepare_conjoint_data(data, column_map)
            print(f"✅ Data preparation successful!")
            print(f"Feature matrix shape: {X_tensor.shape}")
            print(f"Choice mask shape: {choice_mask_tensor.shape}")
            
            return True
            
        except Exception as e:
            print(f"ERROR in data preparation: {e}")
            return False
            
    except Exception as e:
        print(f"ERROR in LLM-based column structure identification: {e}")
        return False

def test_full_cbc_analysis():
    """Test the full CBC analysis with LLM-based mapping."""
    
    print("\n=== Testing Full CBC Analysis with LLM Mapping ===")
    
    # Initialize the CBC MCP
    cbc_mcp = ChoiceBasedConjointMCP()
    
    # Load the data file
    data_file = "uploaded_files/user_demo/d4b30626-22e0-428b-a3e1-b9db6c30fc4c.csv"
    data = pd.read_csv(data_file)
    
    # Create metadata
    metadata = {
        'column_labels': {
            'RespondentID': 'Respondent ID',
            'Task': 'Choice Task',
            'Alternative': 'Alternative/Profile',
            'Chosen': 'Chosen Alternative',
            'Brand': 'Brand',
            'Price': 'Price',
            'Camera': 'Camera Quality',
            'Storage': 'Storage Capacity'
        },
        'value_labels': {},
        'variable_labels': {},
        'data_types': {},
        'unique_values': {}
    }
    
    # Create mock chat model
    mock_chat_model = MockChatModel()
    
    # Create params
    params = {
        'metadata': metadata,
        'chat_model': mock_chat_model
    }
    
    try:
        # Run the full analysis
        result = cbc_mcp.run(data, params)
        print(f"✅ Full CBC analysis successful!")
        print(f"Result keys: {list(result.keys())}")
        print(f"Visualizations: {list(result.get('visualizations', {}).keys())}")
        print(f"Insights length: {len(result.get('insights', ''))}")
        
        return True
        
    except Exception as e:
        print(f"ERROR in full CBC analysis: {e}")
        return False

if __name__ == "__main__":
    print("Starting CBC LLM Mapping Tests...")
    
    # Test 1: LLM-based column mapping
    success1 = test_llm_based_mapping()
    
    # Test 2: Full CBC analysis
    success2 = test_full_cbc_analysis()
    
    if success1 and success2:
        print("\n🎉 All tests passed! LLM-based column mapping is working correctly.")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        sys.exit(1) 