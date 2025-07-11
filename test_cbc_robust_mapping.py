#!/usr/bin/env python3
"""
Test script to verify robust column mapping for CBC analysis.
"""

import sys
import os
import pandas as pd
import json

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from app.mcp.choice_based_conjoint import ChoiceBasedConjointMCP

def test_robust_column_mapping():
    """Test the improved column mapping with the actual data file."""
    
    print("=== Testing Robust Column Mapping for CBC Analysis ===")
    
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
    
    # Create metadata (simulate what the data loader would create)
    metadata = {
        'columns': list(data.columns),
        'column_labels': {},
        'value_labels': {},
        'variable_labels': {},
        'variable_values': {},
        'variable_formats': {},
        'variable_measure': {},
        'variable_roles': {},
        'variable_alignment': {},
        'variable_width': {},
        'variable_decimals': {},
        'row_count': len(data),
        'column_count': len(data.columns),
        'data_types': data.dtypes.to_dict(),
        'missing_values': {col: {'count': data[col].isnull().sum(), 'percentage': data[col].isnull().sum() / len(data) * 100} for col in data.columns},
        'unique_values': {col: {'count': data[col].nunique(), 'values': sorted(data[col].unique().tolist())} for col in data.columns},
        'basic_stats': {},
        'file_type': 'csv',
        'conjoint_structure': {
            'detected': True,
            'structure_type': 'long_format',
            'identified_columns': {
                'respondent_id': 'RespondentID',
                'task_columns': ['Task'],
                'num_respondents': data['RespondentID'].nunique(),
                'num_tasks': data['Task'].nunique(),
                'found_columns': ['RespondentID', 'Task']
            }
        }
    }
    
    print("\n=== Testing Column Structure Identification ===")
    
    # Test the column structure identification
    try:
        column_map = cbc_mcp._identify_conjoint_structure(data, metadata)
        print(f"Identified column map: {column_map}")
        
        # Check if all required columns are mapped
        required_keys = ['respondent_id', 'task_id', 'alternative_id', 'choice']
        missing_keys = [key for key in required_keys if not column_map.get(key)]
        
        if missing_keys:
            print(f"ERROR: Missing required column mappings: {missing_keys}")
            return False
        
        print("✅ All required columns successfully mapped!")
        
        # Test data preparation
        print("\n=== Testing Data Preparation ===")
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
        print(f"ERROR in column structure identification: {e}")
        return False

def test_direct_column_detection():
    """Test direct column detection without metadata."""
    
    print("\n=== Testing Direct Column Detection ===")
    
    # Initialize the CBC MCP
    cbc_mcp = ChoiceBasedConjointMCP()
    
    # Load the data file
    data_file = "uploaded_files/user_demo/d4b30626-22e0-428b-a3e1-b9db6c30fc4c.csv"
    data = pd.read_csv(data_file)
    
    # Test with empty metadata (fallback to pattern-based detection)
    empty_metadata = {
        'columns': list(data.columns),
        'column_labels': {},
        'value_labels': {},
        'conjoint_structure': {'detected': False}
    }
    
    try:
        column_map = cbc_mcp._identify_conjoint_structure(data, empty_metadata)
        print(f"Direct detection column map: {column_map}")
        
        # Check if all required columns are mapped
        required_keys = ['respondent_id', 'task_id', 'alternative_id', 'choice']
        missing_keys = [key for key in required_keys if not column_map.get(key)]
        
        if missing_keys:
            print(f"ERROR: Missing required column mappings: {missing_keys}")
            return False
        
        print("✅ Direct column detection successful!")
        return True
        
    except Exception as e:
        print(f"ERROR in direct column detection: {e}")
        return False

if __name__ == "__main__":
    print("Starting CBC Robust Mapping Tests...")
    
    # Test 1: Metadata-based detection
    success1 = test_robust_column_mapping()
    
    # Test 2: Direct pattern-based detection
    success2 = test_direct_column_detection()
    
    if success1 and success2:
        print("\n🎉 All tests passed! Column mapping is now robust.")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        sys.exit(1) 