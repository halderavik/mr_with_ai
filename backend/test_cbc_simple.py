#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import pandas as pd
import numpy as np

# Add the current directory to Python path
sys.path.append('.')

def test_cbc_mcp_basic():
    """Test the CBC MCP with basic functionality"""
    print("Testing CBC MCP basic functionality...")
    
    try:
        # Test 1: Import the MCP
        from app.mcp.choice_based_conjoint import ChoiceBasedConjointMCP
        print("✓ Import successful!")
        
        # Test 2: Create instance
        mcp = ChoiceBasedConjointMCP()
        print(f"✓ MCP name: {mcp.name}")
        print(f"✓ MCP description: {mcp.description}")
        print(f"✓ Required columns: {mcp.required_columns}")
        print(f"✓ Device: {mcp.device}")
        
        # Test 3: Create sample data
        np.random.seed(42)
        n_respondents = 20
        n_tasks = 4
        n_alternatives = 3
        
        data = []
        for respondent in range(n_respondents):
            for task in range(n_tasks):
                for alt in range(n_alternatives):
                    choice = 1 if alt == np.random.randint(0, n_alternatives) else 0
                    data.append({
                        'respondent_id': respondent,
                        'task_id': task,
                        'alternative_id': alt,
                        'choice': choice,
                        'brand': np.random.choice(['Brand A', 'Brand B', 'Brand C']),
                        'price': np.random.choice(['$10', '$15', '$20']),
                        'color': np.random.choice(['Red', 'Blue', 'Green'])
                    })
        
        test_data = pd.DataFrame(data)
        print(f"✓ Created test data with shape: {test_data.shape}")
        
        # Test 4: Test conjoint structure identification
        metadata = {
            'column_labels': {
                'choice': 'Which alternative did you choose?',
                'task_id': 'Task number',
                'alternative_id': 'Alternative number',
                'brand': 'Brand name',
                'price': 'Price level',
                'color': 'Color option'
            }
        }
        
        conjoint_structure = mcp._identify_conjoint_structure(test_data, metadata)
        print(f"✓ Identified conjoint structure: {conjoint_structure}")
        
        # Test 5: Test data preparation
        column_map = {
            'choice': 'choice',
            'task_id': 'task_id', 
            'alternative_id': 'alternative_id',
            'attributes': ['brand', 'price', 'color']
        }
        
        X, choice_mask = mcp._prepare_conjoint_data(test_data, column_map)
        print(f"✓ Prepared data - X shape: {X.shape}, choice_mask shape: {choice_mask.shape}")
        
        # Test 6: Test full analysis
        print("\nRunning full CBC analysis...")
        result = mcp.run(test_data, {'metadata': metadata})
        
        if result and 'status' in result:
            print(f"✓ Analysis completed with status: {result['status']}")
            if 'visualizations' in result:
                print(f"✓ Generated {len(result['visualizations'])} visualizations")
            if 'insights' in result:
                print(f"✓ Generated insights: {len(result['insights'])} characters")
        else:
            print("✗ Analysis failed or returned unexpected result")
        
        print("\n✓ All basic tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_cbc_mcp_basic()
    if success:
        print("\n🎉 CBC MCP is working correctly!")
    else:
        print("\n❌ CBC MCP has issues that need to be fixed.") 