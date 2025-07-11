#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
sys.path.append('.')

try:
    from app.mcp.choice_based_conjoint import ChoiceBasedConjointMCP
    print("✓ Import successful!")
    
    # Test creating an instance
    mcp = ChoiceBasedConjointMCP()
    print(f"✓ MCP name: {mcp.name}")
    print(f"✓ MCP description: {mcp.description}")
    print(f"✓ Required columns: {mcp.required_columns}")
    print(f"✓ Device: {mcp.device}")
    
    # Test with sample conjoint data
    import pandas as pd
    import numpy as np
    
    # Create sample CBC data
    np.random.seed(42)
    n_respondents = 50
    n_tasks = 8
    n_alternatives = 3
    n_attributes = 4
    
    # Generate sample data
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
                    'color': np.random.choice(['Red', 'Blue', 'Green']),
                    'size': np.random.choice(['Small', 'Medium', 'Large'])
                })
    
    test_data = pd.DataFrame(data)
    print(f"✓ Created test data with shape: {test_data.shape}")
    
    # Test conjoint structure identification
    metadata = {
        'column_labels': {
            'choice': 'Which alternative did you choose?',
            'task_id': 'Task number',
            'alternative_id': 'Alternative number',
            'brand': 'Brand name',
            'price': 'Price level',
            'color': 'Color option',
            'size': 'Size option'
        }
    }
    
    conjoint_structure = mcp._identify_conjoint_structure(test_data, metadata)
    print(f"✓ Identified conjoint structure: {conjoint_structure}")
    
    # Test data preparation
    column_map = {
        'choice': 'choice',
        'task_id': 'task_id', 
        'alternative_id': 'alternative_id',
        'attributes': ['brand', 'price', 'color', 'size']
    }
    
    X, choice_mask = mcp._prepare_conjoint_data(test_data, column_map)
    print(f"✓ Prepared data - X shape: {X.shape}, choice_mask shape: {choice_mask.shape}")
    
    print("✓ All tests passed! The CBC MCP is working correctly.")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc() 