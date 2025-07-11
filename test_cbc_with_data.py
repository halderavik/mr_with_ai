#!/usr/bin/env python3
"""
Test script to run CBC analysis with the provided cbc_choices.csv data
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Add the backend directory to the path
backend_dir = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_dir))

# Import the CBC MCP
from app.mcp.choice_based_conjoint import ChoiceBasedConjointMCP

def load_cbc_data():
    """Load the CBC choices data from CSV"""
    try:
        # Load the data
        df = pd.read_csv('cbc_choices.csv')
        print(f"Loaded data with shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"First few rows:")
        print(df.head())
        
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def run_cbc_analysis():
    """Run the CBC analysis with the loaded data"""
    print("=" * 60)
    print("CBC ANALYSIS WITH PROVIDED DATA")
    print("=" * 60)
    
    # Load the data
    data = load_cbc_data()
    if data is None:
        print("Failed to load data. Exiting.")
        return
    
    # Create the CBC MCP instance
    cbc_mcp = ChoiceBasedConjointMCP()
    
    # Prepare the data for analysis
    print("\n" + "=" * 40)
    print("PREPARING DATA FOR CBC ANALYSIS")
    print("=" * 40)
    
    try:
        # Convert the data to the format expected by the MCP
        # The data has Respondent_ID and Task_1 through Task_15
        # We need to reshape this for the analysis
        
        # Extract choice data (excluding Respondent_ID)
        choice_columns = [col for col in data.columns if col.startswith('Task_')]
        choice_data = data[choice_columns].values
        
        # Create a simple design matrix for testing
        # In a real scenario, this would come from the experimental design
        n_respondents = len(data)
        n_tasks = len(choice_columns)
        n_alternatives = 5  # Based on the data (0-4, where 0 is no choice)
        n_features = 3  # Example: price, brand, quality
        
        # Create a simple design matrix
        # This is a placeholder - in reality, this would be the experimental design
        np.random.seed(42)  # For reproducibility
        design_matrix = np.random.rand(n_respondents, n_tasks, n_alternatives, n_features)
        
        # Create choice mask (indicating which alternatives were available)
        choice_mask = np.ones((n_respondents, n_tasks, n_alternatives), dtype=bool)
        # Set no-choice option (index 0) as always available
        choice_mask[:, :, 0] = True
        
        print(f"Data prepared:")
        print(f"  - Respondents: {n_respondents}")
        print(f"  - Tasks: {n_tasks}")
        print(f"  - Alternatives: {n_alternatives}")
        print(f"  - Features: {n_features}")
        print(f"  - Choice data shape: {choice_data.shape}")
        print(f"  - Design matrix shape: {design_matrix.shape}")
        print(f"  - Choice mask shape: {choice_mask.shape}")
        
        # Create the analysis input
        analysis_input = {
            'design_matrix': design_matrix,
            'choice_data': choice_data,
            'choice_mask': choice_mask,
            'feature_names': ['Price', 'Brand', 'Quality'],
            'alternative_names': ['No Choice', 'Alternative 1', 'Alternative 2', 'Alternative 3', 'Alternative 4']
        }
        
        print("\n" + "=" * 40)
        print("RUNNING CBC ANALYSIS")
        print("=" * 40)
        
        # Run the analysis
        result = cbc_mcp.run_analysis(analysis_input)
        
        print("\n" + "=" * 40)
        print("ANALYSIS RESULTS")
        print("=" * 40)
        
        if result:
            print("Analysis completed successfully!")
            print(f"Result keys: {list(result.keys())}")
            
            # Print some key results
            if 'insights' in result:
                print("\nKey Insights:")
                for insight in result['insights'][:5]:  # Show first 5 insights
                    print(f"  - {insight}")
            
            if 'visualizations' in result:
                print(f"\nGenerated {len(result['visualizations'])} visualizations")
                for i, viz in enumerate(result['visualizations']):
                    print(f"  {i+1}. {viz.get('title', 'Untitled')}")
            
            if 'segments' in result:
                print(f"\nFound {len(result['segments'])} segments")
                for i, segment in enumerate(result['segments']):
                    print(f"  Segment {i+1}: {segment.get('description', 'No description')}")
            
        else:
            print("Analysis failed or returned no results.")
            
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_cbc_analysis() 