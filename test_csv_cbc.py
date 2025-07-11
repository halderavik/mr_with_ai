#!/usr/bin/env python3
"""
Test script to verify CSV metadata extraction and CBC analysis
"""

import sys
import os
from pathlib import Path

# Add the backend directory to the path
backend_dir = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_dir))

def test_csv_metadata_extraction():
    """Test CSV metadata extraction"""
    print("Testing CSV metadata extraction...")
    
    try:
        from app.services.data_loader import load_metadata, test_metadata_loading
        
        # Test with the uploaded CSV file
        user_id = "user_demo"
        dataset_id = "6c2e7fb3-6f1f-43c1-ba20-2a1cb1ef74aa"
        
        print(f"Testing metadata extraction for {user_id}/{dataset_id}")
        
        # Load metadata
        metadata = test_metadata_loading(user_id, dataset_id)
        
        print("\n✅ CSV metadata extraction successful!")
        print(f"   File type: {metadata.get('file_type', 'unknown')}")
        print(f"   Columns: {len(metadata.get('columns', []))}")
        print(f"   Rows: {metadata.get('row_count', 0)}")
        
        # Check for conjoint structure
        if 'conjoint_structure' in metadata:
            conjoint = metadata['conjoint_structure']
            print(f"   Conjoint structure detected: {conjoint['detected']}")
            if conjoint['detected']:
                print(f"   Structure type: {conjoint['structure_type']}")
                print(f"   Identified columns: {conjoint['identified_columns']}")
        
        return metadata
        
    except Exception as e:
        print(f"❌ Error testing CSV metadata extraction: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_cbc_with_csv():
    """Test CBC analysis with CSV data"""
    print("\nTesting CBC analysis with CSV data...")
    
    try:
        from app.services.data_loader import load_dataset, load_metadata
        from app.mcp.choice_based_conjoint import ChoiceBasedConjointMCP
        
        # Load the CSV data
        user_id = "user_demo"
        dataset_id = "6c2e7fb3-6f1f-43c1-ba20-2a1cb1ef74aa"
        
        print(f"Loading CSV data for {user_id}/{dataset_id}")
        df = load_dataset(user_id, dataset_id)
        metadata = load_metadata(user_id, dataset_id)
        
        print(f"Loaded data shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Initialize CBC MCP
        cbc_mcp = ChoiceBasedConjointMCP()
        
        # Run analysis
        print("\nRunning CBC analysis...")
        result = cbc_mcp.run(df, {
            "metadata": metadata,
            "user_id": user_id,
            "dataset_id": dataset_id
        })
        
        print("\n✅ CBC analysis with CSV data successful!")
        print(f"   Reply: {result.get('reply', 'No reply')}")
        print(f"   Insights: {result.get('insights', 'No insights')[:200]}...")
        
        # Check visualizations
        visualizations = result.get('visualizations', {})
        if visualizations:
            charts = visualizations.get('charts', [])
            tables = visualizations.get('tables', [])
            print(f"   Generated {len(charts)} charts and {len(tables)} tables")
        
        return result
        
    except Exception as e:
        print(f"❌ Error testing CBC analysis with CSV: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Run all CSV tests"""
    print("========================================")
    print("CSV METADATA & CBC ANALYSIS TESTS")
    print("========================================")
    
    # Test metadata extraction
    metadata = test_csv_metadata_extraction()
    
    if metadata:
        # Test CBC analysis
        result = test_cbc_with_csv()
        
        print("\n========================================")
        print("TEST SUMMARY")
        print("========================================")
        
        if result:
            print("🎉 All CSV tests passed!")
            print("   - CSV metadata extraction working")
            print("   - CBC analysis with CSV data working")
            print("   - The system can now handle CSV files properly")
        else:
            print("❌ CBC analysis test failed")
    else:
        print("❌ CSV metadata extraction test failed")

if __name__ == "__main__":
    main() 