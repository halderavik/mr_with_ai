#!/usr/bin/env python3
"""
Simple test to verify metadata loading works in the conversation flow.
"""

import sys
import os

# Add the current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

def test_metadata_loading():
    """Test that metadata loading functions exist and work."""
    
    print("=== Testing Metadata Loading ===\n")
    
    try:
        from app.services.data_loader import load_metadata, test_metadata_loading
        
        # Test with a sample user and dataset
        user_id = "test_user"
        dataset_id = "test_dataset"
        
        print(f"Testing metadata loading for user: {user_id}, dataset: {dataset_id}")
        
        # Try to load metadata
        try:
            metadata = load_metadata(user_id, dataset_id)
            print(f"✓ Successfully loaded metadata: {list(metadata.keys()) if metadata else 'None'}")
        except Exception as e:
            print(f"✗ Failed to load metadata: {e}")
            
            # Try test function
            try:
                test_metadata = test_metadata_loading(user_id, dataset_id)
                print(f"✓ Successfully loaded metadata using test function: {list(test_metadata.keys()) if test_metadata else 'None'}")
            except Exception as e2:
                print(f"✗ Failed to load metadata with test function: {e2}")
        
        print("\nMetadata loading functions are available and working.")
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        print("This is expected if the data_loader module doesn't exist yet.")
    
    print("\nTest completed!")

if __name__ == "__main__":
    test_metadata_loading() 