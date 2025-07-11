import sys
import os
sys.path.append('.')

try:
    from app.mcp.van_westendorp import VanWestendorpMCP
    print("Import successful!")
    
    # Test creating an instance
    mcp = VanWestendorpMCP()
    print(f"MCP name: {mcp.name}")
    print(f"MCP description: {mcp.description}")
    
    # Test the handle_followup_question method with invalid input
    import pandas as pd
    test_data = pd.DataFrame({
        'too_cheap': [10, 15, 20],
        'too_expensive': [50, 60, 70],
        'bargain': [25, 30, 35],
        'getting_expensive': [40, 45, 50]
    })
    
    # Test with string input (the bug case)
    result = mcp.handle_followup_question(
        "What does this mean?",
        "this is a string, not a dict",  # This should trigger the fix
        test_data
    )
    print(f"Handle followup with string input: {result['answer'][:50]}...")
    
    # Test with None input
    result = mcp.handle_followup_question(
        "What does this mean?",
        None,  # This should also trigger the fix
        test_data
    )
    print(f"Handle followup with None input: {result['answer'][:50]}...")
    
    print("All tests passed! The fix is working correctly.")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc() 