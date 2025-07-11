#!/usr/bin/env python3
# -*- coding: utf-8 -*-

try:
    from app.mcp.van_westendorp import VanWestendorpMCP
    print("Import successful!")
    
    # Test creating an instance
    mcp = VanWestendorpMCP()
    print(f"MCP name: {mcp.name}")
    print(f"MCP description: {mcp.description}")
    
    # Test the run method
    import pandas as pd
    test_data = pd.DataFrame({
        'too_cheap': [10, 15, 20],
        'too_expensive': [50, 60, 70],
        'bargain': [25, 30, 35],
        'getting_expensive': [40, 45, 50]
    })
    
    result = mcp.run(test_data)
    print(f"Run result: {result}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc() 