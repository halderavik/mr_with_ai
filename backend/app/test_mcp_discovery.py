#!/usr/bin/env python3

import sys
import os

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from agent_controller import AgentController
    
    print("Creating AgentController...")
    ac = AgentController()
    
    print(f"Available MCPs: {list(ac.mcp_mapping.keys())}")
    
    # Test van westendorp detection
    test_message = "run van westendorp analysis"
    analysis_info = ac._extract_analysis_type_with_deepseek(test_message)
    
    print(f"Test message: {test_message}")
    print(f"Detected analysis type: {analysis_info['analysis_type']}")
    
    if analysis_info['analysis_type'] in ac.mcp_mapping:
        print(f"✅ Analysis type '{analysis_info['analysis_type']}' found in MCP mapping!")
    else:
        print(f"❌ Analysis type '{analysis_info['analysis_type']}' NOT found in MCP mapping!")
        print(f"Available keys: {list(ac.mcp_mapping.keys())}")
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc() 