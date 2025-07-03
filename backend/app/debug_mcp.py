#!/usr/bin/env python3

import os
import sys
import importlib
import inspect

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from app.services.mcp_base import MCPBase
    print("✅ MCPBase imported successfully")
except Exception as e:
    print(f"❌ Failed to import MCPBase: {e}")

# Test MCP discovery
mcp_folder = os.path.join(os.path.dirname(__file__), "mcp")
print(f"MCP folder: {mcp_folder}")
print(f"Contents: {os.listdir(mcp_folder)}")

for filename in os.listdir(mcp_folder):
    if filename.endswith(".py") and not filename.startswith("__"):
        mod_name = filename[:-3]
        print(f"\nProcessing: {mod_name}")
        
        try:
            # Try direct import
            mod_path = os.path.join(mcp_folder, filename)
            with open(mod_path, 'r', encoding='utf-8') as f:
                mod_code = f.read()
            
            # Create a namespace for the module
            mod_globals = {
                '__name__': f'app.mcp.{mod_name}',
                '__file__': mod_path,
                'MCPBase': MCPBase
            }
            
            # Execute the module code
            exec(mod_code, mod_globals)
            
            # Find MCP classes
            for name, obj in inspect.getmembers(type('module', (), mod_globals), inspect.isclass):
                print(f"  Found class: {name}")
                if issubclass(obj, MCPBase) and obj is not MCPBase:
                    print(f"  ✅ Found MCP class: {name}")
                    # Convert to key
                    key = name.replace("MCP", "").lower()
                    key = ''.join(['_' + c.lower() if c.isupper() else c for c in key]).lstrip('_')
                    print(f"  Key: {key}")
                    
        except Exception as e:
            print(f"  ❌ Error processing {mod_name}: {e}")
            import traceback
            traceback.print_exc() 