#!/usr/bin/env python3
"""
Update the workflow PNG with the new enhanced visualization.
This script uses the existing workflow_builder to generate an updated crag_graph.png
"""

import os
import sys
from datetime import datetime

# Add src to path to import our modules
sys.path.append('src')

try:
    from src.core.workflow_builder import build_graph
    print("✅ Successfully imported workflow_builder")
except ImportError as e:
    print(f"❌ Failed to import workflow_builder: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)

def main():
    """Generate updated workflow PNG."""
    print("🚀 Updating Legal AI System Workflow PNG...")
    print("=" * 60)
    
    try:
        # Build the graph (this will automatically generate crag_graph.png)
        print("🔧 Building LangGraph workflow...")
        graph_app = build_graph()
        
        # Check if PNG was created
        if os.path.exists("crag_graph.png"):
            print("✅ Updated crag_graph.png successfully!")
            
            # Get file info
            file_size = os.path.getsize("crag_graph.png")
            print(f"📊 File size: {file_size:,} bytes")
            
            # Rename old PNG if it exists
            backup_name = f"crag_graph_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            if os.path.exists("crag_graph_old.png"):
                os.rename("crag_graph_old.png", backup_name)
                print(f"📁 Backed up old PNG as: {backup_name}")
        else:
            print("⚠️ PNG file was not generated. Check if pygraphviz is installed:")
            print("   pip install pygraphviz")
            print("   or")
            print("   uv add pygraphviz")
            
    except Exception as e:
        print(f"❌ Error generating workflow PNG: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Make sure pygraphviz is installed: pip install pygraphviz")
        print("2. Ensure Graphviz is installed on your system")
        print("3. Check that all dependencies are available")
        return False
    
    print("\n" + "=" * 60)
    print("✅ Workflow PNG update completed!")
    print(f"📁 Location: {os.path.abspath('crag_graph.png')}")
    print("🎯 This PNG now shows the complete CRAG + Hybrid system workflow")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 