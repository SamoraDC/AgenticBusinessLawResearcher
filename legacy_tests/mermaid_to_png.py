#!/usr/bin/env python3
"""
Convert Legal AI Workflow Mermaid diagram to PNG
Creates a high-quality PNG image from the Mermaid diagram
"""

import os
import sys
import subprocess
import tempfile
from pathlib import Path

def extract_mermaid_from_md(md_file: str) -> str:
    """Extract Mermaid code from markdown file."""
    with open(md_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find mermaid code block
    start_marker = "```mermaid"
    end_marker = "```"
    
    start_idx = content.find(start_marker)
    if start_idx == -1:
        raise ValueError("No mermaid code block found in the markdown file")
    
    start_idx += len(start_marker)
    end_idx = content.find(end_marker, start_idx)
    
    if end_idx == -1:
        raise ValueError("Unclosed mermaid code block")
    
    mermaid_code = content[start_idx:end_idx].strip()
    return mermaid_code

def create_mermaid_file(mermaid_code: str, output_file: str):
    """Create a standalone .mmd file with the Mermaid code."""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(mermaid_code)

def convert_with_mermaid_cli(mermaid_file: str, png_file: str) -> bool:
    """Convert using Mermaid CLI (mmdc)."""
    try:
        # Check if mmdc is installed
        result = subprocess.run(['mmdc', '--version'], 
                              capture_output=True, text=True)
        if result.returncode != 0:
            return False
        
        print(f"📦 Found Mermaid CLI version: {result.stdout.strip()}")
        
        # Convert to PNG with high quality settings
        cmd = [
            'mmdc',
            '-i', mermaid_file,
            '-o', png_file,
            '-t', 'default',
            '-w', '1920',  # Width
            '-H', '1080',  # Height
            '-b', 'white', # Background
            '--scale', '2' # High DPI
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ Successfully converted to PNG using Mermaid CLI")
            return True
        else:
            print(f"❌ Mermaid CLI error: {result.stderr}")
            return False
            
    except FileNotFoundError:
        print("⚠️ Mermaid CLI (mmdc) not found")
        return False
    except Exception as e:
        print(f"❌ Error with Mermaid CLI: {e}")
        return False

def convert_with_playwright(mermaid_code: str, png_file: str) -> bool:
    """Convert using Playwright browser automation."""
    try:
        from playwright.sync_api import sync_playwright
        
        print("🌐 Using Playwright browser method...")
        
        # Create HTML with Mermaid
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <script src="https://cdn.jsdelivr.net/npm/mermaid@10.6.1/dist/mermaid.min.js"></script>
    <style>
        body {{
            margin: 0;
            padding: 20px;
            background: white;
            font-family: 'Segoe UI', sans-serif;
        }}
        .mermaid {{
            text-align: center;
        }}
    </style>
</head>
<body>
    <div class="mermaid">
{mermaid_code}
    </div>
    <script>
        mermaid.initialize({{
            startOnLoad: true,
            theme: 'default',
            flowchart: {{
                useMaxWidth: false,
                htmlLabels: true,
                curve: 'basis'
            }},
            themeVariables: {{
                primaryColor: '#667eea',
                primaryTextColor: '#333',
                primaryBorderColor: '#764ba2',
                lineColor: '#666'
            }}
        }});
    </script>
</body>
</html>
"""
        
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(viewport={'width': 1920, 'height': 1080})
            page.set_content(html_content)
            
            # Wait for Mermaid to render
            page.wait_for_timeout(3000)
            
            # Take screenshot of the diagram
            mermaid_element = page.locator('.mermaid svg')
            mermaid_element.screenshot(path=png_file)
            
            browser.close()
            
        print(f"✅ Successfully converted to PNG using Playwright")
        return True
        
    except ImportError:
        print("⚠️ Playwright not installed")
        return False
    except Exception as e:
        print(f"❌ Error with Playwright: {e}")
        return False

def convert_with_python_graphviz(mermaid_code: str, png_file: str) -> bool:
    """Convert using Python libraries (basic fallback)."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        from matplotlib.patches import FancyBboxPatch
        
        print("🐍 Using Python matplotlib fallback...")
        
        # Create a basic visual representation
        fig, ax = plt.subplots(figsize=(16, 12))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
        
        # Add title
        ax.text(5, 9.5, 'Legal AI System - Workflow Diagram', 
               fontsize=20, fontweight='bold', ha='center')
        
        # Add a note about the diagram
        note = """
This is a simplified representation of the Legal AI Workflow.
For the full interactive diagram, please use:
• legal_ai_workflow_visualization.html (browser)
• legal_ai_workflow.md (Mermaid code)
• Install Mermaid CLI for full PNG generation
        """
        
        ax.text(5, 5, note, fontsize=12, ha='center', va='center',
               bbox=dict(boxstyle="round,pad=1", facecolor='lightblue', alpha=0.8))
        
        # Add workflow steps
        steps = [
            "1. User Query Input",
            "2. CRAG Document Retrieval", 
            "3. Document Grading",
            "4. LexML Jurisprudence Search",
            "5. Hybrid AI Processing",
            "6. Quality Validation",
            "7. Final Response"
        ]
        
        for i, step in enumerate(steps):
            y_pos = 8 - i * 0.8
            ax.text(5, y_pos, step, fontsize=11, ha='center',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.7))
        
        plt.tight_layout()
        plt.savefig(png_file, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"✅ Created basic diagram using matplotlib")
        return True
        
    except ImportError:
        print("⚠️ Matplotlib not available")
        return False
    except Exception as e:
        print(f"❌ Error with matplotlib: {e}")
        return False

def main():
    """Main function to convert Mermaid to PNG."""
    input_file = "legal_ai_workflow.md"
    output_file = "legal_ai_workflow.png"
    
    print("🚀 Converting Legal AI Workflow Mermaid to PNG...")
    print("=" * 60)
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"❌ Input file not found: {input_file}")
        return False
    
    try:
        # Extract Mermaid code
        print(f"📄 Reading Mermaid code from {input_file}...")
        mermaid_code = extract_mermaid_from_md(input_file)
        print(f"✅ Extracted {len(mermaid_code)} characters of Mermaid code")
        
        # Try different conversion methods
        success = False
        
        # Method 1: Mermaid CLI (best quality)
        if not success:
            temp_mmd = "temp_workflow.mmd"
            create_mermaid_file(mermaid_code, temp_mmd)
            success = convert_with_mermaid_cli(temp_mmd, output_file)
            if os.path.exists(temp_mmd):
                os.remove(temp_mmd)
        
        # Method 2: Playwright (good quality)
        if not success:
            success = convert_with_playwright(mermaid_code, output_file)
        
        # Method 3: Python fallback (basic)
        if not success:
            success = convert_with_python_graphviz(mermaid_code, output_file)
        
        if success and os.path.exists(output_file):
            file_size = os.path.getsize(output_file)
            print(f"\n✅ PNG created successfully!")
            print(f"📁 Output file: {os.path.abspath(output_file)}")
            print(f"📊 File size: {file_size:,} bytes")
            
            print(f"\n🎯 Installation recommendations for better quality:")
            print(f"1. Install Mermaid CLI: npm install -g @mermaid-js/mermaid-cli")
            print(f"2. Install Playwright: pip install playwright && playwright install")
            
            return True
        else:
            print(f"\n❌ Failed to create PNG with all available methods")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 