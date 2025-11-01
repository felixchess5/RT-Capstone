#!/usr/bin/env python3
"""
Simple launcher for the Intelligent-Assignment-Grading-System Gradio Web Interface.

Run this script to start the web interface for assignment grading.
"""

import os
import sys

# Add src to Python path
src_path = os.path.join(os.path.dirname(__file__), 'src')
sys.path.insert(0, src_path)

# Import and run the Gradio app
try:
    from gradio_app import main

    if __name__ == "__main__":
        print("🎓 Intelligent-Assignment-Grading-System Assignment Grading System")
        print("=" * 50)
        print("Starting Gradio web interface...")
        print("📱 The interface will open in your browser automatically")
        print("🌐 Default URL: http://localhost:7860")
        print("⏹️  Press Ctrl+C to stop the server")
        print("=" * 50)

        main()

except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're in the correct directory and have installed dependencies:")
    print("pip install -r requirements.txt")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error starting Gradio interface: {e}")
    sys.exit(1)
