"""
Simple startup script for Fast Image Restoration Server
Run this to start the FastAPI server
"""
import subprocess
import sys
import os

# Install dependencies
print("Checking dependencies...")
try:
    import torch
    import fastapi
    import uvicorn
    print("All dependencies already installed!")
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Please run: pip install -r requirements.txt")
    sys.exit(1)

# Change to project directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Start the server
print("\n" + "="*50)
print("Starting Improved Image Restoration Server")
print("="*50)
print("\nOpen your browser and go to: http://localhost:8002")
print("\nPress Ctrl+C to stop the server")
print("="*50 + "\n")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app_final_working:app", host="127.0.0.1", port=8002, reload=False)
