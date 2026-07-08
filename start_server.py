#!/usr/bin/env python
"""Start the server and keep it running"""
import subprocess
import sys
import os

# Change to project directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

print("Starting server...")
print("Open http://localhost:8000 in your browser")

# Run the server
result = subprocess.run(
    [sys.executable, "-c", """
import uvicorn
uvicorn.run('app:app', host='127.0.0.1', port=8000)
"""],
    cwd=os.getcwd()
)
