import sys
import os

# Ensure the project root directory is in the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pipeline.main import app

if __name__ == "__main__":
    import uvicorn
    # Standardized to IPv4 loopback to resolve macOS localhost IPv6 proxy issues
    uvicorn.run("pipeline.main:app", host="127.0.0.1", port=8011, reload=True)
