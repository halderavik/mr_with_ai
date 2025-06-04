# backend/app/config.py

import os
from pathlib import Path

# Base directory of this file:
BASE_DIR = Path(__file__).parent

# Where to store uploaded files (per user):
UPLOAD_DIR = BASE_DIR.parent / "uploaded_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Deepseek (or other) Chat API key:
# You can set this environment variable before launching uvicorn
DEESEEK_API_KEY = os.getenv("DEESEEK_API_KEY", "<YOUR_DEESEEK_API_KEY>")

# Maximum file size in bytes (e.g. 50 MB):
MAX_FILE_SIZE = 50 * 1024 * 1024

# Allowed file extensions:
ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls", ".sav"}
