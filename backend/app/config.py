# backend/app/config.py

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base directory of this file:
BASE_DIR = Path(__file__).parent

# Where to store uploaded files (per user):
UPLOAD_DIR = BASE_DIR.parent / "uploaded_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# API Configuration
PORT = int(os.getenv("PORT", "8000"))
HOST = os.getenv("HOST", "0.0.0.0")
DEBUG = os.getenv("DEBUG", "True").lower() == "true"
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")

# Security
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:8000")

# File Upload Settings
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", "52428800"))  # 50MB in bytes
STATIC_DIR = BASE_DIR.parent / "static"
os.makedirs(STATIC_DIR, exist_ok=True)

# Deepseek (or other) Chat API key:
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "<YOUR_DEESEEK_API_KEY>")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

# Allowed file extensions:
ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls", ".sav"}

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = os.getenv("LOG_FILE", str(BASE_DIR.parent / "logs" / "app.log"))
os.makedirs(Path(LOG_FILE).parent, exist_ok=True)
