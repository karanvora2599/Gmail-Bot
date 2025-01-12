# logger_config.py

import logging
from logging.handlers import RotatingFileHandler
import os

# Define log directory and file
LOG_DIR = "logs"
LOG_FILE = "app.log"

# Create log directory if it doesn't exist
os.makedirs(LOG_DIR, exist_ok=True)

# Configure the logger
logger = logging.getLogger("pdf_vector_store_manager")
logger.setLevel(logging.DEBUG)  # Set the root logger level to DEBUG

# Formatter to include timestamp, log level, and message
formatter = logging.Formatter(
    fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Console handler for real-time logs
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)  # Set console log level to INFO
console_handler.setFormatter(formatter)

# File handler for persistent logs with rotation
file_handler = RotatingFileHandler(
    filename=os.path.join(LOG_DIR, LOG_FILE),
    maxBytes=5 * 1024 * 1024,  # 5 MB per log file
    backupCount=5,              # Keep up to 5 backup log files
)
file_handler.setLevel(logging.DEBUG)  # File handler captures all levels
file_handler.setFormatter(formatter)

# Add handlers to the logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)