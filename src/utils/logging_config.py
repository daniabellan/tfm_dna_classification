# logging_config.py
from loguru import logger
from pathlib import Path
import time

# =============================
# Log Configuration
# =============================

# Create a directory for logs if it doesn't exist
log_dir = "logs"
Path(log_dir).mkdir(exist_ok=True)

# Define a persistent log file
log_file = Path(log_dir) / f"training_{time.strftime('%Y-%m-%d_%H-%M-%S')}.log"

# Remove default logger and add custom file & console logging
logger.remove()
logger.add(
    log_file, 
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    level="INFO",
    rotation="10 MB",  # Auto-create a new log file if it exceeds 10MB
    compression="zip"  # Optional: Compress old logs
)

# Expose log file path for MLflow and local saving
def get_log_file():
    return log_file

logger.info("🚀 Global logger initialized. Logs saved to disk and MLflow.")
