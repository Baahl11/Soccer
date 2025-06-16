"""Configure logging with proper UTF-8 encoding."""
import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler

def setup_logging():
    """Setup logging with UTF-8 encoding and proper formatting."""
    # Create logs directory if it doesn't exist
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    # Configure the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Create formatters    
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Create and configure console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    console_handler.stream.reconfigure(encoding='utf-8')

    # Create and configure file handler with rotation
    file_handler = RotatingFileHandler(
        'logs/app.log',
        maxBytes=10485760,  # 10MB
        backupCount=5, 
        encoding='utf-8'
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(file_formatter)

    # Remove any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Add new handlers
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    # Configure specific loggers
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)

    # Initialize logger for this module
    logger = logging.getLogger(__name__)
    logger.info("🔧 Logging system initialized with UTF-8 encoding")
