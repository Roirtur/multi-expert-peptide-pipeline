import logging
import sys
import os
from typing import Optional
from logging.handlers import RotatingFileHandler
from datetime import datetime

_DEFAULT_FORMATTER = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
_logger_name = "peptide_pipeline"

class ColoredFormatter(logging.Formatter):
    """
    Custom formatter to add colors to the log output based on the log level.
    """
    # ANSI escape codes
    RESET = "\033[0m"
    RED_BOLD = "\033[1;31m"
    YELLOW = "\033[33m"
    GREEN = "\033[32m"
    BLUE = "\033[34m"

    FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    FORMATS = {
        logging.DEBUG: BLUE + FORMAT + RESET,
        logging.INFO: GREEN + FORMAT + RESET,
        logging.WARNING: YELLOW + FORMAT + RESET,
        logging.ERROR: RED_BOLD + FORMAT + RESET,
        logging.CRITICAL: RED_BOLD + FORMAT + RESET,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno, self.FORMAT)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

def get_logger(name: str = _logger_name, level: int = logging.DEBUG, log_file: Optional[str] = None) -> logging.Logger:
    """
    Get a configured logger instance.
    Logs to console (with colors) and to file (defaulting to a timestamped file in logs/).
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(ColoredFormatter())
        logger.addHandler(console_handler)

        if not log_file:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            log_file = os.path.join("logs", f"{timestamp}_pipeline.log")

        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        file_handler = RotatingFileHandler(log_file, maxBytes=10_000_000, backupCount=5)
        file_handler.setLevel(level)
        file_handler.setFormatter(_DEFAULT_FORMATTER)
        logger.addHandler(file_handler)

    return logger

def set_level(level: int) -> None:
    """
    Set the logging level for the default logger and its handlers.
    """
    logger = logging.getLogger(_logger_name)
    logger.setLevel(level)
    for handler in logger.handlers:
        handler.setLevel(level)
