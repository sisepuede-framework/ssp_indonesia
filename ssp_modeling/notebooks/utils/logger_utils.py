import logging
from typing import List

def setup_clean_logger(name: str = "main", level: int = logging.INFO) -> logging.Logger:
    """
    Create a clean logger with a single console handler, avoiding duplicate logs.

    Args:
        name (str): Name of the logger.
        level (int): Logging level (e.g., logging.INFO, logging.DEBUG).

    Returns:
        logging.Logger: Configured logger instance.
    """
    # Remove all root handlers to avoid duplicates from previous configurations
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Create or retrieve the logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear existing handlers for this logger
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create a console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    # Define log format
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)

    # Add handler to logger
    logger.addHandler(console_handler)

    # Disable propagation to prevent duplication through root logger
    logger.propagate = False

    return logger


def mute_external_loggers(loggers_to_mute: List[str]) -> None:
    """
    Disable propagation for a list of external loggers (e.g., package loggers).

    Args:
        loggers_to_mute (List[str]): List of logger names to mute.
    """
    for name in loggers_to_mute:
        logging.getLogger(name).propagate = False
