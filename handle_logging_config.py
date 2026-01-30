# System imports
from datetime import datetime
import logging
import os
import pathlib
import sys


#logging info in file logger.info(f"temp_desc:\n" f"{temp_desc}")
#call this file before
def setup_logging(
    script_name: str,
) -> logging.Logger:
    """
    Sets up logging for a script.
    This will log to a file in the 'logs' directory and to stdout.

    Args:
        script_name (str):
            Name of the script to create a log file for.

    Returns:
        logging.Logger: Configured logger instance.
    """
    # Format the current date and time as a string
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    # Define log file path
    logfile_path = pathlib.Path(
        "logs",
        f"{script_name}_{timestamp}.log",
    )

    # Create log directory if it does not exist
    logfile_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Basic configuration for logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[
            logging.FileHandler(logfile_path),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )

    # Get logger
    logger = logging.getLogger(script_name)
    logger.info(
        f"Logger initialised for file logfile_path:\n" f"{logfile_path}\nand stdout"
    )


    return logger
