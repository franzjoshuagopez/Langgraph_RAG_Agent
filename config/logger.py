import logging
import sys
from pathlib import Path

#sets path to project root folder
LOG_FILE = Path(__file__).resolve().parent.parent / "agent.log"

def get_logger(name : str) -> logging.Logger:
    """
        returns a logger instance/object that handles both console logging and file logging.
    """
    logger = logging.getLogger(name)
    #this prevents adding multiple handlers
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        
        #console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')) #format of log in this instance e.g. 2025-09-06 16:00:00,123 [INFO] Some logging text

        #File handler
        file_handler = logging.FileHandler(LOG_FILE, mode='w') #mode='w' is overwriting file so that logging is per run not the overall life of the application spanning multiple runs
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')) #format of log in this instance e.g. 2025-09-06 16:00:00,123 [INFO] Some logging text

        logger.addHandler(console_handler)
        logger.addHandler(file_handler)


    return logger