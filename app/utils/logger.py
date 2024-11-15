# logger.py

import logging


# Set up a logger for the entire application
def setup_logger():
    logger = logging.getLogger(__name__)  # Use module name for the logger
    logger.setLevel(logging.DEBUG)  # You can change this to INFO or WARNING for less verbosity

    # Create a console handler and set the logging level
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    # Create a file handler for logging to a file (optional)
    file_handler = logging.FileHandler('app.log')
    file_handler.setLevel(logging.INFO)

    # Create a formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


# Initialize logger
logger = setup_logger()
