
import logging

def setup_logger(logger: logging.Logger = None,) -> logging.Logger:
    # Add file handler to the logger
    file_handler = logging.FileHandler('{}.log'.format(logger.name))
    file_handler.setLevel(logging.INFO)
    # Add console handler to the logger
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    # Set the logging format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler) 
    logger.setLevel(logging.INFO)

    return logger