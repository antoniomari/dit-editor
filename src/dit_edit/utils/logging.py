import logging


def setup_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)

    # Add file handler to the logger
    file_handler = logging.FileHandler(f"{name}.log")
    file_handler.setLevel(logging.INFO)
    # Add console handler to the logger
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    # Set the logging format
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)

    return logger
