import logging
import os


def get_logger(
    output_dir: str, log_file: str, level: int = logging.INFO
) -> logging.Logger:
    """Create and configure the training logger."""
    logger = logging.getLogger("training_logger")
    logger.setLevel(level)

    # Clear existing handlers to prevent duplicate logs
    logger.handlers.clear()

    # Ensure output directory exists
    log_path = os.path.join(output_dir, log_file)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    # Configure file handler
    handler = logging.FileHandler(log_path)
    handler.setLevel(level)

    # Set formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    return logger
