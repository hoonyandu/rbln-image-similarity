import logging


def setup_logger(name: str = None) -> logging.Logger:
    """
    Setup logger
    Args:
        name: logger name

    Returns:
        logger: logger
    """
    logger = logging.getLogger(name) if name else logging.getLogger(__name__)

    # 이미 핸들러가 있으면 반환
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    # 핸들러 추가
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    return logger


def get_logger(name: str = None) -> logging.Logger:
    """
    Get logger
    Args:
        name: logger name

    Returns:
        logger: logger
    """
    logger = logging.getLogger(name)
    logger = logger if logger.handlers else setup_logger(name)

    return logger


def set_log_level(logger: logging.Logger, level: int) -> None:
    """
    Set log level
    Args:
        logger: logger
        level: log level
    """
    logger.setLevel(level)

    for handler in logger.handlers:
        handler.setLevel(level)
