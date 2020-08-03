import logging
import sys
import time

_logger_instance = None

def identity(x):
    return x

def setup_logger():
    global _logger_instance

    logger = logging.getLogger("twitter_sentiment")
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s %(name)s [%(levelname).5s]: %(message)s')

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.INFO)
    stdout_handler.setFormatter(formatter)

    filename = f"/tmp/twitter_sentiment_{int(time.time())}.log"
    file_handler = logging.FileHandler(filename)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.addHandler(stdout_handler)
    logger.addHandler(file_handler)
    _logger_instance = logger

def get_logger():
    """
    Return configured logger
    """
    global _logger_instance

    if not _logger_instance:
        setup_logger()

    return _logger_instance

def download_data():
    pass

def upload_data():
    pass

def download_models():
    pass

def upload_models():
    pass
