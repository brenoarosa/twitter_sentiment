import logging
import sys
import time


def get_logger():
    """
    Return configured logger
    """
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
    return logger

def download_data():
    pass

def upload_data():
    pass

def download_models():
    pass

def upload_models():
    pass
