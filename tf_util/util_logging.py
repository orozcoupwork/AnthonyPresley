import configparser
import logging
import logging.config
import os

def init_logging(name):
    if "AIRFLOW_CONTEXT" not in os.environ:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(name)
    return logger

