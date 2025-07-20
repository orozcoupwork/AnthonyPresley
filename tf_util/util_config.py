""""
This module provides utility functions for loading environment variables and Airflow variables.

The functions defined in this module are designed to facilitate the retrieval of configuration values
from the system's environment variables or Airflow's variable store. When values are not set or missing,
appropriate exceptions are raised and corresponding errors are logged. Airflow is intended for production,
while OS environment variables are useful for development or adhoc troubleshooting in production.

Functions:
-----------
1. load_env_variable(key: str) -> str:
    Retrieves the value of the specified environment variable. Raises a ValueError if the variable is not set.

2. load_airflow_variable(key: str) -> str:
    Fetches the value of the specified Airflow variable. Raises an AirflowNotFoundException if the variable does not exist.

Dependencies:
--------------
- `os`: Used for interacting with environment variables.
- `logging`: For logging error messages.
- `airflow.exceptions.AirflowNotFoundException`: For indicating missing Airflow variables.
- `airflow.models.Variable`: For accessing Airflow variable values.
"""

import logging
import os

from airflow.exceptions import AirflowNotFoundException
from airflow.models import Variable

logger = logging.getLogger(__name__)

def get_work_dir() -> str:
    """
    Constructs and ensures the existence of a work directory for Airflow tasks. The
    directory is created on the provided base directory and Airflow environment
    context variables: DAG ID, Run ID, Task ID, and Try Number so that each task writes
    its files to a unique directory that does not conflict with any other dag runs, tasks
    or trys/attempts. If the directory does not exist, it is created. Logs the path of the
    created or existing directory.

    For development or non-Airflow environments, the work_directory is loaded from the
    WORK_DIR environment variable

    :return: The full path to the constructed work directory.
    :rtype: str
    """
    if 'AIRFLOW_CTX_DAG_ID' in os.environ:
        base_work_dir = load_airflow_variable("WORK_DIR")
        dag_id = load_env_variable("AIRFLOW_CTX_DAG_ID")
        run_id = load_env_variable("AIRFLOW_CTX_DAG_RUN_ID")
        task_id = load_env_variable("AIRFLOW_CTX_TASK_ID")
        try_number = load_env_variable("AIRFLOW_CTX_TRY_NUMBER")
        work_dir = os.path.join(base_work_dir, dag_id, run_id, task_id, try_number)
        os.makedirs(work_dir, exist_ok=True)
    else:
        work_dir = load_env_variable("WORK_DIR")
    logger.info(f"Work directory: {work_dir}")
    return work_dir

def load_env_variable(key: str) -> str:
    """
    Retrieve the value of an environment variable.

    This function fetches the value of a specified environment variable by its key.
    If the environment variable is not set, an error is logged, and a ValueError
    is raised. The returned value is the corresponding value of the environment
    variable.

    :param key: The name of the environment variable to retrieve.
    :type key: str
    :return: The value of the specified environment variable.
    :rtype: str
    :raises ValueError: If the environment variable is not set.
    """
    value = os.getenv(key)
    if not value:
        logger.error(f"Environment variable {key} is not set")
        raise ValueError(f"Missing environment variable: {key}")
    return value


def load_airflow_variable(key: str) -> str:
    """
    Loads the value of an Airflow variable by its key. If the variable is not set
    in Airflow, it logs an error message and propagates the exception.

    :param key: The key of the Airflow variable to fetch.
    :type key: str
    :return: The value associated with the given key in Airflow variables.
    :rtype: str
    :raises AirflowNotFoundException: If the Airflow variable with the provided
        key is not set.
    """
    try:
        value =  Variable.get(key)
        return value
    except AirflowNotFoundException:
        logger.error(f"Airflow variable {key} is not set")
        raise

def load_variable(key: str) -> str:
    if 'AIRFLOW_CTX_DAG_ID' in os.environ:
        return load_airflow_variable(key)
    else:
        return load_env_variable(key)
