import os
import sys
import json
import yaml
import joblib
from typing import Any
from pathlib import Path
from box import ConfigBox
from ensure import ensure_annotations
from box.exceptions import BoxValueError
from src.CategorizeCreditRisk.logger import logging
from src.CategorizeCreditRisk.exception import CustomException


@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """
    Reads yaml file and returns ConfigBox type object

    Args:
        path_to_yaml (str): path like input

    Raises:
        ValueError: if yaml file is empty
        e: empty file

    Returns:
        ConfigBox: ConfigBox typ
    """
    try:
        logging.info(f"Reading yaml file: {path_to_yaml}")

        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)

            logging.info(f"Yaml file read successfully: {path_to_yaml}")

            return ConfigBox(content)

    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        logging.error(f"Error reading yaml file: {path_to_yaml}")
        raise CustomException(e, sys)


@ensure_annotations
def create_directories(path_to_directories: list):
    """
    Create list of directories

    Args:
        path_to_directories (list): list of path of directories

    Returns:
        None
    """
    try:
        for path in path_to_directories:
            if not os.path.exists(path):
                logging.info(f"Creating directory: {path}")

                os.makedirs(path)

                logging.info(f"Directory created successfully: {path}")

            else:
                logging.info(f"Directory already exists: {path}. Skipping creating directory!")

    except Exception as e:
        logging.error(f"Error creating directories: {path_to_directories}")
        raise CustomException(e, sys)


@ensure_annotations
def save_json(path: Path, data: dict):
    """
    Saves json data

    Args:
        path (Path): path to json file
        data (dict): data to be saved in json file

    Returns:
        None
    """
    try:
        logging.info(f"Saving json file to: {path}")

        with open(path, "w") as f:
            json.dump(data, f, indent=4)

        logging.info(f"json file saved at: {path}")

    except Exception as e:
        logging.error(f"Error saving json file to: {path} ")
        raise CustomException(e, sys)


@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """
    Load json file/s data

    Args:
        path (Path): path to json file

    Returns:
        ConfigBox: data as class attributes instead of dict
    """
    try:
        logging.info(f"Loading json file from: {path}")

        with open(path) as f:
            content = json.load(f)

        logging.info(f"json file loaded successfully from: {path}")
        return ConfigBox(content)

    except Exception as e:
        logging.error(f"Error loading json file from: {path}")
        raise CustomException(e, sys)


@ensure_annotations
def save_bin(data: Any, path: Path):
    """
    Saves binary file

    Args:
        data (Any): data to be saved as binary
        path (Path): path to binary file
    """
    try:
        logging.info(f"Saving binary file to: {path}")

        joblib.dump(value=data, filename=path)

        logging.info(f"Binary file saved at: {path}")

    except Exception as e:
        logging.error(f"Error saving binary file to: {path}")
        raise CustomException(e, sys)


@ensure_annotations
def load_bin(path: Path) -> Any:
    """
    Loads binary data

    Args:
        path (Path): path to binary file

    Returns:
        Any: object stored in the file
    """
    try:
        logging.info(f"Loading binary file from: {path}")

        data = joblib.load(path)

        logging.info(f"binary file loaded successfully from: {path}")
        return data

    except Exception as e:
        logging.error(f"Error loading binary file from: {path}")
        raise CustomException(e, sys)


@ensure_annotations
def get_size(path: Path) -> str:
    """
    Get size in KB

    Args:
        path (Path): path of the file

    Returns:
        str: size in KB
    """
    try:
        logging.info(f"Getting size of file: {path}")

        size_in_kb = round(os.path.getsize(path) / 1024)

        return f"~ {size_in_kb} KB"

    except Exception as e:
        logging.error(f"Error getting size of file: {path}")
        raise CustomException(e, sys)