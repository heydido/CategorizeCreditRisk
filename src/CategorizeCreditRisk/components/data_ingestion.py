import os
import sys
import zipfile
import requests
from pathlib import Path
from src.CategorizeCreditRisk.logger import logging
from src.CategorizeCreditRisk.exception import CustomException
from src.CategorizeCreditRisk.utils.common import get_size
from src.CategorizeCreditRisk.config.configuration import ConfigurationManager
from src.CategorizeCreditRisk.entity.config_entity import DataIngestionConfig


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self):
        try:
            if not os.path.exists(self.config.data_file):
                logging.info("Downloading data file from source URL:")

                response = requests.get(self.config.source_URL)
                if response.status_code == 200:
                    with open(self.config.data_file, 'wb') as f:
                        f.write(response.content)
                    logging.info(f"Data file downloaded and saved as: {self.config.data_file}")
                else:
                    logging.error(f"Failed to download data file, status code: {response.status_code}")

            else:
                logging.info(f"Data file already exists of size: {get_size(Path(self.config.data_file))}")

        except Exception as e:
            logging.error(f"Error occurred while downloading data file!")
            raise CustomException(e, sys)

    def unzip_file(self):
        try:
            os.makedirs(self.config.unzip_dir, exist_ok=True)

            if not os.path.exists(self.config.raw_dataset):
                logging.info("Unzipping data file:")

                with zipfile.ZipFile(self.config.data_file, 'r') as zip_ref:
                    zip_ref.extractall(self.config.unzip_dir)

                logging.info(f"Data file unzipped successfully at: {self.config.unzip_dir}")

            else:
                logging.info(f"Raw (unzipped) file already present at: {self.config.raw_dataset}")

        except Exception as e:
            logging.error(f"Error occurred while unzipping data file!")
            raise CustomException(e, sys)


if __name__ == "__main__":
    config_manager = ConfigurationManager()
    data_ingestion_config = config_manager.get_data_ingestion_config()

    data_ingestion = DataIngestion(config=data_ingestion_config)
    data_ingestion.download_file()
    data_ingestion.unzip_file()
