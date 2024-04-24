import sys
from src.CategorizeCreditRisk.logger import logging
from src.CategorizeCreditRisk.exception import CustomException
from src.CategorizeCreditRisk.config.configuration import ConfigurationManager
from src.CategorizeCreditRisk.components.data_validation import DataValidation


STAGE_NAME = "Data Validation"


class DataValidationPipeline:
    def __init__(self):
        pass

    @staticmethod
    def main():
        config_manager = ConfigurationManager()
        data_validation_config = config_manager.get_data_validation_config()
        data_validation = DataValidation(config=data_validation_config)

        data_validation.validate_schema()


if __name__ == '__main__':
    try:
        logging.info(f">>>>>> stage '{STAGE_NAME}' started <<<<<<")

        data_validator = DataValidationPipeline()
        data_validator.main()

        logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<")

    except Exception as e:
        logging.error(f"Error occurred while running {STAGE_NAME}!")
        raise CustomException(e, sys)
