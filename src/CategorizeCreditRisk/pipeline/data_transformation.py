import sys
from src.CategorizeCreditRisk.logger import logging
from src.CategorizeCreditRisk.exception import CustomException
from src.CategorizeCreditRisk.config.configuration import ConfigurationManager
from src.CategorizeCreditRisk.components.data_transformation import DataTransformation


STAGE_NAME = "Data Transformation"


class DataTransformationPipeline:
    def __init__(self):
        pass

    @staticmethod
    def main():
        config_manager = ConfigurationManager()
        data_transformation_config = config_manager.get_data_transformer_config()
        data_transformation = DataTransformation(config=data_transformation_config)

        data_transformation.get_transformed_data()


if __name__ == '__main__':
    try:
        logging.info(f">>>>>> stage '{STAGE_NAME}' started <<<<<<")

        data_transformer = DataTransformationPipeline()
        data_transformer.main()

        logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<")

    except Exception as e:
        logging.error(f"Error occurred while running {STAGE_NAME}!")
        raise CustomException(e, sys)