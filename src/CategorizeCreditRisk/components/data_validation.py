import sys
import pandas as pd
from src.CategorizeCreditRisk.logger import logging
from src.CategorizeCreditRisk.exception import CustomException
from src.CategorizeCreditRisk.config.configuration import ConfigurationManager
from src.CategorizeCreditRisk.entity.config_entity import DataValidationConfig


class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config

    def validate_schema(self):
        try:
            logging.info("Validating schema of internal/external raw data files:")

            internal_data_validation_status = True
            external_data_validation_status = True

            internal_data = pd.read_excel(self.config.internal_raw_file)
            internal_data.columns = [col.lower() for col in internal_data.columns]
            internal_data_columns = sorted(internal_data.columns)
            internal_data_schema = sorted(self.config.internal_data_schema.keys())

            external_data = pd.read_excel(self.config.external_raw_file)
            external_data.columns = [col.lower() for col in external_data.columns]
            external_data_columns = sorted(external_data.columns)
            external_data_schema = sorted(self.config.external_data_schema.keys())

            if internal_data_columns != internal_data_schema:
                extra = set(internal_data_columns) - set(internal_data_schema)
                missing = set(internal_data_schema) - set(internal_data_columns)

                logging.error(f"Columns - {extra} are extra!" if extra else f"Columns - {missing} are missing!")

                internal_data_validation_status = False

            with open(self.config.internal_file_val_status, 'w') as f:
                f.write(f"Validation status: {internal_data_validation_status}")

            logging.info(f"Internal Data - Final validation status: {internal_data_validation_status}")


            if external_data_columns != external_data_schema:
                extra = set(external_data_columns) - set(external_data_schema)
                missing = set(external_data_schema) - set(external_data_columns)

                logging.error(f"Columns - {extra} are extra!" if extra else f"Columns - {missing} are missing!")

                external_data_validation_status = False

            with open(self.config.external_file_val_status, 'w') as f:
                f.write(f"Validation status: {external_data_validation_status}")

            logging.info(f"External Data - Final validation status: {external_data_validation_status}")

            return internal_data_validation_status, external_data_validation_status

        except Exception as e:
            logging.error(f"Error occurred while validating schema of internal/external raw data files!")
            raise CustomException(e, sys)


if __name__ == '__main__':
    config_manager = ConfigurationManager()
    data_validation_config = config_manager.get_data_validation_config()
    data_validation = DataValidation(config=data_validation_config)
    data_validation.validate_schema()
