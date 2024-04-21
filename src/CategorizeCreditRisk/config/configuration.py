import sys
from src.CategorizeCreditRisk.constants import *
from src.CategorizeCreditRisk.logger import logging
from src.CategorizeCreditRisk.exception import CustomException
from src.CategorizeCreditRisk.entity.config_entity import (DataIngestionConfig,
                                                           DataValidationConfig,
                                                           DataPreprocessingConfig,
                                                           DataTransformationConfig,
                                                           ModelTrainingConfig,
                                                           ModelEvaluationConfig,
                                                           PredictionConfig)
from src.CategorizeCreditRisk.utils.common import read_yaml, create_directories


class ConfigurationManager:
    def __init__(self,
                 config_filepath=CONFIG_FILE_PATH,
                 internal_raw_data_schema_filepath=INTERNAL_RAW_DATA_SCHEMA_FILE_PATH,
                 external_raw_data_schema_filepath=EXTERNAL_RAW_DATA_SCHEMA_FILE_PATH,
                 processed_data_schema_filepath=PROCESSED_DATA_SCHEMA_FILE_PATH,
                 params_filepath=PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.internal_raw_data_schema = read_yaml(internal_raw_data_schema_filepath)
        self.external_raw_data_schema = read_yaml(external_raw_data_schema_filepath)
        self.processed_data_schema = read_yaml(processed_data_schema_filepath)
        self.params = read_yaml(params_filepath)

        # Artifact root directory
        create_directories([self.config.root.artifact])

    def get_data_ingestion_config(self, log=True) -> DataIngestionConfig:
        try:
            if log:
                logging.info("Getting data ingestion configuration:")

            config = self.config.data_ingestion

            create_directories([config.root_dir])

            data_ingestion_config = DataIngestionConfig(
                root_dir=config.root_dir,
                source_URL=config.source_URL,
                data_file=config.data_file,
                unzip_dir=config.unzip_dir,
                internal_raw_file=config.internal_raw_file,
                external_raw_file=config.external_raw_file
            )

            if log:
                logging.info("Data ingestion configuration loaded successfully!")

            return data_ingestion_config

        except Exception as e:
            if log:
                logging.error(f"Error occurred while getting data ingestion configuration!")
            raise CustomException(e, sys)

    def get_data_validation_config(self, log=True) -> DataValidationConfig:
        try:
            if log:
                logging.info("Getting data validation configuration:")

            config = self.config.data_validation
            internal_raw_data_schema = self.internal_raw_data_schema.features
            external_raw_data_schema = self.external_raw_data_schema.features

            create_directories([config.root_dir])

            data_validation_config = DataValidationConfig(
                root_dir=config.root_dir,
                internal_raw_file=config.internal_raw_file,
                external_raw_file=config.external_raw_file,
                internal_file_val_status=config.internal_file_val_status,
                external_file_val_status=config.external_file_val_status,
                internal_data_schema=internal_raw_data_schema,
                external_data_schema=external_raw_data_schema
            )

            if log:
                logging.info("Data validation configuration loaded successfully!")

            return data_validation_config

        except Exception as e:
            if log:
                logging.error(f"Error occurred while getting data validation configuration!")
            raise CustomException(e, sys)

    def get_data_preprocessing_config(self, log=True) -> DataPreprocessingConfig:
        try:
            if log:
                logging.info("Getting data preprocessing configuration:")

            config = self.config.data_preprocessing
            cat_features = self.processed_data_schema.cat_features
            num_features = self.processed_data_schema.num_features
            target_variable = self.processed_data_schema.target_variable

            create_directories([config.root_dir])

            data_preprocessing_config = DataPreprocessingConfig(
                root_dir=config.root_dir,
                internal_raw_file=config.internal_raw_file,
                external_raw_file=config.external_raw_file,
                cleaned_raw_dataset=config.cleaned_raw_dataset,
                preprocessed_dataset=config.preprocessed_dataset,
                cat_features=cat_features,
                num_features=num_features,
                target_variable=target_variable
            )

            if log:
                logging.info("Data preprocessing configuration loaded successfully!")

            return data_preprocessing_config

        except Exception as e:
            if log:
                logging.error(f"Error occurred while getting data preprocessing configuration!")
            raise CustomException(e, sys)

    def get_data_transformer_config(self, log=True) -> DataTransformationConfig:
        try:
            if log:
                logging.info("Getting data transformation configuration:")

            config = self.config.data_transformation
            cat_features = self.processed_data_schema.cat_features
            num_features = self.processed_data_schema.num_features
            target_variable = self.processed_data_schema.target_variable

            create_directories([config.root_dir])

            data_transformation_config = DataTransformationConfig(
                root_dir=config.root_dir,
                preprocessed_dataset=config.preprocessed_dataset,
                cat_features=cat_features,
                num_features=num_features,
                target_variable=target_variable,
                data_transformer=config.data_transformer
            )

            if log:
                logging.info("Data transformation configuration loaded successfully!")

            return data_transformation_config

        except Exception as e:
            if log:
                logging.error(f"Error occurred while getting data transformation configuration!")
            raise CustomException(e, sys)

    def get_model_training_config(self, log=True) -> ModelTrainingConfig:
        try:
            if log:
                logging.info("Getting model training configuration:")

            config = self.config.model_training
            model_params = self.params.XGBClassifier

            create_directories([config.root_dir])

            model_training_config = ModelTrainingConfig(
                root_dir=config.root_dir,
                model_params=model_params,
                experiment_name=config.experiment_name,
                latest_run_id=config.latest_run_id
            )

            if log:
                logging.info("Model training configuration loaded successfully!")

            return model_training_config

        except Exception as e:
            if log:
                logging.error(f"Error occurred while getting model training configuration!")
            raise CustomException(e, sys)

    def get_model_evaluation_config(self, log=True) -> ModelEvaluationConfig:
        try:
            if log:
                logging.info("Getting model evaluation configuration:")

            config = self.config.model_evaluation

            create_directories([config.root_dir])

            model_evaluation_config = ModelEvaluationConfig(
                root_dir=config.root_dir,
                latest_run_id=config.latest_run_id,
                experiment_name=config.experiment_name,
                train_metrics=config.train_metrics,
                test_metrics=config.test_metrics
            )

            if log:
                logging.info("Model evaluation configuration loaded successfully!")

            return model_evaluation_config

        except Exception as e:
            if log:
                logging.error(f"Error occurred while getting model evaluation configuration!")
            raise CustomException(e, sys)

    def get_prediction_config(self, log=True) -> PredictionConfig:
        try:
            if log:
                logging.info("Getting prediction configuration:")

            config = self.config.prediction

            prediction_config = PredictionConfig(
                latest_run_id=config.latest_run_id,
                experiment_name=config.experiment_name,
                data_transformer=config.data_transformer
            )

            if log:
                logging.info("Prediction configuration loaded successfully!")

            return prediction_config

        except Exception as e:
            if log:
                logging.error(f"Error occurred while getting prediction configuration!")
            raise CustomException(e, sys)
