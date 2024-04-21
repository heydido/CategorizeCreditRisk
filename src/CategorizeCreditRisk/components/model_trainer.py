import os
import sys

import mlflow
import dagshub
from urllib.parse import urlparse

from xgboost import XGBClassifier

from src.CategorizeCreditRisk.logger import logging
from src.CategorizeCreditRisk.exception import CustomException
from src.CategorizeCreditRisk.entity.config_entity import ModelTrainingConfig
from src.CategorizeCreditRisk.config.configuration import ConfigurationManager
from src.CategorizeCreditRisk.components.data_transformation import DataTransformation


class ModelTrainer:
    def __init__(self, config: ModelTrainingConfig):
        self.config = config

    @staticmethod
    def get_data(log=True):
        try:
            if log:
                logging.info("> Getting data for model training:")

            config = ConfigurationManager()
            data_transformation_config = config.get_data_transformer_config(log=False)
            transformer = DataTransformation(config=data_transformation_config)
            _, x_train, x_test, y_train, y_test = transformer.get_transformed_data(log=False)

            if log:
                logging.info("Data is ready for model training!")

            return x_train, x_test, y_train, y_test

        except Exception as e:
            if log:
                logging.error(f"Error occurred while getting data for model training!")
            raise CustomException(e, sys)

    def train_model(self):
        try:
            logging.info("> Training model:")

            x_train, x_test, y_train, y_test = self.get_data()

            # Initialize DagsHub
            # Note: Comment below line to run experiment/save model locally
            dagshub.init("CategorizeCreditRisk", "heydido", mlflow=True)

            # Set the experiment name
            mlflow.set_experiment(self.config.experiment_name)

            with mlflow.start_run() as run:
                # All the parameters are logged in mlflow
                model_params = self.config.model_params

                mlflow.log_param("alpha", model_params.alpha)
                mlflow.log_param("colsample_bytree", model_params.colsample_bytree)
                mlflow.log_param("learning_rate", model_params.learning_rate)
                mlflow.log_param("max_depth", model_params.max_depth)
                mlflow.log_param("n_estimators", model_params.n_estimators)

                logging.info("Logged model parameters successfully! Training Started....")

                xgbc = XGBClassifier(**model_params, n_jobs=-1)
                xgbc.fit(x_train, y_train)

                logging.info("Model trained successfully!")

                # Note: Comment below two lines to run experiment/save model locally
                remote_server_uri = "https://dagshub.com/heydido/CategorizeCreditRisk.mlflow"
                mlflow.set_tracking_uri(remote_server_uri)

                tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
                if tracking_url_type_store != "file":
                    logging.info("> Saving model - Mode: Remote")

                    mlflow.sklearn.log_model(xgbc, "model", registered_model_name="XGBoostClassifier")

                    model_uri = mlflow.get_artifact_uri("model")

                    logging.info(f"Model saved successfully at: {model_uri}")

                else:
                    logging.info("> Saving model - Mode: Local")

                    mlflow.sklearn.log_model(xgbc, "model")

                    run_id = run.info.run_id
                    experiment_id = mlflow.get_experiment_by_name(self.config.experiment_name).experiment_id
                    model_path = f"mlruns/{experiment_id}/{run_id}/artifacts/model"

                    logging.info(f"Model saved successfully at: {model_path}!")

                # Save run id to track evaluation metrics
                logging.info("> Saving run ID:")

                run_id = run.info.run_id

                run_id_path = os.path.join(self.config.root_dir, "latest_run_id.txt")
                with open(run_id_path, "w") as f:
                    f.write(run_id)

                logging.info(f"Run ID: {run_id} saved successfully at: {run_id_path}")

            # End the run
            mlflow.end_run()

            logging.info("Run ended successfully!")

        except Exception as e:
            logging.error(f"Could not save model, error occurred while training model")
            raise CustomException(e, sys)


if __name__ == '__main__':
    config_manager = ConfigurationManager()
    model_training_config = config_manager.get_model_training_config()
    model_trainer = ModelTrainer(config=model_training_config)
    model_trainer.train_model()