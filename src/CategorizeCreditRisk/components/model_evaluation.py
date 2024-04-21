import sys
import json

import mlflow
from urllib.parse import urlparse

from sklearn.metrics import accuracy_score

from src.CategorizeCreditRisk.logger import logging
from src.CategorizeCreditRisk.exception import CustomException
from src.CategorizeCreditRisk.config.configuration import ConfigurationManager
from src.CategorizeCreditRisk.components.model_trainer import ModelTrainer


class ModelEvaluator:
    def __init__(self, config):
        self.config = config

    @staticmethod
    def _evaluate_model(true, predicted, log=True):
        try:
            if log:
                logging.info("> Evaluating model:")

            accuracy = accuracy_score(true, predicted)

            if log:
                logging.info("Model evaluated successfully!")

            return accuracy

        except Exception as e:
            if log:
                logging.error(f"Error occurred while evaluating model!")
            raise CustomException(e, sys)

    def _get_run_id_and_model(self):
        try:
            logging.info("> Getting model:")

            with open(self.config.latest_run_id, 'r') as file:
                run_id = file.read().strip()
                logging.info(f"run_id: {run_id}")

            # Note: Comment below two lines to run do prediction using a local model
            remote_server_uri = "https://dagshub.com/heydido/CategorizeCreditRisk.mlflow"
            mlflow.set_tracking_uri(remote_server_uri)

            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

            if tracking_url_type_store != "file":
                logging.info("Mode - Remote")

                model_uri = f"runs:/{run_id}/model"
                model = mlflow.sklearn.load_model(model_uri)

                logging.info(f"Model loaded successfully from: {model_uri}")

                return run_id, model

            else:
                logging.info("Mode - Local")

                with mlflow.start_run(run_id=run_id):
                    experiment_id = mlflow.get_experiment_by_name(self.config.experiment_name).experiment_id
                    model_path = f"mlruns/{experiment_id}/{run_id}/artifacts/model"
                    model = mlflow.sklearn.load_model(model_path)

                    logging.info(f"Model loaded successfully from: {model_path}")

                    return run_id, model

        except Exception as e:
            logging.error(f"Error occurred while getting model!")
            raise CustomException(e, sys)

    def get_model_metrics(self):
        try:
            logging.info("> Getting model metrics:")

            run_id, model = self._get_run_id_and_model()

            config = ConfigurationManager()
            model_training_config = config.get_model_training_config(log=False)
            trainer = ModelTrainer(config=model_training_config)

            x_train, x_test, y_train, y_test = trainer.get_data(log=False)

            y_pred_train = model.predict(x_train)
            y_pred_test = model.predict(x_test)

            train_accuracy_score = self._evaluate_model(y_train, y_pred_train, log=False)
            test_accuracy_score = self._evaluate_model(y_test, y_pred_test, log=False)

            with mlflow.start_run(run_id=run_id):
                mlflow.log_metric("train_accuracy_score", train_accuracy_score)
                logging.info(f"Train metrics: Accuracy Score: {train_accuracy_score}")

                mlflow.log_metric("test_accuracy_score", test_accuracy_score)
                logging.info(f"Test metrics: test_accuracy_score: {test_accuracy_score}")

            logging.info("Model metrics logged successfully! Ending run....")

            # End Run
            mlflow.end_run()

            train_metrics = json.dumps(
                {
                    "train_accuracy_score": train_accuracy_score
                },
                indent=4
            )

            test_metrics = json.dumps(
                {
                    "test_accuracy_score": test_accuracy_score
                },
                indent=4
            )

            train_metrics_path = self.config.train_metrics
            test_metrics_path = self.config.test_metrics

            with open(train_metrics_path, 'w') as file:
                file.write(train_metrics)

            with open(test_metrics_path, 'w') as file:
                file.write(test_metrics)

            logging.info(f"Model metrics are ready. Saved at: {train_metrics_path}, {train_metrics_path}!")

        except Exception as e:
            logging.error(f"Error occurred while getting model metrics!")
            raise CustomException(e, sys)


if __name__ == '__main__':
    config_manager = ConfigurationManager()
    model_evaluation_config = config_manager.get_model_evaluation_config()
    model_evaluator = ModelEvaluator(config=model_evaluation_config)
    model_evaluator.get_model_metrics()
