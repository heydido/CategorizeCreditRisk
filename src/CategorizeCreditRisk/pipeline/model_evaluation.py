import sys
from src.CategorizeCreditRisk.logger import logging
from src.CategorizeCreditRisk.exception import CustomException
from src.CategorizeCreditRisk.config.configuration import ConfigurationManager
from src.CategorizeCreditRisk.components.model_evaluation import ModelEvaluator


STAGE_NAME = "Model Evaluation"


class ModelEvaluationPipeline:
    def __init__(self):
        pass

    @staticmethod
    def main():
        config_manager = ConfigurationManager()
        model_evaluation_config = config_manager.get_model_evaluation_config()
        model_evaluator = ModelEvaluator(config=model_evaluation_config)

        model_evaluator.get_model_metrics()


if __name__ == '__main__':
    try:
        logging.info(f">>>>>> stage '{STAGE_NAME}' started <<<<<<")

        model_evaluation = ModelEvaluationPipeline()
        model_evaluation.main()

        logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<")

    except Exception as e:
        logging.error(f"Error occurred while running {STAGE_NAME}!")
        raise CustomException(e, sys)
