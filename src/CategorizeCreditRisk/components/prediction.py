import json
import sys
import pickle
import pandas as pd

import mlflow
from urllib.parse import urlparse

from src.CategorizeCreditRisk.logger import logging
from src.CategorizeCreditRisk.exception import CustomException
from src.CategorizeCreditRisk.entity.config_entity import PredictionConfig
from src.CategorizeCreditRisk.config.configuration import ConfigurationManager

import warnings
warnings.filterwarnings("ignore")


class Predictor:
    def __init__(self, config: PredictionConfig):
        self.config = config
        self.data_transformer = self._load_data_transformer()
        self.model = self._load_model()

    def _load_data_transformer(self):
        try:
            logging.info("> Loading the data transformer:")

            data_transformer_path = self.config.data_transformer
            with open(data_transformer_path, 'rb') as file:
                data_transformer = pickle.load(file)

            logging.info("Data transformer loaded successfully!")

            return data_transformer

        except Exception as e:
            logging.error(f"Error in loading the data transformer!")
            raise CustomException(e, sys)

    def _load_model(self):
        try:
            logging.info("> Loading the model:")

            # Note: Comment below two lines to run do prediction using a local model
            remote_server_uri = "https://dagshub.com/heydido/CategorizeCreditRisk.mlflow"
            mlflow.set_tracking_uri(remote_server_uri)

            experiment_name = self.config.experiment_name
            experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

            runs = mlflow.search_runs(experiment_ids=experiment_id)
            metric_name = "test_accuracy_score"
            best_run = runs.sort_values(by=['metrics.' + metric_name], ascending=True).iloc[0]
            run_id = best_run.run_id
            run_name = runs[run_id == runs.run_id]["tags.mlflow.runName"].values[0]

            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
            if tracking_url_type_store != "file":
                logging.info("Mode - Remote")

                model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")

                logging.info(
                    f"Best model loaded successfully with following info: \n"
                    f" - experiment_id: {experiment_id} \n"
                    f" - run_id: {run_id} \n"
                    f" - run_name: {run_name}"
                )

                return model

            else:
                logging.info("Mode - Local")

                model_path = f"mlruns/{experiment_id}/{run_id}/artifacts/model"

                model = mlflow.sklearn.load_model(model_path)

                logging.info(f"Model loaded successfully. experiment_id: {experiment_id}, run_id: {run_id}")

                return model

        except Exception as e:
            logging.error(f"Error in loading the model!")
            raise CustomException(e, sys)

    def predict(self, prediction_datapoint):
        try:
            logging.info("> Getting prediction:")

            data_transformer = self.data_transformer
            model = self.model

            prediction_datapoint = data_transformer.transform(prediction_datapoint)

            prediction = model.predict(prediction_datapoint)[0]

            get_risk_category = {
                1: "P1 - Least Risk",
                2: "P2 - Some Risk",
                3: "P3 - More Risk",
                4: "P4 - Highest Risk"
            }

            risk_category = get_risk_category[prediction]

            logging.info(f"Prediction done successfully! RiskCategory: {risk_category}")

            return prediction

        except Exception as e:
            logging.error(f"Error in getting prediction!")
            raise CustomException(e, sys)


class CustomData:
    def __init__(
            self,
            maritalstatus, gender, last_prod_enq2, first_prod_enq2, pct_tl_open_l6m, pct_tl_closed_l6m,
            tot_tl_closed_l12m, pct_tl_closed_l12m, tot_missed_pmnt, cc_tl, home_tl,  pl_tl, secured_tl, unsecured_tl,
            other_tl, age_oldest_tl, age_newest_tl, time_since_recent_payment, max_recent_level_of_deliq,
            num_deliq_6_12mts, num_times_60p_dpd, num_std_12mts, num_sub, num_sub_6mts, num_sub_12mts, num_dbt,
            num_dbt_12mts, num_lss, recent_level_of_deliq, cc_enq_l12m, pl_enq_l12m, time_since_recent_enq, enq_l3m,
            netmonthlyincome,  time_with_curr_empr, cc_flag, pl_flag, pct_pl_enq_l6m_of_ever, pct_cc_enq_l6m_of_ever,
            hl_flag, gl_flag, education,
    ):
        self.input_data = {
            "maritalstatus": maritalstatus,
            "gender": gender,
            "last_prod_enq2": last_prod_enq2,
            "first_prod_enq2": first_prod_enq2,
            "pct_tl_open_l6m": pct_tl_open_l6m,
            "pct_tl_closed_l6m": pct_tl_closed_l6m,
            "tot_tl_closed_l12m": tot_tl_closed_l12m,
            "pct_tl_closed_l12m": pct_tl_closed_l12m,
            "tot_missed_pmnt": tot_missed_pmnt,
            "cc_tl": cc_tl,
            "home_tl": home_tl,
            "pl_tl": pl_tl,
            "secured_tl": secured_tl,
            "unsecured_tl": unsecured_tl,
            "other_tl": other_tl,
            "age_oldest_tl": age_oldest_tl,
            "age_newest_tl": age_newest_tl,
            "time_since_recent_payment": time_since_recent_payment,
            "max_recent_level_of_deliq": max_recent_level_of_deliq,
            "num_deliq_6_12mts": num_deliq_6_12mts,
            "num_times_60p_dpd": num_times_60p_dpd,
            "num_std_12mts": num_std_12mts,
            "num_sub": num_sub,
            "num_sub_6mts": num_sub_6mts,
            "num_sub_12mts": num_sub_12mts,
            "num_dbt": num_dbt,
            "num_dbt_12mts": num_dbt_12mts,
            "num_lss": num_lss,
            "recent_level_of_deliq": recent_level_of_deliq,
            "cc_enq_l12m": cc_enq_l12m,
            "pl_enq_l12m": pl_enq_l12m,
            "time_since_recent_enq": time_since_recent_enq,
            "enq_l3m": enq_l3m,
            "netmonthlyincome": netmonthlyincome,
            "time_with_curr_empr": time_with_curr_empr,
            "cc_flag": cc_flag,
            "pl_flag": pl_flag,
            "pct_pl_enq_l6m_of_ever": pct_pl_enq_l6m_of_ever,
            "pct_cc_enq_l6m_of_ever": pct_cc_enq_l6m_of_ever,
            "hl_flag": hl_flag,
            "gl_flag": gl_flag,
            "education": education
        }

    def get_data_as_df(self):
        try:
            logging.info("> Getting data for prediction:")

            data = pd.DataFrame([self.input_data])

            logging.info("Data ready for prediction!")

            return data

        except Exception as e:
            logging.error(f"Error in getting data for prediction!")
            raise CustomException(e, sys)


if __name__ == '__main__':

    # Data for prediction
    custom_data = CustomData(
        maritalstatus='Married',
        gender='M',
        last_prod_enq2='PL',
        first_prod_enq2='AL',
        pct_tl_open_l6m=0.01,
        pct_tl_closed_l6m=0.0,
        tot_tl_closed_l12m=0,
        pct_tl_closed_l12m=0,
        tot_missed_pmnt=0,
        cc_tl=3,
        home_tl=0,
        pl_tl=0,
        secured_tl=0,
        unsecured_tl=3,
        other_tl=0,
        age_oldest_tl=30,
        age_newest_tl=7,
        time_since_recent_payment=1,
        max_recent_level_of_deliq= 0,
        num_deliq_6_12mts=0,
        num_times_60p_dpd=0,
        num_std_12mts=23,
        num_sub=0,
        num_sub_6mts=0,
        num_sub_12mts=0,
        num_dbt=0,
        num_dbt_12mts=0,
        num_lss=0,
        recent_level_of_deliq=0,
        cc_enq_l12m=3,
        pl_enq_l12m=0,
        time_since_recent_enq=3,
        enq_l3m=1,
        netmonthlyincome=40000,
        time_with_curr_empr=260,
        cc_flag=1,
        pl_flag=0,
        pct_pl_enq_l6m_of_ever=0,
        pct_cc_enq_l6m_of_ever=8,
        hl_flag=0,
        gl_flag=0,
        education=3
    )

    logging.info(f"Predicting RiskCategory for:\n {json.dumps(custom_data.input_data, indent=4)}")

    # Input Data
    input_data = custom_data.get_data_as_df()

    # Get Prediction
    config_manager = ConfigurationManager()
    prediction_config = config_manager.get_prediction_config()
    predictor = Predictor(config=prediction_config)

    predictor.predict(input_data)  # P1 - Least Risk
