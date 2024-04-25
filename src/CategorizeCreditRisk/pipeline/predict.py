import sys
import json
from src.CategorizeCreditRisk.logger import logging
from src.CategorizeCreditRisk.exception import CustomException

from src.CategorizeCreditRisk.config.configuration import ConfigurationManager
from src.CategorizeCreditRisk.components.prediction import CustomData, Predictor


STAGE_NAME = "Prediction"


class PredictionPipeline:
    def __init__(self):
        self.config_manager = ConfigurationManager()
        self.prediction_config = self.config_manager.get_prediction_config()
        self.predictor = Predictor(config=self.prediction_config)

    def predict(self, custom_data):
        # TODO: logging.info(f"Predicting RiskCategory for:\n {json.dumps(custom_data.input_data, indent=4)}")

        # Input Data
        input_data = custom_data.get_data_as_df()

        return self.predictor.predict(input_data)


if __name__ == '__main__':
    try:
        logging.info(f">>>>>> stage '{STAGE_NAME}' started <<<<<<")

        prediction = PredictionPipeline()

        # Data for prediction
        pred_datapoint = CustomData(
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
            max_recent_level_of_deliq=0,
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

        loan_amount = prediction.predict(pred_datapoint)

        logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<")

    except Exception as e:
        logging.error(f"Error occurred while running {STAGE_NAME}!")
        raise CustomException(e, sys)
