import sys
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from src.CategorizeCreditRisk.logger import logging
from src.CategorizeCreditRisk.exception import CustomException
from src.CategorizeCreditRisk.config.configuration import ConfigurationManager
from src.CategorizeCreditRisk.entity.config_entity import DataTransformationConfig


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.processed_data = pd.read_csv(self.config.preprocessed_dataset)
        self.cat_features = list(self.config.cat_features.keys())
        self.num_features = list(self.config.num_features.keys())
        self.target_variable = list(self.config.target_variable.keys())[0]

    def _split_data(self, log=True) -> tuple:
        try:
            if log:
                logging.info("> Splitting data into train and test sets:")

            df = self.processed_data

            x, y = df.drop(self.target_variable, axis=1), df[self.target_variable]

            mapping = {'P1': 0, 'P2': 1, 'P3': 2, 'P4': 3}
            y = y.map(mapping)

            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

            if log:
                logging.info("Data splited into train/test successfully!")

            return x_train, x_test, y_train, y_test

        except Exception as e:
            if log:
                logging.error(f"Error occurred while splitting data!")
            raise CustomException(e, sys)

    def _get_data_transformer(self, log=True) -> ColumnTransformer:
        try:
            if log:
                logging.info("> Getting data transformer:")

            num_transformer = StandardScaler()
            cat_transformer = OneHotEncoder(drop='first')

            data_transformer = ColumnTransformer(
                [
                    ("OneHotEncoder", cat_transformer, self.cat_features),
                    ("StandardScaler", num_transformer, self.num_features),
                ]
            )

            if log:
                logging.info("Data transformer is ready!")

            return data_transformer

        except Exception as e:
            if log:
                logging.error(f"Error occurred while getting data transformer!")
            raise CustomException(e, sys)

    def get_transformed_data(self, log=True, save_transformer=True) -> tuple:
        try:
            if log:
                logging.info("> Transforming data:")

            _x_train, _x_test, y_train, y_test = self._split_data()
            data_transformer = self._get_data_transformer()

            x_train = data_transformer.fit_transform(_x_train)
            x_train = pd.DataFrame(x_train, columns=data_transformer.get_feature_names_out())

            x_test = data_transformer.transform(_x_test)
            x_test = pd.DataFrame(x_test, columns=data_transformer.get_feature_names_out())

            if log:
                logging.info("Data transformed successfully!")

            if save_transformer:
                try:
                    logging.info("Saving data transformer:")
                    with open(self.config.data_transformer, 'wb') as file:
                        pickle.dump(data_transformer, file)
                    logging.info("Data transformer saved successfully!")
                except Exception as e:
                    logging.error("Error occurred while saving data transformer!")
                    raise CustomException(e, sys)

            return data_transformer, x_train, x_test, y_train, y_test

        except Exception as e:
            if log:
                logging.error(f"> Error occurred while transforming data!")
            raise CustomException(e, sys)


if __name__ == '__main__':
    config_manager = ConfigurationManager()
    data_transformation_config = config_manager.get_data_transformer_config()
    data_transformation = DataTransformation(config=data_transformation_config)
    data_transformation.get_transformed_data()
