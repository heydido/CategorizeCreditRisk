import sys
import pandas as pd
from scipy.stats import f_oneway
from scipy.stats import chi2_contingency
from statsmodels.stats.outliers_influence import variance_inflation_factor
from src.CategorizeCreditRisk.logger import logging
from src.CategorizeCreditRisk.exception import CustomException
from src.CategorizeCreditRisk.config.configuration import ConfigurationManager
from src.CategorizeCreditRisk.entity.config_entity import DataPreprocessingConfig


class DataPreprocessing:
    def __init__(self, config: DataPreprocessingConfig):
        self.config = config

        self.internal_data = pd.read_excel(self.config.internal_raw_file)
        self.internal_data.columns = [col.lower() for col in self.internal_data.columns]

        self.external_data = pd.read_excel(self.config.external_raw_file)
        self.external_data.columns = [col.lower() for col in self.external_data.columns]

        self.cat_features = self.config.cat_features
        self.num_features = self.config.num_features
        self.target_variable = self.config.target_variable

    def _impute_missing_values(self) -> tuple[pd.DataFrame, pd.DataFrame, list]:
        try:
            logging.info("Imputing missing values:")

            # Internal data
            logging.info("Cleaning internal data:")
            df1 = self.internal_data.copy()
            logging.info(
                f"Internal Data (pre-cleanup) - Shape: {df1.shape}, | Rows: {df1.shape[0]}, Columns: {df1.shape[1]}"
            )
            df1 = df1[df1['age_oldest_tl'] != -99999]
            logging.info(
                f"Internal Data (post-cleanup) - Shape: {df1.shape}, | Rows: {df1.shape[0]}, Columns: {df1.shape[1]}"
            )

            # External data
            logging.info("Cleaning external data:")

            df2 = self.external_data.copy()

            logging.info(
                f"External Data (pre-cleanup) - Shape: {df2.shape}, | Rows: {df2.shape[0]}, Columns: {df2.shape[1]}"
            )

            columns_to_be_removed = []
            for col in df2.columns:
                if df2.loc[df2[col] == -99999].shape[0] > 10000:
                    columns_to_be_removed.append(col)
            logging.info(f"All features to be removed ({len(columns_to_be_removed)}): {columns_to_be_removed}")

            df2 = df2.drop(columns_to_be_removed, axis=1)

            for col in df2.columns:
                df2 = df2.loc[df2[col] != -99999]

            logging.info(
                f"External Data (post-cleanup) - Shape: {df2.shape}, | Rows: {df2.shape[0]}, Columns: {df2.shape[1]}"
            )

            common_features = [feature for feature in df1.columns if feature in df2.columns]
            logging.info(f'All common features ({len(common_features)}): {common_features}')

            return df1, df2, common_features

        except Exception as e:
            logging.error("Imputing missing values failed!")
            raise CustomException(e, sys)

    def _merge_dataframes(self, log=True) -> pd.DataFrame:
        try:
            df1, df2, common_features = self._impute_missing_values()

            if log:
                logging.info("Merging external and internal datasets:")

            df = pd.merge(df1, df2, how='inner', left_on=common_features, right_on=common_features)

            logging.info("Merged external and internal datasets successfully!")
            logging.info(f"Merged Dataset - Shape: {df.shape}, | Rows: {df.shape[0]}, Columns: {df.shape[1]}")

            return df

        except Exception as e:
            if log:
                logging.error(f"Merging external and internal datasets failed!")
            raise CustomException(e, sys)

    def save_merged_data(self) -> None:
        try:
            df = self._merge_dataframes(log=False)

            logging.info("Saving [CleanedRaw] merged data:")

            df.to_csv(self.config.cleaned_raw_dataset, index=False)

            logging.info("[CleanedRaw] Merged data saved successfully!")

        except Exception as e:
            logging.error(f"Error occurred while saving [CleanedRaw] merged data!")
            raise CustomException(e, sys)

    def _preprocess_cat_features(self) -> pd.DataFrame:
        try:
            cr_df = pd.read_csv(self.config.cleaned_raw_dataset)

            logging.info("Preprocessing categorical features:")

            assert cr_df.isnull().sum().sum() == 0, "Stopping execution as there are missing values in cleanedRaw data!"
            assert cr_df.duplicated().sum() == 0, "Stopping execution as there are duplicate values in cleanedRaw data!"

            all_raw_features = list(cr_df.columns)
            dependent_feature = list(self.target_variable)
            raw_independent_features = [feature for feature in all_raw_features if feature not in dependent_feature]
            raw_cat_features = [feature for feature in raw_independent_features if cr_df[feature].dtype == 'O']
            raw_num_features = [
                feature for feature in raw_independent_features if feature not in raw_cat_features + ['prospectid']
            ]

            logging.info(
                f"CleanedRaw Data:\n"
                f" - All Features ({len(all_raw_features)}): {all_raw_features} \n"
                f" - Numerical Features ({len(raw_num_features)}):  {raw_num_features} \n" 
                f" - Categorical Features ({len(raw_cat_features)}): {raw_cat_features} \n"
            )

            logging.info("Performing ChiSq test on cat_features:")

            keep_these, drop_these = [], []
            for feature in raw_cat_features:
                chi2, pval, _, _ = chi2_contingency(pd.crosstab(cr_df[feature], cr_df[dependent_feature[0]]))
                if pval < 0.05:
                    keep_these.append(feature)
                    logging.info(f"[+] Feature: {feature}, pval: {round(pval, 4)}")
                else:
                    drop_these.append(feature)
                    logging.info("[-] Feature: {feature}, pval: {pval}")

            if drop_these:
                logging.info(f"Features to be dropped ({len(drop_these)}): {drop_these}")
                cr_df = cr_df.drop(drop_these, axis=1)
                logging.info(f"Features dropped. Shape: {cr_df.shape}")

            else:
                logging.info("We fail to reject the null hypothesis. No features to be dropped!")

            logging.info("Categorical features preprocessed successfully!")

            return cr_df, keep_these, raw_num_features, dependent_feature

        except Exception as e:
            logging.error("Preprocessing categorical features failed!")
            raise CustomException(e, sys)

    def _preprocess_num_features(self) -> pd.DataFrame:
        try:
            cr_df, retained_cat_features, raw_num_features, dependent_feature = self._preprocess_cat_features()

            logging.info("Preprocessing numerical features:")

            # VIF check
            logging.info("Checking for multi-collinearity among numerical features:")

            vif_data = cr_df[raw_num_features]
            logging.info(
                f"VIF Data (pre-VIF) - Shape: {vif_data.shape} | Rows:{vif_data.shape[0]}, Columns:{vif_data.shape[1]}"
            )

            columns_to_be_kept, columns_to_be_dropped, column_index = [], [], 0
            for i in range(vif_data.shape[1]):
                vif_value = variance_inflation_factor(vif_data, column_index)

                if vif_value <= 6:
                    logging.info(f"[+] {raw_num_features[i]}")
                    columns_to_be_kept.append(raw_num_features[i])
                    column_index += 1
                else:
                    logging.info(f"[-] {raw_num_features[i]}")
                    columns_to_be_dropped.append(raw_num_features[i])
                    vif_data = vif_data.drop(raw_num_features[i], axis=1)

            logging.info("Multi-collinearity checked successfully!")
            logging.info(f"All columns to be kept ({len(columns_to_be_kept)}): {columns_to_be_kept}")
            logging.info(f"All columns dropped ({len(columns_to_be_dropped)}): {columns_to_be_dropped}")

            logging.info(
                f"VIF Data (post-VIF) - Shape: {vif_data.shape} | Rows:{vif_data.shape[0]}, Columns:{vif_data.shape[1]}"
            )

            # ANOVA test
            logging.info("Performing ANOVA test on post VIF numerical features:")

            retained_num_features, discarded_num_feature = [], []
            for feature in columns_to_be_kept:
                a = list(cr_df[feature])
                b = list(cr_df[dependent_feature[0]])

                group_p1 = [value for value, group in zip(a, b) if group == 'P1']
                group_p2 = [value for value, group in zip(a, b) if group == 'P2']
                group_p3 = [value for value, group in zip(a, b) if group == 'P3']
                group_p4 = [value for value, group in zip(a, b) if group == 'P4']

                f_statistic, p_value = f_oneway(group_p1, group_p2, group_p3, group_p4)
                logging.info(f"[+] feature: {feature}" if p_value <= 0.05 else f"[-] feature: {feature}")
                retained_num_features.append(feature) if p_value <= 0.05 else discarded_num_feature.append(feature)

            logging.info("ANOVA test performed successfully!")
            logging.info(f"Numerical features retained ({len(retained_num_features)}): {retained_num_features}")
            logging.info(f"Numerical features discarded ({len(discarded_num_feature)}): {discarded_num_feature}")

            logging.info("Numerical features preprocessed successfully!")

            return cr_df, retained_cat_features, retained_num_features

        except Exception as e:
            logging.error("Preprocessing numerical features failed!")
            raise CustomException(e, sys)

    def get_preprocessed_data(self, save_csv=True) -> None:
        try:
            cr_df, retained_cat_features, retained_num_features = self._preprocess_num_features()

            logging.info("Dropping unused columns and getting preprocessed data:")

            selected_features = retained_cat_features + retained_num_features
            df_selected_features = cr_df[selected_features]

            # Label Encode Education Variable
            education_mapping = {
                1: ['SSC', 'OTHERS'],
                2: ['12TH'],
                3: ['UNDER GRADUATE', 'GRADUATE', 'PROFESSIONAL'],
                4: ['POST-GRADUATE']
            }

            def __apply_education_map(education_level: str) -> int:
                for tuned, to_be_tuned in education_mapping.items():
                    education_level = tuned if education_level in to_be_tuned else education_level

                return int(education_level)

            df_selected_features.loc[:, 'education'] = df_selected_features['education'].apply(__apply_education_map)

            retained_cat_features.remove('education') if 'education' in retained_cat_features else retained_cat_features
            assert 'education' not in retained_cat_features, "Education feature not removed from cat_features!"

            retained_num_features = retained_num_features if 'education' in retained_num_features \
                else retained_num_features + ['education']
            assert 'education' in retained_num_features, "Education feature not added to num_features!"

            assert sorted(list(self.cat_features)) == sorted(retained_cat_features), \
                "Final cat_features does not match with its schema!"
            assert sorted(list(self.num_features)) == sorted(retained_num_features), \
                "Final num_features does not match with its schema!"

            logging.info("Dropped unused columns. Preprocessed data ready to export!")

            if save_csv:
                logging.info("Exporting preprocessed data:")
                df_selected_features.to_csv(self.config.preprocessed_dataset, index=False)
                logging.info("Preprocessed data exported successfully!")

            return df_selected_features

        except Exception as e:
            logging.error(f"Error occurred while getting preprocessed data!")
            raise CustomException(e, sys)


if __name__ == '__main__':
    config_manager = ConfigurationManager()
    data_preprocessing_config = config_manager.get_data_preprocessing_config()
    data_preprocessing = DataPreprocessing(config=data_preprocessing_config)
    data_preprocessing.save_merged_data()
    data_preprocessing.get_preprocessed_data(save_csv=True)
