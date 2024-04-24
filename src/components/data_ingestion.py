from src.logger import logging
from src.exception import CustomException
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
import os
import sys
import pandas as pd
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import ModelTrainer


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("artificats", "train.csv")
    test_data_path: str = os.path.join("artificats", "test.csv")
    raw_data_path: str = os.path.join("artificats", "data.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initation_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df = pd.read_csv("notebook/Data/stud.csv")
            logging.info("Read the dataset as dataframe")
            os.makedirs(
                os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True
            )
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("Train Test Split is initated")
            train_set, test_set = train_test_split(df, test_size=0.20, random_state=44)
            train_set.to_csv(
                self.ingestion_config.train_data_path, index=False, header=True
            )
            test_set.to_csv(
                self.ingestion_config.test_data_path, index=False, header=True
            )
            logging.info("Ingestion of data completed")
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initation_data_ingestion()
    data_transformation = DataTransformation()
    preprocessed_train_df, preprocessed_test_df, preprocessor_path = (
        data_transformation.initiate_data_transformation(
            train_path=train_data, test_path=test_data
        )
    )
    data_obj = ModelTrainer()
    predict_score = data_obj.initiate_model_trainer(
        preprocessed_train_df, preprocessed_test_df, preprocessor_path
    )
    print(predict_score)
