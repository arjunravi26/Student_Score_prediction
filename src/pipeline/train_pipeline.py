from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.logger import logging
from src.exception import CustomException
import sys


class TrainingPipeline:
    def __init__(self) -> None:
        pass

    def training_pipeline(self):
        try:
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
            return predict_score
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.info("Trainig Pipeline is started")
    traning_pipeline = TrainingPipeline()
    score = traning_pipeline.training_pipeline()
    print(score)
