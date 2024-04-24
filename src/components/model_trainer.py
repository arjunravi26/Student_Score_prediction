import sys
import os
from catboost import CatBoostRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from src.exception import CustomException
from src.logger import logging
from sklearn.ensemble import (
    RandomForestRegressor,
    AdaBoostRegressor,
    GradientBoostingRegressor,
)
from sklearn.metrics import r2_score
from src.utils import save_object, evaluate_model
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
import pandas as pd
from src.components.data_transformation import DataTransformation


@dataclass
class ModelTrainerConfig:
    train_model_file_path = os.path.join("artificats", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_arr, test_arr, preprocessor_path):
        try:
            logging.info("Trainig Testing input data")
            X_train, y_train, X_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1],
            )
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }
            logging.info("starting evaluate model")
            models_report: dict = evaluate_model(
                X_train, y_train, X_test, y_test, models
            )
            best_model_scores = max(sorted(models_report.values()))
            best_model_name = list(models_report.keys())[
                list(models_report.values()).index(best_model_scores)
            ]
            best_model = models[best_model_name]
            if best_model_scores < 0.6:
                raise Exception("No best model found")
            logging.info(
                f"Best model is {best_model_name} and its test score is {best_model_scores}"
            )

            save_object(
                file_path=self.model_trainer_config.train_model_file_path,
                obj=best_model,
            )
            predicted = best_model.predict(X_test)
            predicted_score = r2_score(y_test, predicted)
            return predicted_score
        except Exception as e:
            raise CustomException(e, sys)
