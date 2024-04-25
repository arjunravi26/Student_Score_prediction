import sys
import os
from catboost import CatBoostRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
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
            params = {
                "Random Forest": {
                    "criterion": [
                        "squared_error",
                        "friedman_mse",
                        "absolute_error",
                        "poisson",
                    ],
                    "max_features": ["sqrt", "log2", None],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                },
                "Decision Tree": {
                    "criterion": [
                        "squared_error",
                        "friedman_mse",
                        "absolute_error",
                        "poisson",
                    ],
                    "splitter": ["best", "random"],
                    "max_features": ["sqrt", "log2"],
                },
                "Gradient Boosting": {
                    "n_estimators": [100, 200, 300],
                    "learning_rate": [0.05, 0.1, 0.2],
                    "max_depth": [3, 4, 5],
                },
                "Linear Regression": {},
                "XGBRegressor": {
                    "learning_rate": [0.1, 0.01, 0.05, 0.001],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                },
                "CatBoosting Regressor": {
                    "depth": [6, 8, 10],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "iterations": [30, 50, 100],
                },
                "AdaBoost Regressor": {
                    "learning_rate": [0.1, 0.01, 0.5, 0.001],
                    "loss": ["linear", "square", "exponential"],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                },
            }

            logging.info("starting evaluate model")
            models_report: dict = evaluate_model(
                X_train, y_train, X_test, y_test, models, params
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
