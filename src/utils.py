import numpy as np
import pandas as pd
from src.exception import CustomException
import os, sys
import dill
from sklearn.metrics import r2_score
from src.logger import logging
from sklearn.model_selection import GridSearchCV
import pickle


def save_object(file_path: str, obj: object):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_model(X_train, y_train, X_test, y_test, models, params_dict):
    try:
        report = {}
        for i in range(len(list(models))):

            model = list(models.values())[i]
            params = list(params_dict.values())[i]

            rs = GridSearchCV(model, param_grid=params, cv=5)
            rs.fit(X_train, y_train)
            model.set_params(**rs.best_params_)
            model.fit(X_train, y_train)

            # y_train_pred = model.predict(X_train)
            # train_model_score = r2_score(y_train, y_train_pred)
            y_test_pred = model.predict(X_test)
            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
