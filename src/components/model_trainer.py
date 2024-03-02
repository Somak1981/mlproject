import os
import sys

from dataclasses import dataclass

from catboost import CatBoostRegressor

from sklearn.ensemble import (AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor)

from sklearn.linear_model import LinearRegression, Ridge, Lasso

from sklearn.metrics import r2_score

from sklearn.neighbors import KNeighborsRegressor

from sklearn.tree import DecisionTreeRegressor

from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path:str=os.path.join('artifacts',"model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("splitting training and test input data")
            X_train, Y_train, X_test, Y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                "LinearRegression":LinearRegression(),
                "Decision Tree":DecisionTreeRegressor(),
                "Gradient Boosting":GradientBoostingRegressor(),
                "Random Forest":RandomForestRegressor(),
                "XGBoost":XGBRegressor(),
                "Cat Boost":CatBoostRegressor(),
                "Ada Boost":AdaBoostRegressor()
            }

            params={
                "Decision Tree":{
                    'criterion':['squared_error','friedman_mse','absolute_error','poisson']
                },
                "LinearRegression":{},
                "Random Forest":{
                    'n_estimators':[8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    'learning_rate':[0.1,0.01,0.001],
                    'subsample':[0.6,0.65,0.7,0.75,0.8,0.85,0.9],
                    'n_estimators':[8,16,32,64,128,256]
                },
                "XGBoost":{
                    'learning_rate':[0.1,0.01,0.001],
                    'n_estimators':[8,16,32,64,128,256]
                },
                "Cat Boost":{
                    'depth':[6,8,10],
                    'learning_rate':[0.1,0.01,0.001],
                    'n_estimators':[8,16,32,64,128,256]
                },
                "Ada Boost":{
                    'learning_rate':[0.1,0.01,0.001],
                    'n_estimators':[8,16,32,64,128,256]
                }
            }

            model_report:dict=evaluate_model(X_train=X_train, Y_train=Y_train, X_test=X_test,Y_test=Y_test, 
                                             models=models,params=params)

            # to get the best model from model_report
            best_model_score = max(sorted(model_report.values()))

            # to get the nest model name from model_report
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No models are good enough")

            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            
            predicted=best_model.predict(X_test)

            r2_square = r2_score(Y_test, predicted)
            return r2_square

        except Exception as e:
            raise CustomException(e,sys)