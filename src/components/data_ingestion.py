import os
import sys
from src.exception import CustomException
from src.logger import logging

import pandas as pd

from sklearn.model_selection import train_test_split

from dataclasses import dataclass

from src.components.data_transformation import DataTranformation
from src.components.data_transformation import DataTransformationConfig

@dataclass
class DataIngestionConfig:
    train_data_path:str=os.path.join('artifacts',"train.csv")
    test_data_path:str=os.path.join('artifacts',"test.csv")
    raw_data_path:str=os.path.join('artifacts',"raw.csv")

# -------------------------------------------------------------------------------------------------#
# 1. Initialize the variables of the class                                                         #
# 2. Read the dataset containing the data                                                          #
# 3. Create the directories for train, test and raw                                                #
# 4. perform train test split using the model selection in sklearn                                 #
# 5. create the raw, train and test dataset in the location created in step 3                      #
# -------------------------------------------------------------------------------------------------#
class DataIngestion:
    def __init__(self) :
        self.ingestion_config=DataIngestionConfig()             #initialize the variables

    def initiate_data_ingestion(self):
        logging.info("Data ingestion started")
        try:
            df=pd.read_csv('notebook\data\student.csv')
            logging.info("Read the input data as dataframe")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Initiate Train test split")

            train_set, test_set=train_test_split(df,test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)

            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e,sys)

if __name__=="__main__":
    obj=DataIngestion()
    train_data, test_data=obj.initiate_data_ingestion()

    data_transformation = DataTranformation()
    data_transformation.initiate_data_transformation(train_data, test_data)


