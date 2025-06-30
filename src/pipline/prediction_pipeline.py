import sys
from src.entity.config_entity import duplicate_question_PredictorConfig
from src.entity.s3_estimator import Proj1Estimator
from src.exception import MyException
from src.logger import logging
from pandas import DataFrame
from dotenv import load_dotenv
load_dotenv()


class question_Data:
    def __init__(self,
                question1: str,
                question2: str,
                ):
        """
        question Data constructor
        Input: all features of the trained model for prediction
        """
        try:
            self.question1 = question1
            self.question2 = question2
            logging.info("Entered question_Data class constructor")
        except Exception as e:
            raise MyException(e, sys) from e

    def get_question_input_data_frame(self)-> DataFrame:
        """
        This function returns a DataFrame from question_Data class input
        """
        try:

            question_input_dict = self.get_question_data_as_dict()
            return DataFrame(question_input_dict)

        except Exception as e:
            raise MyException(e, sys) from e


    def get_question_data_as_dict(self):
        """
        This function returns a dictionary from question class input
        """
        logging.info("Entered get_question_data_as_dict method as question class")

        try:
            input_data = {
                "question1": [self.question1],
                "question2": [self.question2],
            }

            logging.info("Created question data dict")
            logging.info("Exited get_question_data_as_dict method as question class")
            return input_data

        except Exception as e:
            raise MyException(e, sys) from e

class QuestionDataClassifier:
    def __init__(self,prediction_pipeline_config: duplicate_question_PredictorConfig = duplicate_question_PredictorConfig(),) -> None:
        """
        :param prediction_pipeline_config: Configuration for prediction the value
        """
        try:
            self.prediction_pipeline_config = prediction_pipeline_config
        except Exception as e:
            raise MyException(e, sys)

    def predict(self, dataframe) -> str:
        """
        This is the method of QuestionDataClassifier
        Returns: Prediction in string format
        """
        try:
            logging.info("Entered predict method of QuestionDataClassifier class")
            model = Proj1Estimator(
                bucket_name=self.prediction_pipeline_config.model_bucket_name,
                model_path=self.prediction_pipeline_config.model_file_path,
            )
            result =  model.predict(dataframe)
            
            return result
        
        except Exception as e:
            raise MyException(e, sys)