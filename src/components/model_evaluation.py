from src.entity.config_entity import ModelEvaluationConfig
from src.entity.artifact_entity import ModelTrainerArtifact, DataIngestionArtifact, ModelEvaluationArtifact , DataTransformationArtifact
from sklearn.metrics import f1_score
from src.exception import MyException
from src.constants import TARGET_COLUMN, SCHEMA_FILE_PATH
from src.logger import logging
from src.utils.main_utils import load_object
import sys
import pandas as pd
from typing import Optional
from src.entity.s3_estimator import Proj1Estimator
from dataclasses import dataclass

import re
from bs4 import BeautifulSoup
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from fuzzywuzzy import fuzz
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from src.utils.main_utils import read_yaml_file


@dataclass
class EvaluateModelResponse:
    trained_model_f1_score: float
    best_model_f1_score: float
    is_model_accepted: bool
    difference: float


class ModelEvaluation:

    def __init__(self, model_eval_config: ModelEvaluationConfig, data_ingestion_artifact: DataIngestionArtifact,
                 model_trainer_artifact: ModelTrainerArtifact,
                 preprocessing_object_file_path: DataTransformationArtifact,
                 ):
        try:
            self.model_eval_config = model_eval_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.model_trainer_artifact = model_trainer_artifact
            self.preprocessing_object_file_path = preprocessing_object_file_path
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise MyException(e, sys) from e

    def get_best_model(self) -> Optional[Proj1Estimator]:
        """
        Method Name :   get_best_model
        Description :   This function is used to get model from production stage.
        
        Output      :   Returns model object if available in s3 storage
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            bucket_name = self.model_eval_config.bucket_name
            model_path=self.model_eval_config.s3_model_key_path
            proj1_estimator = Proj1Estimator(bucket_name=bucket_name,
                                               model_path=model_path)

            if proj1_estimator.is_model_present(model_path=model_path):
                return proj1_estimator
            return None
        except Exception as e:
            raise  MyException(e,sys)
    #------------------------------------------------------------------------------
    @staticmethod
    def preprocess(q):
        q = str(q).lower().strip()
        q = q.replace('%', 'percentage ').replace('$', 'doller ').replace('@', 'at ').replace('₹', ' rupee ').replace('€', ' euro ')
        q = q.replace('[math]', '')
        q = q.replace(',000,000,000 ', 'b ').replace(',000,000 ', 'm ').replace(',000 ', 'k ')
        q = re.sub(r'([0-9]+)000000000', r'\1b', q)
        q = re.sub(r'([0-9]+)000000', r'\1m', q)
        q = re.sub(r'([0-9]+)000', r'\1k', q)

        contractions = {
            "ain't": "am not", "aren't": "are not", "can't": "can not", "could've": "could have",
            "couldn't": "could not", "didn't": "did not", "doesn't": "does not", "don't": "do not",
            "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he's": "he is",
            "i'm": "i am", "isn't": "is not", "it's": "it is", "let's": "let us", "mightn't": "might not",
            "mustn't": "must not", "shan't": "shall not", "she's": "she is", "shouldn't": "should not",
            "that's": "that is", "there's": "there is", "they're": "they are", "wasn't": "was not",
            "we're": "we are", "weren't": "were not", "what's": "what is", "won't": "will not",
            "wouldn't": "would not", "you're": "you are"
        }

        q = ' '.join([contractions.get(word, word) for word in q.split()])
        q = BeautifulSoup(q, "html.parser").get_text()
        q = re.sub(r'\W', ' ', q).strip()

        lemmatizer = WordNetLemmatizer()
        q = ' '.join([lemmatizer.lemmatize(word) for word in q.split()])

        return q

    @staticmethod
    def test_common_words(q1, q2):
        w1 = set(q1.split())
        w2 = set(q2.split())
        return len(w1 & w2)

    @staticmethod
    def test_total_words(q1, q2):
        return len(set(q1.split())) + len(set(q2.split()))

    @staticmethod
    def token_feature(q1, q2):
        stop_words = set(stopwords.words('english'))
        safe_div = 0.0001
        features = [0.0] * 8

        q1_tokens, q2_tokens = q1.split(), q2.split()
        if not q1_tokens or not q2_tokens:
            return features

        q1_words = set([w for w in q1_tokens if w not in stop_words])
        q2_words = set([w for w in q2_tokens if w not in stop_words])
        q1_stop = set([w for w in q1_tokens if w in stop_words])
        q2_stop = set([w for w in q2_tokens if w in stop_words])

        features[0] = len(q1_words & q2_words) / (min(len(q1_words), len(q2_words)) + safe_div)
        features[1] = len(q1_words & q2_words) / (max(len(q1_words), len(q2_words)) + safe_div)
        features[2] = len(q1_stop & q2_stop) / (min(len(q1_stop), len(q2_stop)) + safe_div)
        features[3] = len(q1_stop & q2_stop) / (max(len(q1_stop), len(q2_stop)) + safe_div)
        features[4] = len(set(q1_tokens) & set(q2_tokens)) / (min(len(q1_tokens), len(q2_tokens)) + safe_div)
        features[5] = len(set(q1_tokens) & set(q2_tokens)) / (max(len(q1_tokens), len(q2_tokens)) + safe_div)
        features[6] = int(q1_tokens[-1] == q2_tokens[-1])
        features[7] = int(q1_tokens[0] == q2_tokens[0])

        return features

    @staticmethod
    def length_features(q1, q2):
        import distance
        features = []
        strs = list(distance.lcsubstrings(q1, q2))
        lcs_ratio = len(strs[0]) / (min(len(q1), len(q2)) + 1) if strs else 0

        len_q1 = len(q1)
        len_q2 = len(q2)
        len_diff = abs(len_q1 - len_q2)
        avg_len = (len_q1 + len_q2) / 2

        features.append(lcs_ratio)
        features.append(len_diff)
        features.append(avg_len)
        return features

    @staticmethod
    def fuzzy_features(q1, q2):
        return [
            fuzz.ratio(q1, q2),
            fuzz.partial_ratio(q1, q2),
            fuzz.token_sort_ratio(q1, q2),
            fuzz.token_set_ratio(q1, q2)
        ]

    def _drop_id_column(self, df):
        """Drop the 'id' column if it exists."""
        logging.info("Dropping 'id' column")
        drop_cols = self._schema_config.get('drop_columns', [])
        for col in drop_cols:
            if col in df.columns:
                df = df.drop(col, axis=1)
        return df
    
    
    
    # --------------------------------------------------------------------------------    
    

    def evaluate_model(self) -> EvaluateModelResponse:
        """
        Method Name :   evaluate_model
        Description :   This function is used to evaluate trained model 
                        with production model and choose best model 
        
        Output      :   Returns bool value based on validation results
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            x, y = test_df.drop(TARGET_COLUMN, axis=1), test_df[TARGET_COLUMN]

            logging.info("Test data loaded and now transforming it for prediction...")

            x = self._drop_id_column(x)
            x['question1'] = x['question1'].astype(str).apply(self.preprocess)
            x['question2'] = x['question2'].astype(str).apply(self.preprocess)

            logging.info("Preprocessing done for all question1 and question2 in train and test sets")
            logging.info(f"shape of test_df: {x.shape}")

            base_features_test =  [
            len(x),  # number of rows
            len(x),  # number of rows (for question2)
            x['question1'].apply(lambda x: len(str(x).split())).mean(),
            x['question2'].apply(lambda x: len(str(x).split())).mean(),
            np.mean([
                self.test_common_words(q1, q2)
                for q1, q2 in zip(x['question1'], x['question2'])
            ]),
            np.mean([
                self.test_total_words(q1, q2)
                for q1, q2 in zip(x['question1'], x['question2'])
            ])
            ]
            logging.info("Base features calculated for  test data")

            ratios_test = [
                self.test_common_words(q1, q2) / self.test_total_words(q1, q2) if self.test_total_words(q1, q2) else 0
                for q1, q2 in zip(x['question1'], x['question2'])
            ]
            base_features_test.append(np.mean(ratios_test))
            logging.info("Ratio of common words to total words calculated for train and test data")

            tf_test = np.array([
                self.token_feature(q1, q2)
                for q1, q2 in zip(x['question1'], x['question2'])
            ])
            lf_test = np.array([
                self.length_features(q1, q2)
                for q1, q2 in zip(x['question1'], x['question2'])
            ])
            ff_test = np.array([
                self.fuzzy_features(q1, q2)
                for q1, q2 in zip(x['question1'], x['question2'])
            ])

            logging.info("Token, Length and Fuzzy features calculated for train and test data")

            # Load the trained model and preprocessing object
            logging.info("Loading trained model and preprocessing object from artifacts...")
            preprocessing_object = load_object(file_path=self.preprocessing_object_file_path.transformed_object_file_path)
            if preprocessing_object is None:
                logging.info("Preprocessing object does not exist. Returning dummy array.")
                dummy = np.zeros((1, 1))
                return dummy
            

            logging.info("Preprocessing object loaded/exists.")
            try:
                cv = preprocessing_object
                

                logging.info("Scaler object loaded/exists.")
                logging.info("Transforming test data using preprocessing object...")

                q1_vecs_test = cv.transform(x['question1']).toarray()
                q2_vecs_test = cv.transform(x['question2']).toarray()
               

                logging.info("Test data transformed successfully.")
                logging.info("Combining all features for test data...")
                num_samples = tf_test.shape[0]
                base_features_test = np.tile(np.array(base_features_test).reshape(1, -1), (num_samples, 1))
                tf_test = tf_test.reshape(num_samples, -1)
                lf_test = lf_test.reshape(num_samples, -1)
                ff_test = ff_test.reshape(num_samples, -1)
                x_vec = x.toarray() if hasattr(x, "toarray") else x

                all_features_test = np.hstack([base_features_test, tf_test, lf_test, ff_test, q1_vecs_test, q2_vecs_test])

                logging.info("All features combined for test data.")
                logging.info(f"Shape of all features for test data: {all_features_test.shape}")
                
            except Exception as e:
                logging.info(f"Error processing row: {e}")
                dummy = np.zeros((1, 1))
                return dummy
            

            # Load the trained model
            logging.info("Starting model evaluation...")
            logging.info("Loading trained model from artifacts...")
            trained_model = load_object(file_path=self.model_trainer_artifact.trained_model_file_path)
            logging.info("Trained model loaded/exists.")
            trained_model_f1_score = self.model_trainer_artifact.metric_artifact.f1_score
            logging.info(f"F1_Score for this model: {trained_model_f1_score}")

            best_model_f1_score=None
            best_model = self.get_best_model()
            if best_model is not None:
                logging.info(f"Computing F1_Score for production model..")
                y_hat_best_model = best_model.predict(x)
                best_model_f1_score = f1_score(y, y_hat_best_model)
                logging.info(f"F1_Score-Production Model: {best_model_f1_score}, F1_Score-New Trained Model: {trained_model_f1_score}")
            
            tmp_best_model_score = 0 if best_model_f1_score is None else best_model_f1_score
            result = EvaluateModelResponse(trained_model_f1_score=trained_model_f1_score,
                                           best_model_f1_score=best_model_f1_score,
                                           is_model_accepted=trained_model_f1_score > tmp_best_model_score,
                                           difference=trained_model_f1_score - tmp_best_model_score
                                           )
            logging.info(f"Result: {result}")
            return result

        except Exception as e:
            raise MyException(e, sys)

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        """
        Method Name :   initiate_model_evaluation
        Description :   This function is used to initiate all steps of the model evaluation
        
        Output      :   Returns model evaluation artifact
        On Failure  :   Write an exception log and then raise an exception
        """  
        try:
            print("------------------------------------------------------------------------------------------------")
            logging.info("Initialized Model Evaluation Component.")
            evaluate_model_response = self.evaluate_model()
            s3_model_path = self.model_eval_config.s3_model_key_path

            model_evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted=evaluate_model_response.is_model_accepted,
                s3_model_path=s3_model_path,
                trained_model_path=self.model_trainer_artifact.trained_model_file_path,
                changed_accuracy=evaluate_model_response.difference)

            logging.info(f"Model evaluation artifact: {model_evaluation_artifact}")
            logging.info("Model evaluation completed successfully.")
            return model_evaluation_artifact
        except Exception as e:
            raise MyException(e, sys) from e