import sys
import numpy as np
import pandas as pd
import re
from bs4 import BeautifulSoup
from fuzzywuzzy import fuzz
import distance
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from src.constants import TARGET_COLUMN, SCHEMA_FILE_PATH
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataTransformationArtifact, DataIngestionArtifact, DataValidationArtifact
from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import save_object, save_numpy_array_data, read_yaml_file
from sklearn.pipeline import Pipeline


class FullQuestionPairPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.cv_q1 = CountVectorizer(max_features=5000)
        self.cv_q2 = CountVectorizer(max_features=5000)
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    def preprocess(self, q):
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
        q = ' '.join([self.lemmatizer.lemmatize(word) for word in q.split()])
        return q

    def test_common_words(self, q1, q2):
        w1 = set(q1.split())
        w2 = set(q2.split())
        return len(w1 & w2)

    def test_total_words(self, q1, q2):
        return len(set(q1.split())) + len(set(q2.split()))

    def token_feature(self, q1, q2):
        safe_div = 0.0001
        features = [0.0] * 8
        q1_tokens, q2_tokens = q1.split(), q2.split()
        if not q1_tokens or not q2_tokens:
            return features
        q1_words = set([w for w in q1_tokens if w not in self.stop_words])
        q2_words = set([w for w in q2_tokens if w not in self.stop_words])
        q1_stop = set([w for w in q1_tokens if w in self.stop_words])
        q2_stop = set([w for w in q2_tokens if w in self.stop_words])
        features[0] = len(q1_words & q2_words) / (min(len(q1_words), len(q2_words)) + safe_div)
        features[1] = len(q1_words & q2_words) / (max(len(q1_words), len(q2_words)) + safe_div)
        features[2] = len(q1_stop & q2_stop) / (min(len(q1_stop), len(q2_stop)) + safe_div)
        features[3] = len(q1_stop & q2_stop) / (max(len(q1_stop), len(q2_stop)) + safe_div)
        features[4] = len(set(q1_tokens) & set(q2_tokens)) / (min(len(q1_tokens), len(q2_tokens)) + safe_div)
        features[5] = len(set(q1_tokens) & set(q2_tokens)) / (max(len(q1_tokens), len(q2_tokens)) + safe_div)
        features[6] = int(q1_tokens[-1] == q2_tokens[-1])
        features[7] = int(q1_tokens[0] == q2_tokens[0])
        return features

    def length_features(self, q1, q2):
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

    def fuzzy_features(self, q1, q2):
        return [
            fuzz.ratio(q1, q2),
            fuzz.partial_ratio(q1, q2),
            fuzz.token_sort_ratio(q1, q2),
            fuzz.token_set_ratio(q1, q2)
        ]

    def fit(self, X, y=None):
        q1_clean = X['question1'].astype(str).apply(self.preprocess)
        q2_clean = X['question2'].astype(str).apply(self.preprocess)
        self.cv_q1.fit(q1_clean)
        self.cv_q2.fit(q2_clean)
        return self

    def transform(self, X):
        q1_clean = X['question1'].astype(str).apply(self.preprocess)
        q2_clean = X['question2'].astype(str).apply(self.preprocess)
        features = []
        for q1, q2 in zip(q1_clean, q2_clean):
            base = [
                len(q1), len(q2),
                len(q1.split()), len(q2.split()),
                self.test_common_words(q1, q2),
                self.test_total_words(q1, q2)
            ]
            ratio = self.test_common_words(q1, q2) / self.test_total_words(q1, q2) if self.test_total_words(q1, q2) else 0
            token = self.token_feature(q1, q2)
            length = self.length_features(q1, q2)
            fuzzy = self.fuzzy_features(q1, q2)
            features.append(base + [ratio] + token + length + fuzzy)
        features = np.array(features, dtype=np.float32)
        q1_vec = self.cv_q1.transform(q1_clean).toarray()
        q2_vec = self.cv_q2.transform(q2_clean).toarray()
        final_features = np.hstack([features, q1_vec, q2_vec])
        return final_features


class DataTransformation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                 data_transformation_config: DataTransformationConfig,
                 data_validation_artifact: DataValidationArtifact):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise MyException(e, sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise MyException(e, sys)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            logging.info("Data Transformation Started !!!")
            if not self.data_validation_artifact.validation_status:
                raise Exception(self.data_validation_artifact.message)

            preprocessor = FullQuestionPairPreprocessor()
            logging.info("Preprocessor object created")

            train_df = self.read_data(file_path=self.data_ingestion_artifact.trained_file_path)
            test_df = self.read_data(file_path=self.data_ingestion_artifact.test_file_path)
            logging.info("Train-Test data loaded")

            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_train_df = train_df[TARGET_COLUMN]
            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_test_df = test_df[TARGET_COLUMN]
            logging.info("Input and Target cols defined for both train and test df.")

            preprocessor.fit(input_feature_train_df[['question1', 'question2']])

            input_feature_train_final = preprocessor.transform(input_feature_train_df[['question1', 'question2']])
            input_feature_test_final = preprocessor.transform(input_feature_test_df[['question1', 'question2']])

            target_feature_train = target_feature_train_df.values.reshape(-1, 1).astype(np.float32)
            target_feature_test = target_feature_test_df.values.reshape(-1, 1).astype(np.float32)

            input_feature_train_final = np.hstack([input_feature_train_final, target_feature_train])
            input_feature_test_final = np.hstack([input_feature_test_final, target_feature_test])

            logging.info(f"Final input features prepared for train and test data")
            logging.info(f"Input feature train shape: {input_feature_train_final.shape}, Input feature test shape: {input_feature_test_final.shape}")

            save_object(self.data_transformation_config.transformed_object_file_path, preprocessor)
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array=input_feature_train_final)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array=input_feature_test_final)
            logging.info("Saving transformation object and transformed files.")

            return DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )

        except Exception as e:
            logging.info("Exception occured before initiating tranformation block")
            raise MyException(e, sys)