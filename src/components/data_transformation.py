import sys
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer

from src.constants import TARGET_COLUMN, SCHEMA_FILE_PATH
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataTransformationArtifact, DataIngestionArtifact, DataValidationArtifact
from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import save_object, save_numpy_array_data, read_yaml_file

import re
from bs4 import BeautifulSoup
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from fuzzywuzzy import fuzz
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.base import BaseEstimator, TransformerMixin


nltk.download('stopwords')
nltk.download('wordnet')


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

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        """
        Initiates the data transformation component for the pipeline.
        """
        try:
            logging.info("Data Transformation Started !!!")
            if not self.data_validation_artifact.validation_status:
                raise Exception(self.data_validation_artifact.message)

            # Load train and test data
            train_df = self.read_data(file_path=self.data_ingestion_artifact.trained_file_path)
            test_df = self.read_data(file_path=self.data_ingestion_artifact.test_file_path)
            logging.info("Train-Test data loaded")
       

            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_train_df = train_df[TARGET_COLUMN]

            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_test_df = test_df[TARGET_COLUMN]
            logging.info("Input and Target cols defined for both train and test df.")
            # Apply custom transformations in specified sequence
            input_feature_train_df['question1'] = input_feature_train_df['question1'].astype(str).apply(self.preprocess)
            input_feature_train_df['question2'] = input_feature_train_df['question2'].astype(str).apply(self.preprocess)
            input_feature_test_df['question1'] = input_feature_test_df['question1'].astype(str).apply(self.preprocess)
            input_feature_test_df['question2'] = input_feature_test_df['question2'].astype(str).apply(self.preprocess)
            logging.info("Preprocessing done for all question1 and question2 in train and test sets")
            logging.info(f"shape of input_feature_train_df: {input_feature_train_df.shape}, shape of input_feature_test_df: {input_feature_test_df.shape}")

            base_features = [
            len(input_feature_train_df),  # number of rows
            len(input_feature_train_df),  # number of rows (for question2)
            input_feature_train_df['question1'].apply(lambda x: len(str(x).split())).mean(),
            input_feature_train_df['question2'].apply(lambda x: len(str(x).split())).mean(),
            np.mean([
                self.test_common_words(q1, q2)
                for q1, q2 in zip(input_feature_train_df['question1'], input_feature_train_df['question2'])
            ]),
            np.mean([
                self.test_total_words(q1, q2)
                for q1, q2 in zip(input_feature_train_df['question1'], input_feature_train_df['question2'])
            ]),
            ]
            base_features_test =  [
            len(input_feature_test_df),  # number of rows
            len(input_feature_test_df),  # number of rows (for question2)
            input_feature_test_df['question1'].apply(lambda x: len(str(x).split())).mean(),
            input_feature_test_df['question2'].apply(lambda x: len(str(x).split())).mean(),
            np.mean([
                self.test_common_words(q1, q2)
                for q1, q2 in zip(input_feature_test_df['question1'], input_feature_test_df['question2'])
            ]),
            np.mean([
                self.test_total_words(q1, q2)
                for q1, q2 in zip(input_feature_test_df['question1'], input_feature_test_df['question2'])
            ])
            ]
            logging.info("Base features calculated for train and test data")

            # For train set
            ratios = [
                self.test_common_words(q1, q2) / self.test_total_words(q1, q2) if self.test_total_words(q1, q2) else 0
                for q1, q2 in zip(input_feature_train_df['question1'], input_feature_train_df['question2'])
            ]
            base_features.append(np.mean(ratios))

            # For test set
            ratios_test = [
                self.test_common_words(q1, q2) / self.test_total_words(q1, q2) if self.test_total_words(q1, q2) else 0
                for q1, q2 in zip(input_feature_test_df['question1'], input_feature_test_df['question2'])
            ]
            base_features_test.append(np.mean(ratios_test))
            logging.info("Ratio of common words to total words calculated for train and test data")

                        # For train set
            tf = np.array([
                self.token_feature(q1, q2)
                for q1, q2 in zip(input_feature_train_df['question1'], input_feature_train_df['question2'])
            ])
            lf = np.array([
                self.length_features(q1, q2)
                for q1, q2 in zip(input_feature_train_df['question1'], input_feature_train_df['question2'])
            ])
            ff = np.array([
                self.fuzzy_features(q1, q2)
                for q1, q2 in zip(input_feature_train_df['question1'], input_feature_train_df['question2'])
            ])

            # For test set
            tf_test = np.array([
                self.token_feature(q1, q2)
                for q1, q2 in zip(input_feature_test_df['question1'], input_feature_test_df['question2'])
            ])
            lf_test = np.array([
                self.length_features(q1, q2)
                for q1, q2 in zip(input_feature_test_df['question1'], input_feature_test_df['question2'])
            ])
            ff_test = np.array([
                self.fuzzy_features(q1, q2)
                for q1, q2 in zip(input_feature_test_df['question1'], input_feature_test_df['question2'])
            ])

            logging.info("Token, Length and Fuzzy features calculated for train and test data")
            
        
            # --- Keep only this vectorization block --

            # --- Vectorization and scaling pipeline for question1 and question2 ---
            class TextVectorizerTransformer(BaseEstimator, TransformerMixin):
                def __init__(self, vectorizer=None):
                    self.vectorizer = vectorizer or CountVectorizer(max_features=5000)
                def fit(self, X, y=None):
                    self.vectorizer.fit(X)
                    return self
                def transform(self, X):
                    return self.vectorizer.transform(X).toarray()

            # Build pipelines for question1 and question2
            q1_pipeline = Pipeline([
                ('vectorizer', TextVectorizerTransformer(CountVectorizer(max_features=5000))),
                ('scaler', StandardScaler())
            ])
            q2_pipeline = Pipeline([
                ('vectorizer', TextVectorizerTransformer(CountVectorizer(max_features=5000))),
                ('scaler', StandardScaler())
            ])

            # Fit on all questions (train+test) for vocabulary consistency
            all_questions = pd.concat([
                input_feature_train_df['question1'], input_feature_train_df['question2'],
                input_feature_test_df['question1'], input_feature_test_df['question2']
            ]).astype(str).apply(self.preprocess)

            q1_pipeline.named_steps['vectorizer'].fit(all_questions)
            q2_pipeline.named_steps['vectorizer'].fit(all_questions)

            # Transform train and test
            q1_vecs = q1_pipeline.fit_transform(input_feature_train_df['question1'].astype(str).apply(self.preprocess))
            q2_vecs = q2_pipeline.fit_transform(input_feature_train_df['question2'].astype(str).apply(self.preprocess))
            q1_vecs_test = q1_pipeline.transform(input_feature_test_df['question1'].astype(str).apply(self.preprocess))
            q2_vecs_test = q2_pipeline.transform(input_feature_test_df['question2'].astype(str).apply(self.preprocess))
            logging.info("Vectorization and scaling done for question1 and question2 in train and test sets")

           
            # Stack all features horizontally for each sample
            all_features = np.hstack([
                np.tile(base_features, (q1_vecs.shape[0], 1)),
                tf, lf, ff, q1_vecs, q2_vecs, target_feature_train_df.values.reshape(-1, 1)
            ])
            all_features_test = np.hstack([
                np.tile(base_features_test, (q1_vecs_test.shape[0], 1)),
                tf_test, lf_test, ff_test, q1_vecs_test, q2_vecs_test, target_feature_test_df.values.reshape(-1, 1)
            ])
            logging.info("All features concatenated for train and test data")
            logging.info(f"All features shape: {all_features.shape}, Test features shape: {all_features_test.shape}")

            input_feature_train_final = all_features
            input_feature_test_final = all_features_test
            logging.info("Final input features prepared for train and test data")
            logging.info(f"Input feature train shape: {input_feature_train_final.shape}, Input feature test shape: {input_feature_test_final.shape}")

            # Save the transformed data as numpy arrays
            save_object(self.data_transformation_config.transformed_object_file_path, q1_pipeline.named_steps['vectorizer'].vectorizer)
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
# ...rest of the code remains unchanged...