import os
import uuid
import numpy as np
import pandas as pd
import joblib

from pandas import json_normalize
from datetime import datetime as dt, timedelta

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from toto_logger.logger import TotoLogger

from util.history import HistoryDownloader
from util.feature import FeatureEngineering

from totoml.model import TrainedModel

logger = TotoLogger()

class Trainer: 

    def __init__(self): 
        pass

    def train(self, model_info, context):

        # Create the folder where to store all the data
        folder = "{tmp}/{model_name}/{fid}".format(tmp=os.environ['TOTO_TMP_FOLDER'], model_name=model_info['name'], fid=uuid.uuid1())
        os.makedirs(name=folder, exist_ok=True)

        # Determine the training period
        # Looking at 3 time periods, to be able to have a flexible model
        training_period = (dt.today() - timedelta(days=10)).strftime("%Y%m%d")
        
        # 1. Download the data
        history_filename = HistoryDownloader(training_period).download(folder, context)

        # 2. Engineer features
        (features_filename, time_cluster_model, id_encoder) = FeatureEngineering().do_for_train(folder, history_filename, context)

        # 3. Train
        logger.compute(context.correlation_id, '[ {context} ] - [ TRAINING ] - Starting model training'.format(context=context.process), 'info')
        
        # Read the features
        features_df = pd.read_csv(features_filename, index_col=0)

        # Extraction of X and y
        X = features_df[['time_cluster', 'staying_home']]
        y = features_df.drop(columns=['time_cluster', 'staying_home', 'time'])

        model = GridSearchCV(RandomForestClassifier(), param_grid={
            "n_estimators"  : [1, 2, 3, 5, 9, 20, 30]
        }, scoring='recall_samples')
    
        model.fit(X, y)

        logger.compute(context.correlation_id, '[ {context} ] - [ TRAINING ] - Model training completed'.format(context=context.process), 'info')
        
        # 4. Score
        logger.compute(context.correlation_id, '[ {context} ] - [ SCORE ] - Scoring trained model'.format(context=context.process), 'info')

        pred = model.predict(features_df[['time_cluster', 'staying_home']])

        p1 = precision_score(y, pred, average='weighted')
        r1 = recall_score(y, pred, average='weighted')
        f1 = f1_score(y, pred, average='weighted')

        score = [
            {"name": "precision", "value" : p1},
            {"name": "recall", "value": r1},
            {"name": "f1", "value": f1}
        ]

        logger.compute(context.correlation_id, '[ {context} ] - [ SCORE ] - Done. Training complete.'.format(context=context.process), 'info')

        # 5. Save all the objects
        model_filepath = "{folder}/model".format(folder=folder)
        time_cluster_model_filepath = "{folder}/time-cluster-model".format(folder=folder)
        id_encoder_filepath = "{folder}/id-encoder".format(folder=folder)

        joblib.dump(model, model_filepath)
        joblib.dump(time_cluster_model, time_cluster_model_filepath)
        joblib.dump(id_encoder, id_encoder_filepath)

        # 6. Return the trained model objects
        return TrainedModel({
            "model": model_filepath, 
            "time-cluster-model": time_cluster_model_filepath, 
            "id-encoder": id_encoder_filepath,
            }, 
            [history_filename, features_filename], 
            score
        )