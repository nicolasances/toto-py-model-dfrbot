import joblib
import pandas as pd
import numpy as np

from util.feature import FeatureEngineering
from toto_logger.logger import TotoLogger

from totoml.model import ModelPrediction

logger = TotoLogger()

class Predictor: 

    def __init__(self): 
        pass

    def predict (self, model, context, data):
        """
        Predicts the list of foods that are most likely to be chosen at a specific time
        Requires the following data to be passed in data: "date" (%Y%m%d), "time" (HH:mm)
        """
        # Load relevant model files
        time_cluster_model = joblib.load(model.files['time-cluster-model'])
        id_encoder = joblib.load(model.files['id-encoder'])

        # 1. Feature engineering
        features_df = FeatureEngineering().do_for_predict(data, time_cluster_model, id_encoder, context)

        # 2. Load model & other required files
        trained_model = joblib.load(model.files['model'])

        # 3. Predict
        pred = trained_model.predict(features_df)

        # 4. Return the prediction
        positives = []
        for i in range(len(pred[0])): 
            if pred[0][i] > 0: positives.append(i)
        
        predicted_aliments = id_encoder.inverse_transform(positives)

        return ModelPrediction(prediction={"aliments": predicted_aliments.tolist()})