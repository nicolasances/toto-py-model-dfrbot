import joblib

from toto_logger.logger import TotoLogger

from totoml.delegate import ModelDelegate
from totoml.model import ModelType

from model.train import Trainer
from model.predict import Predictor
from model.score import Scorer

logger = TotoLogger()

class DFRBOT(ModelDelegate):

    def __init__(self):
        pass

    def get_model_type(self):
        return ModelType.sklearn

    def get_name(self): 
        return "dfrbot"

    def predict(self, model, context, data):

        return Predictor().predict(model, context, data)

    def predict_batch(self, model, context, data=None):

        logger.compute(context.correlation_id, 'Predict Batch is not supported!', 'error')

    def score(self, model, context):

        return Scorer().score(model, context)        

    def train(self, model_info, context):

        return Trainer().train(model_info, context)
