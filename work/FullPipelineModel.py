
import utils
import FeatureEnginner
import MoisturePipeline

class FullPipelineModel(mlflow.pyfunc.PythonModel):

    def __init__(self, pipe):
        self.pipe = pipe
    
    def load_context(self, context):
        import pickle
        with open(context.artifacts["pipe"], "rb") as f:
            self.pipe = pickle.load(f)

    def predict(self, context, model_input):
        return self.pipe.predict(model_input)