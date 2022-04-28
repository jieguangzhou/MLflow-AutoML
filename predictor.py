from mlflow.pyfunc import PythonModel


class PredictorWrapper(PythonModel):
    def load_context(self, context):
        model_path = context.artifacts["model_path"]
        if "autosklearn" in model_path:
            self.predictor = self.load_autosklearn_predictor(model_path)
        else:
            assert f"cant not load model from path {model_path}"

    def predict(self, context, model_input):
        results = self.predictor.predict(model_input)
        return {"results": results}

    def load_autosklearn_predictor(self, path):
        from automl.mod_autosklearn import Predictor

        predictor = Predictor(path)
        return predictor
