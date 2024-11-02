class ModelWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model

    def predict(self, context, model_input):
        # The error suggests model_input is a DataFrame but we're trying to call predict on it
        # Instead, we should use self.model to make predictions
        return self.model.predict(model_input) 