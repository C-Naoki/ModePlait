class BaseModel(object):
    """Template of Time Series Forecasting Methods"""

    def __init__(self):
        print("Initializing RealtimeForecasting object")

    def fit(self, data, params):
        raise NotImplementedError("Define 'fit' function for your model")

    def predict(self, data, length):
        raise NotImplementedError("Define 'predict' function for your model")
