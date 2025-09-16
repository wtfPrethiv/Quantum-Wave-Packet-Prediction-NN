import joblib

class Preprocessor:
    def __init__(self, scaler_X_path, scaler_y_path):
        self.scaler_X = joblib.load(scaler_X_path)
        self.scaler_y = joblib.load(scaler_y_path)

    def scale_X(self, X):
        return self.scaler_X.transform(X)

    def descale_y(self, y_scaled):
        return self.scaler_y.inverse_transform(y_scaled.reshape(-1, 1))
