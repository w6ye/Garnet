# This is the modulus for utilities like load model
import pickle
from sklearn.externals import joblib

def load_model(model_path):
    """
    Load model using joblib
    Args:
        model_path(str): Path to load the model
    Returns:
        sklearn model
    """
    model = joblib.load(model_path)
    return model

