import joblib

# Load the pre-trained model
def load_model(model_path='INX_Future_Inc.pkl'):
    return joblib.load(model_path)
