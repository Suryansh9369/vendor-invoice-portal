import joblib
import pandas as pd
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "predict_freight_model.pkl")

def load_data(model_path: str=MODEL_PATH):
    """
    Load trained freight prediction model
    """
    with open(model_path, "rb") as f:
        model = joblib.load(f)
    return model

def predict_freight_cost(input_data):
    """
    Predict freight cost for new vendors invoice

    Parameters
    ----------
    input_data : dict

    Returns
    -------
    pd.DataFrame with predicted freight cost
    """
    model = load_data()
    input_df = pd.DataFrame(input_data)
    input_df['Predicted_Freight'] = model.predict(input_df).round()
    return input_df

if __name__ == "__main__":
    #Example inference runs for local testing
    sample_data = {
        "Dollars": [18500, 9000, 3000, 200],
        "Quantity" : [100, 500, 1000, 5000]
    }
    prediction = predict_freight_cost(sample_data)
    print(prediction)