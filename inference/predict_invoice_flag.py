import joblib
import pandas as pd

MODEL_PATH = "C:/Users/surya/Desktop/predicting frieght cost/models/predict_flag_invoice.pkl"

def load_model(model_path: str=MODEL_PATH):
    """
    Load trained classifier model
    """
    with open(model_path, "rb") as f:
        model = joblib.load(f)
    return model

def predict_flag_invoice(input_data):
    """
    Predict invoice flag for new vendor invoices.

    Parameters
    ----------
    input_data : dict

    Return
    ------
    pd.DataFrame with Predicted flag
    """
    model = load_model()
    input_df = pd.DataFrame(input_data)
    input_df['Predict_Flag'] = model.predict(input_df)
    return input_df

if __name__ == "__main__":
    print("completed")