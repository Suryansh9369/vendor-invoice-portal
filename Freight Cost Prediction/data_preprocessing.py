import pandas as pd
import sqlite3
from sklearn.model_selection import train_test_split

def load_vendor_invoice(db_path: str):
    """
    load vendor invoice from sqlite database.
    """
    conn = sqlite3.connect(db_path)
    query = "SELECT * FROM vendor_invoice"
    df = pd.read_sql_query(query, conn)
    conn.close
    return df

def prepare_feature(df: pd.DataFrame):
    """
    Select features and target variables
    """
    X= df[["Dollars", "Quantity"]]
    y= df["Freight"]

    return X,y

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Spliting data into training and testing sets.
    """
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )