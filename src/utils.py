import pandas as pd
import os

# Define the base path relative to this script
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')

def load_data():
    """
    Loads the training and test sets from the raw data directory.
    Returns:
        X_train, y_train, X_test
    """
    print(f"Loading data from {DATA_DIR}...")
    
    X_train = pd.read_csv(os.path.join(DATA_DIR, 'input_train_set.csv'), index_col=0)
    y_train = pd.read_csv(os.path.join(DATA_DIR, 'output_train_set.csv'), index_col=0)
    X_test = pd.read_csv(os.path.join(DATA_DIR, 'input_test_set.csv'), index_col=0)
    
    return X_train, y_train, X_test