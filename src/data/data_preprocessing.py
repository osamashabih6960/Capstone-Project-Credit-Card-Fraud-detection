import numpy as np
import pandas as pd
import os
import sys
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer
from sklearn.utils import resample
from src.logger import logging
import yaml

def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logging.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logging.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logging.error('YAML error: %s', e)
        raise
    except Exception as e:
        logging.error('Unexpected error: %s', e)
        raise

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the data by handling fraud detection and applying transformations."""
    try:
        logging.info("Pre-processing fraud detection data...")
        
        if 'Class' not in df.columns:
            raise KeyError("Missing required column: 'Class'")
        
        normal = df[df['Class'] == 0]  # Get all normal transactions
        fraud = df[df['Class'] == 1]   # Get all fraud transactions
        
        logging.info(f"Original normal transactions: {normal.shape}, fraud transactions: {fraud.shape}")
        
        normal_under_sample = resample(normal, replace=False, n_samples=len(fraud), random_state=27)
        df_balanced = pd.concat([normal_under_sample, fraud]).sample(frac=1, random_state=42).reset_index(drop=True)
        
        logging.info(f"New dataset shape after undersampling: {df_balanced.shape}")
        
        return df_balanced
    except KeyError as e:
        logging.error('Missing column in the dataframe: %s', e)
        raise
    except Exception as e:
        logging.error('Unexpected error during preprocessing: %s', e)
        raise


def main():
    try:
        params = load_params('./params.yaml')
        test_size = params['data_preprocessing']['test_size']
        # test_size = .20

        # Fetch the raw data
        df = pd.read_csv('./data/raw/data.csv')
        # df = pd.read_csv('./data/raw/creditcard.csv')
        logging.info('Raw data loaded successfully')
        
        # Preprocess the data
        df_balanced = preprocess_data(df)
        logging.info(f"Length of balanced data: {len(df_balanced)}")
        
        # Train-test split
        train_data, test_data = train_test_split(df_balanced, test_size=test_size, random_state=42, stratify=df_balanced['Class'])
        logging.info(f"Train data shape: {train_data.shape}, Test data shape: {test_data.shape}")
        
        # Save processed data after train-test split
        processed_data_path = os.path.join("./data", "interim")
        os.makedirs(processed_data_path, exist_ok=True)
        
        train_data.to_csv(os.path.join(processed_data_path, "train_preprocessed.csv"), index=False)
        test_data.to_csv(os.path.join(processed_data_path, "test_preprocessed.csv"), index=False)
        
        logging.info('Processed train and test data saved successfully in %s', processed_data_path)
    except Exception as e:
        logging.error('Failed to complete the data preprocessing process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
