import numpy as np
import pandas as pd
import os
import sys
import logging
from sklearn.preprocessing import PowerTransformer
from src.logger import logging  # Ensure your logger is properly set up
from sklearn.preprocessing import PowerTransformer


def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        logging.info('Data loaded successfully from %s', file_path)
        return df
    except Exception as e:
        logging.error('Error loading data from %s: %s', file_path, e)
        raise


def apply_power_transformer(train_data: pd.DataFrame, test_data: pd.DataFrame) -> tuple:
    """Apply PowerTransformer (Yeo-Johnson) to normalize numerical features separately for X_train and X_test."""
    try:
        logging.info("Applying Power Transformer to numerical features...")

        # Separate features (X) and target (y)
        X_train = train_data.drop(columns=['Class'])
        y_train = train_data['Class']
        X_test = test_data.drop(columns=['Class'])
        y_test = test_data['Class']

        # Identify numerical features
        numerical_features = X_train.select_dtypes(include=['number']).columns.tolist()

        # Apply Power Transformer
        pt = PowerTransformer(method='yeo-johnson')
        X_train[numerical_features] = pt.fit_transform(X_train[numerical_features])
        X_test[numerical_features] = pt.transform(X_test[numerical_features])

        # Reconstruct DataFrames
        train_transformed = pd.concat([X_train, y_train], axis=1)
        test_transformed = pd.concat([X_test, y_test], axis=1)

        logging.info("Power Transformation applied successfully.")

        return train_transformed, test_transformed
    except Exception as e:
        logging.error("Error during Power Transformation: %s", e)
        raise


def save_data(df: pd.DataFrame, file_path: str) -> None:
    """Save the dataframe to a CSV file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)
        logging.info('Data saved to %s', file_path)
    except Exception as e:
        logging.error("Error saving data to %s: %s", file_path, e)
        raise

def main():
    try:
        # Load processed train & test data from interim
        train_data = load_data('./data/interim/train_preprocessed.csv')
        test_data = load_data('./data/interim/test_preprocessed.csv')

        # Apply Power Transformer
        train_transformed, test_transformed = apply_power_transformer(train_data, test_data)

        # Save the transformed data
        save_data(train_transformed, './data/processed/train_final.csv')
        save_data(test_transformed, './data/processed/test_final.csv')

        logging.info("Feature engineering completed successfully.")
    except Exception as e:
        logging.error("Feature engineering failed: %s", e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
