"""
This script trains several baseline models on a credit card fraud detection dataset while applying different feature engineering techniques.
Performance is logged in MLflow, and the best model is selected for hyperparameter tuning.
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import dagshub
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
from dotenv import load_dotenv
load_dotenv()
import os

warnings.filterwarnings("ignore")

# ========================== CONFIGURATION ==========================
CONFIG = {
    "data_path": "notebooks/data.csv",
    "test_size": 0.2,
    "mlflow_tracking_uri": "https://dagshub.com/osamashabih6960/Capstone-Project-Credit-Card-Fraud-detection.mlflow",
    "dagshub_repo_owner": "osamashabih6960",
    "dagshub_repo_name": "Capstone-Project-Credit-Card-Fraud-detection",
    "experiment_name": "feature-egg-model-exp"
}

# ========================== SETUP MLflow & DAGSHUB ==========================
mlflow.set_tracking_uri(CONFIG["mlflow_tracking_uri"])
dagshub.init(repo_owner=CONFIG["dagshub_repo_owner"], repo_name=CONFIG["dagshub_repo_name"], mlflow=True)
mlflow.set_experiment(CONFIG["experiment_name"])

# ========================== FEATURE ENGINEERING TECHNIQUES ==========================
FE_TECH = {
    'StandardScaler': StandardScaler(),
    'MinMaxScaler': MinMaxScaler(),
    'PowerTransformer': PowerTransformer(method='yeo-johnson')
}

# ========================== MODELS ==========================
ALGORITHMS = {
    'LogisticRegression': LogisticRegression(),
    'RandomForest': RandomForestClassifier(),
    'GradientBoosting': GradientBoostingClassifier(),
}

# ========================== LOAD DATA ==========================
def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

# ========================== TRAIN & EVALUATE MODELS ==========================
def train_and_evaluate(df):
    X = df.drop(columns=['Class'])
    y = df['Class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=CONFIG["test_size"], random_state=42)
    
    with mlflow.start_run(run_name="Feature Engineering & Models") as parent_run:
        for fe_name, fe_method in FE_TECH.items():
            try:
                X_train_transformed = fe_method.fit_transform(X_train)
                X_test_transformed = fe_method.transform(X_test)
                
                for algo_name, algorithm in ALGORITHMS.items():
                    with mlflow.start_run(run_name=f"{algo_name} with {fe_name}", nested=True) as child_run:
                        try:
                            model = algorithm
                            model.fit(X_train_transformed, y_train)
                            y_pred = model.predict(X_test_transformed)
                            
                            # Log preprocessing parameters
                            mlflow.log_params({
                                "feature engineering technique": fe_name,
                                "algorithm": algo_name,
                                "test_size": CONFIG["test_size"]
                            })


                            metrics = {
                                "accuracy": accuracy_score(y_test, y_pred),
                                "precision": precision_score(y_test, y_pred),
                                "recall": recall_score(y_test, y_pred),
                                "f1_score": f1_score(y_test, y_pred)
                            }
                            
                            mlflow.log_metrics(metrics)
                            log_model_params(algo_name, model)
                            mlflow.sklearn.log_model(model, "model")
                            
                            print(f"\nFeature Engineering: {fe_name} | Algorithm: {algo_name}")
                            print(f"Metrics: {metrics}")
                        except Exception as e:
                            print(f"Error training {algo_name} with {fe_name}: {e}")
                            mlflow.log_param("error", str(e))
            except Exception as fe_error:
                print(f"Error applying {fe_name}: {fe_error}")
                mlflow.log_param("fe_error", str(fe_error))

# ========================== LOG MODEL PARAMETERS ==========================
def log_model_params(algo_name, model):
    params_to_log = {}
    if algo_name == 'LogisticRegression':
        params_to_log["C"] = model.C
    elif algo_name == 'RandomForest':
        params_to_log["n_estimators"] = model.n_estimators
        params_to_log["max_depth"] = model.max_depth
    elif algo_name == 'GradientBoosting':
        params_to_log["n_estimators"] = model.n_estimators
        params_to_log["learning_rate"] = model.learning_rate
        params_to_log["max_depth"] = model.max_depth
    mlflow.log_params(params_to_log)

# ========================== EXECUTION ==========================
if __name__ == "__main__":
    df = load_data(CONFIG["data_path"])
    train_and_evaluate(df)



