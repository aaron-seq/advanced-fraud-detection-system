# src/advanced_fraud_detector.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
import optuna
import joblib
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AdvancedFraudDetector:
    """
    A comprehensive class for detecting credit card fraud using an ensemble of advanced models.
    It includes data preprocessing, hyperparameter optimization with Optuna,
    model training, and evaluation.
    """

    def __init__(self, config: dict):
        """
        Initializes the detector with a given configuration.
        
        Args:
            config (dict): A dictionary containing configuration parameters for the detector,
                           including model choices and optimization settings.
        """
        self.config = config
        self.models = {}
        self.scaler = StandardScaler()
        self.stacking_classifier = None
        logging.info("AdvancedFraudDetector initialized with config: %s", config)

    def load_and_preprocess_data(self, dataset_path: str) -> tuple:
        """
        Loads and preprocesses the transaction data from a CSV file.
        
        Args:
            dataset_path (str): The path to the credit card transactions CSV file.
            
        Returns:
            tuple: A tuple containing scaled features (X), labels (y), and the original DataFrame.
        """
        logging.info("Loading and preprocessing data from %s", dataset_path)
        try:
            transaction_df = pd.read_csv(dataset_path)
        except FileNotFoundError:
            logging.error("Dataset not found at path: %s", dataset_path)
            return None, None, None
            
        # Drop unnecessary columns if they exist
        if 'Time' in transaction_df.columns:
            transaction_df = transaction_df.drop('Time', axis=1)

        features = transaction_df.drop('Class', axis=1)
        labels = transaction_df['Class']

        # Scale numerical features
        features_scaled = self.scaler.fit_transform(features)
        
        logging.info("Data preprocessing complete. Features shape: %s", features_scaled.shape)
        return features_scaled, labels, transaction_df

    def _optimize_model_hyperparameters(self, X_train, y_train, model_name: str) -> dict:
        """
        Optimizes hyperparameters for a given model using Optuna.
        
        Args:
            X_train: Training features.
            y_train: Training labels.
            model_name (str): The name of the model to optimize ('xgb', 'lgb', 'catboost', 'rf').
            
        Returns:
            dict: The best hyperparameters found by Optuna.
        """
        def objective(trial):
            if model_name == 'xgb':
                params = {
                    'objective': 'binary:logistic',
                    'eval_metric': 'auc',
                    'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                }
                model = xgb.XGBClassifier(**params, use_label_encoder=False)
            elif model_name == 'lgb':
                params = {
                    'objective': 'binary',
                    'metric': 'auc',
                    'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'num_leaves': trial.suggest_int('num_leaves', 20, 300),
                }
                model = lgb.LGBMClassifier(**params)
            # Add other models here
            else:
                return 0

            model.fit(X_train, y_train, eval_set=[(X_train, y_train)], early_stopping_rounds=50, verbose=False)
            predictions = model.predict_proba(X_train)[:, 1]
            return roc_auc_score(y_train, predictions)

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.config.get('optimization_trials', 50))
        logging.info("Best hyperparameters for %s: %s", model_name, study.best_params)
        return study.best_params

    def train_ensemble_models(self, X_train, y_train):
        """
        Trains an ensemble of models, optionally with hyperparameter optimization.
        
        Args:
            X_train: Training features.
            y_train: Training labels.
            
        Returns:
            dict: A dictionary of trained models.
        """
        for model_name in self.config.get('ensemble_methods', []):
            logging.info("Training %s model...", model_name)
            
            best_params = self._optimize_model_hyperparameters(X_train, y_train, model_name)

            if model_name == 'xgb':
                model = xgb.XGBClassifier(**best_params, use_label_encoder=False)
            elif model_name == 'lgb':
                model = lgb.LGBMClassifier(**best_params)
            elif model_name == 'catboost':
                model = cb.CatBoostClassifier(**best_params, verbose=0)
            elif model_name == 'rf':
                model = RandomForestClassifier(**best_params)
            else:
                continue

            model.fit(X_train, y_train)
            self.models[model_name] = model
            logging.info("%s model training complete.", model_name.upper())
            
        return self.models

    def create_and_train_stacking_classifier(self, X_train, y_train):
        """
        Creates and trains a stacking classifier using the trained base models.
        """
        estimators = [(name, model) for name, model in self.models.items()]
        final_estimator = LogisticRegression()
        
        self.stacking_classifier = StackingClassifier(
            estimators=estimators, final_estimator=final_estimator, cv=5
        )
        self.stacking_classifier.fit(X_train, y_train)
        logging.info("Stacking classifier training complete.")

    def evaluate_models(self, X_test, y_test):
        """
        Evaluates all trained models and the stacking ensemble on the test set.
        
        Args:
            X_test: Test features.
            y_test: Test labels.
            
        Returns:
            dict: A dictionary containing evaluation results for each model.
        """
        results = {}
        for name, model in self.models.items():
            predictions = model.predict(X_test)
            proba = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, proba)
            report = classification_report(y_test, predictions, output_dict=True)
            results[name] = {'auc': auc, 'report': report}
            logging.info("Evaluation for %s - AUC: %.4f", name, auc)

        if self.stacking_classifier:
            stack_predictions = self.stacking_classifier.predict(X_test)
            stack_proba = self.stacking_classifier.predict_proba(X_test)[:, 1]
            stack_auc = roc_auc_score(y_test, stack_proba)
            stack_report = classification_report(y_test, stack_predictions, output_dict=True)
            results['ensemble'] = {'auc': stack_auc, 'report': stack_report}
            logging.info("Evaluation for Stacking Ensemble - AUC: %.4f", stack_auc)

        return results
        
    def save_model(self, model_name: str, path: str):
        """Saves a trained model to a file."""
        if model_name in self.models:
            joblib.dump(self.models[model_name], path)
            logging.info("Model '%s' saved to %s", model_name, path)
        elif model_name == 'ensemble' and self.stacking_classifier:
            joblib.dump(self.stacking_classifier, path)
            logging.info("Ensemble model saved to %s", path)
        else:
            logging.error("Model '%s' not found.", model_name)

if __name__ == '__main__':
    # Example Usage
    config = {
        'ensemble_methods': ['xgb', 'lgb'],
        'optimization_trials': 50  # Number of Optuna trials
    }

    detector = AdvancedFraudDetector(config)
    X, y, df = detector.load_and_preprocess_data('data/creditcard.csv')

    if X is not None:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Train base models
        detector.train_ensemble_models(X_train, y_train)
        
        # Train stacking classifier
        detector.create_and_train_stacking_classifier(X_train, y_train)

        # Evaluate all models
        evaluation_results = detector.evaluate_models(X_test, y_test)
        print("\nEvaluation Results:")
        for model_name, result in evaluation_results.items():
            print(f"\n--- {model_name.upper()} ---")
            print(f"AUC: {result['auc']:.4f}")
            print("Classification Report:")
            print(pd.DataFrame(result['report']).transpose())

        # Save the final ensemble model
        detector.save_model('ensemble', 'models/ensemble/stacking_fraud_detector.joblib')
