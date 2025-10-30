# src/ensemble_optimizer.py

import optuna
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from typing import Dict, Any
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class EnsembleOptimizer:
    """
    Optimizes hyperparameters for an ensemble of fraud detection models using Optuna.
    """

    def __init__(self, X, y, models: Dict[str, Any]):
        self.X = X
        self.y = y
        self.models = models

    def optimize(self, n_trials: int = 50) -> Dict[str, Any]:
        """
        Runs the hyperparameter optimization for each model in the ensemble.

        Args:
            n_trials (int): The number of optimization trials to run for each model.

        Returns:
            Dict[str, Any]: A dictionary containing the best hyperparameters for each model.
        """
        best_params = {}
        for name in self.models.keys():
            study = optuna.create_study(direction='maximize')
            study.optimize(lambda trial: self._objective(trial, name), n_trials=n_trials)
            best_params[name] = study.best_params
            logging.info(f"Best params for {name}: {study.best_params}")

        return best_params

    def _objective(self, trial, model_name: str) -> float:
        """
        The objective function for Optuna to maximize (e.g., cross-validated AUC).
        """
        if model_name == 'xgboost':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            }
            model = xgb.XGBClassifier(**params, random_state=42)
        elif model_name == 'lightgbm':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'num_leaves': trial.suggest_int('num_leaves', 20, 300),
            }
            model = lgb.LGBMClassifier(**params, random_state=42, verbose=-1)
        # Add other models here...
        else:
            return 0.0

        # Use cross-validation to get a robust estimate of performance
        score = cross_val_score(model, self.X, self.y, n_jobs=-1, cv=3, scoring='roc_auc').mean()
        return score

def main():
    """
    Example usage of the EnsembleOptimizer.
    """
    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, n_redundant=0, random_state=42)

    models = {
        'xgboost': xgb.XGBClassifier(random_state=42),
        'lightgbm': lgb.LGBMClassifier(random_state=42, verbose=-1),
    }

    optimizer = EnsembleOptimizer(X, y, models)
    best_hyperparams = optimizer.optimize(n_trials=20)

    logging.info(f"Best hyperparameters found: {best_hyperparams}")

if __name__ == "__main__":
    main()
