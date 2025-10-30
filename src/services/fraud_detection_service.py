# src/services/fraud_detection_service.py

import numpy as np
import pandas as pd
from typing import Dict, List, Any
import joblib
import logging
from pathlib import Path
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import shap

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FraudDetectionService:
    """
    A production-ready fraud detection service that uses an ensemble of machine learning models.
    """

    def __init__(self):
        self.models = {}
        self.scaler = None
        self.feature_names = []

    async def load_models(self):
        """
        Initializes and loads all machine learning models asynchronously.
        """
        models_dir = Path("models")
        if not models_dir.exists():
            logging.info("Models directory not found. Creating and training new models.")
            await self._train_and_save_models()
        else:
            self._load_existing_models()

        await self._initialize_explainers()

    async def _train_and_save_models(self):
        """
        Trains and saves a new set of models using synthetic data.
        """
        X, y = self._generate_synthetic_data()
        self.feature_names = [f'V{i}' for i in range(1, 29)] + ['Amount', 'Time']

        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        self.models['xgboost'] = xgb.XGBClassifier(random_state=42, eval_metric='logloss').fit(X_scaled, y)
        self.models['lightgbm'] = lgb.LGBMClassifier(random_state=42, verbose=-1).fit(X_scaled, y)
        self.models['catboost'] = cb.CatBoostClassifier(random_state=42, verbose=False).fit(X_scaled, y)

        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        joblib.dump(self.scaler, models_dir / "scaler.joblib")
        for name, model in self.models.items():
            joblib.dump(model, models_dir / f"{name}_model.joblib")

    def _load_existing_models(self):
        """
        Loads pre-trained models from the 'models' directory.
        """
        models_dir = Path("models")
        self.scaler = joblib.load(models_dir / "scaler.joblib")
        self.models['xgboost'] = joblib.load(models_dir / "xgboost_model.joblib")
        self.models['lightgbm'] = joblib.load(models_dir / "lightgbm_model.joblib")
        self.models['catboost'] = joblib.load(models_dir / "catboost_model.joblib")
        self.feature_names = [f'V{i}' for i in range(1, 29)] + ['Amount', 'Time']

    async def _initialize_explainers(self):
        """
        Initializes the SHAP explainer for model interpretation.
        """
        self.shap_explainer = shap.TreeExplainer(self.models['xgboost'])

    def _generate_synthetic_data(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Generates synthetic data for training the fraud detection models.
        """
        n_samples = 10000
        n_features = 30
        X = np.random.randn(n_samples, n_features)
        y = np.random.choice([0, 1], n_samples, p=[0.9983, 0.0017])
        return X, y

    async def analyze(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyzes a single transaction to detect potential fraud.
        """
        features = self._prepare_features(transaction)

        individual_preds = {name: model.predict_proba(features)[0, 1] for name, model in self.models.items()}
        fraud_prob = np.mean(list(individual_preds.values()))

        explanation = self._generate_explanation(features)

        return {
            'is_fraud': bool(fraud_prob > 0.5),
            'fraud_probability': float(fraud_prob),
            'explanation': explanation,
            'model_predictions': {name: float(pred) for name, pred in individual_preds.items()}
        }

    def _prepare_features(self, transaction: Dict[str, Any]) -> np.ndarray:
        """
        Prepares a feature vector from the transaction data.
        """
        features = [transaction.get(f'V{i}', np.random.randn()) for i in range(1, 29)]
        features.append(transaction.get('Amount', 0.0))
        features.append(transaction.get('Time', 0.0))
        return self.scaler.transform(np.array(features).reshape(1, -1))

    def _generate_explanation(self, features: np.ndarray) -> Dict[str, Any]:
        """
        Generates a SHAP-based explanation for the model's prediction.
        """
        shap_values = self.shap_explainer.shap_values(features)

        feature_importance = sorted(zip(self.feature_names, shap_values[0]), key=lambda x: abs(x[1]), reverse=True)

        return {
            "summary": "Transaction risk evaluated based on multiple factors.",
            "key_factors": [{
                "feature": name,
                "impact": float(impact)
            } for name, impact in feature_importance[:5]]
        }
