"""
Enhanced Fraud Detection Service
Production-grade machine learning service with ensemble models and real-time processing.
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import joblib
import logging
from datetime import datetime, timedelta
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.metrics import precision_recall_curve, roc_auc_score

# Explainable AI
import shap
import lime
from lime.lime_tabular import LimeTabularExplainer

# Async processing
from concurrent.futures import ThreadPoolExecutor
import asyncio

logger = logging.getLogger(__name__)

class EnhancedFraudDetectionService:
    """
    Production-ready fraud detection service with ensemble learning and explainable AI.
    """
    
    def __init__(self):
        """Initialize the fraud detection service"""
        self.models = {}
        self.scalers = {}
        self.feature_names = []
        self.model_metadata = {}
        self.explainer = None
        self.shap_explainer = None
        self.lime_explainer = None
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Model configuration
        self.model_config = {
            'xgboost': {
                'weight': 0.35,
                'threshold': 0.5
            },
            'lightgbm': {
                'weight': 0.35,
                'threshold': 0.5
            },
            'catboost': {
                'weight': 0.30,
                'threshold': 0.5
            }
        }
        
        logger.info("Enhanced Fraud Detection Service initialized")

    async def initialize_models(self) -> bool:
        """
        Initialize and load all ML models asynchronously
        """
        try:
            logger.info("Loading ML models...")
            
            # Create models directory if it doesn't exist
            models_dir = Path("models")
            models_dir.mkdir(exist_ok=True)
            
            # Check if pre-trained models exist, otherwise create them
            if not self._models_exist():
                logger.info("Pre-trained models not found. Creating new models...")
                await self._create_and_train_models()
            else:
                await self._load_existing_models()
            
            # Initialize explainers
            await self._initialize_explainers()
            
            logger.info(f"Successfully loaded {len(self.models)} models")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing models: {str(e)}")
            return False

    def _models_exist(self) -> bool:
        """Check if pre-trained models exist"""
        models_dir = Path("models")
        required_files = ['xgboost_model.joblib', 'lightgbm_model.joblib', 'catboost_model.joblib']
        return all((models_dir / file).exists() for file in required_files)

    async def _create_and_train_models(self):
        """Create and train models with synthetic data for demonstration"""
        logger.info("Creating synthetic training data...")
        
        # Generate synthetic credit card transaction data
        np.random.seed(42)
        n_samples = 10000
        n_features = 30
        
        # Simulate PCA-transformed features (like in the original credit card dataset)
        X_synthetic = np.random.randn(n_samples, n_features)
        
        # Create realistic fraud labels (0.17% fraud rate)
        fraud_indices = np.random.choice(n_samples, size=int(n_samples * 0.0017), replace=False)
        y_synthetic = np.zeros(n_samples)
        y_synthetic[fraud_indices] = 1
        
        # Make fraudulent transactions more extreme
        X_synthetic[fraud_indices] *= 2
        
        # Feature names
        self.feature_names = [f'V{i}' for i in range(1, 29)] + ['Amount', 'Time']
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_synthetic)
        
        # Train models
        await self._train_ensemble_models(X_scaled, y_synthetic, scaler)

    async def _train_ensemble_models(self, X_train: np.ndarray, y_train: np.ndarray, scaler):
        """Train ensemble of ML models"""
        
        # XGBoost
        logger.info("Training XGBoost model...")
        xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='auc'
        )
        xgb_model.fit(X_train, y_train)
        
        # LightGBM
        logger.info("Training LightGBM model...")
        lgb_model = lgb.LGBMClassifier(
            n_estimators=100,
            learning_rate=0.1,
            num_leaves=31,
            random_state=42,
            verbose=-1
        )
        lgb_model.fit(X_train, y_train)
        
        # CatBoost
        logger.info("Training CatBoost model...")
        cat_model = cb.CatBoostClassifier(
            iterations=100,
            learning_rate=0.1,
            depth=6,
            random_state=42,
            verbose=False
        )
        cat_model.fit(X_train, y_train)
        
        # Store models
        self.models = {
            'xgboost': xgb_model,
            'lightgbm': lgb_model,
            'catboost': cat_model
        }
        
        self.scalers['main'] = scaler
        
        # Save models
        models_dir = Path("models")
        joblib.dump(xgb_model, models_dir / 'xgboost_model.joblib')
        joblib.dump(lgb_model, models_dir / 'lightgbm_model.joblib')
        joblib.dump(cat_model, models_dir / 'catboost_model.joblib')
        joblib.dump(scaler, models_dir / 'scaler.joblib')
        
        logger.info("All models trained and saved successfully")

    async def _load_existing_models(self):
        """Load pre-trained models from disk"""
        models_dir = Path("models")
        
        self.models = {
            'xgboost': joblib.load(models_dir / 'xgboost_model.joblib'),
            'lightgbm': joblib.load(models_dir / 'lightgbm_model.joblib'),
            'catboost': joblib.load(models_dir / 'catboost_model.joblib')
        }
        
        self.scalers['main'] = joblib.load(models_dir / 'scaler.joblib')
        self.feature_names = [f'V{i}' for i in range(1, 29)] + ['Amount', 'Time']
        
        logger.info("Existing models loaded successfully")

    async def _initialize_explainers(self):
        """Initialize SHAP and LIME explainers"""
        try:
            # Create sample data for explainers
            sample_data = np.random.randn(100, len(self.feature_names))
            sample_data = self.scalers['main'].transform(sample_data)
            
            # SHAP explainer for XGBoost
            self.shap_explainer = shap.TreeExplainer(self.models['xgboost'])
            
            # LIME explainer
            self.lime_explainer = LimeTabularExplainer(
                sample_data,
                feature_names=self.feature_names,
                class_names=['Legitimate', 'Fraud'],
                mode='classification'
            )
            
            logger.info("Explainers initialized successfully")
            
        except Exception as e:
            logger.warning(f"Could not initialize explainers: {str(e)}")

    async def analyze_transaction(self, transaction_data: dict, user_context: dict = None) -> Dict[str, Any]:
        """
        Analyze a single transaction for fraud detection
        
        Args:
            transaction_data: Transaction features and metadata
            user_context: User context for personalization
            
        Returns:
            Dict containing fraud analysis results
        """
        start_time = datetime.utcnow()
        
        try:
            # Prepare transaction features
            features = await self._prepare_transaction_features(transaction_data)
            
            # Get ensemble prediction
            fraud_probability, individual_predictions = await self._get_ensemble_prediction(features)
            
            # Calculate risk score
            risk_score = self._calculate_risk_score(fraud_probability, individual_predictions)
            
            # Determine fraud classification
            is_fraud = fraud_probability > 0.5
            
            # Generate explanation
            explanation = await self._generate_explanation(features, fraud_probability)
            
            # Calculate processing time
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            result = {
                'is_fraud': is_fraud,
                'probability': round(float(fraud_probability), 4),
                'risk_score': round(float(risk_score), 3),
                'confidence': self._calculate_confidence(individual_predictions),
                'explanation': explanation,
                'model_version': '2.0.0',
                'processing_time': round(processing_time, 2),
                'individual_model_predictions': {
                    name: round(float(pred), 4) 
                    for name, pred in individual_predictions.items()
                }
            }
            
            logger.debug(f"Transaction analyzed: fraud_prob={fraud_probability:.4f}, time={processing_time:.2f}ms")
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing transaction: {str(e)}")
            raise

    async def analyze_transaction_batch(self, transactions: List[dict], user_context: dict = None) -> List[Dict[str, Any]]:
        """
        Analyze multiple transactions in batch for improved efficiency
        """
        try:
            logger.info(f"Processing batch of {len(transactions)} transactions")
            
            # Process transactions in parallel
            tasks = [
                self.analyze_transaction(transaction, user_context) 
                for transaction in transactions
            ]
            
            results = await asyncio.gather(*tasks)
            
            logger.info(f"Batch processing completed: {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Error in batch processing: {str(e)}")
            raise

    async def _prepare_transaction_features(self, transaction_data: dict) -> np.ndarray:
        """
        Prepare transaction features for model input
        """
        try:
            # Extract features (mock implementation for demo)
            # In production, this would extract real features from transaction
            features = []
            
            # Mock PCA features V1-V28
            for i in range(1, 29):
                features.append(transaction_data.get(f'V{i}', np.random.randn()))
            
            # Amount and Time features
            features.append(transaction_data.get('Amount', 0.0))
            features.append(transaction_data.get('Time', 0.0))
            
            # Convert to numpy array and scale
            feature_array = np.array(features).reshape(1, -1)
            scaled_features = self.scalers['main'].transform(feature_array)
            
            return scaled_features
            
        except Exception as e:
            logger.error(f"Error preparing features: {str(e)}")
            raise

    async def _get_ensemble_prediction(self, features: np.ndarray) -> Tuple[float, Dict[str, float]]:
        """
        Get ensemble prediction from all models
        """
        try:
            individual_predictions = {}
            weighted_sum = 0.0
            total_weight = 0.0
            
            # Get predictions from each model
            for model_name, model in self.models.items():
                pred_proba = model.predict_proba(features)[0, 1]  # Fraud probability
                individual_predictions[model_name] = pred_proba
                
                weight = self.model_config[model_name]['weight']
                weighted_sum += pred_proba * weight
                total_weight += weight
            
            # Calculate ensemble prediction
            ensemble_prediction = weighted_sum / total_weight if total_weight > 0 else 0.0
            
            return ensemble_prediction, individual_predictions
            
        except Exception as e:
            logger.error(f"Error getting ensemble prediction: {str(e)}")
            raise

    def _calculate_risk_score(self, fraud_probability: float, individual_predictions: Dict[str, float]) -> float:
        """
        Calculate composite risk score based on predictions and model agreement
        """
        # Base risk from ensemble probability
        base_risk = fraud_probability * 100
        
        # Agreement factor (how much models agree)
        predictions = list(individual_predictions.values())
        agreement_std = np.std(predictions)
        agreement_factor = 1 - min(agreement_std * 2, 1.0)  # Higher agreement = higher confidence
        
        # Final risk score
        risk_score = base_risk * (0.7 + 0.3 * agreement_factor)
        
        return min(risk_score, 100.0)

    def _calculate_confidence(self, individual_predictions: Dict[str, float]) -> str:
        """
        Calculate confidence level based on model agreement
        """
        predictions = list(individual_predictions.values())
        std_dev = np.std(predictions)
        
        if std_dev < 0.1:
            return "high"
        elif std_dev < 0.3:
            return "medium"
        else:
            return "low"

    async def _generate_explanation(self, features: np.ndarray, fraud_probability: float) -> Dict[str, Any]:
        """
        Generate explainable AI insights for the prediction
        """
        try:
            explanation = {
                "summary": self._get_prediction_summary(fraud_probability),
                "key_factors": [],
                "risk_indicators": [],
                "confidence_factors": []
            }
            
            # Generate SHAP explanations if available
            if self.shap_explainer:
                try:
                    shap_values = self.shap_explainer.shap_values(features)
                    
                    # Get top contributing features
                    feature_importance = list(zip(self.feature_names, shap_values[0]))
                    feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
                    
                    explanation["key_factors"] = [
                        {
                            "feature": name,
                            "impact": float(impact),
                            "description": self._get_feature_description(name, impact)
                        }
                        for name, impact in feature_importance[:5]
                    ]
                    
                except Exception as e:
                    logger.warning(f"Could not generate SHAP explanation: {str(e)}")
            
            # Add risk indicators
            if fraud_probability > 0.8:
                explanation["risk_indicators"].append("Very high fraud probability detected")
            elif fraud_probability > 0.5:
                explanation["risk_indicators"].append("Elevated fraud risk identified")
            
            return explanation
            
        except Exception as e:
            logger.error(f"Error generating explanation: {str(e)}")
            return {"summary": "Explanation unavailable", "key_factors": []}

    def _get_prediction_summary(self, fraud_probability: float) -> str:
        """Generate human-readable prediction summary"""
        if fraud_probability > 0.8:
            return "Transaction shows strong indicators of fraudulent activity"
        elif fraud_probability > 0.5:
            return "Transaction exhibits suspicious patterns requiring review"
        elif fraud_probability > 0.3:
            return "Transaction shows some unusual characteristics but appears legitimate"
        else:
            return "Transaction appears normal with low fraud risk"

    def _get_feature_description(self, feature_name: str, impact: float) -> str:
        """Get human-readable description of feature impact"""
        impact_direction = "increases" if impact > 0 else "decreases"
        
        if feature_name == "Amount":
            return f"Transaction amount {impact_direction} fraud likelihood"
        elif feature_name == "Time":
            return f"Transaction timing {impact_direction} fraud probability"
        else:
            return f"Feature {feature_name} {impact_direction} fraud risk"

    async def get_analytics_summary(self, days_back: int = 7, user_id: str = None) -> Dict[str, Any]:
        """
        Get analytics summary for dashboard
        """
        # Mock analytics data - in production this would query the database
        return {
            "transactions_processed": 15847,
            "fraud_detected": 267,
            "fraud_rate": 1.68,
            "avg_processing_time_ms": 45.2,
            "model_accuracy": 99.87,
            "daily_stats": [
                {"date": "2024-10-10", "transactions": 2341, "fraud": 39},
                {"date": "2024-10-09", "transactions": 2198, "fraud": 35},
                {"date": "2024-10-08", "transactions": 2445, "fraud": 41},
            ]
        }

    def get_model_health(self) -> Dict[str, Any]:
        """Get model health status"""
        return {
            "loaded": len(self.models) > 0,
            "count": len(self.models),
            "models": list(self.models.keys()),
            "scalers_loaded": len(self.scalers) > 0,
            "explainers_ready": self.shap_explainer is not None
        }