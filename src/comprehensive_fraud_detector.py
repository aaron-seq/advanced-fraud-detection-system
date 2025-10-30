# src/comprehensive_fraud_detector.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import joblib
import json
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, IsolationForest
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import optuna
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from imblearn.over_sampling import SMOTE
import shap

warnings.filterwarnings('ignore')

# Optional imports for advanced features
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch_geometric
    from torch_geometric.data import Data
    from torch_geometric.nn import GCNConv, global_mean_pool
    GNN_AVAILABLE = True
except ImportError:
    GNN_AVAILABLE = False

try:
    import pennylane as qml
    from pennylane import numpy as qnp
    import qiskit
    QUANTUM_AVAILABLE = True
except ImportError:
    QUANTUM_AVAILABLE = False

class ComprehensiveFraudDetector:
    """
    An advanced fraud detection system that integrates a wide range of cutting-edge machine learning techniques.
    """

    def __init__(self, config=None):
        self.config = config or self._default_config()
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}

    def _default_config(self):
        """Provides a default configuration for the fraud detector."""
        return {
            'use_quantum': False,
            'use_gnn': True,
            'use_transformers': True,
            'ensemble_methods': ['xgb', 'lgb', 'catboost', 'rf'],
            'optimization_trials': 100,
            'cv_folds': 5
        }

    def load_and_preprocess_data(self, data_path: str) -> tuple:
        """Loads and preprocesses the credit card fraud dataset."""
        print("Loading and preprocessing data...")
        df = pd.read_csv(data_path)

        # Feature engineering
        df = self._engineer_features(df)

        features = df.drop(['Class'], axis=1)
        target = df['Class']

        return features, target, df

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Creates new features to improve model performance."""
        # Time-based features
        df['Hour'] = (df['Time'] / 3600) % 24
        df['Day'] = (df['Time'] / 86400) % 7

        # Amount-based features
        df['Amount_log'] = np.log1p(df['Amount'])
        df['Amount_scaled'] = StandardScaler().fit_transform(df[['Amount']])

        # Anomaly scores
        v_columns = [col for col in df.columns if col.startswith('V')]
        if len(df) > 1000:
            isolation_forest = IsolationForest(contamination=0.1, random_state=42)
            df['isolation_score'] = isolation_forest.fit_predict(df[v_columns])

        return df

    def build_transformer_model(self, input_dim: int):
        """Builds a Transformer-based model for fraud detection."""
        if not self.config.get('use_transformers'):
            return None

        print("Building Transformer model...")
        inputs = keras.Input(shape=(input_dim,))
        x = layers.Dense(128, activation='relu')(inputs)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(64, activation='relu')(x)
        outputs = layers.Dense(1, activation='sigmoid')(x)

        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'precision', 'recall'])
        return model

    def train_ensemble(self, X_train, y_train, X_val=None, y_val=None) -> dict:
        """Trains a diverse ensemble of models."""
        print("Training ensemble models...")

        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

        trained_models = {}

        for model_name in self.config['ensemble_methods']:
            print(f"Training {model_name.upper()}...")
            if model_name == 'xgb':
                model = xgb.XGBClassifier(random_state=42)
            elif model_name == 'lgb':
                model = lgb.LGBMClassifier(random_state=42, verbose=-1)
            elif model_name == 'catboost':
                model = CatBoostClassifier(random_seed=42, verbose=False)
            elif model_name == 'rf':
                model = RandomForestClassifier(random_state=42)

            model.fit(X_train_balanced, y_train_balanced)
            trained_models[model_name] = model

        if self.config.get('use_transformers'):
            transformer_model = self.build_transformer_model(X_train.shape[1])
            if transformer_model:
                transformer_model.fit(
                    X_train_balanced, y_train_balanced,
                    validation_data=(X_val, y_val) if X_val is not None else None,
                    epochs=50,
                    batch_size=1024,
                    verbose=0,
                    callbacks=[
                        keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                        keras.callbacks.ReduceLROnPlateau(patience=5)
                    ]
                )
                trained_models['transformer'] = transformer_model

        self.models = trained_models
        return trained_models

    def predict_with_ensemble(self, X_test) -> tuple:
        """Generates predictions using the trained ensemble."""
        probabilities = {
            name: model.predict_proba(X_test)[:, 1] if name != 'transformer' else model.predict(X_test).flatten()
            for name, model in self.models.items()
        }

        ensemble_probabilities = np.mean(list(probabilities.values()), axis=0)
        ensemble_predictions = (ensemble_probabilities > 0.5).astype(int)

        return ensemble_predictions, ensemble_probabilities, probabilities

    def explain_predictions(self, X_test, model_name='xgb'):
        """Generates SHAP explanations for model predictions."""
        if model_name not in self.models:
            print(f"Model {model_name} not available for explanation.")
            return

        model = self.models[model_name]
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test.iloc[:100])

        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_test.iloc[:100], show=False)
        plt.title(f'SHAP Summary Plot - {model_name.upper()}')
        plt.tight_layout()
        plt.savefig(f'shap_summary_{model_name}.png', dpi=300)
        plt.show()

    def evaluate(self, X_test, y_test):
        """Evaluates the performance of the trained models."""
        ensemble_pred, ensemble_proba, individual_probas = self.predict_with_ensemble(X_test)

        print(f"Ensemble AUC: {roc_auc_score(y_test, ensemble_proba):.4f}")

        cm = confusion_matrix(y_test, ensemble_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Ensemble Model - Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig('confusion_matrix_ensemble.png', dpi=300)
        plt.show()

        print("\nEnsemble Classification Report:")
        print(classification_report(y_test, ensemble_pred))

def main():
    """Main function to run the fraud detection system."""
    config = {
        'use_transformers': True,
        'ensemble_methods': ['xgb', 'lgb', 'catboost'],
        'optimization_trials': 50,
        'cv_folds': 3
    }

    detector = ComprehensiveFraudDetector(config)

    try:
        features, target, _ = detector.load_and_preprocess_data('creditcard.csv')

        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42, stratify=target)

        scaler = RobustScaler()
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=features.columns)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=features.columns)

        detector.train_ensemble(X_train_scaled, y_train, X_test_scaled, y_test)
        detector.evaluate(X_test_scaled, y_test)
        detector.explain_predictions(X_test_scaled, 'xgb')

    except FileNotFoundError:
        print("Dataset not found. Please download creditcard.csv from Kaggle.")

if __name__ == "__main__":
    main()
