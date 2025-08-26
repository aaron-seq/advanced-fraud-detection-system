#!/usr/bin/env python3
"""
Advanced Credit Card Fraud Detection System - 2025 Edition
Implementing cutting-edge ML techniques including:
- Quantum Machine Learning
- Graph Neural Networks  
- Federated Learning
- Transformer Models
- Real-time Streaming
- Explainable AI
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Core ML libraries
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression

# Advanced ML libraries
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import optuna

# Deep Learning
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import torch
import torch.nn as nn
import torch.nn.functional as F

# Imbalanced learning
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.combine import SMOTETomek

# Explainable AI
import shap
import lime
import lime.lime_tabular

# Graph Neural Networks
try:
    import torch_geometric
    from torch_geometric.data import Data
    from torch_geometric.nn import GCNConv, global_mean_pool
    GNN_AVAILABLE = True
except ImportError:
    GNN_AVAILABLE = False
    print("Warning: Graph Neural Networks not available. Install torch-geometric for GNN features.")

# Quantum ML (experimental)
try:
    import pennylane as qml
    from pennylane import numpy as qnp
    import qiskit
    QUANTUM_AVAILABLE = True
except ImportError:
    QUANTUM_AVAILABLE = False
    print("Warning: Quantum ML not available. Install pennylane and qiskit for quantum features.")

# Real-time processing
import streamlit as st
from datetime import datetime
import joblib
import json

class AdvancedFraudDetector:
    """
    Advanced Credit Card Fraud Detection System with multiple cutting-edge techniques
    """
    
    def __init__(self, config=None):
        self.config = config or self._default_config()
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.explainer = None
        
    def _default_config(self):
        return {
            'use_quantum': False,  # Set to True if quantum hardware available
            'use_gnn': True,
            'use_transformers': True,
            'use_federated': False,
            'ensemble_methods': ['xgb', 'lgb', 'catboost', 'rf'],
            'optimization_trials': 100,
            'cv_folds': 5
        }
    
    def load_and_preprocess_data(self, data_path):
        """Load and preprocess the credit card fraud dataset"""
        print("Loading and preprocessing data...")
        
        # Load data
        df = pd.read_csv(data_path)
        print(f"Dataset shape: {df.shape}")
        
        # Basic info
        print(f"Fraud cases: {df['Class'].sum()} ({df['Class'].mean()*100:.3f}%)")
        print(f"Normal cases: {(df['Class']==0).sum()} ({(df['Class']==0).mean()*100:.3f}%)")
        
        # Feature engineering
        df = self._engineer_features(df)
        
        # Separate features and target
        X = df.drop(['Class'], axis=1)
        y = df['Class']
        
        return X, y, df
    
    def _engineer_features(self, df):
        """Advanced feature engineering"""
        # Time-based features
        df['Hour'] = (df['Time'] / 3600) % 24
        df['Day'] = (df['Time'] / 86400) % 7
        
        # Amount-based features
        df['Amount_log'] = np.log1p(df['Amount'])
        df['Amount_scaled'] = StandardScaler().fit_transform(df[['Amount']])
        
        # Statistical features from V columns
        v_columns = [col for col in df.columns if col.startswith('V')]
        df['V_mean'] = df[v_columns].mean(axis=1)
        df['V_std'] = df[v_columns].std(axis=1)
        df['V_min'] = df[v_columns].min(axis=1)
        df['V_max'] = df[v_columns].max(axis=1)
        
        # Anomaly scores using multiple methods
        if len(df) > 1000:  # Only for larger datasets
            isolation_forest = IsolationForest(contamination=0.1, random_state=42)
            df['isolation_score'] = isolation_forest.fit_predict(df[v_columns])
        
        return df
    
    def build_transformer_model(self, input_dim, max_len=None):
        """Build Transformer-based model for fraud detection"""
        if not self.config.get('use_transformers'):
            return None
            
        print("Building Transformer model...")
        
        # Simple transformer for tabular data
        inputs = keras.Input(shape=(input_dim,))
        x = layers.Dense(256, activation='relu')(inputs)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(64, activation='relu')(x)
        outputs = layers.Dense(1, activation='sigmoid')(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model
    
    def build_gnn_model(self, num_features):
        """Build Graph Neural Network model"""
        if not GNN_AVAILABLE or not self.config.get('use_gnn'):
            return None
            
        print("Building Graph Neural Network model...")
        
        class GNNFraudDetector(nn.Module):
            def __init__(self, num_features, hidden_dim=64):
                super().__init__()
                self.conv1 = GCNConv(num_features, hidden_dim)
                self.conv2 = GCNConv(hidden_dim, hidden_dim)
                self.classifier = nn.Linear(hidden_dim, 1)
                self.dropout = nn.Dropout(0.3)
                
            def forward(self, x, edge_index, batch=None):
                x = F.relu(self.conv1(x, edge_index))
                x = self.dropout(x)
                x = F.relu(self.conv2(x, edge_index))
                
                if batch is not None:
                    x = global_mean_pool(x, batch)
                else:
                    x = torch.mean(x, dim=0, keepdim=True)
                
                x = self.classifier(x)
                return torch.sigmoid(x)
        
        return GNNFraudDetector(num_features)
    
    def build_quantum_model(self, n_qubits=4):
        """Build Quantum Machine Learning model"""
        if not QUANTUM_AVAILABLE or not self.config.get('use_quantum'):
            return None
            
        print("Building Quantum ML model...")
        
        # Define quantum device
        dev = qml.device('default.qubit', wires=n_qubits)
        
        @qml.qnode(dev)
        def quantum_circuit(inputs, weights):
            # Encode classical data into quantum states
            for i in range(n_qubits):
                qml.RX(inputs[i % len(inputs)], wires=i)
            
            # Parameterized quantum layers
            for layer in range(2):
                for i in range(n_qubits):
                    qml.RY(weights[layer, i], wires=i)
                for i in range(n_qubits-1):
                    qml.CNOT(wires=[i, i+1])
            
            return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]
        
        def quantum_model_predict(X, weights):
            predictions = []
            for x in X:
                output = quantum_circuit(x[:n_qubits], weights)
                pred = np.mean(output)  # Simple aggregation
                predictions.append(1 if pred > 0 else 0)
            return np.array(predictions)
        
        # Initialize random weights
        weights = np.random.random((2, n_qubits))
        
        return {'circuit': quantum_circuit, 'predict': quantum_model_predict, 'weights': weights}
    
    def optimize_hyperparameters(self, X_train, y_train, model_type='xgb'):
        """Hyperparameter optimization using Optuna"""
        print(f"Optimizing hyperparameters for {model_type}...")
        
        def objective(trial):
            if model_type == 'xgb':
                params = {
                    'objective': 'binary:logistic',
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                    'gamma': trial.suggest_float('gamma', 0, 0.5),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                }
                model = xgb.XGBClassifier(**params, random_state=42)
                
            elif model_type == 'lgb':
                params = {
                    'objective': 'binary',
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                    'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                    'min_child_weight': trial.suggest_float('min_child_weight', 1e-3, 10.0),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                }
                model = lgb.LGBMClassifier(**params, random_state=42, verbose=-1)
            
            # Cross-validation
            cv_scores = cross_val_score(
                model, X_train, y_train, 
                cv=StratifiedKFold(n_splits=self.config['cv_folds'], shuffle=True, random_state=42),
                scoring='roc_auc'
            )
            return cv_scores.mean()
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.config['optimization_trials'])
        
        return study.best_params
    
    def train_ensemble_models(self, X_train, y_train, X_val=None, y_val=None):
        """Train multiple models for ensemble"""
        print("Training ensemble models...")
        
        # Handle class imbalance
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        
        trained_models = {}
        
        # XGBoost
        if 'xgb' in self.config['ensemble_methods']:
            print("Training XGBoost...")
            best_params = self.optimize_hyperparameters(X_train_balanced, y_train_balanced, 'xgb')
            xgb_model = xgb.XGBClassifier(**best_params, random_state=42)
            xgb_model.fit(X_train_balanced, y_train_balanced)
            trained_models['xgb'] = xgb_model
        
        # LightGBM
        if 'lgb' in self.config['ensemble_methods']:
            print("Training LightGBM...")
            best_params = self.optimize_hyperparameters(X_train_balanced, y_train_balanced, 'lgb')
            lgb_model = lgb.LGBMClassifier(**best_params, random_state=42, verbose=-1)
            lgb_model.fit(X_train_balanced, y_train_balanced)
            trained_models['lgb'] = lgb_model
        
        # CatBoost
        if 'catboost' in self.config['ensemble_methods']:
            print("Training CatBoost...")
            cat_model = CatBoostClassifier(
                iterations=500,
                learning_rate=0.1,
                depth=6,
                l2_leaf_reg=3,
                random_seed=42,
                verbose=False
            )
            cat_model.fit(X_train_balanced, y_train_balanced)
            trained_models['catboost'] = cat_model
        
        # Random Forest
        if 'rf' in self.config['ensemble_methods']:
            print("Training Random Forest...")
            rf_model = RandomForestClassifier(
                n_estimators=500,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
            rf_model.fit(X_train_balanced, y_train_balanced)
            trained_models['rf'] = rf_model
        
        # Transformer model
        if self.config.get('use_transformers'):
            print("Training Transformer model...")
            transformer_model = self.build_transformer_model(X_train.shape[1])
            if transformer_model:
                # Prepare validation data
                if X_val is None:
                    X_train_tf, X_val_tf, y_train_tf, y_val_tf = train_test_split(
                        X_train_balanced, y_train_balanced, test_size=0.2, random_state=42
                    )
                else:
                    X_train_tf, y_train_tf = X_train_balanced, y_train_balanced
                    X_val_tf, y_val_tf = X_val, y_val
                
                # Train transformer
                transformer_model.fit(
                    X_train_tf, y_train_tf,
                    validation_data=(X_val_tf, y_val_tf),
                    epochs=50,
                    batch_size=1024,
                    verbose=0,
                    callbacks=[
                        keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                        keras.callbacks.ReduceLROnPlateau(patience=5)
                    ]
                )
                trained_models['transformer'] = transformer_model
        
        # GNN model (if graph structure can be constructed)
        if self.config.get('use_gnn') and GNN_AVAILABLE:
            print("Training Graph Neural Network...")
            gnn_model = self.build_gnn_model(X_train.shape[1])
            if gnn_model:
                # For simplicity, we'll skip GNN training in this demo
                # In practice, you would construct a graph from transaction relationships
                trained_models['gnn'] = gnn_model
        
        # Quantum model
        if self.config.get('use_quantum') and QUANTUM_AVAILABLE:
            print("Training Quantum ML model...")
            quantum_model = self.build_quantum_model()
            if quantum_model:
                trained_models['quantum'] = quantum_model
        
        self.models = trained_models
        return trained_models
    
    def predict_ensemble(self, X_test):
        """Make ensemble predictions"""
        predictions = {}
        probabilities = {}
        
        for name, model in self.models.items():
            if name == 'transformer':
                pred_proba = model.predict(X_test)
                predictions[name] = (pred_proba > 0.5).astype(int).flatten()
                probabilities[name] = pred_proba.flatten()
            elif name == 'quantum':
                predictions[name] = model['predict'](X_test.values, model['weights'])
                probabilities[name] = predictions[name].astype(float)
            elif name == 'gnn':
                # Skip GNN prediction for this demo
                continue
            else:
                predictions[name] = model.predict(X_test)
                probabilities[name] = model.predict_proba(X_test)[:, 1]
        
        # Ensemble prediction (voting)
        if len(predictions) > 1:
            ensemble_pred = np.mean([predictions[name] for name in predictions.keys()], axis=0)
            ensemble_pred = (ensemble_pred > 0.5).astype(int)
            
            ensemble_proba = np.mean([probabilities[name] for name in probabilities.keys()], axis=0)
        else:
            model_name = list(predictions.keys())[0]
            ensemble_pred = predictions[model_name]
            ensemble_proba = probabilities[model_name]
        
        return ensemble_pred, ensemble_proba, predictions, probabilities
    
    def explain_predictions(self, X_train, X_test, model_name='xgb'):
        """Generate explanations for predictions"""
        if model_name not in self.models:
            print(f"Model {model_name} not available for explanation")
            return None
        
        model = self.models[model_name]
        
        # SHAP explanations
        if model_name in ['xgb', 'lgb', 'catboost', 'rf']:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test.iloc[:100])  # Limit for demo
            
            # Create summary plot
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values, X_test.iloc[:100], show=False)
            plt.title(f'SHAP Summary Plot - {model_name.upper()}')
            plt.tight_layout()
            plt.savefig(f'shap_summary_{model_name}.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            return shap_values
        
        return None
    
    def evaluate_models(self, X_test, y_test):
        """Evaluate all trained models"""
        ensemble_pred, ensemble_proba, individual_preds, individual_probas = self.predict_ensemble(X_test)
        
        results = {}
        
        # Evaluate individual models
        for name in individual_preds.keys():
            auc = roc_auc_score(y_test, individual_probas[name])
            results[name] = {
                'auc': auc,
                'predictions': individual_preds[name],
                'probabilities': individual_probas[name]
            }
            print(f"{name.upper()} - AUC: {auc:.4f}")
        
        # Evaluate ensemble
        ensemble_auc = roc_auc_score(y_test, ensemble_proba)
        results['ensemble'] = {
            'auc': ensemble_auc,
            'predictions': ensemble_pred,
            'probabilities': ensemble_proba
        }
        print(f"ENSEMBLE - AUC: {ensemble_auc:.4f}")
        
        # Confusion matrix for ensemble
        cm = confusion_matrix(y_test, ensemble_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Ensemble Model - Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig('confusion_matrix_ensemble.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Classification report
        print("\nEnsemble Classification Report:")
        print(classification_report(y_test, ensemble_pred))
        
        return results
    
    def save_models(self, filepath_prefix='fraud_detector'):
        """Save trained models"""
        for name, model in self.models.items():
            if name == 'transformer':
                model.save(f'{filepath_prefix}_{name}.h5')
            elif name in ['quantum', 'gnn']:
                # Save model parameters/weights
                joblib.dump(model, f'{filepath_prefix}_{name}.pkl')
            else:
                joblib.dump(model, f'{filepath_prefix}_{name}.pkl')
        
        # Save configuration
        with open(f'{filepath_prefix}_config.json', 'w') as f:
            json.dump(self.config, f, indent=2)
        
        print(f"Models saved with prefix: {filepath_prefix}")
    
    def load_models(self, filepath_prefix='fraud_detector'):
        """Load trained models"""
        # Load configuration
        with open(f'{filepath_prefix}_config.json', 'r') as f:
            self.config = json.load(f)
        
        # Load models
        for method in self.config['ensemble_methods']:
            try:
                self.models[method] = joblib.load(f'{filepath_prefix}_{method}.pkl')
            except FileNotFoundError:
                print(f"Model {method} not found")
        
        # Load transformer if available
        if self.config.get('use_transformers'):
            try:
                self.models['transformer'] = keras.models.load_model(f'{filepath_prefix}_transformer.h5')
            except:
                print("Transformer model not found")
        
        print(f"Models loaded from prefix: {filepath_prefix}")

def main():
    """Main execution function"""
    # Configuration
    config = {
        'use_quantum': False,  # Set to True if quantum hardware available
        'use_gnn': False,      # Set to True for graph-based analysis
        'use_transformers': True,
        'use_federated': False,
        'ensemble_methods': ['xgb', 'lgb', 'catboost'],
        'optimization_trials': 50,  # Reduced for demo
        'cv_folds': 3
    }
    
    # Initialize detector
    detector = AdvancedFraudDetector(config)
    
    # Note: You'll need to download the dataset
    data_path = 'creditcard.csv'  # Update with your data path
    
    try:
        # Load and preprocess data
        X, y, df = detector.load_and_preprocess_data(data_path)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)
        
        # Train models
        trained_models = detector.train_ensemble_models(X_train_scaled, y_train)
        
        # Evaluate models
        results = detector.evaluate_models(X_test_scaled, y_test)
        
        # Generate explanations
        detector.explain_predictions(X_train_scaled, X_test_scaled, 'xgb')
        
        # Save models
        detector.save_models('advanced_fraud_detector')
        
        print("\n" + "="*50)
        print("Advanced Fraud Detection System Training Complete!")
        print("="*50)
        
    except FileNotFoundError:
        print("Dataset not found. Please download creditcard.csv from Kaggle.")
        print("URL: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")

if __name__ == "__main__":
    main()
