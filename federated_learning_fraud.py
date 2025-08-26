#!/usr/bin/env python3
"""
Federated Learning for Credit Card Fraud Detection
Privacy-preserving distributed learning across multiple financial institutions
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import logging
from typing import List, Dict, Tuple, Optional
import json
import time
from pathlib import Path

# Federated Learning
try:
    import flwr as fl
    FEDERATED_AVAILABLE = True
except ImportError:
    FEDERATED_AVAILABLE = False
    print("Warning: Federated Learning not available. Install flower (flwr) for federated capabilities.")

# Standard ML libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

class FederatedFraudDetector:
    """Federated Learning system for fraud detection across multiple institutions"""
    
    def __init__(self, institution_id: str, config: Dict = None):
        self.institution_id = institution_id
        self.config = config or self._default_config()
        self.model = None
        self.scaler = StandardScaler()
        self.local_data = None
        
        # Privacy and security settings
        self.differential_privacy = self.config.get('differential_privacy', True)
        self.noise_multiplier = self.config.get('noise_multiplier', 0.1)
        
        # Federated learning parameters
        self.aggregation_rounds = self.config.get('aggregation_rounds', 10)
        self.local_epochs = self.config.get('local_epochs', 5)
        self.clients_per_round = self.config.get('clients_per_round', 3)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(f"Institution_{institution_id}")
    
    def _default_config(self):
        return {
            'model_type': 'neural_network',
            'differential_privacy': True,
            'noise_multiplier': 0.1,
            'aggregation_rounds': 10,
            'local_epochs': 5,
            'clients_per_round': 3,
            'min_clients': 2,
            'learning_rate': 0.001,
            'batch_size': 512
        }
    
    def load_local_data(self, data_path: str = None, synthetic_data: bool = True):
        """Load local institution data (with privacy preservation)"""
        if synthetic_data:
            # Generate synthetic data for demonstration
            n_samples = np.random.randint(10000, 50000)
            n_features = 30
            
            # Create synthetic transaction data
            X = np.random.randn(n_samples, n_features)
            
            # Make fraud cases rare (0.5-2% depending on institution)
            fraud_rate = np.random.uniform(0.005, 0.02)
            y = np.random.choice([0, 1], size=n_samples, p=[1-fraud_rate, fraud_rate])
            
            # Make fraudulent transactions more distinct
            fraud_indices = y == 1
            X[fraud_indices] += np.random.randn(fraud_indices.sum(), n_features) * 2
            
            self.logger.info(f"Generated {n_samples} synthetic transactions with {fraud_rate*100:.2f}% fraud rate")
            
        else:
            if data_path and Path(data_path).exists():
                # Load real data
                df = pd.read_csv(data_path)
                X = df.drop(['Class'], axis=1).values
                y = df['Class'].values
                self.logger.info(f"Loaded {len(X)} real transactions from {data_path}")
            else:
                raise ValueError("Data path not found and synthetic_data=False")
        
        # Store local data
        self.local_data = {
            'X': X,
            'y': y,
            'fraud_rate': y.mean()
        }
        
        return X, y
    
    def create_model(self, input_dim: int):
        """Create neural network model for fraud detection"""
        model = keras.Sequential([
            keras.layers.Dense(128, activation='relu', input_shape=(input_dim,)),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.config['learning_rate']),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model
    
    def add_differential_privacy_noise(self, gradients: List[np.ndarray]) -> List[np.ndarray]:
        """Add differential privacy noise to gradients"""
        if not self.differential_privacy:
            return gradients
        
        noisy_gradients = []
        for gradient in gradients:
            noise = np.random.normal(0, self.noise_multiplier, gradient.shape)
            noisy_gradient = gradient + noise
            noisy_gradients.append(noisy_gradient)
        
        return noisy_gradients
    
    def train_local_model(self, global_weights: Optional[List[np.ndarray]] = None) -> Dict:
        """Train model on local data"""
        if self.local_data is None:
            raise ValueError("No local data loaded. Call load_local_data() first.")
        
        X, y = self.local_data['X'], self.local_data['y']
        
        # Handle class imbalance with SMOTE
        if y.mean() < 0.1:  # If fraud rate < 10%
            smote = SMOTE(random_state=42)
            X_balanced, y_balanced = smote.fit_resample(X, y)
            self.logger.info(f"Applied SMOTE: {len(X)} -> {len(X_balanced)} samples")
        else:
            X_balanced, y_balanced = X, y
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_balanced)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced
        )
        
        # Create model
        if self.model is None:
            self.model = self.create_model(X_train.shape[1])
        
        # Set global weights if provided (for federated aggregation)
        if global_weights is not None:
            self.model.set_weights(global_weights)
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.local_epochs,
            batch_size=self.config['batch_size'],
            verbose=0,
            callbacks=[
                keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
            ]
        )
        
        # Evaluate model
        val_loss, val_accuracy, val_precision, val_recall = self.model.evaluate(X_val, y_val, verbose=0)
        val_predictions = self.model.predict(X_val)
        val_auc = roc_auc_score(y_val, val_predictions)
        
        # Get model weights (with differential privacy)
        local_weights = self.model.get_weights()
        if self.differential_privacy:
            local_weights = self.add_differential_privacy_noise(local_weights)
        
        training_results = {
            'institution_id': self.institution_id,
            'num_samples': len(X_train),
            'fraud_rate': y_train.mean(),
            'val_loss': val_loss,
            'val_accuracy': val_accuracy,
            'val_precision': val_precision,
            'val_recall': val_recall,
            'val_auc': val_auc,
            'model_weights': local_weights,
            'training_history': {
                'loss': history.history['loss'],
                'val_loss': history.history['val_loss']
            }
        }
        
        self.logger.info(f"Local training complete - AUC: {val_auc:.4f}, Accuracy: {val_accuracy:.4f}")
        
        return training_results

class FederatedAggregator:
    """Central aggregator for federated learning"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.global_model = None
        self.aggregation_history = []
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("FederatedAggregator")
    
    def federated_averaging(self, client_results: List[Dict]) -> List[np.ndarray]:
        """Perform FedAvg aggregation of client model weights"""
        if not client_results:
            raise ValueError("No client results provided")
        
        # Extract weights and sample counts
        all_weights = [result['model_weights'] for result in client_results]
        sample_counts = [result['num_samples'] for result in client_results]
        
        # Calculate total samples
        total_samples = sum(sample_counts)
        
        # Initialize aggregated weights
        aggregated_weights = []
        num_layers = len(all_weights[0])
        
        for layer_idx in range(num_layers):
            # Weighted average of weights for this layer
            layer_weights = []
            for client_idx, client_weights in enumerate(all_weights):
                weight = sample_counts[client_idx] / total_samples
                weighted_layer = client_weights[layer_idx] * weight
                layer_weights.append(weighted_layer)
            
            # Sum all weighted contributions
            aggregated_layer = np.sum(layer_weights, axis=0)
            aggregated_weights.append(aggregated_layer)
        
        self.logger.info(f"Aggregated weights from {len(client_results)} clients")
        
        return aggregated_weights
    
    def evaluate_global_model(self, test_data: Tuple[np.ndarray, np.ndarray], 
                            global_weights: List[np.ndarray]) -> Dict:
        """Evaluate global model on test data"""
        X_test, y_test = test_data
        
        # Create model with global weights
        if self.global_model is None:
            self.global_model = self._create_global_model(X_test.shape[1])
        
        self.global_model.set_weights(global_weights)
        
        # Make predictions
        predictions = self.global_model.predict(X_test, verbose=0)
        binary_predictions = (predictions > 0.5).astype(int)
        
        # Calculate metrics
        test_loss, test_accuracy, test_precision, test_recall = self.global_model.evaluate(
            X_test, y_test, verbose=0
        )
        test_auc = roc_auc_score(y_test, predictions)
        
        results = {
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_auc': test_auc,
            'predictions': predictions.flatten(),
            'binary_predictions': binary_predictions.flatten()
        }
        
        self.logger.info(f"Global model evaluation - AUC: {test_auc:.4f}, Accuracy: {test_accuracy:.4f}")
        
        return results
    
    def _create_global_model(self, input_dim: int):
        """Create global model architecture"""
        model = keras.Sequential([
            keras.layers.Dense(128, activation='relu', input_shape=(input_dim,)),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model

def run_federated_learning_simulation():
    """Run a complete federated learning simulation"""
    print("Starting Federated Learning Fraud Detection Simulation")
    print("=" * 60)
    
    # Configuration
    config = {
        'num_institutions': 5,
        'aggregation_rounds': 10,
        'local_epochs': 3,
        'differential_privacy': True,
        'noise_multiplier': 0.05
    }
    
    # Create institutions (clients)
    institutions = []
    for i in range(config['num_institutions']):
        institution = FederatedFraudDetector(
            institution_id=f"bank_{i+1}",
            config=config
        )
        # Load synthetic data for each institution
        institution.load_local_data(synthetic_data=True)
        institutions.append(institution)
    
    print(f"Created {len(institutions)} participating institutions")
    
    # Create aggregator
    aggregator = FederatedAggregator(config)
    
    # Generate global test data
    print("\nGenerating global test dataset...")
    X_test = np.random.randn(5000, 30)
    y_test = np.random.choice([0, 1], size=5000, p=[0.98, 0.02])
    fraud_indices = y_test == 1
    X_test[fraud_indices] += np.random.randn(fraud_indices.sum(), 30) * 2
    
    # Federated learning rounds
    global_weights = None
    results_history = []
    
    for round_num in range(config['aggregation_rounds']):
        print(f"\n--- Federated Learning Round {round_num + 1} ---")
        
        # Select participating institutions (simulate partial participation)
        participating = np.random.choice(institutions, size=min(3, len(institutions)), replace=False)
        
        # Local training
        client_results = []
        for institution in participating:
            print(f"Training at {institution.institution_id}...")
            result = institution.train_local_model(global_weights)
            client_results.append(result)
        
        # Aggregate weights
        global_weights = aggregator.federated_averaging(client_results)
        
        # Evaluate global model
        scaler = StandardScaler()
        X_test_scaled = scaler.fit_transform(X_test)
        global_results = aggregator.evaluate_global_model((X_test_scaled, y_test), global_weights)
        
        # Store results
        round_results = {
            'round': round_num + 1,
            'participating_institutions': [inst.institution_id for inst in participating],
            'global_metrics': global_results,
            'client_metrics': [
                {
                    'institution': result['institution_id'],
                    'samples': result['num_samples'],
                    'auc': result['val_auc'],
                    'accuracy': result['val_accuracy']
                }
                for result in client_results
            ]
        }
        results_history.append(round_results)
        
        print(f"Global Model Performance - AUC: {global_results['test_auc']:.4f}, "
              f"Accuracy: {global_results['test_accuracy']:.4f}")
    
    # Final results
    print("\n" + "=" * 60)
    print("FEDERATED LEARNING COMPLETE")
    print("=" * 60)
    
    final_results = results_history[-1]['global_metrics']
    print(f"Final Global Model Performance:")
    print(f"  AUC: {final_results['test_auc']:.4f}")
    print(f"  Accuracy: {final_results['test_accuracy']:.4f}")
    print(f"  Precision: {final_results['test_precision']:.4f}")
    print(f"  Recall: {final_results['test_recall']:.4f}")
    
    # Privacy analysis
    print(f"\nPrivacy Protection:")
    print(f"  Differential Privacy: {'Enabled' if config['differential_privacy'] else 'Disabled'}")
    print(f"  Noise Multiplier: {config['noise_multiplier']}")
    print(f"  Data remained local: ✓")
    print(f"  Only model weights shared: ✓")
    
    return results_history, global_weights

def main():
    """Main function"""
    if not FEDERATED_AVAILABLE:
        print("Running simulation without Flower framework...")
    
    # Run federated learning simulation
    results, final_weights = run_federated_learning_simulation()
    
    # Save results
    with open('federated_learning_results.json', 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = []
        for result in results:
            serializable_result = result.copy()
            # Remove non-serializable items
            if 'predictions' in serializable_result['global_metrics']:
                del serializable_result['global_metrics']['predictions']
            if 'binary_predictions' in serializable_result['global_metrics']:
                del serializable_result['global_metrics']['binary_predictions']
            serializable_results.append(serializable_result)
        
        json.dump(serializable_results, f, indent=2)
    
    print("\nResults saved to federated_learning_results.json")

if __name__ == "__main__":
    main()
