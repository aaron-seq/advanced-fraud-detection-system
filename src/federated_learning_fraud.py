# src/federated_learning_fraud.py

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import logging
from typing import List, Dict, Tuple, Optional
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE
import warnings

warnings.filterwarnings('ignore')

# Configure logging for clear and informative output
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FraudDetectionClient:
    """
    Represents a client in the federated learning system, responsible for local model training.
    """

    def __init__(self, client_id: str, config: Dict = None):
        self.client_id = client_id
        self.config = config or self._default_config()
        self.model = None
        self.scaler = StandardScaler()
        self.local_data = None

        # Differential privacy settings
        self.use_differential_privacy = self.config.get('use_differential_privacy', True)
        self.noise_multiplier = self.config.get('noise_multiplier', 0.1)

    def _default_config(self) -> Dict:
        """Provides a default configuration for the client."""
        return {
            'use_differential_privacy': True,
            'noise_multiplier': 0.1,
            'local_epochs': 5,
            'learning_rate': 0.001,
            'batch_size': 512
        }

    def load_local_data(self, data_path: str = None, is_synthetic: bool = True):
        """Loads local data for the client, with an option to generate synthetic data."""
        if is_synthetic:
            num_samples = np.random.randint(10000, 50000)
            num_features = 30
            X = np.random.randn(num_samples, num_features)
            fraud_rate = np.random.uniform(0.005, 0.02)
            y = np.random.choice([0, 1], size=num_samples, p=[1 - fraud_rate, fraud_rate])

            fraud_indices = y == 1
            X[fraud_indices] += np.random.randn(fraud_indices.sum(), num_features) * 2
        else:
            if data_path and Path(data_path).exists():
                df = pd.read_csv(data_path)
                X = df.drop(['Class'], axis=1).values
                y = df['Class'].values
            else:
                raise ValueError("Data path not found and is_synthetic is False.")

        self.local_data = {'X': X, 'y': y}

    def create_model(self, input_dim: int):
        """Creates a neural network model for fraud detection."""
        model = keras.Sequential([
            keras.layers.Dense(128, activation='relu', input_shape=(input_dim,)),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.config['learning_rate']),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        self.model = model

    def train_local_model(self, global_weights: Optional[List[np.ndarray]] = None) -> Dict:
        """Trains the model on local data and returns the results."""
        if self.local_data is None:
            raise ValueError("Local data not loaded. Call load_local_data() first.")

        X, y = self.local_data['X'], self.local_data['y']

        smote = SMOTE(random_state=42)
        X_balanced, y_balanced = smote.fit_resample(X, y)

        X_scaled = self.scaler.fit_transform(X_balanced)
        X_train, X_val, y_train, y_val = train_test_split(X_scaled, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced)

        if self.model is None:
            self.create_model(X_train.shape[1])

        if global_weights:
            self.model.set_weights(global_weights)

        self.model.fit(X_train, y_train, validation_data=(X_val, y_val),
                       epochs=self.config['local_epochs'], batch_size=self.config['batch_size'], verbose=0)

        val_loss, val_accuracy = self.model.evaluate(X_val, y_val, verbose=0)
        val_auc = roc_auc_score(y_val, self.model.predict(X_val))

        local_weights = self.model.get_weights()
        if self.use_differential_privacy:
            local_weights = [w + np.random.normal(0, self.noise_multiplier, w.shape) for w in local_weights]

        return {
            'client_id': self.client_id,
            'num_samples': len(X_train),
            'val_auc': val_auc,
            'val_accuracy': val_accuracy,
            'model_weights': local_weights
        }

class FederatedServer:
    """
    Coordinates the federated learning process, including model aggregation and evaluation.
    """

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.global_model = None

    def aggregate_weights(self, client_results: List[Dict]) -> List[np.ndarray]:
        """Performs federated averaging on client model weights."""
        if not client_results:
            raise ValueError("No client results provided for aggregation.")

        total_samples = sum(result['num_samples'] for result in client_results)

        aggregated_weights = []
        for i in range(len(client_results[0]['model_weights'])):
            weighted_sum = np.zeros_like(client_results[0]['model_weights'][i])
            for result in client_results:
                weight = result['num_samples'] / total_samples
                weighted_sum += result['model_weights'][i] * weight
            aggregated_weights.append(weighted_sum)

        return aggregated_weights

    def evaluate_global_model(self, test_data: Tuple[np.ndarray, np.ndarray], global_weights: List[np.ndarray]) -> Dict:
        """Evaluates the global model on a centralized test dataset."""
        X_test, y_test = test_data

        if self.global_model is None:
            self.global_model = self._create_global_model(X_test.shape[1])

        self.global_model.set_weights(global_weights)

        predictions = self.global_model.predict(X_test, verbose=0).flatten()
        test_auc = roc_auc_score(y_test, predictions)

        return {'test_auc': test_auc}

    def _create_global_model(self, input_dim: int):
        """Creates the global model architecture."""
        model = keras.Sequential([
            keras.layers.Dense(128, activation='relu', input_shape=(input_dim,)),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

def main():
    """Main function to run the federated learning simulation."""
    config = {
        'num_clients': 5,
        'num_rounds': 10,
        'local_epochs': 3,
        'use_differential_privacy': True,
        'noise_multiplier': 0.05
    }

    clients = [FraudDetectionClient(client_id=f"client_{i+1}", config=config) for i in range(config['num_clients'])]
    for client in clients:
        client.load_local_data(is_synthetic=True)

    server = FederatedServer(config)

    X_test = np.random.randn(5000, 30)
    y_test = np.random.choice([0, 1], size=5000, p=[0.98, 0.02])

    global_weights = None
    for round_num in range(config['num_rounds']):
        logging.info(f"--- Round {round_num + 1} ---")

        client_results = [client.train_local_model(global_weights) for client in clients]
        global_weights = server.aggregate_weights(client_results)

        scaler = StandardScaler()
        X_test_scaled = scaler.fit_transform(X_test)
        global_metrics = server.evaluate_global_model((X_test_scaled, y_test), global_weights)

        logging.info(f"Global Model AUC: {global_metrics['test_auc']:.4f}")

if __name__ == "__main__":
    main()
