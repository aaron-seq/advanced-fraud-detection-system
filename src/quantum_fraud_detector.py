# src/quantum_fraud_detector.py

import numpy as np
import pandas as pd
from typing import Dict, Tuple
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.ensemble import RandomForestClassifier

try:
    import pennylane as qml
    from pennylane import numpy as qnp
    QUANTUM_AVAILABLE = True
except ImportError:
    QUANTUM_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class QuantumFraudDetector:
    """
    A Quantum Machine Learning system for fraud detection, designed for modularity and clarity.
    """

    def __init__(self, config: Dict = None):
        self.config = config or self._default_config()
        self.quantum_model = None
        self.scaler = MinMaxScaler(feature_range=(0, np.pi))

        if QUANTUM_AVAILABLE:
            self.device = qml.device(self.config['quantum_device'], wires=self.config['n_qubits'])
        else:
            logging.warning("Quantum libraries not available. Using classical fallback.")

    def _default_config(self) -> Dict:
        """Provides a default configuration for the quantum detector."""
        return {
            'n_qubits': 8,
            'n_layers': 3,
            'quantum_device': 'default.qubit',
            'learning_rate': 0.01,
            'epochs': 50
        }

    def preprocess_data(self, X: np.ndarray) -> np.ndarray:
        """Preprocesses data for quantum circuits."""
        from sklearn.decomposition import PCA
        pca = PCA(n_components=self.config['n_qubits'])
        X_reduced = pca.fit_transform(X)
        return self.scaler.fit_transform(X_reduced)

    def create_hybrid_model(self):
        """Creates a hybrid quantum-classical model."""
        if not QUANTUM_AVAILABLE:
            return RandomForestClassifier(n_estimators=100, random_state=42)

        @qml.qnode(self.device)
        def quantum_circuit(inputs, weights):
            for i in range(self.config['n_qubits']):
                qml.RX(inputs[i], wires=i)
            for layer in range(self.config['n_layers']):
                for i in range(self.config['n_qubits'] - 1):
                    qml.CNOT(wires=[i, i + 1])
                for i in range(self.config['n_qubits']):
                    qml.RY(weights[layer, i], wires=i)
            return [qml.expval(qml.PauliZ(wires=i)) for i in range(self.config['n_qubits'])]

        self.quantum_model = {'circuit': quantum_circuit}
        return self.quantum_model

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Trains the quantum classifier."""
        X_processed = self.preprocess_data(X_train)

        self.create_hybrid_model()

        if not QUANTUM_AVAILABLE:
            self.quantum_model.fit(X_processed, y_train)
            return

        weights = np.random.uniform(0, 2 * np.pi, size=(self.config['n_layers'], self.config['n_qubits']))
        opt = qml.AdamOptimizer(stepsize=self.config['learning_rate'])

        def cost(weights):
            predictions = np.array([1 / (1 + np.exp(-np.sum(self.quantum_model['circuit'](x, weights)))) for x in X_processed])
            return -np.mean(y_train * np.log(predictions) + (1 - y_train) * np.log(1 - predictions))

        for epoch in range(self.config['epochs']):
            weights, loss = opt.step_and_cost(cost, weights)
            if epoch % 10 == 0:
                logging.info(f"Epoch {epoch}: Loss = {loss:.4f}")

        self.quantum_model['weights'] = weights

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Makes predictions using the trained quantum model."""
        X_processed = self.preprocess_data(X_test)

        if not QUANTUM_AVAILABLE:
            return self.quantum_model.predict(X_processed)

        predictions = [1 / (1 + np.exp(-np.sum(self.quantum_model['circuit'](x, self.quantum_model['weights'])))) for x in X_processed]
        return (np.array(predictions) > 0.5).astype(int)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Evaluates the performance of the quantum model."""
        predictions = self.predict(X_test)

        auc = roc_auc_score(y_test, predictions)
        report = classification_report(y_test, predictions, output_dict=True)

        return {'auc': auc, 'classification_report': report}

def main():
    """Main function to run the quantum fraud detection demo."""
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=2000, n_features=16, n_informative=8, n_redundant=0, random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    detector = QuantumFraudDetector()
    detector.train(X_train, y_train)
    results = detector.evaluate(X_test, y_test)

    logging.info(f"Quantum Model AUC: {results['auc']:.4f}")
    logging.info(f"Classification Report:\n{pd.DataFrame(results['classification_report']).transpose()}")

if __name__ == "__main__":
    main()
