#!/usr/bin/env python3
"""
Quantum Machine Learning for Credit Card Fraud Detection
Implementing variational quantum circuits and hybrid quantum-classical models
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
import json
import time

# Classical ML libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import roc_auc_score, classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

# Quantum computing libraries
try:
    import pennylane as qml
    from pennylane import numpy as qnp
    import qiskit
    from qiskit import QuantumCircuit, Aer, transpile
    from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
    from qiskit.algorithms.optimizers import SPSA, COBYLA
    from qiskit_machine_learning.algorithms import VQC
    from qiskit_machine_learning.neural_networks import CircuitQNN
    QUANTUM_AVAILABLE = True
except ImportError:
    QUANTUM_AVAILABLE = False
    print("Warning: Quantum libraries not available. Install qiskit and pennylane for quantum features.")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QuantumFraudDetector:
    """Quantum Machine Learning system for fraud detection"""
    
    def __init__(self, config: Dict = None):
        self.config = config or self._default_config()
        self.quantum_model = None
        self.classical_model = None
        self.scaler = MinMaxScaler(feature_range=(0, np.pi))  # Scale to quantum range
        
        # Quantum parameters
        self.n_qubits = self.config.get('n_qubits', 8)
        self.n_layers = self.config.get('n_layers', 3)
        self.quantum_device = self.config.get('quantum_device', 'default.qubit')
        
        # Setup quantum device
        if QUANTUM_AVAILABLE:
            self.device = qml.device(self.quantum_device, wires=self.n_qubits)
            logger.info(f"Initialized quantum device: {self.quantum_device} with {self.n_qubits} qubits")
        else:
            logger.warning("Quantum libraries not available - using classical fallback")
    
    def _default_config(self):
        return {
            'n_qubits': 8,
            'n_layers': 3,
            'quantum_device': 'default.qubit',
            'optimizer': 'adam',
            'learning_rate': 0.01,
            'max_iterations': 200,
            'use_hybrid': True,
            'feature_reduction': 'pca',
            'n_components': 8
        }
    
    def preprocess_data(self, X: np.ndarray, y: np.ndarray = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Preprocess data for quantum circuits"""
        logger.info("Preprocessing data for quantum circuits...")
        
        # Feature reduction for quantum efficiency
        if self.config.get('feature_reduction') == 'pca':
            from sklearn.decomposition import PCA
            pca = PCA(n_components=self.config.get('n_components', self.n_qubits))
            X_reduced = pca.fit_transform(X)
            logger.info(f"Reduced features from {X.shape[1]} to {X_reduced.shape[1]} using PCA")
        else:
            # Select first n_qubits features
            X_reduced = X[:, :self.n_qubits]
        
        # Scale features to quantum range [0, π]
        X_scaled = self.scaler.fit_transform(X_reduced)
        
        return X_scaled, y
    
    def create_quantum_circuit(self) -> callable:
        """Create variational quantum circuit for classification"""
        if not QUANTUM_AVAILABLE:
            raise RuntimeError("Quantum libraries not available")
        
        @qml.qnode(self.device)
        def quantum_circuit(inputs, weights):
            # Data encoding layer
            for i in range(self.n_qubits):
                qml.RX(inputs[i], wires=i)
            
            # Variational layers
            for layer in range(self.n_layers):
                # Entangling layer
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                qml.CNOT(wires=[self.n_qubits - 1, 0])  # Circular entanglement
                
                # Rotation layer
                for i in range(self.n_qubits):
                    qml.RY(weights[layer, i, 0], wires=i)
                    qml.RZ(weights[layer, i, 1], wires=i)
            
            # Measurement
            return [qml.expval(qml.PauliZ(wires=i)) for i in range(self.n_qubits)]
        
        return quantum_circuit
    
    def create_hybrid_model(self, input_dim: int):
        """Create hybrid quantum-classical model"""
        if not QUANTUM_AVAILABLE:
            logger.warning("Creating classical fallback model")
            return RandomForestClassifier(n_estimators=100, random_state=42)
        
        logger.info("Creating hybrid quantum-classical model...")
        
        class HybridQuantumClassifier:
            def __init__(self, quantum_circuit, n_qubits, n_layers):
                self.quantum_circuit = quantum_circuit
                self.n_qubits = n_qubits
                self.n_layers = n_layers
                
                # Initialize quantum weights
                self.quantum_weights = np.random.uniform(
                    0, 2*np.pi, size=(n_layers, n_qubits, 2)
                )
                
                # Classical post-processing layer
                self.classical_weights = np.random.randn(n_qubits, 1)
                self.classical_bias = np.random.randn(1)
                
                self.training_history = []
            
            def forward(self, X):
                """Forward pass through hybrid model"""
                batch_size = X.shape[0]
                quantum_outputs = np.zeros((batch_size, self.n_qubits))
                
                # Process each sample through quantum circuit
                for i, x in enumerate(X):
                    quantum_outputs[i] = self.quantum_circuit(x, self.quantum_weights)
                
                # Classical post-processing
                classical_output = np.dot(quantum_outputs, self.classical_weights) + self.classical_bias
                
                # Sigmoid activation
                probabilities = 1 / (1 + np.exp(-classical_output.flatten()))
                
                return probabilities
            
            def fit(self, X, y, epochs=100, learning_rate=0.01):
                """Train the hybrid model"""
                logger.info(f"Training hybrid quantum-classical model for {epochs} epochs...")
                
                for epoch in range(epochs):
                    # Forward pass
                    predictions = self.forward(X)
                    
                    # Calculate loss (binary cross-entropy)
                    loss = -np.mean(y * np.log(predictions + 1e-8) + 
                                  (1 - y) * np.log(1 - predictions + 1e-8))
                    
                    # Calculate accuracy
                    binary_predictions = (predictions > 0.5).astype(int)
                    accuracy = np.mean(binary_predictions == y)
                    
                    # Store metrics
                    self.training_history.append({
                        'epoch': epoch,
                        'loss': loss,
                        'accuracy': accuracy
                    })
                    
                    # Simple gradient estimation and update (parameter-shift rule approximation)
                    if epoch % 20 == 0:
                        logger.info(f"Epoch {epoch}: Loss = {loss:.4f}, Accuracy = {accuracy:.4f}")
                    
                    # Update weights using finite differences
                    self._update_weights(X, y, learning_rate)
                
                logger.info("Training completed!")
                return self.training_history
            
            def _update_weights(self, X, y, learning_rate):
                """Update quantum and classical weights"""
                # Update quantum weights using parameter-shift rule approximation
                shift = np.pi / 2
                
                for layer in range(self.n_layers):
                    for qubit in range(self.n_qubits):
                        for param in range(2):
                            # Forward shift
                            self.quantum_weights[layer, qubit, param] += shift
                            forward_pred = self.forward(X)
                            forward_loss = -np.mean(y * np.log(forward_pred + 1e-8) + 
                                                  (1 - y) * np.log(1 - forward_pred + 1e-8))
                            
                            # Backward shift
                            self.quantum_weights[layer, qubit, param] -= 2 * shift
                            backward_pred = self.forward(X)
                            backward_loss = -np.mean(y * np.log(backward_pred + 1e-8) + 
                                                   (1 - y) * np.log(1 - backward_pred + 1e-8))
                            
                            # Restore original value
                            self.quantum_weights[layer, qubit, param] += shift
                            
                            # Gradient approximation
                            gradient = (forward_loss - backward_loss) / 2
                            
                            # Update weight
                            self.quantum_weights[layer, qubit, param] -= learning_rate * gradient
                
                # Update classical weights
                predictions = self.forward(X)
                quantum_outputs = np.zeros((X.shape[0], self.n_qubits))
                
                for i, x in enumerate(X):
                    quantum_outputs[i] = self.quantum_circuit(x, self.quantum_weights)
                
                # Classical gradient
                error = predictions - y
                classical_gradient = np.dot(quantum_outputs.T, error) / X.shape[0]
                bias_gradient = np.mean(error)
                
                # Update classical weights
                self.classical_weights -= learning_rate * classical_gradient.reshape(-1, 1)
                self.classical_bias -= learning_rate * bias_gradient
            
            def predict(self, X):
                """Make predictions"""
                probabilities = self.forward(X)
                return (probabilities > 0.5).astype(int)
            
            def predict_proba(self, X):
                """Get prediction probabilities"""
                probabilities = self.forward(X)
                return np.column_stack([1 - probabilities, probabilities])
        
        quantum_circuit = self.create_quantum_circuit()
        return HybridQuantumClassifier(quantum_circuit, self.n_qubits, self.n_layers)
    
    def create_qiskit_vqc(self, input_dim: int):
        """Create Variational Quantum Classifier using Qiskit"""
        if not QUANTUM_AVAILABLE:
            return None
        
        try:
            from qiskit_machine_learning.algorithms import VQC
            from qiskit.circuit.library import TwoLocal
            from qiskit.algorithms.optimizers import SPSA
            
            logger.info("Creating Qiskit VQC model...")
            
            # Feature map
            feature_map = ZZFeatureMap(feature_dimension=self.n_qubits, reps=2)
            
            # Ansatz (variational form)
            ansatz = TwoLocal(
                num_qubits=self.n_qubits,
                rotation_blocks=['ry', 'rz'],
                entanglement_blocks='cz',
                entanglement='linear',
                reps=self.n_layers
            )
            
            # Optimizer
            optimizer = SPSA(maxiter=self.config.get('max_iterations', 100))
            
            # Create VQC
            vqc = VQC(
                feature_map=feature_map,
                ansatz=ansatz,
                optimizer=optimizer,
                quantum_instance=Aer.get_backend('qasm_simulator')
            )
            
            return vqc
            
        except Exception as e:
            logger.error(f"Error creating Qiskit VQC: {e}")
            return None
    
    def train_quantum_classifier(self, X_train: np.ndarray, y_train: np.ndarray, 
                                validation_data: Optional[Tuple] = None):
        """Train the quantum classifier"""
        logger.info("Training quantum fraud classifier...")
        
        # Preprocess data
        X_train_processed, y_train_processed = self.preprocess_data(X_train, y_train)
        
        if self.config.get('use_hybrid', True):
            # Train hybrid quantum-classical model
            self.quantum_model = self.create_hybrid_model(X_train_processed.shape[1])
            
            if hasattr(self.quantum_model, 'fit'):
                training_history = self.quantum_model.fit(
                    X_train_processed, 
                    y_train_processed,
                    epochs=self.config.get('max_iterations', 100),
                    learning_rate=self.config.get('learning_rate', 0.01)
                )
                return training_history
            else:
                # Classical fallback
                self.quantum_model.fit(X_train_processed, y_train_processed)
                return None
        else:
            # Pure quantum approach with Qiskit VQC
            self.quantum_model = self.create_qiskit_vqc(X_train_processed.shape[1])
            
            if self.quantum_model:
                try:
                    self.quantum_model.fit(X_train_processed, y_train_processed)
                    logger.info("Qiskit VQC training completed!")
                except Exception as e:
                    logger.error(f"Error training Qiskit VQC: {e}")
                    # Fallback to classical
                    self.quantum_model = RandomForestClassifier(n_estimators=100, random_state=42)
                    self.quantum_model.fit(X_train_processed, y_train_processed)
            
            return None
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Make predictions using trained quantum model"""
        if self.quantum_model is None:
            raise ValueError("Model not trained. Call train_quantum_classifier() first.")
        
        # Preprocess test data
        X_test_processed, _ = self.preprocess_data(X_test)
        
        # Make predictions
        predictions = self.quantum_model.predict(X_test_processed)
        
        return predictions
    
    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """Get prediction probabilities"""
        if self.quantum_model is None:
            raise ValueError("Model not trained. Call train_quantum_classifier() first.")
        
        # Preprocess test data
        X_test_processed, _ = self.preprocess_data(X_test)
        
        # Get probabilities
        if hasattr(self.quantum_model, 'predict_proba'):
            probabilities = self.quantum_model.predict_proba(X_test_processed)
        else:
            # For models without predict_proba, use decision function or predictions
            predictions = self.quantum_model.predict(X_test_processed)
            probabilities = np.column_stack([1 - predictions, predictions])
        
        return probabilities
    
    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Evaluate quantum model performance"""
        logger.info("Evaluating quantum fraud detection model...")
        
        # Make predictions
        predictions = self.predict(X_test)
        probabilities = self.predict_proba(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, predictions)
        auc = roc_auc_score(y_test, probabilities[:, 1])
        
        # Classification report
        class_report = classification_report(y_test, predictions, output_dict=True)
        
        results = {
            'accuracy': accuracy,
            'auc': auc,
            'classification_report': class_report,
            'predictions': predictions,
            'probabilities': probabilities[:, 1]
        }
        
        logger.info(f"Quantum Model - Accuracy: {accuracy:.4f}, AUC: {auc:.4f}")
        
        return results
    
    def quantum_advantage_analysis(self, X_train: np.ndarray, y_train: np.ndarray,
                                 X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Analyze potential quantum advantage"""
        logger.info("Analyzing quantum advantage...")
        
        results = {
            'quantum_metrics': {},
            'classical_metrics': {},
            'comparison': {}
        }
        
        # Train and evaluate quantum model
        start_time = time.time()
        self.train_quantum_classifier(X_train, y_train)
        quantum_train_time = time.time() - start_time
        
        start_time = time.time()
        quantum_results = self.evaluate_model(X_test, y_test)
        quantum_inference_time = time.time() - start_time
        
        results['quantum_metrics'] = {
            'accuracy': quantum_results['accuracy'],
            'auc': quantum_results['auc'],
            'training_time': quantum_train_time,
            'inference_time': quantum_inference_time
        }
        
        # Train and evaluate classical baseline
        classical_model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        start_time = time.time()
        classical_model.fit(X_train, y_train)
        classical_train_time = time.time() - start_time
        
        start_time = time.time()
        classical_predictions = classical_model.predict(X_test)
        classical_probabilities = classical_model.predict_proba(X_test)[:, 1]
        classical_inference_time = time.time() - start_time
        
        classical_accuracy = accuracy_score(y_test, classical_predictions)
        classical_auc = roc_auc_score(y_test, classical_probabilities)
        
        results['classical_metrics'] = {
            'accuracy': classical_accuracy,
            'auc': classical_auc,
            'training_time': classical_train_time,
            'inference_time': classical_inference_time
        }
        
        # Comparison
        results['comparison'] = {
            'accuracy_improvement': quantum_results['accuracy'] - classical_accuracy,
            'auc_improvement': quantum_results['auc'] - classical_auc,
            'training_speedup': classical_train_time / quantum_train_time,
            'inference_speedup': classical_inference_time / quantum_inference_time,
            'quantum_advantage': quantum_results['auc'] > classical_auc
        }
        
        logger.info(f"Quantum vs Classical Comparison:")
        logger.info(f"  Accuracy: {quantum_results['accuracy']:.4f} vs {classical_accuracy:.4f}")
        logger.info(f"  AUC: {quantum_results['auc']:.4f} vs {classical_auc:.4f}")
        logger.info(f"  Quantum Advantage: {results['comparison']['quantum_advantage']}")
        
        return results
    
    def save_model(self, filepath: str):
        """Save quantum model"""
        model_data = {
            'config': self.config,
            'scaler_params': {
                'scale_': self.scaler.scale_.tolist() if hasattr(self.scaler, 'scale_') else None,
                'min_': self.scaler.min_.tolist() if hasattr(self.scaler, 'min_') else None,
                'data_min_': self.scaler.data_min_.tolist() if hasattr(self.scaler, 'data_min_') else None,
                'data_max_': self.scaler.data_max_.tolist() if hasattr(self.scaler, 'data_max_') else None,
            },
            'model_type': 'quantum' if QUANTUM_AVAILABLE and hasattr(self.quantum_model, 'quantum_weights') else 'classical'
        }
        
        # Save quantum weights if available
        if hasattr(self.quantum_model, 'quantum_weights'):
            model_data['quantum_weights'] = self.quantum_model.quantum_weights.tolist()
            model_data['classical_weights'] = self.quantum_model.classical_weights.tolist()
            model_data['classical_bias'] = self.quantum_model.classical_bias.tolist()
        
        with open(filepath, 'w') as f:
            json.dump(model_data, f, indent=2)
        
        logger.info(f"Quantum model saved to {filepath}")

def generate_synthetic_fraud_data(n_samples: int = 5000) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic fraud detection data"""
    logger.info(f"Generating {n_samples} synthetic fraud transactions...")
    
    # Generate features
    X = np.random.randn(n_samples, 16)  # Reduced features for quantum efficiency
    
    # Create fraud pattern (complex non-linear relationship)
    fraud_pattern = (
        (X[:, 0] * X[:, 1] > 1.5) |  # Feature interaction
        (np.sum(X[:, :4], axis=1) > 3) |  # Sum threshold
        (np.linalg.norm(X[:, 4:8], axis=1) > 2.5)  # Distance threshold
    )
    
    # Add noise and make fraud rare
    y = fraud_pattern.astype(int)
    fraud_indices = np.where(y == 1)[0]
    
    # Keep only 2% as fraud
    n_fraud = int(0.02 * n_samples)
    if len(fraud_indices) > n_fraud:
        keep_fraud = np.random.choice(fraud_indices, n_fraud, replace=False)
        y = np.zeros(n_samples, dtype=int)
        y[keep_fraud] = 1
    
    logger.info(f"Generated data with {y.sum()} fraud cases ({y.mean()*100:.2f}%)")
    
    return X, y

def main():
    """Main function for quantum fraud detection demo"""
    print("Quantum Machine Learning Credit Card Fraud Detection")
    print("=" * 60)
    
    if not QUANTUM_AVAILABLE:
        print("Quantum libraries not available. Running classical simulation.")
    
    # Configuration
    config = {
        'n_qubits': 8,
        'n_layers': 3,
        'quantum_device': 'default.qubit',
        'max_iterations': 50,  # Reduced for demo
        'learning_rate': 0.01,
        'use_hybrid': True
    }
    
    # Generate synthetic data
    X, y = generate_synthetic_fraud_data(n_samples=2000)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Initialize quantum detector
    quantum_detector = QuantumFraudDetector(config)
    
    try:
        # Quantum advantage analysis
        advantage_results = quantum_detector.quantum_advantage_analysis(
            X_train, y_train, X_test, y_test
        )
        
        print("\n" + "="*60)
        print("QUANTUM ADVANTAGE ANALYSIS RESULTS")
        print("="*60)
        
        print(f"Quantum Model Performance:")
        print(f"  Accuracy: {advantage_results['quantum_metrics']['accuracy']:.4f}")
        print(f"  AUC: {advantage_results['quantum_metrics']['auc']:.4f}")
        
        print(f"\nClassical Baseline Performance:")
        print(f"  Accuracy: {advantage_results['classical_metrics']['accuracy']:.4f}")
        print(f"  AUC: {advantage_results['classical_metrics']['auc']:.4f}")
        
        print(f"\nComparison:")
        print(f"  Accuracy Improvement: {advantage_results['comparison']['accuracy_improvement']:.4f}")
        print(f"  AUC Improvement: {advantage_results['comparison']['auc_improvement']:.4f}")
        print(f"  Quantum Advantage: {advantage_results['comparison']['quantum_advantage']}")
        
        # Save results
        with open('quantum_fraud_results.json', 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_results = {}
            for key, value in advantage_results.items():
                if isinstance(value, dict):
                    serializable_results[key] = {}
                    for k, v in value.items():
                        if isinstance(v, np.ndarray):
                            serializable_results[key][k] = v.tolist()
                        else:
                            serializable_results[key][k] = v
                else:
                    serializable_results[key] = value
            
            json.dump(serializable_results, f, indent=2)
        
        # Save model
        quantum_detector.save_model('quantum_fraud_model.json')
        
        print("\nResults and model saved successfully!")
        
    except Exception as e:
        logger.error(f"Error in quantum fraud detection: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
