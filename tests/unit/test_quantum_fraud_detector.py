# tests/unit/test_quantum_fraud_detector.py

import unittest
import numpy as np
from src.quantum_fraud_detector import QuantumFraudDetector

class TestQuantumFraudDetector(unittest.TestCase):
    """
    Unit tests for the QuantumFraudDetector class.
    """

    def setUp(self):
        """
        Set up the test environment before each test.
        """
        self.config = {
            'n_qubits': 4,
            'n_layers': 1,
            'epochs': 2,
            'quantum_device': 'default.qubit',
            'learning_rate': 0.01
        }
        self.detector = QuantumFraudDetector(self.config)

        # Create synthetic data for testing
        self.X_train = np.random.rand(20, 8)
        self.y_train = np.random.randint(0, 2, 20)

    def test_preprocess_data(self):
        """
        Test the quantum-specific data preprocessing.
        """
        processed_X = self.detector.preprocess_data(self.X_train)

        self.assertEqual(processed_X.shape[1], self.config['n_qubits'])
        # Check that data is scaled to [0, pi]
        self.assertTrue(np.all(processed_X >= 0))
        self.assertTrue(np.all(processed_X <= np.pi))

    def test_train(self):
        """
        Test the training process of the quantum model.
        """
        # This will test the classical fallback if quantum libraries are not installed
        self.detector.train(self.X_train, self.y_train)
        self.assertIsNotNone(self.detector.quantum_model)

if __name__ == '__main__':
    unittest.main()
