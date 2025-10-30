# tests/unit/test_gnn_system.py

import unittest
import pandas as pd
import numpy as np
from src.graph_neural_network_fraud import GNNSystem

class TestGNNSystem(unittest.TestCase):
    """
    Unit tests for the GNNSystem class.
    """

    def setUp(self):
        """
        Set up the test environment before each test.
        """
        self.config = {
            'model_type': 'GCN',
            'hidden_dim': 16,
            'num_layers': 2,
            'epochs': 2,  # Keep epochs low for testing
            'dropout': 0.3,
            'learning_rate': 0.01
        }
        self.gnn_system = GNNSystem(self.config)

        # Create a dummy dataset for testing
        self.dummy_data = pd.DataFrame({
            'Time': np.arange(100),
            'V1': np.random.rand(100),
            'Amount': np.random.rand(100) * 1000,
            'Class': np.random.randint(0, 2, 100)
        })

    def test_train_and_evaluate(self):
        """
        Test the full training and evaluation pipeline of the GNN.
        """
        results = self.gnn_system.train(self.dummy_data)

        self.assertIn('auc', results)
        self.assertIn('classification_report', results)
        self.assertGreater(results['auc'], 0.5) # Should be better than random

if __name__ == '__main__':
    unittest.main()
