# tests/unit/test_federated_learning.py

import unittest
import numpy as np
from src.federated_learning_fraud import FraudDetectionClient, FederatedServer

class TestFederatedLearning(unittest.TestCase):
    """
    Unit tests for the federated learning components.
    """

    def setUp(self):
        """
        Set up the test environment before each test.
        """
        self.client_config = {
            'local_epochs': 1,
            'batch_size': 32,
            'learning_rate': 0.001
        }
        self.client = FraudDetectionClient('test_client', self.client_config)
        self.server = FederatedServer()

    def test_client_data_loading(self):
        """
        Test that the client can load synthetic data.
        """
        self.client.load_local_data(is_synthetic=True)
        self.assertIsNotNone(self.client.local_data)
        self.assertIn('X', self.client.local_data)

    def test_client_model_training(self):
        """
        Test the local model training process on a client.
        """
        self.client.load_local_data(is_synthetic=True)
        training_results = self.client.train_local_model()

        self.assertIn('model_weights', training_results)
        self.assertGreater(training_results['val_auc'], 0.5)

    def test_server_aggregation(self):
        """
        Test the server's ability to aggregate model weights.
        """
        # Simulate results from two clients
        client1 = FraudDetectionClient('client1', self.client_config)
        client1.load_local_data(is_synthetic=True)
        results1 = client1.train_local_model()

        client2 = FraudDetectionClient('client2', self.client_config)
        client2.load_local_data(is_synthetic=True)
        results2 = client2.train_local_model()

        aggregated_weights = self.server.aggregate_weights([results1, results2])

        # Check that the aggregated weights have the correct structure
        self.assertEqual(len(aggregated_weights), len(results1['model_weights']))
        self.assertEqual(aggregated_weights[0].shape, results1['model_weights'][0].shape)

if __name__ == '__main__':
    unittest.main()
