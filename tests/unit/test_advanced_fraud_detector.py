# tests/unit/test_advanced_fraud_detector.py

import unittest
import numpy as np
import pandas as pd
from src.advanced_fraud_detector import AdvancedFraudDetector

class TestAdvancedFraudDetector(unittest.TestCase):
    """
    Unit tests for the AdvancedFraudDetector class.
    """

    def setUp(self):
        """
        Set up the test environment before each test.
        """
        self.config = {
            'ensemble_methods': ['xgb', 'lgb'],
            'optimization_trials': 2  # Keep trials low for testing
        }
        self.detector = AdvancedFraudDetector(self.config)

        # Create a dummy dataset for testing
        self.dummy_data = pd.DataFrame({
            'Time': np.arange(100),
            'V1': np.random.rand(100),
            'V2': np.random.rand(100),
            'Amount': np.random.rand(100) * 1000,
            'Class': np.random.randint(0, 2, 100)
        })
        self.dummy_data.to_csv('dummy_transactions.csv', index=False)

    def test_load_and_preprocess_data(self):
        """
        Test that data is loaded and preprocessed correctly.
        """
        features, target, df = self.detector.load_and_preprocess_data('dummy_transactions.csv')

        self.assertIsNotNone(features)
        self.assertIsNotNone(target)
        self.assertEqual(features.shape[0], 100)
        self.assertEqual(target.shape[0], 100)
        # Ensure 'Time' column is dropped
        self.assertNotIn('Time', df.columns)

    def test_train_ensemble(self):
        """
        Test the ensemble training process.
        """
        features, target, _ = self.detector.load_and_preprocess_data('dummy_transactions.csv')

        # A small subset for faster testing
        X_train, y_train = features[:50], target[:50]

        trained_models = self.detector.train_ensemble(X_train, y_train)

        # Check that models are trained and stored
        self.assertIn('xgb', trained_models)
        self.assertIn('lgb', trained_models)
        self.assertIsNotNone(trained_models['xgb'])
        self.assertIsNotNone(trained_models['lgb'])

    def tearDown(self):
        """
        Clean up the test environment after each test.
        """
        import os
        os.remove('dummy_transactions.csv')

if __name__ == '__main__':
    unittest.main()
