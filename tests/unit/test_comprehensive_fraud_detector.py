# tests/unit/test_comprehensive_fraud_detector.py

import unittest
import numpy as np
import pandas as pd
from src.comprehensive_fraud_detector import ComprehensiveFraudDetector

class TestComprehensiveFraudDetector(unittest.TestCase):
    """
    Unit tests for the ComprehensiveFraudDetector class.
    """

    def setUp(self):
        """
        Set up the test environment before each test.
        """
        self.config = {
            'use_transformers': False,
            'ensemble_methods': ['xgb', 'lgb'],
        }
        self.detector = ComprehensiveFraudDetector(self.config)

        # Create a dummy dataset for testing
        self.dummy_data = pd.DataFrame({
            'Time': np.arange(100),
            'V1': np.random.rand(100),
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
        self.assertIn('Amount_log', df.columns) # Check feature engineering

    def test_train_ensemble(self):
        """
        Test the ensemble training process.
        """
        features, target, _ = self.detector.load_and_preprocess_data('dummy_transactions.csv')
        X_train, y_train = features.iloc[:50], target.iloc[:50]

        trained_models = self.detector.train_ensemble(X_train, y_train)

        self.assertIn('xgb', trained_models)
        self.assertIn('lgb', trained_models)

    def tearDown(self):
        """
        Clean up the test environment after each test.
        """
        import os
        os.remove('dummy_transactions.csv')

if __name__ == '__main__':
    unittest.main()
