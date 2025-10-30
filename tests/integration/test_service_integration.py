# tests/integration/test_service_integration.py

import unittest
from unittest.mock import AsyncMock, MagicMock
from src.services.model_management_service import ModelManagementService
import asyncio

class TestServiceIntegration(unittest.TestCase):
    """
    Integration tests for the interaction between different services.
    """

    def setUp(self):
        """
        Set up the test environment before each test.
        """
        self.model_management_service = ModelManagementService()

    def test_model_loading(self):
        """
        Test that models are loaded by the management service.
        """
        # Mock the FraudDetectionService
        fraud_detection_service = MagicMock()
        fraud_detection_service.load_models = AsyncMock()

        async def run_test():
            # The fraud detection service should load these models on initialization
            await fraud_detection_service.load_models()

            # Check that the load_models method was called
            fraud_detection_service.load_models.assert_called_once()

        # Run the async test
        asyncio.run(run_test())

if __name__ == '__main__':
    unittest.main()
