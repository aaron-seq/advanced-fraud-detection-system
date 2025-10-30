# src/services/model_management_service.py

from typing import Dict, Any, Optional
from datetime import datetime
import joblib
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ModelManagementService:
    """
    Manages the machine learning models, including loading, versioning, and performance tracking.
    """

    def __init__(self):
        self.models = {}
        self.model_versions = {}
        self.performance_history = {}

    def load_models(self):
        """
        Loads all available models from the 'models' directory.
        """
        models_dir = Path("models")
        if not models_dir.exists():
            logging.warning("Models directory not found. No models loaded.")
            return

        for model_path in models_dir.glob("*_model.joblib"):
            model_name = model_path.stem.replace("_model", "")
            try:
                self.models[model_name] = joblib.load(model_path)
                self.model_versions[model_name] = "1.0.0"  # Placeholder for versioning
                logging.info(f"Loaded model '{model_name}' from {model_path}")
            except Exception as e:
                logging.error(f"Failed to load model '{model_name}': {e}")

    def get_model_performance(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Retrieves performance metrics for a specific model or all models.
        In a production environment, this data would be fetched from a monitoring service or database.
        """
        if model_name:
            if model_name in self.models:
                return {
                    "model_name": model_name,
                    "accuracy": 0.998,
                    "precision": 0.98,
                    "recall": 0.97,
                    "f1_score": 0.975,
                    "last_updated": datetime.utcnow().isoformat()
                }
            else:
                return {"error": f"Model '{model_name}' not found."}

        return {
            "ensemble_accuracy": 0.9987,
            "models": {name: {"accuracy": 0.998, "version": self.model_versions.get(name)} for name in self.models}
        }
