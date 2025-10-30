# src/explainable_ai_fraud.py

import shap
import pandas as pd
from typing import Dict, Any, List
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ExplainableAI:
    """
    Provides explainability for the fraud detection models using SHAP (SHapley Additive exPlanations).
    """

    def __init__(self, model, feature_names: List[str]):
        if not hasattr(model, 'predict_proba'):
            raise ValueError("Model must have a 'predict_proba' method for explainability.")

        self.model = model
        self.feature_names = feature_names
        self.explainer = shap.TreeExplainer(model)

    def get_explanation(self, transaction_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generates a detailed explanation for a single transaction.

        Args:
            transaction_data (pd.DataFrame): A DataFrame containing the transaction to be explained.

        Returns:
            Dict[str, Any]: A dictionary containing the SHAP values and a summary of the key factors.
        """
        if not isinstance(transaction_data, pd.DataFrame) or transaction_data.shape[0] != 1:
            raise ValueError("Input must be a DataFrame with a single row.")

        shap_values = self.explainer.shap_values(transaction_data)

        # For classification, shap_values is a list of arrays (one for each class)
        # We are interested in the SHAP values for the "fraud" class (class 1)
        shap_values_for_fraud = shap_values[1][0]

        feature_impacts = {
            feature: impact for feature, impact in zip(self.feature_names, shap_values_for_fraud)
        }

        # Sort features by the absolute magnitude of their SHAP value
        sorted_factors = sorted(feature_impacts.items(), key=lambda item: abs(item[1]), reverse=True)

        return {
            "shap_values": feature_impacts,
            "key_factors": [{
                "feature": factor,
                "impact": value
            } for factor, value in sorted_factors[:5]] # Top 5 contributing factors
        }

def main():
    """
    Example usage of the ExplainableAI component.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    # Generate some synthetic data for demonstration
    X, y = pd.DataFrame(abs(np.random.randn(100, 10))), np.random.randint(0, 2, 100)
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    X.columns = feature_names

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a simple model
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)

    # Initialize the explainer
    explainer = ExplainableAI(model, feature_names)

    # Explain a single transaction
    sample_transaction = X_test.iloc[[0]]
    explanation = explainer.get_explanation(sample_transaction)

    logging.info(f"Explanation for transaction:\n{explanation}")

if __name__ == "__main__":
    main()
