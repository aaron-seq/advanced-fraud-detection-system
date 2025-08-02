# Updated Credit Card Fraud Detection System
# Author: Aaron Sequeira
# Date: August 2025
# Description: Improved fraud detection system with proper structure, comments, and best practices

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import logging
from typing import Tuple, Dict, Any, Optional
from datetime import datetime

# Machine Learning Libraries
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix, 
                            roc_auc_score, precision_recall_curve, 
                            f1_score, precision_score, recall_score)
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# PyOD Libraries for Outlier Detection
from pyod.models.abod import ABOD
from pyod.models.cblof import CBLOF
from pyod.models.feature_bagging import FeatureBagging
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.ecod import ECOD

import joblib
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fraud_detection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

class FraudDetectionConfig:
    """Configuration class for fraud detection parameters"""
    
    # Data parameters
    DATA_PATH = "creditcard.csv"
    RANDOM_STATE = 42
    TEST_SIZE = 0.3
    
    # Outlier detection parameters
    OUTLIER_FRACTION = 0.05
    N_NEIGHBORS = 35
    
    # Model parameters
    N_ESTIMATORS = 100
    CV_FOLDS = 5
    
    # Visualization parameters
    FIGURE_SIZE = (12, 8)
    
    # Output parameters
    MODEL_PATH = "fraud_model_improved.pkl"
    RESULTS_PATH = "model_results.csv"

class DataPreprocessor:
    """Handles data loading and preprocessing operations"""
    
    def __init__(self, config: FraudDetectionConfig):
        self.config = config
        self.scaler = StandardScaler()
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load the credit card fraud dataset
        
        Args:
            file_path (str): Path to the CSV file
            
        Returns:
            pd.DataFrame: Loaded dataset
        """
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Dataset loaded successfully. Shape: {df.shape}")
            logger.info(f"Columns: {list(df.columns)}")
            
            # Basic data validation
            if 'Class' not in df.columns:
                raise ValueError("Target column 'Class' not found in dataset")
                
            fraud_count = df['Class'].sum()
            total_count = len(df)
            fraud_rate = fraud_count / total_count * 100
            
            logger.info(f"Dataset statistics:")
            logger.info(f"  Total transactions: {total_count:,}")
            logger.info(f"  Fraudulent transactions: {fraud_count:,}")
            logger.info(f"  Fraud rate: {fraud_rate:.3f}%")
            
            return df
            
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def explore_data(self, df: pd.DataFrame) -> None:
        """
        Perform exploratory data analysis
        
        Args:
            df (pd.DataFrame): Input dataset
        """
        logger.info("Performing exploratory data analysis...")
        
        # Display basic information
        print("\nDataset Info:")
        print(f"Shape: {df.shape}")
        print(f"Missing values: {df.isnull().sum().sum()}")
        print(f"Data types:\n{df.dtypes.value_counts()}")
        
        # Class distribution
        class_dist = df['Class'].value_counts()
        print(f"\nClass Distribution:")
        print(f"Normal (0): {class_dist[0]:,} ({class_dist[0]/len(df)*100:.2f}%)")
        print(f"Fraud (1): {class_dist[1]:,} ({class_dist[1]/len(df)*100:.2f}%)")
        
    def create_visualizations(self, df: pd.DataFrame) -> None:
        """
        Create data visualization plots
        
        Args:
            df (pd.DataFrame): Input dataset
        """
        plt.style.use('seaborn-v0_8')
        
        # Class distribution plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Class distribution
        df['Class'].value_counts().plot(kind='bar', ax=axes[0,0], color=['skyblue', 'salmon'])
        axes[0,0].set_title('Class Distribution')
        axes[0,0].set_xlabel('Class (0: Normal, 1: Fraud)')
        axes[0,0].set_ylabel('Count')
        
        # 2. Amount distribution by class
        df[df['Class'] == 0]['Amount'].hist(bins=50, alpha=0.7, ax=axes[0,1], label='Normal', color='skyblue')
        df[df['Class'] == 1]['Amount'].hist(bins=50, alpha=0.7, ax=axes[0,1], label='Fraud', color='salmon')
        axes[0,1].set_title('Transaction Amount Distribution by Class')
        axes[0,1].set_xlabel('Amount')
        axes[0,1].set_ylabel('Frequency')
        axes[0,1].legend()
        axes[0,1].set_yscale('log')
        
        # 3. Time distribution
        df['Time_hours'] = df['Time'] / 3600  # Convert to hours
        df[df['Class'] == 0]['Time_hours'].hist(bins=50, alpha=0.7, ax=axes[1,0], label='Normal', color='skyblue')
        df[df['Class'] == 1]['Time_hours'].hist(bins=50, alpha=0.7, ax=axes[1,0], label='Fraud', color='salmon')
        axes[1,0].set_title('Transaction Time Distribution by Class')
        axes[1,0].set_xlabel('Time (hours)')
        axes[1,0].set_ylabel('Frequency')
        axes[1,0].legend()
        
        # 4. Correlation heatmap (sample of features)
        # Select a subset of features for correlation analysis
        features_to_plot = ['Amount', 'Time'] + [col for col in df.columns if col.startswith('V')][:10]
        corr_matrix = df[features_to_plot].corr()
        sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', ax=axes[1,1])
        axes[1,1].set_title('Feature Correlation Matrix (Sample)')
        
        plt.tight_layout()
        plt.savefig('fraud_detection_eda.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info("Visualizations created and saved as 'fraud_detection_eda.png'")
    
    def preprocess_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess the data for modeling
        
        Args:
            df (pd.DataFrame): Input dataset
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Features (X) and target (y)
        """
        logger.info("Preprocessing data...")
        
        # Separate features and target
        X = df.drop(['Class'], axis=1)
        y = df['Class']
        
        # Feature scaling for Amount and Time (other features are already scaled)
        if 'Amount' in X.columns:
            X['Amount'] = StandardScaler().fit_transform(X[['Amount']])
        if 'Time' in X.columns:
            X['Time'] = StandardScaler().fit_transform(X[['Time']])
            
        logger.info(f"Features shape: {X.shape}")
        logger.info(f"Target shape: {y.shape}")
        
        return X.values, y.values

class OutlierDetector:
    """Handles outlier detection using various PyOD algorithms"""
    
    def __init__(self, config: FraudDetectionConfig):
        self.config = config
        self.detectors = self._initialize_detectors()
        
    def _initialize_detectors(self) -> Dict[str, Any]:
        """
        Initialize outlier detection algorithms
        
        Returns:
            Dict[str, Any]: Dictionary of detector instances
        """
        random_state = np.random.RandomState(self.config.RANDOM_STATE)
        
        detectors = {
            'Isolation Forest': IForest(
                contamination=self.config.OUTLIER_FRACTION,
                random_state=self.config.RANDOM_STATE
            ),
            'Local Outlier Factor': LOF(
                n_neighbors=self.config.N_NEIGHBORS,
                contamination=self.config.OUTLIER_FRACTION
            ),
            'K-Nearest Neighbors': KNN(
                contamination=self.config.OUTLIER_FRACTION,
                n_neighbors=self.config.N_NEIGHBORS
            ),
            'ECOD': ECOD(contamination=self.config.OUTLIER_FRACTION),
            'Feature Bagging': FeatureBagging(
                base_estimator=LOF(n_neighbors=self.config.N_NEIGHBORS),
                contamination=self.config.OUTLIER_FRACTION,
                random_state=self.config.RANDOM_STATE
            )
        }
        
        return detectors
    
    def detect_outliers(self, X: np.ndarray) -> Dict[str, Dict[str, Any]]:
        """
        Perform outlier detection using multiple algorithms
        
        Args:
            X (np.ndarray): Feature matrix (excluding target variable)
            
        Returns:
            Dict[str, Dict[str, Any]]: Results from each detector
        """
        logger.info("Starting outlier detection...")
        results = {}
        
        for name, detector in self.detectors.items():
            try:
                logger.info(f"Running {name}...")
                
                # Fit the detector
                detector.fit(X)
                
                # Get predictions and scores
                predictions = detector.predict(X)
                scores = detector.decision_function(X)
                
                # Calculate metrics
                n_outliers = np.sum(predictions == 1)
                n_inliers = np.sum(predictions == 0)
                outlier_percentage = (n_outliers / len(predictions)) * 100
                
                results[name] = {
                    'predictions': predictions,
                    'scores': scores,
                    'n_outliers': n_outliers,
                    'n_inliers': n_inliers,
                    'outlier_percentage': outlier_percentage,
                    'detector': detector
                }
                
                logger.info(f"{name} - Outliers: {n_outliers}, Inliers: {n_inliers}, "
                           f"Percentage: {outlier_percentage:.2f}%")
                
            except Exception as e:
                logger.error(f"Error with {name}: {str(e)}")
                continue
        
        return results
    
    def visualize_outliers(self, X: np.ndarray, results: Dict[str, Dict[str, Any]], 
                          feature_names: Optional[list] = None) -> None:
        """
        Visualize outlier detection results
        
        Args:
            X (np.ndarray): Feature matrix
            results (Dict[str, Dict[str, Any]]): Outlier detection results
            feature_names (Optional[list]): Names of features for plotting
        """
        n_detectors = len(results)
        if n_detectors == 0:
            logger.warning("No outlier detection results to visualize")
            return
            
        # Create subplot grid
        n_cols = min(3, n_detectors)
        n_rows = (n_detectors + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)
            
        axes = axes.flatten()
        
        # Use first two principal components for visualization if more than 2 features
        if X.shape[1] > 2:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2, random_state=self.config.RANDOM_STATE)
            X_viz = pca.fit_transform(X)
            xlabel, ylabel = 'First Principal Component', 'Second Principal Component'
        else:
            X_viz = X
            xlabel = feature_names[0] if feature_names else 'Feature 1'
            ylabel = feature_names[1] if feature_names else 'Feature 2'
        
        for idx, (name, result) in enumerate(results.items()):
            ax = axes[idx]
            
            # Separate inliers and outliers
            inliers = X_viz[result['predictions'] == 0]
            outliers = X_viz[result['predictions'] == 1]
            
            # Plot points
            ax.scatter(inliers[:, 0], inliers[:, 1], c='skyblue', 
                      alpha=0.6, label=f'Inliers ({len(inliers)})', s=20)
            ax.scatter(outliers[:, 0], outliers[:, 1], c='red', 
                      alpha=0.8, label=f'Outliers ({len(outliers)})', s=30)
            
            ax.set_title(f'{name}\n{result["outlier_percentage"]:.2f}% outliers')
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(n_detectors, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('outlier_detection_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info("Outlier detection visualizations saved as 'outlier_detection_results.png'")

class FraudClassifier:
    """Handles supervised learning for fraud detection"""
    
    def __init__(self, config: FraudDetectionConfig):
        self.config = config
        self.pipeline = None
        self.results = {}
        
    def build_pipeline(self) -> ImbPipeline:
        """
        Build machine learning pipeline with SMOTE and Random Forest
        
        Returns:
            ImbPipeline: Complete ML pipeline
        """
        pipeline = ImbPipeline([
            ('scaler', StandardScaler()),
            ('smote', SMOTE(random_state=self.config.RANDOM_STATE)),
            ('classifier', RandomForestClassifier(
                n_estimators=self.config.N_ESTIMATORS,
                class_weight='balanced',
                random_state=self.config.RANDOM_STATE,
                n_jobs=-1
            ))
        ])
        
        return pipeline
    
    def train_and_evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Train and evaluate the fraud detection model
        
        Args:
            X (np.ndarray): Feature matrix
            y (np.ndarray): Target vector
            
        Returns:
            Dict[str, Any]: Evaluation results
        """
        logger.info("Training fraud detection model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config.TEST_SIZE, 
            stratify=y, random_state=self.config.RANDOM_STATE
        )
        
        # Build and train pipeline
        self.pipeline = self.build_pipeline()
        self.pipeline.fit(X_train, y_train)
        
        # Make predictions
        y_pred = self.pipeline.predict(X_test)
        y_pred_proba = self.pipeline.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        results = {
            'accuracy': self.pipeline.score(X_test, y_test),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred),
            'y_test': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        # Cross-validation
        cv_scores = cross_val_score(
            self.pipeline, X_train, y_train, 
            cv=StratifiedKFold(n_splits=self.config.CV_FOLDS, shuffle=True, 
                              random_state=self.config.RANDOM_STATE),
            scoring='f1'
        )
        results['cv_f1_mean'] = cv_scores.mean()
        results['cv_f1_std'] = cv_scores.std()
        
        self.results = results
        
        logger.info("Model training completed")
        logger.info(f"Test Accuracy: {results['accuracy']:.4f}")
        logger.info(f"Test F1-Score: {results['f1']:.4f}")
        logger.info(f"Test ROC-AUC: {results['roc_auc']:.4f}")
        logger.info(f"CV F1-Score: {results['cv_f1_mean']:.4f} ± {results['cv_f1_std']:.4f}")
        
        return results
    
    def create_evaluation_plots(self) -> None:
        """Create model evaluation visualizations"""
        if not self.results:
            logger.warning("No results available for plotting")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Confusion Matrix
        cm = self.results['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0,0])
        axes[0,0].set_title('Confusion Matrix')
        axes[0,0].set_xlabel('Predicted')
        axes[0,0].set_ylabel('Actual')
        
        # 2. Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(
            self.results['y_test'], self.results['y_pred_proba']
        )
        axes[0,1].plot(recall, precision, marker='.')
        axes[0,1].set_title('Precision-Recall Curve')
        axes[0,1].set_xlabel('Recall')
        axes[0,1].set_ylabel('Precision')
        axes[0,1].grid(True)
        
        # 3. ROC Curve
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(self.results['y_test'], self.results['y_pred_proba'])
        axes[1,0].plot(fpr, tpr, label=f'ROC AUC = {self.results["roc_auc"]:.3f}')
        axes[1,0].plot([0, 1], [0, 1], 'k--', label='Random')
        axes[1,0].set_title('ROC Curve')
        axes[1,0].set_xlabel('False Positive Rate')
        axes[1,0].set_ylabel('True Positive Rate')
        axes[1,0].legend()
        axes[1,0].grid(True)
        
        # 4. Feature Importance (if available)
        if hasattr(self.pipeline.named_steps['classifier'], 'feature_importances_'):
            importances = self.pipeline.named_steps['classifier'].feature_importances_
            # Show top 15 features
            top_indices = np.argsort(importances)[-15:]
            axes[1,1].barh(range(len(top_indices)), importances[top_indices])
            axes[1,1].set_title('Top 15 Feature Importances')
            axes[1,1].set_xlabel('Importance')
            axes[1,1].set_yticks(range(len(top_indices)))
            axes[1,1].set_yticklabels([f'Feature_{i}' for i in top_indices])
        else:
            axes[1,1].text(0.5, 0.5, 'Feature importance not available', 
                          ha='center', va='center', transform
