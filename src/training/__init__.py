"""
Model training and evaluation.

Turns the "not measured" entries in the README into measured ones, using a
methodology that survives deployment: forward-in-time splits, preprocessing
fitted on training rows only, and PR-AUC rather than accuracy as the headline.
"""

from src.training.dataset import Dataset, DataSource, load_dataset
from src.training.evaluation import EvaluationReport, evaluate
from src.training.pipeline import TrainingConfig, TrainingResult, save_artifacts, train

__all__ = [
    "DataSource",
    "Dataset",
    "EvaluationReport",
    "TrainingConfig",
    "TrainingResult",
    "evaluate",
    "load_dataset",
    "save_artifacts",
    "train",
]
