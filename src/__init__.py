"""Enterprise MLOps Platform with Feature Stores."""

__version__ = "1.0.0"
__author__ = "techdevworks"

from src.feature_store.client import FeatureStoreClient
from src.training.trainer import ModelTrainer
from src.serving.model_server import ModelServer
from src.monitoring.drift_detector import DriftDetector

__all__ = [
    "FeatureStoreClient",
    "ModelTrainer",
    "ModelServer",
    "DriftDetector",
]
