"""REST API server for model serving."""

from typing import Dict, Any, Optional, List
from datetime import datetime
import logging
import json

logger = logging.getLogger(__name__)


class ModelServer:
    """
    REST API server for serving model predictions in real-time.
    
    Handles model loading, feature caching, and prediction serving.
    """

    def __init__(self, model_name: str = "default", port: int = 8000):
        """
        Initialize model server.
        
        Args:
            model_name: Name of the model to serve
            port: Port to serve on
        """
        self.model_name = model_name
        self.port = port
        self.model = None
        self.feature_cache = {}
        self.prediction_cache = {}
        self.metrics = {
            "total_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "avg_latency_ms": 0,
            "error_count": 0
        }
        self.request_log = []

    def load_model(self, model_path: str) -> Dict[str, Any]:
        """
        Load a trained model.
        
        Args:
            model_path: Path to saved model
            
        Returns:
            Model metadata
        """
        import joblib
        
        self.model = joblib.load(model_path)
        metadata = {
            "model_name": self.model_name,
            "model_path": model_path,
            "loaded_at": datetime.utcnow().isoformat()
        }
        
        logger.info(f"Loaded model from {model_path}")
        return metadata

    def predict(
        self,
        features: Dict[str, Any],
        use_cache: bool = True,
        explain: bool = False
    ) -> Dict[str, Any]:
        """
        Make a prediction on input features.
        
        Args:
            features: Input features dictionary
            use_cache: Whether to use prediction cache
            explain: Whether to include prediction explanation
            
        Returns:
            Prediction result with optional explanation
        """
        request_start = datetime.utcnow()
        self.metrics["total_requests"] += 1

        # Check cache
        cache_key = json.dumps(features, sort_keys=True)
        if use_cache and cache_key in self.prediction_cache:
            self.metrics["cache_hits"] += 1
            logger.debug(f"Cache hit for prediction")
            return self.prediction_cache[cache_key]

        self.metrics["cache_misses"] += 1

        try:
            if self.model is None:
                raise ValueError("No model loaded")

            # Convert features to model input format
            import numpy as np
            feature_vector = np.array([list(features.values())])

            # Get prediction
            prediction = self.model.predict(feature_vector)[0]
            
            # Get prediction probability
            try:
                probability = float(self.model.predict_proba(feature_vector)[0].max())
            except:
                probability = float(prediction)

            result = {
                "prediction": float(prediction),
                "confidence": probability,
                "model_version": "1.0.0",
                "served_at": datetime.utcnow().isoformat()
            }

            # Add explanation if requested
            if explain and hasattr(self.model, 'feature_importances_'):
                result["feature_importance"] = {
                    f"feature_{i}": float(imp)
                    for i, imp in enumerate(self.model.feature_importances_)
                }

            # Cache result
            if use_cache:
                self.prediction_cache[cache_key] = result

            latency_ms = (datetime.utcnow() - request_start).total_seconds() * 1000
            result["latency_ms"] = latency_ms

            # Update metrics
            self._update_metrics(latency_ms)

            return result

        except Exception as e:
            self.metrics["error_count"] += 1
            logger.error(f"Prediction error: {str(e)}")
            raise

    def batch_predict(
        self,
        features_list: List[Dict[str, Any]],
        return_explanations: bool = False
    ) -> Dict[str, Any]:
        """
        Make batch predictions.
        
        Args:
            features_list: List of feature dictionaries
            return_explanations: Whether to include explanations
            
        Returns:
            Batch predictions
        """
        logger.info(f"Starting batch prediction for {len(features_list)} samples")
        
        predictions = []
        errors = []

        for i, features in enumerate(features_list):
            try:
                pred = self.predict(features, use_cache=False, explain=return_explanations)
                pred["sample_index"] = i
                predictions.append(pred)
            except Exception as e:
                errors.append({"sample_index": i, "error": str(e)})

        result = {
            "total_samples": len(features_list),
            "successful_predictions": len(predictions),
            "failed_predictions": len(errors),
            "predictions": predictions
        }

        if errors:
            result["errors"] = errors

        logger.info(f"Batch prediction completed. Success: {len(predictions)}, Errors: {len(errors)}")
        return result

    def health_check(self) -> Dict[str, Any]:
        """
        Check server health status.
        
        Returns:
            Health status
        """
        return {
            "status": "healthy" if self.model is not None else "unhealthy",
            "model_loaded": self.model is not None,
            "model_name": self.model_name,
            "total_requests": self.metrics["total_requests"],
            "cache_hit_rate": (
                self.metrics["cache_hits"] / self.metrics["total_requests"]
                if self.metrics["total_requests"] > 0
                else 0
            ),
            "error_rate": (
                self.metrics["error_count"] / self.metrics["total_requests"]
                if self.metrics["total_requests"] > 0
                else 0
            ),
            "avg_latency_ms": self.metrics["avg_latency_ms"]
        }

    def _update_metrics(self, latency_ms: float) -> None:
        """Update server metrics."""
        # Update moving average latency
        total_requests = self.metrics["total_requests"]
        current_avg = self.metrics["avg_latency_ms"]
        
        self.metrics["avg_latency_ms"] = (
            (current_avg * (total_requests - 1) + latency_ms) / total_requests
        )

    def get_metrics(self) -> Dict[str, Any]:
        """Get detailed server metrics."""
        return self.metrics.copy()

    def clear_cache(self) -> None:
        """Clear prediction cache."""
        self.prediction_cache.clear()
        self.metrics["cache_hits"] = 0
        self.metrics["cache_misses"] = 0
        logger.info("Cleared prediction cache")
