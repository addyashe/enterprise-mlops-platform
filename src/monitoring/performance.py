"""Model performance monitoring and tracking."""

from typing import Dict, Any, List
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """
    Monitors model performance metrics over time.
    
    Tracks accuracy, latency, and other metrics in production.
    """

    def __init__(self, model_name: str):
        """Initialize performance monitor."""
        self.model_name = model_name
        self.metrics_log = []
        self.performance_alerts = []

    def log_prediction(
        self,
        prediction: Any,
        actual: Any,
        latency_ms: float,
        features: Dict[str, Any] = None
    ) -> None:
        """
        Log a prediction for monitoring.
        
        Args:
            prediction: Model prediction
            actual: Actual/true label (optional)
            latency_ms: Prediction latency in milliseconds
            features: Input features (optional)
        """
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "prediction": prediction,
            "actual": actual,
            "latency_ms": latency_ms,
            "is_correct": prediction == actual if actual is not None else None,
            "features": features or {}
        }

        self.metrics_log.append(entry)

    def get_current_metrics(
        self,
        window_size: int = 100
    ) -> Dict[str, float]:
        """
        Get current performance metrics over recent predictions.
        
        Args:
            window_size: Number of recent predictions to consider
            
        Returns:
            Current performance metrics
        """
        if not self.metrics_log:
            return {}

        recent_logs = self.metrics_log[-window_size:]

        # Calculate metrics
        correct_predictions = sum(
            1 for log in recent_logs
            if log["is_correct"] is True
        )
        accuracy = correct_predictions / len(recent_logs) if recent_logs else 0

        latencies = [log["latency_ms"] for log in recent_logs]
        avg_latency = sum(latencies) / len(latencies) if latencies else 0

        p95_latency = sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0
        p99_latency = sorted(latencies)[int(len(latencies) * 0.99)] if latencies else 0

        metrics = {
            "accuracy": accuracy,
            "avg_latency_ms": avg_latency,
            "p95_latency_ms": p95_latency,
            "p99_latency_ms": p99_latency,
            "total_predictions": len(self.metrics_log),
            "window_predictions": len(recent_logs)
        }

        logger.info(f"Current metrics: Accuracy={accuracy:.4f}, Avg Latency={avg_latency:.2f}ms")
        return metrics

    def check_performance_degradation(
        self,
        baseline_metrics: Dict[str, float],
        degradation_threshold: float = 0.05
    ) -> Dict[str, Any]:
        """
        Check if current metrics have degraded from baseline.
        
        Args:
            baseline_metrics: Baseline metric values
            degradation_threshold: Threshold for degradation detection
            
        Returns:
            Degradation check results
        """
        current_metrics = self.get_current_metrics()

        if not current_metrics:
            return {"status": "no_data"}

        degradation_results = {
            "checked_at": datetime.utcnow().isoformat(),
            "degraded_metrics": [],
            "healthy_metrics": []
        }

        for metric_name, baseline_value in baseline_metrics.items():
            if metric_name not in current_metrics:
                continue

            current_value = current_metrics[metric_name]

            # Calculate degradation percentage
            if baseline_value > 0:
                degradation_pct = abs(current_value - baseline_value) / baseline_value
            else:
                degradation_pct = 0

            is_degraded = degradation_pct > degradation_threshold

            metric_result = {
                "metric_name": metric_name,
                "baseline_value": baseline_value,
                "current_value": current_value,
                "degradation_pct": degradation_pct,
                "is_degraded": is_degraded
            }

            if is_degraded:
                degradation_results["degraded_metrics"].append(metric_result)
                logger.warning(f"Degradation detected in {metric_name}: {degradation_pct:.2%}")
            else:
                degradation_results["healthy_metrics"].append(metric_result)

        degradation_results["has_degradation"] = len(degradation_results["degraded_metrics"]) > 0

        return degradation_results

    def get_metrics_history(self, limit: int = None) -> List[Dict[str, Any]]:
        """Get metrics history."""
        if limit:
            return self.metrics_log[-limit:]
        return self.metrics_log.copy()

    def export_metrics(self, format: str = "dict") -> Any:
        """Export metrics in specified format."""
        if format == "dict":
            return {
                "model": self.model_name,
                "metrics_count": len(self.metrics_log),
                "current_metrics": self.get_current_metrics()
            }
        else:
            raise ValueError(f"Unsupported format: {format}")
