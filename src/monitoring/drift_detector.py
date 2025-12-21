"""Data drift detection for monitoring feature distributions."""

from typing import Dict, List, Any, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class DriftDetector:
    """
    Detects statistical drift in feature distributions.
    
    Compares baseline vs current distributions to identify data drift.
    """

    def __init__(self, threshold: float = 0.05):
        """
        Initialize drift detector.
        
        Args:
            threshold: P-value threshold for drift detection
        """
        self.threshold = threshold
        self.baseline_statistics = {}
        self.drift_history = []

    def compute_baseline_statistics(
        self,
        features: List[Dict[str, float]]
    ) -> Dict[str, Any]:
        """
        Compute baseline statistics from reference data.
        
        Args:
            features: List of feature dictionaries
            
        Returns:
            Baseline statistics
        """
        import statistics
        
        statistics_dict = {}

        if not features:
            return statistics_dict

        # Get all numeric features
        all_keys = set()
        for feature_dict in features:
            all_keys.update(feature_dict.keys())

        for key in all_keys:
            values = [
                f[key] for f in features
                if key in f and isinstance(f[key], (int, float))
            ]

            if values:
                statistics_dict[key] = {
                    "mean": statistics.mean(values),
                    "median": statistics.median(values),
                    "stdev": statistics.stdev(values) if len(values) > 1 else 0,
                    "min": min(values),
                    "max": max(values),
                    "count": len(values)
                }

        self.baseline_statistics = statistics_dict
        logger.info(f"Computed baseline statistics for {len(statistics_dict)} features")
        return statistics_dict

    def detect_drift(
        self,
        current_features: List[Dict[str, float]],
        feature_names: List[str] = None
    ) -> Dict[str, Any]:
        """
        Detect drift in current features vs baseline.
        
        Args:
            current_features: List of current feature dictionaries
            feature_names: Optional list of features to check
            
        Returns:
            Drift detection results
        """
        if not self.baseline_statistics:
            raise ValueError("Baseline statistics not computed. Call compute_baseline_statistics first.")

        if not current_features:
            return {"status": "no_data"}

        import statistics
        from scipy import stats

        feature_names = feature_names or list(self.baseline_statistics.keys())
        drift_results = {
            "detected_at": datetime.utcnow().isoformat(),
            "features_checked": len(feature_names),
            "features_with_drift": [],
            "features_without_drift": []
        }

        for feature_name in feature_names:
            if feature_name not in self.baseline_statistics:
                continue

            # Get current values
            current_values = [
                f[feature_name] for f in current_features
                if feature_name in f and isinstance(f[feature_name], (int, float))
            ]

            if not current_values:
                continue

            # Get baseline stats
            baseline = self.baseline_statistics[feature_name]

            # Compute current statistics
            current_mean = statistics.mean(current_values)
            current_stdev = statistics.stdev(current_values) if len(current_values) > 1 else 0

            # Perform KS test
            baseline_data = [baseline[feature_name] for _ in range(len(current_values))]
            ks_stat, p_value = stats.ks_2samp(current_values, baseline_data)

            is_drift = p_value < self.threshold

            feature_drift = {
                "feature_name": feature_name,
                "baseline_mean": baseline["mean"],
                "current_mean": current_mean,
                "baseline_stdev": baseline["stdev"],
                "current_stdev": current_stdev,
                "ks_statistic": ks_stat,
                "p_value": p_value,
                "is_drift": is_drift,
                "mean_shift_pct": (current_mean - baseline["mean"]) / baseline["mean"] * 100 if baseline["mean"] else 0
            }

            if is_drift:
                drift_results["features_with_drift"].append(feature_drift)
                logger.warning(f"Drift detected in feature '{feature_name}' (p={p_value:.4f})")
            else:
                drift_results["features_without_drift"].append(feature_drift)

        drift_results["has_drift"] = len(drift_results["features_with_drift"]) > 0

        self.drift_history.append(drift_results)
        logger.info(f"Drift detection completed. Drift in {len(drift_results['features_with_drift'])} features")

        return drift_results

    def get_drift_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent drift detection results."""
        return self.drift_history[-limit:]

    def get_drift_summary(self) -> Dict[str, Any]:
        """Get summary of drift history."""
        if not self.drift_history:
            return {"total_checks": 0}

        total_checks = len(self.drift_history)
        checks_with_drift = sum(1 for h in self.drift_history if h.get("has_drift", False))

        return {
            "total_checks": total_checks,
            "checks_with_drift": checks_with_drift,
            "drift_rate": checks_with_drift / total_checks if total_checks > 0 else 0,
            "last_check": self.drift_history[-1].get("detected_at") if self.drift_history else None
        }
