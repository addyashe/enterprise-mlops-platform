"""Utility functions for feature store operations."""

from typing import Any, List, Dict
from datetime import datetime
import hashlib


def hash_feature_vector(features: Dict[str, Any]) -> str:
    """
    Generate hash of feature vector for caching and deduplication.
    
    Args:
        features: Dictionary of features
        
    Returns:
        SHA256 hash of feature vector
    """
    feature_str = str(sorted(features.items()))
    return hashlib.sha256(feature_str.encode()).hexdigest()


def validate_feature_schema(
    data: Dict[str, Any],
    schema: Dict[str, type]
) -> bool:
    """
    Validate that data matches expected feature schema.
    
    Args:
        data: Feature data to validate
        schema: Expected schema (column_name -> type)
        
    Returns:
        True if valid, raises ValueError otherwise
    """
    for column, expected_type in schema.items():
        if column not in data:
            raise ValueError(f"Missing required feature: {column}")
        
        if not isinstance(data[column], expected_type):
            raise ValueError(
                f"Feature '{column}' has type {type(data[column])}, "
                f"expected {expected_type}"
            )
    
    return True


def compute_feature_statistics(
    feature_values: List[Any]
) -> Dict[str, float]:
    """
    Compute descriptive statistics for numeric feature.
    
    Args:
        feature_values: List of feature values
        
    Returns:
        Dictionary with min, max, mean, std, median
    """
    import statistics
    
    numeric_values = [v for v in feature_values if isinstance(v, (int, float))]
    
    if not numeric_values:
        return {}
    
    return {
        "min": min(numeric_values),
        "max": max(numeric_values),
        "mean": statistics.mean(numeric_values),
        "median": statistics.median(numeric_values),
        "stdev": statistics.stdev(numeric_values) if len(numeric_values) > 1 else 0,
        "count": len(numeric_values)
    }


def apply_feature_transformations(
    features: Dict[str, Any],
    transformations: Dict[str, callable]
) -> Dict[str, Any]:
    """
    Apply transformations to features.
    
    Args:
        features: Input features
        transformations: Dict mapping feature name to transformation function
        
    Returns:
        Transformed features
    """
    transformed = features.copy()
    
    for feature_name, transform_fn in transformations.items():
        if feature_name in transformed:
            try:
                transformed[feature_name] = transform_fn(transformed[feature_name])
            except Exception as e:
                raise ValueError(f"Error transforming {feature_name}: {str(e)}")
    
    return transformed


def detect_feature_drift(
    baseline: List[float],
    current: List[float],
    threshold: float = 0.1
) -> Dict[str, Any]:
    """
    Detect statistical drift in feature distribution.
    
    Args:
        baseline: Baseline feature values
        current: Current feature values
        threshold: Drift threshold
        
    Returns:
        Dictionary with drift detection results
    """
    import statistics
    
    baseline_mean = statistics.mean(baseline)
    current_mean = statistics.mean(current)
    
    baseline_std = statistics.stdev(baseline) if len(baseline) > 1 else 0
    current_std = statistics.stdev(current) if len(current) > 1 else 0
    
    mean_shift = abs(current_mean - baseline_mean) / (baseline_std + 1e-6)
    
    return {
        "baseline_mean": baseline_mean,
        "current_mean": current_mean,
        "baseline_std": baseline_std,
        "current_std": current_std,
        "mean_shift_std": mean_shift,
        "is_drift": mean_shift > threshold,
        "threshold": threshold
    }
