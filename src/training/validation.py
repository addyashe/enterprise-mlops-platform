"""Model validation and evaluation utilities."""

from typing import Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)


class ModelValidator:
    """Validates model performance and data quality."""

    @staticmethod
    def validate_model_performance(
        metrics: Dict[str, float],
        thresholds: Dict[str, float]
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate that model meets performance thresholds.
        
        Args:
            metrics: Model metrics dictionary
            thresholds: Minimum acceptable metric values
            
        Returns:
            (is_valid, validation_results)
        """
        validation_results = {
            "valid": True,
            "metrics_checked": {},
            "failed_checks": []
        }

        for metric_name, threshold in thresholds.items():
            if metric_name not in metrics:
                validation_results["failed_checks"].append(
                    f"Metric '{metric_name}' not found"
                )
                validation_results["valid"] = False
                continue

            metric_value = metrics[metric_name]
            validation_results["metrics_checked"][metric_name] = metric_value

            if metric_value < threshold:
                validation_results["failed_checks"].append(
                    f"{metric_name}: {metric_value:.4f} < {threshold:.4f}"
                )
                validation_results["valid"] = False

        if validation_results["valid"]:
            logger.info("Model performance validation passed")
        else:
            logger.warning("Model performance validation failed")

        return validation_results["valid"], validation_results

    @staticmethod
    def validate_input_schema(
        data: Dict[str, Any],
        required_fields: list,
        expected_types: Dict[str, type]
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate input data schema.
        
        Args:
            data: Input data
            required_fields: List of required fields
            expected_types: Mapping of field names to expected types
            
        Returns:
            (is_valid, validation_results)
        """
        validation_results = {
            "valid": True,
            "errors": []
        }

        # Check required fields
        for field in required_fields:
            if field not in data:
                validation_results["errors"].append(f"Missing required field: {field}")
                validation_results["valid"] = False

        # Check field types
        for field, expected_type in expected_types.items():
            if field in data and not isinstance(data[field], expected_type):
                validation_results["errors"].append(
                    f"Field '{field}' has type {type(data[field])}, "
                    f"expected {expected_type}"
                )
                validation_results["valid"] = False

        if validation_results["valid"]:
            logger.info("Input schema validation passed")
        else:
            logger.warning(f"Input schema validation failed: {validation_results['errors']}")

        return validation_results["valid"], validation_results

    @staticmethod
    def cross_validate_model(
        model: Any,
        X: Any,
        y: Any,
        cv_folds: int = 5
    ) -> Dict[str, Any]:
        """
        Perform cross-validation on model.
        
        Args:
            model: Trained model
            X: Features
            y: Target
            cv_folds: Number of folds
            
        Returns:
            Cross-validation results
        """
        from sklearn.model_selection import cross_validate
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        scoring = {
            "accuracy": "accuracy",
            "precision": "precision",
            "recall": "recall",
            "f1": "f1"
        }

        cv_results = cross_validate(model, X, y, cv=cv_folds, scoring=scoring)

        results = {
            "cv_folds": cv_folds,
            "accuracy_mean": cv_results["test_accuracy"].mean(),
            "accuracy_std": cv_results["test_accuracy"].std(),
            "precision_mean": cv_results["test_precision"].mean(),
            "recall_mean": cv_results["test_recall"].mean(),
            "f1_mean": cv_results["test_f1"].mean(),
        }

        logger.info(f"Cross-validation completed ({cv_folds} folds)")
        logger.info(f"Mean accuracy: {results['accuracy_mean']:.4f} (+/- {results['accuracy_std']:.4f})")

        return results

    @staticmethod
    def check_data_quality(
        data: Any,
        max_missing_pct: float = 0.1
    ) -> Dict[str, Any]:
        """
        Check data quality metrics.
        
        Args:
            data: Input data
            max_missing_pct: Maximum acceptable missing value percentage
            
        Returns:
            Data quality report
        """
        import pandas as pd
        import numpy as np

        if not isinstance(data, pd.DataFrame):
            return {}

        report = {
            "rows": len(data),
            "columns": len(data.columns),
            "numeric_cols": len(data.select_dtypes(include=[np.number]).columns),
            "missing_values": {},
            "duplicates": len(data.duplicates()),
            "quality_issues": []
        }

        for col in data.columns:
            missing_pct = data[col].isnull().sum() / len(data)
            report["missing_values"][col] = missing_pct

            if missing_pct > max_missing_pct:
                report["quality_issues"].append(
                    f"Column '{col}' has {missing_pct:.1%} missing values"
                )

        logger.info(f"Data quality check completed. Issues found: {len(report['quality_issues'])}")
        return report
