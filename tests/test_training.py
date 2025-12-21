"""Unit tests for model training module."""

import pytest
import numpy as np
from sklearn.datasets import make_classification
from src.training.trainer import ModelTrainer
from src.training.preprocessing import DataPreprocessor
from src.training.validation import ModelValidator


class TestModelTrainer:
    """Tests for ModelTrainer."""

    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        X, y = make_classification(n_samples=200, n_features=10, random_state=42)
        return X, y

    def test_train_model(self, sample_data):
        """Test training a model."""
        X, y = sample_data
        X_train, X_val = X[:150], X[150:]
        y_train, y_val = y[:150], y[150:]
        
        trainer = ModelTrainer("test_experiment", "gradient_boosting")
        results = trainer.train(X_train, y_train, X_val, y_val)
        
        assert results["model_type"] == "gradient_boosting"
        assert "metrics" in results
        assert results["data_size"] == 150

    def test_model_evaluation(self, sample_data):
        """Test model evaluation metrics."""
        X, y = sample_data
        X_train, X_test = X[:150], X[150:]
        y_train, y_test = y[:150], y[150:]
        
        trainer = ModelTrainer("test", "logistic_regression")
        trainer.train(X_train, y_train)
        
        metrics = trainer._evaluate(trainer.scaler.transform(X_test), y_test, "test")
        
        assert "test_accuracy" in metrics
        assert 0 <= metrics["test_accuracy"] <= 1

    def test_hyperparameter_tuning(self, sample_data):
        """Test hyperparameter tuning."""
        X, y = sample_data
        trainer = ModelTrainer("test", "random_forest")
        
        param_grid = {"n_estimators": [10, 20], "max_depth": [3, 5]}
        results = trainer.hyperparameter_tuning(X, y, param_grid, cv_folds=3)
        
        assert "best_params" in results
        assert "best_score" in results

    def test_feature_importance(self, sample_data):
        """Test getting feature importances."""
        X, y = sample_data
        trainer = ModelTrainer("test", "gradient_boosting")
        trainer.train(X, y)
        
        importances = trainer.get_feature_importance()
        
        assert len(importances) == 10
        assert all(v >= 0 for v in importances.values())

    def test_save_and_load_model(self, sample_data, tmp_path):
        """Test saving and loading models."""
        X, y = sample_data
        trainer = ModelTrainer("test", "random_forest")
        trainer.train(X, y)
        
        model_path = tmp_path / "model.pkl"
        trainer.save_model(str(model_path))
        
        trainer2 = ModelTrainer("test2", "random_forest")
        loaded_model = trainer2.load_model(str(model_path))
        
        assert loaded_model is not None


class TestDataPreprocessor:
    """Tests for DataPreprocessor."""

    def test_handle_missing_values(self):
        """Test handling missing values."""
        import pandas as pd
        
        df = pd.DataFrame({
            "a": [1, 2, None, 4],
            "b": [5.0, None, 7.0, 8.0]
        })
        
        preprocessor = DataPreprocessor(handle_missing="mean")
        result = preprocessor.handle_missing_values(df, strategy="mean")
        
        assert not result.isnull().any().any()

    def test_normalize_features(self):
        """Test feature normalization."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        
        preprocessor = DataPreprocessor()
        X_normalized, _ = preprocessor.normalize_features(X, fit=True)
        
        assert X_normalized.min() >= 0
        assert X_normalized.max() <= 1

    def test_remove_outliers(self):
        """Test outlier removal."""
        import pandas as pd
        
        df = pd.DataFrame({
            "value": [1, 2, 3, 4, 5, 100]  # 100 is outlier
        })
        
        preprocessor = DataPreprocessor()
        result = preprocessor.remove_outliers(df, method="iqr")
        
        assert len(result) < len(df)
        assert 100 not in result["value"].values


class TestModelValidator:
    """Tests for ModelValidator."""

    def test_validate_performance(self):
        """Test performance validation."""
        metrics = {"accuracy": 0.95, "precision": 0.92}
        thresholds = {"accuracy": 0.90, "precision": 0.90}
        
        is_valid, results = ModelValidator.validate_model_performance(metrics, thresholds)
        
        assert is_valid
        assert len(results["failed_checks"]) == 0

    def test_validate_performance_failure(self):
        """Test performance validation failure."""
        metrics = {"accuracy": 0.85, "precision": 0.80}
        thresholds = {"accuracy": 0.90, "precision": 0.90}
        
        is_valid, results = ModelValidator.validate_model_performance(metrics, thresholds)
        
        assert not is_valid
        assert len(results["failed_checks"]) > 0

    def test_validate_input_schema(self):
        """Test input schema validation."""
        data = {"age": 30, "income": 50000}
        required = ["age"]
        types = {"age": int, "income": int}
        
        is_valid, results = ModelValidator.validate_input_schema(data, required, types)
        
        assert is_valid

    def test_validate_input_schema_failure(self):
        """Test input schema validation failure."""
        data = {"age": "thirty"}
        required = ["age"]
        types = {"age": int}
        
        is_valid, results = ModelValidator.validate_input_schema(data, required, types)
        
        assert not is_valid
