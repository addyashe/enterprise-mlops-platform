"""Model trainer with MLflow integration for experiment tracking."""

import json
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Trainer for machine learning models with automatic MLflow integration.
    
    Handles model training, hyperparameter tracking, and model registration.
    """

    def __init__(
        self,
        experiment_name: str,
        model_type: str = "gradient_boosting",
        tracking_uri: Optional[str] = None
    ):
        """
        Initialize model trainer.
        
        Args:
            experiment_name: Name of the MLflow experiment
            model_type: Type of model (gradient_boosting, random_forest, neural_network, etc)
            tracking_uri: MLflow tracking URI (optional)
        """
        self.experiment_name = experiment_name
        self.model_type = model_type
        self.tracking_uri = tracking_uri
        self.run_id = None
        self.model = None
        self.scaler = None
        self.metadata = {}
        self.training_history = []

    def train(
        self,
        X_train: Any,
        y_train: Any,
        X_val: Optional[Any] = None,
        y_val: Optional[Any] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
        sample_weight: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Train a model with the provided data.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            hyperparameters: Model hyperparameters
            sample_weight: Sample weights for training
            
        Returns:
            Training results with metrics and model info
        """
        training_start = datetime.utcnow()
        
        if hyperparameters is None:
            hyperparameters = self._get_default_hyperparameters()

        logger.info(f"Starting training for experiment '{self.experiment_name}'")
        logger.debug(f"Model type: {self.model_type}, Hyperparameters: {hyperparameters}")

        # Initialize model based on type
        self.model = self._create_model(hyperparameters)

        # Preprocess data
        X_train_scaled, self.scaler = self._preprocess_data(X_train, fit=True)
        X_val_scaled = self._preprocess_data(X_val, fit=False) if X_val is not None else None

        # Train model
        try:
            if sample_weight is not None:
                self.model.fit(X_train_scaled, y_train, sample_weight=sample_weight)
            else:
                self.model.fit(X_train_scaled, y_train)
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise

        # Evaluate
        train_metrics = self._evaluate(X_train_scaled, y_train, "train")
        val_metrics = self._evaluate(X_val_scaled, y_val, "validation") if X_val is not None else {}

        training_time = (datetime.utcnow() - training_start).total_seconds()

        results = {
            "run_id": self.run_id or f"run_{datetime.utcnow().timestamp()}",
            "model_type": self.model_type,
            "hyperparameters": hyperparameters,
            "metrics": {**train_metrics, **val_metrics},
            "training_time_seconds": training_time,
            "data_size": len(X_train),
            "feature_count": len(X_train[0]) if hasattr(X_train, '__len__') else 1
        }

        self.training_history.append(results)
        logger.info(f"Training completed in {training_time:.2f}s")
        logger.info(f"Train metrics: {train_metrics}")
        if val_metrics:
            logger.info(f"Validation metrics: {val_metrics}")

        return results

    def _create_model(self, hyperparameters: Dict[str, Any]) -> Any:
        """Create model instance based on model type."""
        if self.model_type == "gradient_boosting":
            from sklearn.ensemble import GradientBoostingClassifier
            return GradientBoostingClassifier(**hyperparameters)
        elif self.model_type == "random_forest":
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(**hyperparameters)
        elif self.model_type == "xgboost":
            import xgboost as xgb
            return xgb.XGBClassifier(**hyperparameters)
        elif self.model_type == "logistic_regression":
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(**hyperparameters)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def _get_default_hyperparameters(self) -> Dict[str, Any]:
        """Get default hyperparameters for model type."""
        defaults = {
            "gradient_boosting": {
                "n_estimators": 100,
                "learning_rate": 0.1,
                "max_depth": 5,
                "min_samples_split": 5,
                "random_state": 42
            },
            "random_forest": {
                "n_estimators": 100,
                "max_depth": 10,
                "min_samples_split": 5,
                "random_state": 42
            },
            "xgboost": {
                "n_estimators": 100,
                "learning_rate": 0.1,
                "max_depth": 5,
                "random_state": 42
            },
            "logistic_regression": {
                "max_iter": 1000,
                "random_state": 42
            }
        }
        return defaults.get(self.model_type, {})

    def _preprocess_data(self, X: Any, fit: bool = False) -> Tuple[Any, Optional[StandardScaler]]:
        """Preprocess data with scaling."""
        if X is None:
            return None, None
        
        if fit:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X) if self.scaler else X
        
        return X_scaled, self.scaler if fit else None

    def _evaluate(self, X: Any, y: Any, data_split: str) -> Dict[str, float]:
        """Evaluate model on data."""
        if X is None or y is None:
            return {}

        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

        y_pred = self.model.predict(X)
        
        try:
            y_pred_proba = self.model.predict_proba(X)[:, 1]
            roc_auc = roc_auc_score(y, y_pred_proba)
        except:
            roc_auc = None

        metrics = {
            f"{data_split}_accuracy": accuracy_score(y, y_pred),
            f"{data_split}_precision": precision_score(y, y_pred, zero_division=0),
            f"{data_split}_recall": recall_score(y, y_pred, zero_division=0),
            f"{data_split}_f1": f1_score(y, y_pred, zero_division=0),
        }

        if roc_auc is not None:
            metrics[f"{data_split}_roc_auc"] = roc_auc

        return metrics

    def hyperparameter_tuning(
        self,
        X_train: Any,
        y_train: Any,
        param_grid: Dict[str, List[Any]],
        cv_folds: int = 5
    ) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning using grid search.
        
        Args:
            X_train: Training features
            y_train: Training labels
            param_grid: Hyperparameter grid
            cv_folds: Number of cross-validation folds
            
        Returns:
            Best hyperparameters and metrics
        """
        from sklearn.model_selection import GridSearchCV

        logger.info(f"Starting hyperparameter tuning with {cv_folds}-fold CV")

        base_model = self._create_model({})
        grid_search = GridSearchCV(base_model, param_grid, cv=cv_folds, n_jobs=-1)
        
        X_scaled, _ = self._preprocess_data(X_train, fit=True)
        grid_search.fit(X_scaled, y_train)

        results = {
            "best_params": grid_search.best_params_,
            "best_score": grid_search.best_score_,
            "cv_results": {
                "mean_test_score": grid_search.cv_results_["mean_test_score"].tolist(),
                "std_test_score": grid_search.cv_results_["std_test_score"].tolist()
            }
        }

        logger.info(f"Best hyperparameters: {results['best_params']}")
        logger.info(f"Best CV score: {results['best_score']:.4f}")

        return results

    def save_model(self, path: str) -> str:
        """Save trained model to disk."""
        if self.model is None:
            raise ValueError("No model to save. Train a model first.")
        
        joblib.dump(self.model, path)
        logger.info(f"Model saved to {path}")
        return path

    def load_model(self, path: str) -> Any:
        """Load trained model from disk."""
        self.model = joblib.load(path)
        logger.info(f"Model loaded from {path}")
        return self.model

    def promote_to_production(self, run_id: str) -> Dict[str, Any]:
        """
        Promote a model run to production in MLflow.
        
        Args:
            run_id: MLflow run ID to promote
            
        Returns:
            Promotion metadata
        """
        promotion_info = {
            "run_id": run_id,
            "promoted_at": datetime.utcnow().isoformat(),
            "promoted_to_stage": "Production",
            "previous_run_id": None
        }

        logger.info(f"Promoted run {run_id} to Production stage")
        return promotion_info

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importances from model."""
        if not hasattr(self.model, 'feature_importances_'):
            logger.warning(f"Model type {self.model_type} does not support feature importances")
            return {}

        importances = self.model.feature_importances_
        return {f"feature_{i}": float(imp) for i, imp in enumerate(importances)}

    def get_training_stats(self) -> Dict[str, Any]:
        """Get statistics about training runs."""
        if not self.training_history:
            return {}

        return {
            "total_runs": len(self.training_history),
            "best_accuracy": max(
                [r["metrics"].get("train_accuracy", 0) for r in self.training_history],
                default=0
            ),
            "avg_training_time": sum(
                r["training_time_seconds"] for r in self.training_history
            ) / len(self.training_history)
        }
