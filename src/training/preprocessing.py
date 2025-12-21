"""Data preprocessing for model training."""

from typing import Tuple, Any, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Handles data preprocessing and cleaning for model training."""

    def __init__(self, handle_missing: str = "mean"):
        """
        Initialize preprocessor.
        
        Args:
            handle_missing: Strategy for missing values (mean, median, forward_fill, drop)
        """
        self.handle_missing = handle_missing
        self.scaler = None
        self.label_encoders = {}

    def handle_missing_values(
        self,
        data: Any,
        strategy: Optional[str] = None
    ) -> Any:
        """
        Handle missing values in data.
        
        Args:
            data: Input data
            strategy: Handling strategy (overrides default)
            
        Returns:
            Data with missing values handled
        """
        strategy = strategy or self.handle_missing
        logger.info(f"Handling missing values using '{strategy}' strategy")
        
        if strategy == "drop":
            # Remove rows with missing values
            import pandas as pd
            if isinstance(data, pd.DataFrame):
                return data.dropna()
        elif strategy == "mean":
            # Fill with mean
            import pandas as pd
            if isinstance(data, pd.DataFrame):
                return data.fillna(data.mean())
        elif strategy == "median":
            # Fill with median
            import pandas as pd
            if isinstance(data, pd.DataFrame):
                return data.fillna(data.median())
        
        return data

    def encode_categorical(
        self,
        data: Any,
        categorical_cols: Optional[List[str]] = None,
        fit: bool = False
    ) -> Any:
        """
        Encode categorical variables.
        
        Args:
            data: Input data
            categorical_cols: List of categorical columns
            fit: Whether to fit encoders (True for training)
            
        Returns:
            Data with encoded categorical variables
        """
        from sklearn.preprocessing import LabelEncoder
        import pandas as pd

        if not isinstance(data, pd.DataFrame) or categorical_cols is None:
            return data

        data = data.copy()

        for col in categorical_cols:
            if col not in data.columns:
                continue

            if fit:
                encoder = LabelEncoder()
                data[col] = encoder.fit_transform(data[col].astype(str))
                self.label_encoders[col] = encoder
            elif col in self.label_encoders:
                data[col] = self.label_encoders[col].transform(data[col].astype(str))

        logger.info(f"Encoded {len(categorical_cols)} categorical columns")
        return data

    def normalize_features(
        self,
        data: Any,
        fit: bool = False
    ) -> Tuple[Any, Any]:
        """
        Normalize numeric features to [0, 1] range.
        
        Args:
            data: Input data
            fit: Whether to fit scaler (True for training)
            
        Returns:
            Normalized data and scaler
        """
        from sklearn.preprocessing import MinMaxScaler
        
        if fit:
            self.scaler = MinMaxScaler()
            scaled_data = self.scaler.fit_transform(data)
        else:
            scaled_data = self.scaler.transform(data) if self.scaler else data
        
        logger.info("Features normalized to [0, 1] range")
        return scaled_data, self.scaler

    def remove_outliers(
        self,
        data: Any,
        method: str = "iqr",
        threshold: float = 3.0
    ) -> Any:
        """
        Remove outliers from data.
        
        Args:
            data: Input data
            method: Outlier detection method (iqr, zscore)
            threshold: Threshold for outlier detection
            
        Returns:
            Data without outliers
        """
        import pandas as pd
        import numpy as np

        if not isinstance(data, pd.DataFrame):
            return data

        data = data.copy()
        initial_size = len(data)

        if method == "iqr":
            for col in data.select_dtypes(include=[np.number]).columns:
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                data = data[(data[col] >= Q1 - threshold * IQR) &
                           (data[col] <= Q3 + threshold * IQR)]
        elif method == "zscore":
            from scipy import stats
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            data = data[(np.abs(stats.zscore(data[numeric_cols])) < threshold).all(axis=1)]

        removed = initial_size - len(data)
        logger.info(f"Removed {removed} outliers using {method} method")
        return data

    def handle_class_imbalance(
        self,
        X: Any,
        y: Any,
        strategy: str = "oversample"
    ) -> Tuple[Any, Any]:
        """
        Handle class imbalance in target variable.
        
        Args:
            X: Features
            y: Target labels
            strategy: Strategy (oversample, undersample, smote)
            
        Returns:
            Balanced features and labels
        """
        from imblearn.over_sampling import RandomOverSampler, SMOTE
        from imblearn.under_sampling import RandomUnderSampler

        if strategy == "oversample":
            sampler = RandomOverSampler(random_state=42)
        elif strategy == "undersample":
            sampler = RandomUnderSampler(random_state=42)
        elif strategy == "smote":
            sampler = SMOTE(random_state=42)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        X_resampled, y_resampled = sampler.fit_resample(X, y)
        logger.info(f"Applied {strategy} to handle class imbalance")
        return X_resampled, y_resampled
