"""Airflow configuration."""

import os
from datetime import datetime

# Airflow settings
AIRFLOW_HOME = os.path.expanduser("~/airflow")
DAGS_FOLDER = os.path.join(AIRFLOW_HOME, "dags")
BASE_LOG_FOLDER = os.path.join(AIRFLOW_HOME, "logs")

# Default arguments
DEFAULT_ARGS = {
    "owner": "mlops",
    "start_date": datetime(2025, 1, 1),
    "retries": 2,
    "retry_delay_minutes": 5,
    "email_on_failure": True,
    "email_on_retry": False,
    "email": ["mlops-alerts@company.com"]
}

# Feature store config
FEATURE_STORE_CONFIG = {
    "workspace_url": "https://your-workspace.databricks.com",
    "token": "${DATABRICKS_TOKEN}",
    "compute_cluster": "ml-compute"
}

# Model training config
TRAINING_CONFIG = {
    "experiment_name": "production_models",
    "tracking_uri": "http://mlflow:5000",
    "models": ["gradient_boosting", "xgboost", "random_forest"]
}

# Model serving config
SERVING_CONFIG = {
    "port": 8000,
    "workers": 4,
    "timeout": 60
}

# Monitoring config
MONITORING_CONFIG = {
    "drift_threshold": 0.05,
    "performance_threshold": 0.85,
    "alert_email": "mlops-team@company.com"
}
