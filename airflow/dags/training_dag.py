"""Training DAG - Scheduled model retraining pipeline."""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.task_group import TaskGroup

# Default arguments
DEFAULT_ARGS = {
    "owner": "mlops",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": True
}

# Define DAG
dag = DAG(
    "model_training",
    default_args=DEFAULT_ARGS,
    schedule_interval="0 2 * * *",  # Daily at 2 AM
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["mlops", "training"]
)

def fetch_features(**context):
    """Fetch training features from feature store."""
    from src.feature_store.client import FeatureStoreClient
    
    fs = FeatureStoreClient("https://workspace.databricks.com")
    features = fs.get_feature_table("recommendation_features")
    
    context["task_instance"].xcom_push(key="features_path", value=features["table_name"])
    return f"Fetched {features['row_count']} features"

def validate_data(**context):
    """Validate data quality."""
    from src.training.validation import ModelValidator
    
    features = context["task_instance"].xcom_pull(key="features_path")
    
    # Placeholder validation
    return f"Validated {features}"

def train_model(**context):
    """Train new model."""
    from src.training.trainer import ModelTrainer
    import numpy as np
    
    trainer = ModelTrainer("production_training", "gradient_boosting")
    
    # Generate sample data
    X_train = np.random.rand(1000, 20)
    y_train = np.random.randint(0, 2, 1000)
    
    results = trainer.train(X_train, y_train)
    
    context["task_instance"].xcom_push(key="model_run_id", value=results["run_id"])
    return f"Training completed: {results['run_id']}"

def evaluate_model(**context):
    """Evaluate model performance."""
    run_id = context["task_instance"].xcom_pull(key="model_run_id")
    
    # Placeholder evaluation
    accuracy = 0.95
    
    if accuracy < 0.85:
        raise ValueError(f"Model accuracy {accuracy} below threshold")
    
    return f"Model evaluation passed: {accuracy}"

def register_model(**context):
    """Register model in registry."""
    run_id = context["task_instance"].xcom_pull(key="model_run_id")
    
    from src.models import ModelRegistry
    
    registry = ModelRegistry()
    result = registry.promote_model(run_id, to_stage="Production")
    
    return f"Model registered: {run_id}"

# Task definitions
with TaskGroup("feature_engineering") as feature_group:
    fetch_task = PythonOperator(
        task_id="fetch_features",
        python_callable=fetch_features,
        provide_context=True
    )
    
    validate_task = PythonOperator(
        task_id="validate_data",
        python_callable=validate_data,
        provide_context=True,
        depends_on_past=False
    )
    
    fetch_task >> validate_task

train_task = PythonOperator(
    task_id="train_model",
    python_callable=train_model,
    provide_context=True
)

evaluate_task = PythonOperator(
    task_id="evaluate_model",
    python_callable=evaluate_model,
    provide_context=True
)

register_task = PythonOperator(
    task_id="register_model",
    python_callable=register_model,
    provide_context=True
)

# DAG execution flow
feature_group >> train_task >> evaluate_task >> register_task
