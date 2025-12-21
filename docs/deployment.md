# Deployment Guide

## Local Development Setup

### Prerequisites
- Python 3.9+
- Docker & Docker Compose
- Git

### Step 1: Clone and Setup
```bash
git clone https://github.com/techdevworks/enterprise-mlops-platform.git
cd enterprise-mlops-platform

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### Step 2: Start Development Environment
```bash
docker-compose up -d
```

This starts:
- PostgreSQL (for Airflow metadata)
- Airflow webserver
- Airflow scheduler
- MLflow tracking server

### Step 3: Run Tests
```bash
pytest tests/ -v --cov=src
```

## Kubernetes Deployment

### Prerequisites
- Kubernetes cluster (1.24+)
- kubectl configured
- Docker registry access

### Step 1: Build Docker Image
```bash
docker build -t yourregistry/mlops-platform:1.0.0 .
docker push yourregistry/mlops-platform:1.0.0
```

### Step 2: Deploy to Kubernetes
```bash
# Update image in k8s/deployment.yaml
kubectl apply -f k8s/

# Check deployment status
kubectl get pods
kubectl logs -f deployment/mlops-platform
```

### Step 3: Configure Services
```bash
# Expose serving API
kubectl port-forward svc/mlops-serving 8000:8000

# Access MLflow UI
kubectl port-forward svc/mlflow 5000:5000
```

## Databricks Deployment

### Step 1: Configure Databricks Workspace
```python
from databricks.sdk import WorkspaceClient

wc = WorkspaceClient(
    host="https://your-workspace.cloud.databricks.com",
    token="your-token"
)

# Create cluster
cluster_config = {
    "spark_version": "13.3.x-scala2.12",
    "node_type_id": "i3.xlarge",
    "num_workers": 2,
    "aws_attributes": {"zone_id": "us-west-2a"}
}
```

### Step 2: Deploy Feature Store
```python
from databricks.feature_store import FeatureStoreClient

fs = FeatureStoreClient()

# Create feature table
fs.create_table(
    name="user_features",
    primary_keys=["user_id"],
    timestamp_key="updated_at",
    df=feature_df,
    description="User demographic and behavioral features"
)
```

### Step 3: Configure MLflow
```python
import mlflow

mlflow.set_tracking_uri("databricks")
mlflow.set_experiment("/Users/your-email@company.com/mlops-experiments")
```

## Production Deployment (AWS)

### Architecture
```
ALB → ECS Task → Container with MLOps Platform
     ↓
  RDS (PostgreSQL)
     ↓
  S3 (Model Registry, Features)
     ↓
  ElastiCache (Feature Cache)
```

### Step 1: Create ECS Task Definition
```json
{
  "family": "mlops-platform",
  "taskRoleArn": "arn:aws:iam::ACCOUNT:role/ecs-task-role",
  "containerDefinitions": [
    {
      "name": "mlops",
      "image": "YOUR_REGISTRY/mlops-platform:1.0.0",
      "portMappings": [{"containerPort": 8000, "protocol": "tcp"}],
      "environment": [
        {"name": "MLFLOW_TRACKING_URI", "value": "s3://your-bucket/mlflow"},
        {"name": "FEATURE_STORE_URI", "value": "https://workspace.databricks.com"}
      ]
    }
  ]
}
```

### Step 2: Create ECS Service
```bash
aws ecs create-service \
  --cluster production \
  --service-name mlops-platform \
  --task-definition mlops-platform:1 \
  --desired-count 3 \
  --load-balancers targetGroupArn=arn:aws:elasticloadbalancing:...,containerName=mlops,containerPort=8000
```

### Step 3: Configure Auto-scaling
```bash
aws autoscaling create-auto-scaling-group \
  --auto-scaling-group-name mlops-asg \
  --service-namespace ecs \
  --scalable-dimension ecs:service:DesiredCount \
  --min-capacity 2 \
  --max-capacity 10
```

## Monitoring Setup

### Prometheus Configuration
```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'mlops-platform'
    static_configs:
      - targets: ['localhost:8000']
```

### Grafana Dashboards
1. Model Performance Dashboard
2. Feature Store Health
3. Serving Latency
4. Training Pipeline Status
5. Cost Breakdown

### Alert Rules
```yaml
groups:
  - name: mlops.rules
    rules:
      - alert: HighPredictionLatency
        expr: histogram_quantile(0.99, prediction_latency_ms) > 250
        annotations:
          summary: "Prediction latency exceeds threshold"
      
      - alert: DataDrift
        expr: ks_test_pvalue < 0.05
        annotations:
          summary: "Data drift detected"
```

## CI/CD Pipeline

### GitHub Actions Workflow
```yaml
name: Deploy MLOps Platform

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
      - run: pip install -r requirements.txt
      - run: pytest tests/ --cov=src

  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: docker/build-push-action@v4
        with:
          push: true
          tags: registry/mlops-platform:${{ github.sha }}

  deploy:
    needs: build
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to Kubernetes
        run: |
          kubectl set image deployment/mlops \
            mlops=registry/mlops-platform:${{ github.sha }}
```

## Troubleshooting

### High Memory Usage
```python
# Reduce batch size
server.batch_predict(features, batch_size=100)

# Clear feature cache
server.clear_cache()
```

### Slow Predictions
```python
# Check prediction cache hit rate
metrics = server.get_metrics()
print(f"Cache hit rate: {metrics['cache_hits'] / metrics['total_requests']}")

# Enable feature caching
server.enable_cache(ttl=3600)
```

### Data Drift Detection
```python
# Check drift results
drift_results = detector.detect_drift(current_features)
if drift_results["has_drift"]:
    # Retrain model
    trainer.train(new_features, new_labels)
```

## Rollback Procedures

### Model Rollback
```python
from src.models.registry import ModelRegistry

registry = ModelRegistry()
registry.promote_model("recommendation_v1", to_stage="Production")
```

### Feature Rollback
```python
# Feature store maintains version history
fs.get_feature_table("users", timestamp="2025-01-01")
```

## Scaling Guidelines

| Component | Scaling Metric | Action |
|-----------|---|---|
| API Server | p99 latency > 250ms | Add replicas |
| Feature Store | Query latency > 5s | Increase cache TTL |
| Training | Time > 30min | Increase Spark cluster |
| Database | CPU > 80% | Upgrade instance |

## Backup & Recovery

### Database Backup
```bash
# Daily automated backup to S3
aws rds create-db-snapshot \
  --db-instance-identifier mlops-db \
  --db-snapshot-identifier mlops-db-$(date +%Y%m%d)
```

### Model Registry Backup
```bash
# Backup MLflow artifacts
aws s3 sync \
  s3://your-bucket/mlflow \
  s3://your-bucket/mlflow-backup/$(date +%Y%m%d)
```

### Disaster Recovery Plan
1. **Detection**: Monitoring alerts trigger for service outage
2. **Assessment**: Determine scope (database, service, region)
3. **Failover**: Switch to standby infrastructure
4. **Restore**: Restore from latest backup
5. **Verification**: Run smoke tests
6. **Communication**: Update status page

## Documentation

- [Architecture Guide](./architecture.md)
- [Feature Store Guide](./feature_store_guide.md)
- [Monitoring Guide](./monitoring_guide.md)
- [API Documentation](./api.md)
