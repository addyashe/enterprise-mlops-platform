# Monitoring and Observability Guide

## Overview

Production ML systems require comprehensive monitoring to ensure models perform reliably and catch issues early.

## Key Metrics

### Model Performance Metrics
```python
from src.monitoring.performance import PerformanceMonitor

monitor = PerformanceMonitor("recommendation_v2")

# Log predictions
for prediction, actual, latency in predictions:
    monitor.log_prediction(prediction, actual, latency)

# Get current metrics
metrics = monitor.get_current_metrics(window_size=1000)
print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"Avg Latency: {metrics['avg_latency_ms']:.2f}ms")
print(f"P99 Latency: {metrics['p99_latency_ms']:.2f}ms")
```

### Data Drift Metrics
```python
from src.monitoring.drift_detector import DriftDetector

detector = DriftDetector(threshold=0.05)

# Establish baseline
detector.compute_baseline_statistics(training_features)

# Check for drift
drift_results = detector.detect_drift(serving_features)

if drift_results["has_drift"]:
    for feature in drift_results["features_with_drift"]:
        print(f"Drift in {feature['feature_name']}: "
              f"Baseline mean={feature['baseline_mean']:.2f}, "
              f"Current mean={feature['current_mean']:.2f}")
```

### System Metrics
```
Model Serving:
- Request rate (QPS)
- Response time (p50, p99)
- Error rate
- Cache hit rate
- CPU/Memory usage

Feature Store:
- Query latency
- Cache hit rate
- Data freshness

Training:
- Training duration
- Model quality
- Feature compute time
```

## Alerting Strategy

### Alert Types

**1. Performance Alerts**
```python
from src.monitoring.alerts import AlertManager

manager = AlertManager()

# Define accuracy drop alert
manager.define_alert(
    "accuracy_drop",
    lambda m: m.get("accuracy", 1.0) < 0.85,
    severity="critical"
)

# Define latency alert
manager.define_alert(
    "high_latency",
    lambda m: m.get("p99_latency_ms", 0) > 250,
    severity="warning"
)
```

**2. Data Quality Alerts**
```python
# Missing values alert
manager.define_alert(
    "high_missing_values",
    lambda m: m.get("missing_pct", 0) > 0.1,
    severity="warning"
)

# Feature value range alert
manager.define_alert(
    "feature_out_of_range",
    lambda m: m.get("feature_max", 0) > 1000,
    severity="critical"
)
```

**3. Resource Alerts**
```python
# High memory usage
manager.define_alert(
    "high_memory",
    lambda m: m.get("memory_pct", 0) > 90,
    severity="critical"
)

# Model serving overload
manager.define_alert(
    "queue_depth_high",
    lambda m: m.get("queue_depth", 0) > 100,
    severity="warning"
)
```

### Alert Handlers
```python
# Email handler
def send_email_alert(alert_event):
    send_email(
        to="ml-team@company.com",
        subject=f"Alert: {alert_event['alert_name']}",
        body=f"Severity: {alert_event['severity']}\n{alert_event['description']}"
    )

manager.register_handler("accuracy_drop", send_email_alert)

# Slack handler
def send_slack_alert(alert_event):
    post_to_slack(f":warning: {alert_event['alert_name']}")

manager.register_handler("high_latency", send_slack_alert)

# Auto-remediation
def retrain_model(alert_event):
    trigger_retraining()

manager.register_handler("accuracy_drop", retrain_model)
```

## Dashboards

### Model Performance Dashboard
**Key visualizations**:
- Accuracy over time
- Prediction latency (p50, p99)
- Error rate
- Prediction distribution
- Feature importance

### Data Quality Dashboard
**Key visualizations**:
- Missing value percentage by feature
- Feature value distributions
- Outlier detection results
- Data drift detection

### Business Metrics Dashboard
**Key visualizations**:
- Click-through rate
- Conversion rate
- Revenue impact
- User satisfaction (if available)

### System Health Dashboard
**Key visualizations**:
- API response times
- Error rates
- Cache hit rates
- Resource utilization (CPU, memory)

## Logging and Tracing

### Request Logging
```python
import logging
import json

logger = logging.getLogger(__name__)

def log_prediction_request(request_id, features, prediction):
    log_entry = {
        "request_id": request_id,
        "timestamp": datetime.utcnow().isoformat(),
        "features": features,
        "prediction": prediction,
        "latency_ms": latency
    }
    logger.info(json.dumps(log_entry))
```

### Distributed Tracing
```python
from opentelemetry import trace

tracer = trace.get_tracer(__name__)

with tracer.start_as_current_span("model_prediction") as span:
    # Fetch features
    with tracer.start_as_current_span("feature_retrieval"):
        features = get_features(user_id)
    
    # Make prediction
    with tracer.start_as_current_span("inference"):
        prediction = model.predict(features)
    
    span.set_attribute("prediction", prediction)
```

## Monitoring Best Practices

### 1. Establish Baselines
```python
# Train baseline metrics on historical data
baseline_metrics = compute_metrics(historical_data)

# Use for degradation detection
current_metrics = get_current_metrics()
degradation = check_degradation(baseline_metrics, current_metrics, threshold=0.05)
```

### 2. Automate Alerting
```python
# Check alerts periodically
metrics = get_current_metrics()
triggered_alerts = manager.check_alerts(metrics)

for alert in triggered_alerts:
    logger.critical(f"Alert triggered: {alert['alert_name']}")
```

### 3. Track Metrics Trends
```python
# Store metrics for trend analysis
metrics_history = []

for metrics in periodic_metrics:
    metrics_history.append({
        "timestamp": datetime.utcnow(),
        "accuracy": metrics["accuracy"],
        "latency": metrics["p99_latency_ms"]
    })

# Detect trends
trend = analyze_trend(metrics_history)
```

### 4. Monitor Model Drift
```python
# Regular drift checks
detector = DriftDetector(threshold=0.05)
baseline = detector.compute_baseline_statistics(training_data)

# Check drift periodically
current_data = get_current_serving_data()
drift = detector.detect_drift(current_data)

if drift["has_drift"]:
    # Trigger retraining
    retrain_model()
    # Update baseline
    detector.compute_baseline_statistics(current_data)
```

## Troubleshooting

### High Latency Issues
```python
# Check cache hit rate
metrics = server.get_metrics()
if metrics["cache_hit_rate"] < 0.5:
    # Enable caching or increase TTL
    server.set_cache_ttl(3600)

# Check resource utilization
if cpu_usage > 80:
    # Scale up
    increase_replicas(2)
```

### Accuracy Degradation
```python
# Check for data drift
drift = detector.detect_drift(current_features)
if drift["has_drift"]:
    print("Data drift detected - retraining recommended")

# Check baseline vs current performance
degradation = monitor.check_performance_degradation(baseline)
if degradation["has_degradation"]:
    print("Model degradation detected")
    retrain_model()
```

### False Alerts
```python
# Adjust alert thresholds based on false positive rate
manager.alerts["high_latency"]["threshold"] = 300  # Increase threshold

# Use statistical significance testing
p_value = compute_statistical_significance(control, treatment)
if p_value < 0.05:
    trigger_alert()
```

## References

- [Architecture Guide](./architecture.md)
- [Deployment Guide](./deployment.md)
- [Feature Store Guide](./feature_store_guide.md)
