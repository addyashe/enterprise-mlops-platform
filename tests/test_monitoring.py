"""Unit tests for monitoring module."""

import pytest
from src.monitoring.drift_detector import DriftDetector
from src.monitoring.performance import PerformanceMonitor
from src.monitoring.alerts import AlertManager


class TestDriftDetector:
    """Tests for DriftDetector."""

    def test_compute_baseline(self):
        """Test computing baseline statistics."""
        detector = DriftDetector()
        features = [
            {"feature_1": 1.0, "feature_2": 2.0},
            {"feature_1": 2.0, "feature_2": 3.0},
            {"feature_1": 3.0, "feature_2": 4.0}
        ]
        
        baseline = detector.compute_baseline_statistics(features)
        
        assert "feature_1" in baseline
        assert "mean" in baseline["feature_1"]
        assert baseline["feature_1"]["count"] == 3

    def test_detect_no_drift(self):
        """Test drift detection when there's no drift."""
        detector = DriftDetector()
        baseline_features = [
            {"value": 1.0},
            {"value": 2.0},
            {"value": 3.0}
        ]
        
        detector.compute_baseline_statistics(baseline_features)
        
        current_features = [
            {"value": 1.5},
            {"value": 2.5},
            {"value": 3.5}
        ]
        
        results = detector.detect_drift(current_features)
        
        assert "has_drift" in results

    def test_drift_history(self):
        """Test drift detection history."""
        detector = DriftDetector()
        detector.compute_baseline_statistics([{"value": 1.0}])
        
        detector.detect_drift([{"value": 1.0}])
        detector.detect_drift([{"value": 1.0}])
        
        history = detector.get_drift_history(limit=1)
        
        assert len(history) == 1

    def test_drift_summary(self):
        """Test drift summary."""
        detector = DriftDetector()
        detector.compute_baseline_statistics([{"value": 1.0}])
        
        detector.detect_drift([{"value": 1.0}])
        
        summary = detector.get_drift_summary()
        
        assert summary["total_checks"] == 1


class TestPerformanceMonitor:
    """Tests for PerformanceMonitor."""

    def test_monitor_initialization(self):
        """Test monitor initialization."""
        monitor = PerformanceMonitor("test_model")
        
        assert monitor.model_name == "test_model"
        assert len(monitor.metrics_log) == 0

    def test_log_prediction(self):
        """Test logging a prediction."""
        monitor = PerformanceMonitor("test")
        
        monitor.log_prediction(1, 1, 50.0)
        
        assert len(monitor.metrics_log) == 1
        assert monitor.metrics_log[0]["is_correct"] is True

    def test_get_current_metrics(self):
        """Test getting current metrics."""
        monitor = PerformanceMonitor("test")
        
        for i in range(10):
            monitor.log_prediction(1, 1, 50.0 + i)
        
        metrics = monitor.get_current_metrics()
        
        assert "accuracy" in metrics
        assert "avg_latency_ms" in metrics
        assert metrics["accuracy"] == 1.0  # All predictions correct

    def test_performance_degradation(self):
        """Test performance degradation detection."""
        monitor = PerformanceMonitor("test")
        
        for i in range(10):
            monitor.log_prediction(1, 1, 50.0)
        
        baseline = {"accuracy": 1.0}
        degradation = monitor.check_performance_degradation(baseline, 0.1)
        
        assert "has_degradation" in degradation

    def test_metrics_history(self):
        """Test metrics history."""
        monitor = PerformanceMonitor("test")
        
        monitor.log_prediction(1, 1, 50.0)
        monitor.log_prediction(0, 1, 100.0)
        
        history = monitor.get_metrics_history(limit=1)
        
        assert len(history) == 1


class TestAlertManager:
    """Tests for AlertManager."""

    def test_define_alert(self):
        """Test defining an alert."""
        manager = AlertManager()
        
        alert = manager.define_alert(
            "high_error_rate",
            lambda m: m.get("error_rate", 0) > 0.1,
            severity="critical"
        )
        
        assert alert["name"] == "high_error_rate"
        assert alert["severity"] == "critical"

    def test_check_alerts(self):
        """Test checking alerts."""
        manager = AlertManager()
        manager.define_alert(
            "test_alert",
            lambda m: m.get("value", 0) > 5,
            severity="warning"
        )
        
        triggered = manager.check_alerts({"value": 10})
        
        assert len(triggered) == 1
        assert triggered[0]["alert_name"] == "test_alert"

    def test_alert_handler(self):
        """Test alert handler registration and calling."""
        manager = AlertManager()
        handler_called = []
        
        def test_handler(event):
            handler_called.append(event)
        
        manager.define_alert(
            "test",
            lambda m: True,
            severity="warning"
        )
        manager.register_handler("test", test_handler)
        
        manager.check_alerts({})
        
        assert len(handler_called) == 1

    def test_alert_summary(self):
        """Test alert summary."""
        manager = AlertManager()
        manager.define_alert("alert1", lambda m: False)
        manager.define_alert("alert2", lambda m: False)
        
        summary = manager.get_alert_summary()
        
        assert summary["total_alerts"] == 2

    def test_alert_history(self):
        """Test alert history."""
        manager = AlertManager()
        manager.define_alert("test", lambda m: True)
        
        manager.check_alerts({})
        manager.check_alerts({})
        
        history = manager.get_alert_history()
        
        assert len(history) == 2
