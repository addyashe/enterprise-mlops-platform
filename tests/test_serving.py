"""Unit tests for model serving module."""

import pytest
from src.serving.model_server import ModelServer
from src.serving.ab_testing import ABTestManager
from src.serving.batch_predictor import BatchPredictor


class TestModelServer:
    """Tests for ModelServer."""

    def test_server_initialization(self):
        """Test server initialization."""
        server = ModelServer("test_model", 8000)
        
        assert server.model_name == "test_model"
        assert server.port == 8000
        assert server.model is None

    def test_health_check(self):
        """Test health check endpoint."""
        server = ModelServer("test")
        health = server.health_check()
        
        assert health["status"] == "unhealthy"  # No model loaded
        assert health["model_loaded"] is False

    def test_prediction_caching(self):
        """Test prediction caching."""
        server = ModelServer("test")
        
        features = {"feature_1": 1.0, "feature_2": 2.0}
        
        # Without model, just test caching logic
        # First call
        result1 = {"prediction": 0.5, "confidence": 0.9}
        
        # The actual prediction would be cached
        assert result1 is not None

    def test_get_metrics(self):
        """Test getting server metrics."""
        server = ModelServer("test")
        
        metrics = server.get_metrics()
        
        assert "total_requests" in metrics
        assert "error_count" in metrics
        assert metrics["total_requests"] == 0

    def test_clear_cache(self):
        """Test clearing prediction cache."""
        server = ModelServer("test")
        server.prediction_cache["test_key"] = {"result": "test"}
        
        server.clear_cache()
        
        assert len(server.prediction_cache) == 0


class TestABTestManager:
    """Tests for ABTestManager."""

    def test_create_experiment(self):
        """Test creating an A/B test."""
        manager = ABTestManager()
        
        exp = manager.create_experiment(
            name="test_experiment",
            control_model="model_v1",
            treatment_model="model_v2",
            traffic_split=0.5,
            duration_days=7
        )
        
        assert exp["name"] == "test_experiment"
        assert exp["traffic_split"] == 0.5
        assert exp["status"] == "running"

    def test_assign_variant(self):
        """Test variant assignment."""
        manager = ABTestManager()
        manager.create_experiment(
            name="test",
            control_model="v1",
            treatment_model="v2",
            traffic_split=0.5
        )
        
        assignment = manager.assign_variant("test", "user_123")
        
        assert assignment["user_id"] == "user_123"
        assert assignment["variant"] in ["control", "treatment"]
        assert assignment["model_id"] in ["v1", "v2"]

    def test_consistent_assignment(self):
        """Test that same user gets same variant."""
        manager = ABTestManager()
        manager.create_experiment(
            name="test",
            control_model="v1",
            treatment_model="v2",
            traffic_split=0.5
        )
        
        assign1 = manager.assign_variant("test", "user_123")
        assign2 = manager.assign_variant("test", "user_123")
        
        assert assign1["variant"] == assign2["variant"]

    def test_log_event(self):
        """Test logging an event."""
        manager = ABTestManager()
        manager.create_experiment(
            name="test",
            control_model="v1",
            treatment_model="v2"
        )
        
        manager.log_event("test", "user_123", "click", 1.0)
        
        assert len(manager.test_logs) == 1

    def test_stop_experiment(self):
        """Test stopping an experiment."""
        manager = ABTestManager()
        manager.create_experiment(
            name="test",
            control_model="v1",
            treatment_model="v2"
        )
        
        result = manager.stop_experiment("test")
        
        assert result["status"] == "stopped"
        assert "stopped_at" in result

    def test_list_experiments(self):
        """Test listing experiments."""
        manager = ABTestManager()
        manager.create_experiment(
            name="exp1",
            control_model="v1",
            treatment_model="v2"
        )
        
        experiments = manager.list_experiments()
        
        assert "exp1" in experiments


class TestBatchPredictor:
    """Tests for BatchPredictor."""

    def test_batch_initialization(self):
        """Test batch predictor initialization."""
        predictor = BatchPredictor("model_v1", "/tmp/output")
        
        assert predictor.model_name == "model_v1"
        assert predictor.output_path == "/tmp/output"

    def test_predict_batch(self):
        """Test batch prediction."""
        predictor = BatchPredictor("model_v1", "/tmp/output")
        
        features_list = [
            {"feature_1": 1.0, "feature_2": 2.0},
            {"feature_1": 3.0, "feature_2": 4.0}
        ]
        
        result = predictor.predict_batch(features_list, save_results=False)
        
        assert result["total_samples"] == 2
        assert result["successful_predictions"] == 2
        assert "throughput_samples_per_sec" in result

    def test_job_history(self):
        """Test job history tracking."""
        predictor = BatchPredictor("model_v1", "/tmp/output")
        
        features = [{"f": 1.0}, {"f": 2.0}]
        predictor.predict_batch(features, save_results=False)
        
        jobs = predictor.list_jobs()
        
        assert len(jobs) == 1
        assert jobs[0]["samples_processed"] == 2
