"""Unit tests for feature store module."""

import pytest
from src.feature_store.client import FeatureStoreClient
from src.feature_store.pipeline import FeatureComputationPipeline


class TestFeatureStoreClient:
    """Tests for FeatureStoreClient."""

    def test_create_feature_table(self):
        """Test creating a feature table."""
        client = FeatureStoreClient("https://test.databricks.com")
        
        schema = {"user_id": "int", "age": "int", "country": "string"}
        table = client.create_feature_table(
            name="user_features",
            primary_keys=["user_id"],
            timestamp_key="updated_at",
            schema=schema
        )
        
        assert table["name"] == "user_features"
        assert table["primary_keys"] == ["user_id"]
        assert table["schema"] == schema

    def test_get_feature_table(self):
        """Test retrieving a feature table."""
        client = FeatureStoreClient("https://test.databricks.com")
        client.create_feature_table(
            name="test_features",
            primary_keys=["id"],
            timestamp_key="ts",
            schema={"id": "int", "value": "double"}
        )
        
        result = client.get_feature_table("test_features")
        
        assert result["table_name"] == "test_features"
        assert "features" in result
        assert result["row_count"] == 1000000

    def test_feature_table_not_found(self):
        """Test error when feature table doesn't exist."""
        client = FeatureStoreClient("https://test.databricks.com")
        
        with pytest.raises(ValueError):
            client.get_feature_table("nonexistent_table")

    def test_search_features(self):
        """Test searching for features."""
        client = FeatureStoreClient("https://test.databricks.com")
        client.create_feature_table(
            name="user_features",
            primary_keys=["user_id"],
            timestamp_key="ts",
            schema={"value": "int"}
        )
        
        results = client.search_features("user")
        
        assert len(results) > 0
        assert "user_features" in [r["name"] for r in results]

    def test_register_feature(self):
        """Test registering a feature."""
        client = FeatureStoreClient("https://test.databricks.com")
        
        feature = client.register_feature(
            feature_name="user_age",
            table_name="users",
            column_name="age",
            description="User age in years",
            data_type="int"
        )
        
        assert feature["feature_name"] == "user_age"
        assert feature["data_type"] == "int"

    def test_cache_operations(self):
        """Test feature cache operations."""
        client = FeatureStoreClient("https://test.databricks.com")
        client.create_feature_table(
            name="cached_features",
            primary_keys=["id"],
            timestamp_key="ts",
            schema={"value": "double"}
        )
        
        # Get feature (cache miss)
        result1 = client.get_feature_table("cached_features")
        
        # Get feature again (cache hit)
        result2 = client.get_feature_table("cached_features")
        
        assert result1 == result2
        
        # Clear cache
        client.clear_cache()
        assert len(client.feature_cache) == 0


class TestFeatureComputationPipeline:
    """Tests for FeatureComputationPipeline."""

    def test_create_pipeline(self):
        """Test creating a pipeline."""
        pipeline = FeatureComputationPipeline("test_pipeline", "Test pipeline")
        
        assert pipeline.name == "test_pipeline"
        assert len(pipeline.stages) == 0

    def test_add_stage(self):
        """Test adding stages to pipeline."""
        pipeline = FeatureComputationPipeline("test")
        
        pipeline.add_stage("normalize", lambda x: x / 10)
        pipeline.add_stage("scale", lambda x: x * 2)
        
        assert len(pipeline.stages) == 2

    def test_execute_pipeline(self):
        """Test executing a pipeline."""
        pipeline = FeatureComputationPipeline("test")
        pipeline.add_stage("double", lambda d: d * 2)
        pipeline.add_stage("add_one", lambda d: d + 1)
        
        result = pipeline.execute({"value": 5})
        
        assert "features" in result
        assert result["features"]["double"] == 10
        assert result["features"]["add_one"] == 6

    def test_pipeline_with_dependencies(self):
        """Test pipeline stage dependencies."""
        pipeline = FeatureComputationPipeline("test")
        pipeline.add_stage("compute_a", lambda d: 10)
        pipeline.add_stage("compute_b", lambda d: 20, dependencies=["compute_a"])
        
        plan = pipeline.get_execution_plan()
        
        assert len(plan) == 2
        assert plan[0]["name"] == "compute_a"

    def test_pipeline_validation(self):
        """Test pipeline validation."""
        pipeline = FeatureComputationPipeline("test")
        pipeline.add_stage("stage1", lambda d: d)
        pipeline.add_stage("stage2", lambda d: d, dependencies=["nonexistent"])
        
        validation = pipeline.validate()
        
        assert not validation["valid"]
        assert len(validation["issues"]) > 0
