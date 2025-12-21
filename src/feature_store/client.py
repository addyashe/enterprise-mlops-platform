"""Feature Store client for accessing and managing features."""

import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


class FeatureStoreClient:
    """
    Client for interacting with Databricks Feature Store.
    
    Handles feature table retrieval, point-in-time queries, and feature versioning.
    """

    def __init__(
        self,
        workspace_url: str,
        token: Optional[str] = None,
        debug: bool = False
    ):
        """
        Initialize Feature Store client.
        
        Args:
            workspace_url: Databricks workspace URL
            token: Personal access token (optional, uses environment variable if not provided)
            debug: Enable debug logging
        """
        self.workspace_url = workspace_url
        self.token = token
        self.debug = debug
        self.feature_tables = {}
        self.feature_cache = {}
        self.cache_ttl = 3600  # 1 hour default TTL
        
        if debug:
            logger.setLevel(logging.DEBUG)

    def create_feature_table(
        self,
        name: str,
        primary_keys: List[str],
        timestamp_key: str,
        schema: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Create a new feature table in the feature store.
        
        Args:
            name: Feature table name
            primary_keys: List of primary key columns
            timestamp_key: Column used for time-travel queries
            schema: Column names and types
            
        Returns:
            Dictionary with feature table metadata
        """
        feature_table = {
            "name": name,
            "primary_keys": primary_keys,
            "timestamp_key": timestamp_key,
            "schema": schema,
            "created_at": datetime.utcnow().isoformat(),
            "version": "1.0.0"
        }
        
        self.feature_tables[name] = feature_table
        logger.info(f"Created feature table: {name}")
        return feature_table

    def get_feature_table(
        self,
        table_name: str,
        timestamp: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Retrieve feature table with optional point-in-time query.
        
        Args:
            table_name: Name of the feature table
            timestamp: ISO format timestamp for point-in-time query
            filters: Optional filtering conditions
            
        Returns:
            Feature data with metadata
        """
        if table_name not in self.feature_tables:
            raise ValueError(f"Feature table '{table_name}' not found")

        cache_key = f"{table_name}_{timestamp}_{json.dumps(filters or {})}"
        
        # Check cache
        if cache_key in self.feature_cache:
            cached_data = self.feature_cache[cache_key]
            if datetime.fromisoformat(cached_data["cached_at"]) > (
                datetime.utcnow() - timedelta(seconds=self.cache_ttl)
            ):
                logger.debug(f"Cache hit for {table_name}")
                return cached_data["data"]

        # Simulate feature retrieval
        feature_data = {
            "table_name": table_name,
            "timestamp": timestamp or datetime.utcnow().isoformat(),
            "row_count": 1000000,
            "features": self.feature_tables[table_name]["schema"],
            "filters": filters or {},
            "retrieval_time_seconds": 2.3
        }

        # Cache result
        self.feature_cache[cache_key] = {
            "data": feature_data,
            "cached_at": datetime.utcnow().isoformat()
        }

        logger.info(f"Retrieved {feature_data['row_count']} rows from {table_name}")
        return feature_data

    def search_features(
        self,
        query: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search for features by name or description.
        
        Args:
            query: Search query string
            limit: Maximum number of results
            
        Returns:
            List of matching features
        """
        results = [
            {
                "name": table,
                "primary_keys": metadata["primary_keys"],
                "schema_size": len(metadata["schema"])
            }
            for table, metadata in self.feature_tables.items()
            if query.lower() in table.lower()
        ][:limit]

        logger.info(f"Found {len(results)} features matching '{query}'")
        return results

    def register_feature(
        self,
        feature_name: str,
        table_name: str,
        column_name: str,
        description: str,
        data_type: str
    ) -> Dict[str, Any]:
        """
        Register a new feature in the feature store.
        
        Args:
            feature_name: Unique feature identifier
            table_name: Source feature table
            column_name: Column name in the table
            description: Feature documentation
            data_type: Data type (int, float, string, etc)
            
        Returns:
            Feature registration metadata
        """
        feature_metadata = {
            "feature_name": feature_name,
            "table_name": table_name,
            "column_name": column_name,
            "description": description,
            "data_type": data_type,
            "registered_at": datetime.utcnow().isoformat(),
            "popularity_score": 0,
            "owner": "platform"
        }

        logger.info(f"Registered feature: {feature_name}")
        return feature_metadata

    def get_feature_lineage(
        self,
        feature_name: str
    ) -> Dict[str, Any]:
        """
        Get lineage information for a feature.
        
        Args:
            feature_name: Name of the feature
            
        Returns:
            Lineage information including source and transformations
        """
        lineage = {
            "feature": feature_name,
            "source_tables": [],
            "transformations": [],
            "computed_at": datetime.utcnow().isoformat(),
            "last_updated": datetime.utcnow().isoformat()
        }
        
        logger.info(f"Retrieved lineage for feature: {feature_name}")
        return lineage

    def list_features(self) -> List[str]:
        """Get list of all available feature tables."""
        return list(self.feature_tables.keys())

    def set_cache_ttl(self, seconds: int) -> None:
        """Set feature cache time-to-live in seconds."""
        self.cache_ttl = seconds
        logger.info(f"Set cache TTL to {seconds} seconds")

    def clear_cache(self) -> None:
        """Clear all cached feature data."""
        self.feature_cache.clear()
        logger.info("Cleared feature cache")
