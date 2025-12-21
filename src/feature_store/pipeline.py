"""Feature computation pipelines for automated feature engineering."""

from typing import Dict, List, Optional, Callable, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class FeatureComputationPipeline:
    """
    Orchestrates feature computation from raw data.
    
    Handles transformations, aggregations, and feature computation DAGs.
    """

    def __init__(self, name: str, description: str = ""):
        """
        Initialize feature computation pipeline.
        
        Args:
            name: Pipeline name
            description: Pipeline description
        """
        self.name = name
        self.description = description
        self.stages = []
        self.execution_history = []

    def add_stage(
        self,
        name: str,
        transformation: Callable[[Dict], Dict],
        dependencies: Optional[List[str]] = None
    ) -> "FeatureComputationPipeline":
        """
        Add a transformation stage to the pipeline.
        
        Args:
            name: Stage name
            transformation: Function that transforms data
            dependencies: Names of dependent stages
            
        Returns:
            Self for method chaining
        """
        stage = {
            "name": name,
            "transformation": transformation,
            "dependencies": dependencies or [],
            "order": len(self.stages)
        }
        self.stages.append(stage)
        logger.info(f"Added stage '{name}' to pipeline '{self.name}'")
        return self

    def execute(
        self,
        input_data: Dict[str, Any],
        partial: bool = False
    ) -> Dict[str, Any]:
        """
        Execute the feature computation pipeline.
        
        Args:
            input_data: Input data for feature computation
            partial: Allow partial execution if some stages fail
            
        Returns:
            Computed features
        """
        execution_start = datetime.utcnow()
        results = {"input": input_data, "features": {}}
        failed_stages = []

        for stage in self.stages:
            try:
                # Check dependencies
                for dep in stage["dependencies"]:
                    if dep not in results["features"] and not partial:
                        raise ValueError(f"Dependency '{dep}' not computed")

                # Execute transformation
                feature_value = stage["transformation"](input_data)
                results["features"][stage["name"]] = feature_value
                logger.debug(f"Executed stage '{stage['name']}'")

            except Exception as e:
                error_msg = f"Error in stage '{stage['name']}': {str(e)}"
                logger.warning(error_msg)
                if not partial:
                    raise
                failed_stages.append(stage["name"])

        execution_time = (datetime.utcnow() - execution_start).total_seconds()
        results["execution_time_seconds"] = execution_time
        results["failed_stages"] = failed_stages
        results["executed_at"] = execution_start.isoformat()

        logger.info(
            f"Pipeline '{self.name}' executed in {execution_time:.2f}s "
            f"({len(self.stages) - len(failed_stages)}/{len(self.stages)} stages)"
        )

        return results

    def validate(self) -> Dict[str, Any]:
        """
        Validate pipeline structure and dependencies.
        
        Returns:
            Validation results with any issues found
        """
        validation_results = {
            "valid": True,
            "issues": [],
            "stage_count": len(self.stages)
        }

        # Check for circular dependencies
        stage_names = {s["name"]: s for s in self.stages}
        for stage in self.stages:
            for dep in stage["dependencies"]:
                if dep not in stage_names:
                    validation_results["issues"].append(
                        f"Stage '{stage['name']}' depends on non-existent '{dep}'"
                    )
                    validation_results["valid"] = False

        if validation_results["valid"]:
            logger.info(f"Pipeline '{self.name}' validation passed")
        else:
            logger.warning(f"Pipeline '{self.name}' validation failed")

        return validation_results

    def get_execution_plan(self) -> List[Dict[str, Any]]:
        """
        Get the execution plan (topologically sorted stages).
        
        Returns:
            List of stages in execution order
        """
        # Simple topological sort
        sorted_stages = []
        remaining = self.stages.copy()
        completed = set()

        while remaining:
            for stage in remaining[:]:
                if all(dep in completed for dep in stage["dependencies"]):
                    sorted_stages.append(stage)
                    completed.add(stage["name"])
                    remaining.remove(stage)
                    break
            else:
                raise ValueError("Circular dependency detected in pipeline")

        return sorted_stages

    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        return {
            "name": self.name,
            "stage_count": len(self.stages),
            "execution_count": len(self.execution_history),
            "avg_execution_time": (
                sum(e["execution_time"] for e in self.execution_history) /
                len(self.execution_history)
                if self.execution_history
                else 0
            )
        }
