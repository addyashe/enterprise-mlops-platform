"""A/B testing framework for model comparison."""

from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import random
import logging

logger = logging.getLogger(__name__)


class ABTestManager:
    """
    Manages A/B testing experiments for comparing model variants.
    
    Handles traffic splitting, statistical testing, and result analysis.
    """

    def __init__(self):
        """Initialize A/B test manager."""
        self.experiments = {}
        self.test_logs = []

    def create_experiment(
        self,
        name: str,
        control_model: str,
        treatment_model: str,
        traffic_split: float = 0.5,
        duration_days: int = 7,
        metrics_to_track: list = None
    ) -> Dict[str, Any]:
        """
        Create a new A/B test experiment.
        
        Args:
            name: Experiment name
            control_model: Control model ID
            treatment_model: Treatment model ID
            traffic_split: Fraction of traffic to treatment (0-1)
            duration_days: Duration of test in days
            metrics_to_track: List of metrics to track
            
        Returns:
            Experiment configuration
        """
        if not 0 < traffic_split < 1:
            raise ValueError("traffic_split must be between 0 and 1")

        experiment = {
            "name": name,
            "control_model": control_model,
            "treatment_model": treatment_model,
            "traffic_split": traffic_split,
            "duration_days": duration_days,
            "metrics_to_track": metrics_to_track or ["click_rate", "conversion_rate", "revenue"],
            "created_at": datetime.utcnow().isoformat(),
            "end_date": (datetime.utcnow() + timedelta(days=duration_days)).isoformat(),
            "status": "running",
            "results": {}
        }

        self.experiments[name] = experiment
        logger.info(f"Created A/B test experiment: {name}")
        return experiment

    def assign_variant(
        self,
        experiment_name: str,
        user_id: str
    ) -> Dict[str, Any]:
        """
        Assign a user to a variant (control or treatment).
        
        Args:
            experiment_name: Experiment name
            user_id: User identifier
            
        Returns:
            Assignment details
        """
        if experiment_name not in self.experiments:
            raise ValueError(f"Experiment '{experiment_name}' not found")

        exp = self.experiments[experiment_name]
        
        # Deterministic assignment based on user_id hash
        hash_value = hash(f"{experiment_name}_{user_id}") % 100
        assigned_treatment = hash_value < (exp["traffic_split"] * 100)

        assignment = {
            "experiment_name": experiment_name,
            "user_id": user_id,
            "variant": "treatment" if assigned_treatment else "control",
            "model_id": exp["treatment_model"] if assigned_treatment else exp["control_model"],
            "assigned_at": datetime.utcnow().isoformat()
        }

        return assignment

    def log_event(
        self,
        experiment_name: str,
        user_id: str,
        event_type: str,
        metric_value: float,
        metadata: Dict[str, Any] = None
    ) -> None:
        """
        Log an event for analysis.
        
        Args:
            experiment_name: Experiment name
            user_id: User ID
            event_type: Type of event (click, conversion, etc)
            metric_value: Metric value
            metadata: Optional metadata
        """
        assignment = self.assign_variant(experiment_name, user_id)
        
        log_entry = {
            "experiment_name": experiment_name,
            "user_id": user_id,
            "variant": assignment["variant"],
            "event_type": event_type,
            "metric_value": metric_value,
            "metadata": metadata or {},
            "timestamp": datetime.utcnow().isoformat()
        }

        self.test_logs.append(log_entry)

    def get_results(
        self,
        experiment_name: str,
        alpha: float = 0.05
    ) -> Dict[str, Any]:
        """
        Get analysis results for an experiment.
        
        Args:
            experiment_name: Experiment name
            alpha: Statistical significance level
            
        Returns:
            Analysis results including p-values and recommendations
        """
        if experiment_name not in self.experiments:
            raise ValueError(f"Experiment '{experiment_name}' not found")

        # Filter logs for this experiment
        exp_logs = [l for l in self.test_logs if l["experiment_name"] == experiment_name]

        if not exp_logs:
            logger.warning(f"No data for experiment '{experiment_name}'")
            return {"status": "no_data"}

        # Split by variant
        control_data = [l for l in exp_logs if l["variant"] == "control"]
        treatment_data = [l for l in exp_logs if l["variant"] == "treatment"]

        if not control_data or not treatment_data:
            logger.warning(f"Insufficient data in both variants")
            return {"status": "insufficient_data"}

        # Calculate metrics
        control_metric = sum(l["metric_value"] for l in control_data) / len(control_data)
        treatment_metric = sum(l["metric_value"] for l in treatment_data) / len(treatment_data)

        # Perform t-test
        from scipy import stats
        t_stat, p_value = stats.ttest_ind(
            [l["metric_value"] for l in treatment_data],
            [l["metric_value"] for l in control_data]
        )

        # Determine winner
        if p_value < alpha:
            if treatment_metric > control_metric:
                winner = "treatment"
            else:
                winner = "control"
        else:
            winner = "inconclusive"

        results = {
            "experiment_name": experiment_name,
            "control_samples": len(control_data),
            "treatment_samples": len(treatment_data),
            "control_metric": control_metric,
            "treatment_metric": treatment_metric,
            "metric_lift": (treatment_metric - control_metric) / control_metric if control_metric else 0,
            "t_statistic": t_stat,
            "p_value": p_value,
            "significant_at_alpha": p_value < alpha,
            "winner": winner,
            "recommendation": self._get_recommendation(winner, p_value, control_metric, treatment_metric)
        }

        self.experiments[experiment_name]["results"] = results
        logger.info(f"Analysis completed for experiment '{experiment_name}': winner={winner}")

        return results

    def _get_recommendation(
        self,
        winner: str,
        p_value: float,
        control_metric: float,
        treatment_metric: float
    ) -> str:
        """Generate recommendation based on results."""
        if winner == "inconclusive":
            return "Continue test or increase sample size"
        elif winner == "treatment":
            return "Deploy treatment model to production"
        else:
            return "Keep control model, investigate treatment"

    def stop_experiment(self, experiment_name: str) -> Dict[str, Any]:
        """Stop a running experiment."""
        if experiment_name not in self.experiments:
            raise ValueError(f"Experiment '{experiment_name}' not found")

        self.experiments[experiment_name]["status"] = "stopped"
        self.experiments[experiment_name]["stopped_at"] = datetime.utcnow().isoformat()

        logger.info(f"Stopped experiment: {experiment_name}")
        return self.experiments[experiment_name]

    def list_experiments(self, status: str = None) -> Dict[str, Any]:
        """List all experiments, optionally filtered by status."""
        if status:
            return {
                name: exp for name, exp in self.experiments.items()
                if exp["status"] == status
            }
        return self.experiments
