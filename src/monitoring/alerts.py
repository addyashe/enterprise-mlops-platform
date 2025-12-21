"""Alert management for ML operations."""

from typing import Dict, List, Any, Callable
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class AlertManager:
    """
    Manages alerts for monitoring anomalies and issues.
    
    Handles alert definitions, triggering, and escalation.
    """

    def __init__(self):
        """Initialize alert manager."""
        self.alerts = {}
        self.alert_history = []
        self.handlers = {}

    def define_alert(
        self,
        alert_name: str,
        condition: Callable[[Dict[str, Any]], bool],
        severity: str = "warning",
        description: str = ""
    ) -> Dict[str, Any]:
        """
        Define a new alert rule.
        
        Args:
            alert_name: Unique alert name
            condition: Function that returns True when alert should trigger
            severity: Alert severity (info, warning, critical)
            description: Alert description
            
        Returns:
            Alert definition
        """
        alert_def = {
            "name": alert_name,
            "condition": condition,
            "severity": severity,
            "description": description,
            "created_at": datetime.utcnow().isoformat(),
            "triggered_count": 0
        }

        self.alerts[alert_name] = alert_def
        logger.info(f"Defined alert: {alert_name} (severity: {severity})")
        return alert_def

    def check_alerts(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Check all alerts against metrics.
        
        Args:
            metrics: Metrics to check against
            
        Returns:
            List of triggered alerts
        """
        triggered_alerts = []

        for alert_name, alert_def in self.alerts.items():
            try:
                if alert_def["condition"](metrics):
                    alert_event = {
                        "alert_name": alert_name,
                        "severity": alert_def["severity"],
                        "description": alert_def["description"],
                        "triggered_at": datetime.utcnow().isoformat(),
                        "metrics": metrics
                    }

                    triggered_alerts.append(alert_event)
                    self.alerts[alert_name]["triggered_count"] += 1
                    self.alert_history.append(alert_event)

                    logger.warning(f"Alert triggered: {alert_name} ({alert_def['severity']})")

                    # Call handlers
                    if alert_name in self.handlers:
                        for handler in self.handlers[alert_name]:
                            try:
                                handler(alert_event)
                            except Exception as e:
                                logger.error(f"Error in alert handler: {str(e)}")

            except Exception as e:
                logger.error(f"Error checking alert '{alert_name}': {str(e)}")

        return triggered_alerts

    def register_handler(
        self,
        alert_name: str,
        handler: Callable[[Dict[str, Any]], None]
    ) -> None:
        """
        Register a handler for an alert.
        
        Args:
            alert_name: Alert to handle
            handler: Function to call when alert triggers
        """
        if alert_name not in self.handlers:
            self.handlers[alert_name] = []

        self.handlers[alert_name].append(handler)
        logger.info(f"Registered handler for alert: {alert_name}")

    def get_alert_summary(self) -> Dict[str, Any]:
        """Get summary of all alerts."""
        triggered_count = sum(a["triggered_count"] for a in self.alerts.values())
        
        return {
            "total_alerts": len(self.alerts),
            "total_triggered": triggered_count,
            "alerts": {
                name: {
                    "severity": alert["severity"],
                    "triggered_count": alert["triggered_count"],
                    "description": alert["description"]
                }
                for name, alert in self.alerts.items()
            }
        }

    def get_alert_history(self, limit: int = None) -> List[Dict[str, Any]]:
        """Get alert history."""
        if limit:
            return self.alert_history[-limit:]
        return self.alert_history.copy()

    def clear_history(self) -> None:
        """Clear alert history."""
        self.alert_history.clear()
        logger.info("Cleared alert history")
