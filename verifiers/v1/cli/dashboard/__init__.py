"""Rich live dashboards."""

from verifiers.v1.cli.dashboard.eval import dashboard
from verifiers.v1.cli.dashboard.validate import TaskProgress, validate_dashboard

__all__ = ["TaskProgress", "dashboard", "validate_dashboard"]
