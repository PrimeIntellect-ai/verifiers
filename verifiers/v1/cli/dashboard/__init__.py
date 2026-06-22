"""Rich live dashboards for the eval and validate runs (one frame, refreshed on a tick)."""

from verifiers.v1.cli.dashboard.eval import dashboard
from verifiers.v1.cli.dashboard.validate import TaskProgress, validate_dashboard

__all__ = ["TaskProgress", "dashboard", "validate_dashboard"]
