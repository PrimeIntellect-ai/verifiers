# ABOUTME: MLFlow-based tracker implementation for experiment tracking.
# ABOUTME: Logs metrics, parameters, and artifacts to MLFlow tracking server.

from typing import Any, Optional

from verifiers.tracking.tracker import Tracker


class MLFlowTracker(Tracker):
    """MLFlow-based tracker for experiment tracking."""

    def __init__(
        self,
        experiment_name: Optional[str] = None,
        run_name: Optional[str] = None,
        tracking_uri: Optional[str] = None,
        tags: Optional[dict[str, str]] = None,
        **kwargs,
    ):
        super().__init__(project=experiment_name, name=run_name, **kwargs)
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.tracking_uri = tracking_uri
        self.tags = tags
        self._mlflow = None
        self._run = None

    def init(self, **kwargs) -> None:
        import mlflow

        self._mlflow = mlflow

        if self.tracking_uri:
            self._mlflow.set_tracking_uri(self.tracking_uri)

        if self.experiment_name:
            self._mlflow.set_experiment(self.experiment_name)

        if self._mlflow.active_run() is None:
            self._run = self._mlflow.start_run(run_name=self.run_name, tags=self.tags)
        else:
            self._run = self._mlflow.active_run()

        if self.config:
            self._mlflow.log_params(self.config)

        self._initialized = True

    def log_metrics(
        self, metrics: dict[str, float], step: Optional[int] = None, **kwargs
    ) -> None:
        if self._mlflow is None:
            return
        self._mlflow.log_metrics(metrics, step=step)

    def log_table(
        self,
        table_name: str,
        data: dict[str, list[Any]],
        step: Optional[int] = None,
        **kwargs,
    ) -> None:
        if self._mlflow is None:
            return

        try:
            import pandas as pd

            df = pd.DataFrame(data)

            import tempfile
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=f"_{table_name}.csv", delete=False
            ) as f:
                df.to_csv(f.name, index=False)
                self._mlflow.log_artifact(f.name, artifact_path="tables")
        except Exception as e:
            self.logger.warning(f"Failed to log table {table_name}: {e}")

    def log_completions(
        self,
        prompts: list[str],
        completions: list[str],
        rewards: list[float],
        step: Optional[int] = None,
        **kwargs,
    ) -> None:
        table_data = {
            "prompt": prompts,
            "completion": completions,
            "reward": rewards,
        }
        self.log_table("completions", table_data, step=step)

    def log_config(self, config: dict[str, Any], **kwargs) -> None:
        super().log_config(config)
        if self._mlflow is not None:
            self._mlflow.log_params(config)

    def finish(self, **kwargs) -> None:
        if self._mlflow is not None:
            self._mlflow.end_run()
