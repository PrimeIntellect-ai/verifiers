# ABOUTME: Weights & Biases tracker implementation.
# ABOUTME: Integrates with W&B for experiment tracking and visualization.

from typing import Any, Optional

from verifiers.tracking.tracker import Tracker


class WandbTracker(Tracker):
    """Weights & Biases experiment tracker."""

    def __init__(
        self,
        project: Optional[str] = None,
        name: Optional[str] = None,
        config: Optional[dict[str, Any]] = None,
        entity: Optional[str] = None,
        tags: Optional[list[str]] = None,
        group: Optional[str] = None,
        job_type: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(project=project, name=name, config=config, **kwargs)
        self.entity = entity
        self.tags = tags
        self.group = group
        self.job_type = job_type
        self._wandb = None
        self._run = None

    def init(self, **kwargs) -> None:
        try:
            import wandb
            self._wandb = wandb
        except ImportError:
            raise ImportError(
                "wandb is required for WandbTracker. Install with: pip install wandb"
            )

        if self._wandb.run is None:
            self._run = self._wandb.init(
                project=self.project,
                name=self.name,
                config=self.config,
                entity=self.entity,
                tags=self.tags,
                group=self.group,
                job_type=self.job_type,
                **kwargs,
            )
        else:
            self._run = self._wandb.run
            self.logger.info("Using existing wandb run")

    def log_metrics(
        self, metrics: dict[str, float], step: Optional[int] = None, **kwargs
    ) -> None:
        if self._wandb is None or self._wandb.run is None:
            self.logger.warning("WandbTracker not initialized, skipping log_metrics")
            return

        log_dict = dict(metrics)
        if step is not None:
            log_dict["step"] = step

        self._wandb.log(log_dict, **kwargs)

    def log_table(
        self,
        table_name: str,
        data: dict[str, list[Any]],
        step: Optional[int] = None,
        **kwargs,
    ) -> None:
        if self._wandb is None or self._wandb.run is None:
            self.logger.warning("WandbTracker not initialized, skipping log_table")
            return

        try:
            import pandas as pd

            df = pd.DataFrame(data)
            table = self._wandb.Table(dataframe=df)
            log_dict = {table_name: table}
            if step is not None:
                log_dict["step"] = step
            self._wandb.log(log_dict, **kwargs)
        except ImportError:
            self.logger.warning("pandas required for table logging, skipping")
        except Exception as e:
            self.logger.warning(f"Failed to log table {table_name}: {e}")

    def log_config(self, config: dict[str, Any], **kwargs) -> None:
        super().log_config(config)
        if self._wandb is not None and self._wandb.run is not None:
            self._wandb.config.update(config, **kwargs)

    def finish(self, **kwargs) -> None:
        if self._wandb is not None and self._run is not None:
            self._wandb.finish(**kwargs)
