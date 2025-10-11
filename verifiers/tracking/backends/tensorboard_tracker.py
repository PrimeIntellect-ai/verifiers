# ABOUTME: TensorBoard-based tracker implementation for experiment tracking.
# ABOUTME: Logs metrics, text, and hyperparameters to TensorBoard event files.

from typing import Any, Optional

from verifiers.tracking.tracker import Tracker


class TensorBoardTracker(Tracker):
    """TensorBoard-based tracker for experiment tracking."""

    def __init__(
        self,
        log_dir: str = "./runs",
        comment: str = "",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.log_dir = log_dir
        self.comment = comment
        self._writer = None
        self._hparams_logged = False

    def init(self, **kwargs) -> None:
        from torch.utils.tensorboard import SummaryWriter

        self._writer = SummaryWriter(log_dir=self.log_dir, comment=self.comment)
        self._initialized = True

        if self.config:
            self._log_hparams(self.config)

    def _log_hparams(self, hparams: dict[str, Any]) -> None:
        if self._writer is None or self._hparams_logged:
            return

        try:
            hparam_dict = {}
            for key, value in hparams.items():
                if isinstance(value, (int, float, str, bool)):
                    hparam_dict[key] = value
                else:
                    hparam_dict[key] = str(value)

            metric_dict = {}
            self._writer.add_hparams(hparam_dict, metric_dict)
            self._hparams_logged = True
        except Exception as e:
            self.logger.warning(f"Failed to log hyperparameters: {e}")

    def log_metrics(
        self, metrics: dict[str, float], step: Optional[int] = None, **kwargs
    ) -> None:
        if self._writer is None:
            return

        for key, value in metrics.items():
            self._writer.add_scalar(key, value, global_step=step)

    def log_table(
        self,
        table_name: str,
        data: dict[str, list[Any]],
        step: Optional[int] = None,
        **kwargs,
    ) -> None:
        if self._writer is None:
            return

        try:
            import pandas as pd

            df = pd.DataFrame(data)
            text = df.to_markdown() if hasattr(df, "to_markdown") else df.to_string()
            self._writer.add_text(table_name, text, global_step=step)
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
        self._log_hparams(config)

    def finish(self, **kwargs) -> None:
        if self._writer is not None:
            self._writer.flush()
            self._writer.close()
