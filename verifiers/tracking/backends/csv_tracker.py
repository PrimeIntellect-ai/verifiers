# ABOUTME: CSV-based tracker implementation for local experiment tracking.
# ABOUTME: Writes metrics and tables to CSV files in specified directory.

import csv
import json
from pathlib import Path
from typing import Any, Optional

from verifiers.tracking.tracker import Tracker


class CSVTracker(Tracker):
    """CSV-based tracker for local experiment tracking."""

    def __init__(
        self,
        log_dir: str = "./logs",
        project: Optional[str] = None,
        name: Optional[str] = None,
        config: Optional[dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(project=project, name=name, config=config, **kwargs)
        self.log_dir = Path(log_dir)
        self._metrics_file = None
        self._metrics_writer = None
        self._metrics_fieldnames = None

    def init(self, **kwargs) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)

        if self.config:
            config_file = self.log_dir / "config.json"
            with open(config_file, "w") as f:
                json.dump(self.config, f, indent=2)

    def log_metrics(
        self, metrics: dict[str, float], step: Optional[int] = None, **kwargs
    ) -> None:
        metrics_file = self.log_dir / "metrics.csv"

        row = {"step": step} if step is not None else {}
        row.update(metrics)

        file_exists = metrics_file.exists()
        with open(metrics_file, "a", newline="") as f:
            if self._metrics_fieldnames is None:
                self._metrics_fieldnames = list(row.keys())
                writer = csv.DictWriter(f, fieldnames=self._metrics_fieldnames)
                if not file_exists:
                    writer.writeheader()
            else:
                for key in row.keys():
                    if key not in self._metrics_fieldnames:
                        self._metrics_fieldnames.append(key)
                writer = csv.DictWriter(f, fieldnames=self._metrics_fieldnames)

            writer.writerow(row)

    def log_table(
        self,
        table_name: str,
        data: dict[str, list[Any]],
        step: Optional[int] = None,
        **kwargs,
    ) -> None:
        table_file = self.log_dir / f"{table_name}.csv"

        if not data:
            return

        lengths = [len(v) for v in data.values()]
        if len(set(lengths)) > 1:
            self.logger.warning(f"Inconsistent column lengths in table {table_name}")
            return

        num_rows = lengths[0] if lengths else 0
        rows = []
        for i in range(num_rows):
            row = {}
            for key, values in data.items():
                value = values[i]
                if isinstance(value, (list, dict)):
                    row[key] = json.dumps(value)
                else:
                    row[key] = str(value)
            rows.append(row)

        file_exists = table_file.exists()
        fieldnames = list(data.keys())

        with open(table_file, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerows(rows)

    def log_config(self, config: dict[str, Any], **kwargs) -> None:
        super().log_config(config)
        config_file = self.log_dir / "config.json"
        with open(config_file, "w") as f:
            json.dump(self.config, f, indent=2)

    def finish(self, **kwargs) -> None:
        self.logger.info(f"CSVTracker: Logs saved to {self.log_dir}")
