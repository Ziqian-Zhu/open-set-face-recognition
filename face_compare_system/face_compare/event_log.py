"""Privacy-conscious, machine-readable JSON Lines event logging."""

from __future__ import annotations

import json
import threading
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class JsonlEventLogger:
    """Append operational events without storing face images or features."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.last_error = None

    def write(self, event_type: str, **fields: Any) -> None:
        """Append one UTF-8 JSON record atomically within this process."""

        record = {
            "timestamp": datetime.now(timezone.utc).astimezone().isoformat(timespec="milliseconds"),
            "event": event_type,
            **fields,
        }
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        try:
            with self._lock, self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
            self.last_error = None
        except OSError as error:
            # A committed enrollment must not look failed just because logging failed.
            if self.last_error is None:
                warnings.warn(f"事件日志暂时不可写：{error}", RuntimeWarning)
            self.last_error = str(error)
